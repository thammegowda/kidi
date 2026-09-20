#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <spdlog/cfg/env.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "kidi/cli/argparse.h"
#include "kidi/core/version.h"
#include "kidi/model/package.h"
#include "kidi/inference/translator.h"
#include "kidi/runtime/ynn/graph.h"
#include "kidi/tensor/backend.h"

namespace {

auto inference_backend(const kidi::cli::Namespace& arguments) -> kidi::inference::InferenceBackend {
    const auto& backend = arguments.get<std::string>("backend");
    return backend == "mps" || (backend == "auto" && kidi::module_device == kidi::tensor::Device::apple_gpu())
               ? kidi::inference::InferenceBackend::MPS
               : kidi::inference::InferenceBackend::YNNPACK;
}

auto inspect(const kidi::cli::Namespace& arguments) -> int {
    auto package = kidi::model::Package::load(arguments.get<std::filesystem::path>("model"));
    if (!package) {
        spdlog::error("{}", package.error().message);
        return 1;
    }
    const auto& config = package->config();
    auto validation = kidi::model::TransformerImpl::validate_config(config["model"]);
    if (!validation) {
        spdlog::error("{}", validation.error().message);
        return 1;
    }
    std::cout << "format: " << config["format_version"].as<int>() << '\n'
              << "model: " << config["model"]["type"].as<std::string>() << '\n'
              << "weights: " << config["weights_file"].as<std::string>() << '\n'
              << "weight tensors: " << package->weights().size() << '\n'
              << "vocabularies: " << package->source_tokenizer().vocabulary_size() << ", "
              << package->target_tokenizer().vocabulary_size() << '\n'
              << "beam: " << config["decode"]["beam_size"].as<int>() << '\n'
              << "default device: " << kidi::tensor::to_string(kidi::module_device) << '\n'
              << "cpu features: " << kidi::runtime::ynn::supported_arch_names() << '\n';
    for (const auto& backend : kidi::tensor::BackendRegistry::instance().backends()) {
        std::cout << "tensor backend: " << kidi::tensor::to_string(backend.device_kind) << ' ' << backend.name
                  << " storage=" << (backend.storage_available ? "yes" : "no")
                  << " execution=" << (backend.execution_available ? "yes" : "no");
        if (!backend.storage_available) std::cout << " reason=" << backend.unavailable_reason;
        std::cout << '\n';
    }
    return 0;
}

auto predict(const kidi::cli::Namespace& arguments) -> int {
    const auto& input_type = arguments.get<std::string>("inp_type");
    if (input_type != "text") {
        spdlog::error("input type '{}' is not supported yet", input_type);
        return 2;
    }
    if (arguments.contains("threads")) {
        const auto threads = arguments.get<std::int32_t>("threads");
        if (threads <= 0) {
            spdlog::error("thread count must be positive");
            return 2;
        }
        kidi::runtime::ynn::set_thread_count(static_cast<std::size_t>(threads));
    }

    const auto& input_path = arguments.get<std::string>("input");
    std::ifstream input_file;
    std::istream* input = &std::cin;
    if (input_path != "-") {
        input_file.open(input_path);
        if (!input_file) {
            spdlog::error("cannot open input file: {}", input_path);
            return 1;
        }
        input = &input_file;
    }

    const auto& output_path = arguments.get<std::string>("output");
    std::ofstream output_file;
    std::ostream* output = &std::cout;
    if (output_path != "-") {
        output_file.open(output_path);
        if (!output_file) {
            spdlog::error("cannot open output file: {}", output_path);
            return 1;
        }
        output = &output_file;
    }

    kidi::inference::InferenceStats profile;
    const auto batch_size = arguments.get<std::int32_t>("batch_size");
    const auto batch_window =
        arguments.contains("batch_window") ? arguments.get<std::int32_t>("batch_window") : batch_size;
    if (batch_size < 1 || batch_size > 256) {
        spdlog::error("batch size must be between 1 and 256");
        return 2;
    }
    if (batch_window < batch_size || batch_window > 4096) {
        spdlog::error("batch window must be between batch-size and 4096");
        return 2;
    }
    auto* profile_ptr = arguments.get<bool>("profile") ? &profile : nullptr;
    const auto backend = inference_backend(arguments);
    auto translator = kidi::inference::Translator::load(arguments.get<std::filesystem::path>("model"), backend,
                                                        profile_ptr, batch_size);
    if (!translator) {
        spdlog::error("{}", translator.error().message);
        return 1;
    }

    kidi::inference::DecodeOptions options;
    if (arguments.contains("beam_size")) options.beam_size = arguments.get<std::int32_t>("beam_size");
    if (arguments.contains("maximum_extra_tokens")) {
        options.maximum_extra_tokens = arguments.get<std::int32_t>("maximum_extra_tokens");
    }
    if (arguments.contains("length_penalty")) options.length_penalty = arguments.get<float>("length_penalty");
    options.compute_score = arguments.get<bool>("score");

    std::size_t line_number = 0;
    std::size_t translated_items = 0;
    std::size_t target_tokens = 0;
    std::vector<std::string> pending;
    const auto flush = [&]() -> bool {
        std::vector<std::string> sources;
        for (const auto& line : pending)
            if (!line.empty()) sources.push_back(line);
        auto translations = translator->translate_batch(sources, options, profile_ptr);
        if (!translations) {
            spdlog::error("batch ending at input line {}: {}", line_number, translations.error().message);
            return false;
        }
        std::size_t index = 0;
        for (const auto& line : pending) {
            if (!line.empty()) {
                const auto& translation = (*translations)[index++];
                *output << translation.text;
                if (arguments.get<bool>("score")) *output << '\t' << translation.score;
                ++translated_items;
                target_tokens += translation.token_ids.size();
            }
            *output << '\n';
        }
        output->flush();
        pending.clear();
        if (!*output) {
            spdlog::error("cannot write output: {}", output_path);
            return false;
        }
        return true;
    };
    for (std::string line; std::getline(*input, line);) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        pending.push_back(std::move(line));
        if (pending.size() == static_cast<std::size_t>(batch_window) && !flush()) return 1;
    }
    if (!pending.empty() && !flush()) return 1;
    if (input->bad()) {
        spdlog::error("cannot read input: {}", input_path);
        return 1;
    }
    if (arguments.get<bool>("stats")) {
        std::cerr << "kidi_metrics|input_lines=" << line_number << "|translated_items=" << translated_items
                  << "|target_tokens=" << target_tokens << '\n';
    }
    if (profile_ptr) {
        std::cerr << "kidi_profile|backend=" << kidi::inference::to_string(backend)
                  << "|threads=" << kidi::runtime::ynn::thread_count()
                  << "|ynn_arch_flags=" << kidi::runtime::ynn::supported_arch_flags()
                  << "|ynn_arch=" << kidi::runtime::ynn::supported_arch_names()
                  << "|package_load_ns=" << profile.package_load_ns << "|graph_compile_ns=" << profile.graph_compile_ns
                  << "|source_tokenize_ns=" << profile.source_tokenize_ns << "|encoder_ns=" << profile.encoder_ns
                  << "|source_embedding_ns=" << profile.source_embedding_ns << "|decoder_ns=" << profile.decoder_ns
                  << "|source_projection_ns=" << profile.source_projection_ns
                  << "|target_embedding_ns=" << profile.target_embedding_ns
                  << "|host_search_ns=" << profile.host_search_ns << "|target_decode_ns=" << profile.target_decode_ns
                  << "|translate_ns=" << profile.translate_ns << "|decoder_steps=" << profile.decoder_steps << '\n';
    }
    return 0;
}

} // namespace

auto main(int argc, char** argv) -> int {
    spdlog::set_default_logger(spdlog::stderr_color_mt("kidi"));
    spdlog::set_pattern("[%n] [%l] %v");
    spdlog::cfg::load_env_levels();
    kidi::cli::ArgumentParser parser("kidi", "Run optimized RTG translation models.");
    parser.version("kidi " + std::string(kidi::version()));
    auto& commands = parser.add_subparsers().required();

    auto& inspect_parser = commands.add_parser("inspect", "inspect a model package");
    inspect_parser.description("Inspect an RTG model package.");
    inspect_parser.add_argument("-m", "--model")
        .type<std::filesystem::path>()
        .required()
        .metavar("DIR")
        .help("model package directory");

    auto& predict_parser = commands.add_parser("predict", "translate Moses-tokenized text");
    predict_parser.description("Translate Moses-tokenized text, one sentence per line.");
    predict_parser.add_argument("-b", "--beam-size").type<std::int32_t>().metavar("N").help("beam size");
    predict_parser.add_argument("-x", "--max-extra-tokens")
        .dest("maximum_extra_tokens")
        .type<std::int32_t>()
        .metavar("N")
        .help("maximum target tokens beyond source length");
    predict_parser.add_argument("-a", "--length-penalty")
        .type<float>()
        .metavar("ALPHA")
        .help("Wu length penalty exponent");
    predict_parser.add_argument("--score").action(kidi::cli::Action::STORE_TRUE).help("append hypothesis score");
    predict_parser.add_argument("--stats")
        .action(kidi::cli::Action::STORE_TRUE)
        .help("emit machine-readable generation stats to stderr");
    predict_parser.add_argument("--profile")
        .action(kidi::cli::Action::STORE_TRUE)
        .help("emit machine-readable inference timings to stderr");
    predict_parser.add_argument("--backend")
        .choices({"auto", "ynnpack", "mps"})
        .default_value(std::string("auto"))
        .help("inference backend (auto prefers an available GPU)");
    predict_parser.add_argument("--batch-size")
        .type<std::int32_t>()
        .default_value(std::int32_t{1})
        .metavar("N")
        .help("independent sentences per batch (1 to 256; CPU executes serially)");
    predict_parser.add_argument("--batch-window")
        .type<std::int32_t>()
        .metavar("N")
        .help("bounded input window for length grouping (default: batch-size; maximum 4096)");
    predict_parser.add_argument("-j", "--threads")
        .type<std::int32_t>()
        .metavar("N")
        .help("total YNNPACK threads including the caller");
    predict_parser.add_argument("-i", "--in")
        .dest("input")
        .default_value(std::string("-"))
        .metavar("FILE")
        .help("input file, or '-' for stdin");
    predict_parser.add_argument("-o", "--out")
        .dest("output")
        .default_value(std::string("-"))
        .metavar("FILE")
        .help("output file, or '-' for stdout");
    predict_parser.add_argument("-it", "--inp-type")
        .choices({"text", "jsonl"})
        .default_value(std::string("text"))
        .help("input record format");
    predict_parser.add_argument("-m", "--model")
        .type<std::filesystem::path>()
        .required()
        .metavar("DIR")
        .help("model package directory");

    try {
        const auto arguments = parser.parse_args(argc, argv);
        if (arguments.should_exit()) {
            std::cout << arguments.output();
            return 0;
        }
        const auto& command = arguments.get<std::string>("command");
        if (command == "inspect") return inspect(arguments);
        if (command == "predict") return predict(arguments);
        throw std::logic_error("unhandled command: " + command);
    } catch (const kidi::cli::ParseError& error) {
        std::cerr << error.usage();
        spdlog::error("{}", error.what());
        return 2;
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return 1;
    }
}