#include <cstddef>
#include <cstdint>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include <spdlog/cfg/env.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "kidi/cli/argparse.h"
#include "kidi/core/version.h"
#include "kidi/model/package.h"
#include "kidi/model/config.h"
#include "kidi/inference/translator.h"
#include "kidi/inference/generator.h"
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
    const auto directory = arguments.get<std::filesystem::path>("model");
    auto document = kidi::model::load_config(directory / "model.yaml");
    if (!document) {
        spdlog::error("{}", document.error().message);
        return 1;
    }
    if ((*document)["model"]["type"].as<std::string>() == "gemma4_text") {
        auto validation = kidi::model::Gemma4Impl::validate_config((*document)["model"]);
        if (!validation) {
            spdlog::error("{}", validation.error().message);
            return 1;
        }
        auto weights = kidi::model::Weights::load((*document)["weights_file"].as<std::string>());
        if (!weights) {
            spdlog::error("{}", weights.error().message);
            return 1;
        }
        const bool qat = static_cast<bool>((*document)["model"]["quantization_config"]);
        const auto embedding = weights->tensor(qat ? "model.language_model.embed_tokens.embedding_quantized"
                                                   : "model.language_model.embed_tokens.weight");
        if (!embedding) {
            spdlog::error("{}", embedding.error().message);
            return 1;
        }
        std::cout << "format: " << (*document)["format_version"].as<int>() << '\n'
                  << "model: gemma4_text\n"
                  << "weights: " << (*document)["weights_file"].as<std::string>() << '\n'
                  << "checkpoint tensors (including unused modalities): " << weights->size() << '\n'
                  << "weight dtype: " << kidi::tensor::to_string(embedding->dtype()) << '\n'
                  << "native mobile QAT: " << (qat ? "yes" : "no") << '\n'
                  << "text layers: " << (*document)["model"]["num_hidden_layers"].as<int>() << '\n'
                  << "vocabulary: " << (*document)["model"]["vocab_size"].as<int>() << '\n'
                  << "default device: " << kidi::tensor::to_string(kidi::module_device) << '\n';
        return 0;
    }
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

auto generate_lines(kidi::inference::Generator& generator, const kidi::cli::Namespace& arguments,
                    kidi::inference::GenerationOptions options) -> int {
    try {
        const kidi::inference::ServingOptions limits{.maximum_active = arguments.get<std::size_t>("max_active"),
                                                     .maximum_requests = arguments.get<std::size_t>("queue_size"),
                                                     .cache_token_budget = arguments.get<std::size_t>("cache_tokens"),
                                                     .prefill_tokens_per_step = options.prefill_chunk_size};
        kidi::ops::require(generator.configure_serving(limits));
        const auto& input_path = arguments.get<std::string>("input_lines");
        std::ifstream file;
        std::istream* input = &std::cin;
        if (input_path != "-") {
            file.open(input_path);
            if (!file) throw std::runtime_error("cannot open prompt lines: " + input_path);
            input = &file;
        }
        bool ended = false;
        std::uint64_t prefill_ns = 0, decode_ns = 0, preparation_ns = 0;
        std::size_t completed = 0, peak_cache_tokens = 0;
        const auto started = std::chrono::steady_clock::now();
        while (!ended || generator.pending_requests()) {
            while (!ended && generator.pending_requests() < limits.maximum_requests) {
                std::string prompt;
                if (!std::getline(*input, prompt)) {
                    if (!input->eof()) throw std::runtime_error("failed reading prompt lines");
                    ended = true;
                    break;
                }
                kidi::ops::require(generator.enqueue(prompt, options));
            }
            if (!generator.pending_requests()) break;
            const auto result = kidi::ops::require(generator.step());
            prefill_ns += result.prefill_ns;
            decode_ns += result.decode_ns;
            preparation_ns += result.preparation_ns;
            peak_cache_tokens = std::max(peak_cache_tokens, result.reserved_cache_tokens);
            for (const auto& event : result.events) {
                if (!event.completed) continue;
                const auto& output = *event.completed;
                nlohmann::json record{{"request_id", event.request_id},
                                      {"text", output.text},
                                      {"token_ids", output.generation.token_ids},
                                      {"decoder_steps", output.generation.decoder_steps}};
                if (arguments.get<bool>("profile")) {
                    record["ttft_ns"] = output.stats.time_to_first_token_ns;
                    record["generation_ns"] = output.stats.generation_ns;
                    record["prompt_tokens"] = output.stats.prompt_tokens;
                }
                std::cout << record.dump() << '\n';
                ++completed;
            }
            std::cout.flush();
            if (!std::cout) throw std::runtime_error("failed writing generation results");
        }
        if (arguments.get<bool>("profile")) {
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started)
                    .count();
            std::cerr << "kidi_serving|requests=" << completed << "|max_active=" << limits.maximum_active
                      << "|cache_tokens=" << limits.cache_token_budget
                      << "|peak_reserved_cache_tokens=" << peak_cache_tokens << "|prefill_ns=" << prefill_ns
                      << "|decode_ns=" << decode_ns << "|preparation_ns=" << preparation_ns
                      << "|generation_ns=" << elapsed << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        spdlog::error("{}", error.what());
        return 1;
    }
}
auto generate(const kidi::cli::Namespace& arguments) -> int {
    const auto threads = arguments.get<std::int32_t>("threads");
    const auto runs = arguments.get<std::int32_t>("runs");
    const auto warmups = arguments.get<std::int32_t>("warmups");
    if (threads <= 0 || runs <= 0 || warmups < 0) {
        spdlog::error("threads and runs must be positive; warmups must be non-negative");
        return 2;
    }
    const auto& input_lines = arguments.get<std::string>("input_lines");
    if (!input_lines.empty() && (!arguments.get<std::string>("prompt").empty() || runs != 1 || warmups ||
                                 arguments.get<std::size_t>("prefix_cache_bytes"))) {
        spdlog::error("line generation cannot combine --prompt, repeats, warmups or prefix caching");
        return 2;
    }
    kidi::runtime::ynn::set_thread_count(threads);
    const auto backend = inference_backend(arguments);
    const auto device = backend == kidi::inference::InferenceBackend::MPS ? kidi::tensor::Device::apple_gpu()
                                                                          : kidi::tensor::Device::cpu();
    const auto started = std::chrono::steady_clock::now();
    const auto weight_bits = arguments.get<std::int32_t>("weight_bits");
    const auto group_size = arguments.get<std::int32_t>("group_size");
    const auto packed_prefill = arguments.get<bool>("packed_prefill");
    auto generator = kidi::inference::Generator::load(arguments.get<std::filesystem::path>("model"), device,
                                                      weight_bits, group_size, packed_prefill);
    if (!generator) {
        spdlog::error("{}", generator.error().message);
        return 1;
    }
    const auto load_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count();
    auto prompt = arguments.get<std::string>("prompt");
    if (prompt.empty() && input_lines.empty())
        prompt.assign(std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>());
    const kidi::inference::GenerationOptions options{
        .maximum_new_tokens = arguments.get<std::size_t>("max_new_tokens"),
        .context_size = arguments.get<std::size_t>("context_size"),
        .prefill_chunk_size = arguments.get<std::size_t>("prefill_chunk_size"),
        .prefix_cache_bytes = arguments.get<std::size_t>("prefix_cache_bytes"),
        .raw_prompt = arguments.get<bool>("raw_prompt"),
        .ignore_eos = arguments.get<bool>("ignore_eos"),
        .full_attention_cache = arguments.get<bool>("full_attention_cache")};
    if (!input_lines.empty()) return generate_lines(*generator, arguments, options);
    for (std::int32_t run = -warmups; run < runs; ++run) {
        auto result = generator->generate(prompt, options);
        if (!result) {
            spdlog::error("{}", result.error().message);
            return 1;
        }
        if (run < 0) continue;
        std::cout << result->text << '\n';
        if (arguments.get<bool>("profile")) {
            const auto& stats = result->stats;
            std::cerr << "kidi_generation|backend=" << kidi::inference::to_string(backend)
                      << "|native_qat=" << generator->native_qat() << "|weight_bits=" << weight_bits
                      << "|device_selection=" << stats.device_selection
                      << "|last_token_prefill=" << stats.last_token_prefill << "|group_size=" << group_size
                      << "|shared_prefill_tail=" << stats.shared_prefill_tail << "|packed_prefill=" << packed_prefill
                      << "|crop_local_attention="
                      << (device == kidi::tensor::Device::cpu() && !options.full_attention_cache)
                      << "|batch_size=1|threads=" << threads << "|run=" << run << "|load_ns=" << load_ns
                      << "|prompt_tokens=" << stats.prompt_tokens
                      << "|reused_prompt_tokens=" << stats.reused_prompt_tokens
                      << "|prefix_cache_bytes=" << stats.prefix_cache_bytes
                      << "|prefix_reserved_bytes=" << stats.prefix_reserved_bytes
                      << "|generated_tokens=" << result->generation.decoder_steps
                      << "|decode_tokens=" << stats.decode_tokens << "|tokenize_ns=" << stats.tokenize_ns
                      << "|prefill_ns=" << stats.prefill_ns << "|decode_ns=" << stats.decode_ns
                      << "|preparation_ns=" << stats.preparation_ns << "|ttft_ns=" << stats.time_to_first_token_ns
                      << "|generation_ns=" << stats.generation_ns << "|token_ids=";
            for (auto token : result->generation.token_ids) std::cerr << token << ',';
            std::cerr << '\n';
        }
    }
    return std::cout ? 0 : 1;
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
    kidi::cli::ArgumentParser parser("kidi", "Run translation and text generation models.");
    parser.version("kidi " + std::string(kidi::version()));
    auto& commands = parser.add_subparsers().required();
    auto& generate_parser = commands.add_parser("generate", "generate text with Gemma 4");
    generate_parser.add_argument("-m", "--model").type<std::filesystem::path>().required().metavar("DIR");
    generate_parser.add_argument("--prompt").default_value(std::string{}).help("prompt text; stdin when omitted");
    generate_parser.add_argument("--input-lines")
        .dest("input_lines")
        .default_value(std::string{})
        .help("finite prompt file, one per line (- for stdin); emit completion-order JSONL with 1-based request IDs");
    generate_parser.add_argument("--max-active").dest("max_active").default_value<std::size_t>(4);
    generate_parser.add_argument("--queue-size").dest("queue_size").default_value<std::size_t>(64);
    generate_parser.add_argument("--cache-tokens")
        .dest("cache_tokens")
        .default_value<std::size_t>(8192)
        .help("sum of reserved dense context capacities for line generation");
    generate_parser.add_argument("--backend").default_value(std::string("auto")).choices({"auto", "ynnpack", "mps"});
    generate_parser.add_argument("-j", "--threads").default_value<std::int32_t>(4);
    generate_parser.add_argument("--max-new-tokens").dest("max_new_tokens").default_value<std::size_t>(0);
    generate_parser.add_argument("--context-size").dest("context_size").default_value<std::size_t>(0);
    generate_parser.add_argument("--prefill-chunk-size").dest("prefill_chunk_size").default_value<std::size_t>(128);
    generate_parser.add_argument("--prefix-cache-bytes")
        .dest("prefix_cache_bytes")
        .default_value<std::size_t>(0)
        .help("opt-in one-entry prefix snapshot budget; 0 disables cross-request reuse");
    generate_parser.add_argument("--raw-prompt").dest("raw_prompt").action(kidi::cli::Action::STORE_TRUE);
    generate_parser.add_argument("--ignore-eos").dest("ignore_eos").action(kidi::cli::Action::STORE_TRUE);
    generate_parser.add_argument("--full-attention-cache")
        .dest("full_attention_cache")
        .action(kidi::cli::Action::STORE_TRUE)
        .help("disable CPU local-history cropping for the previous numerical path");
    generate_parser.add_argument("--profile").action(kidi::cli::Action::STORE_TRUE);
    generate_parser.add_argument("--weight-bits")
        .dest("weight_bits")
        .default_value<std::int32_t>(0)
        .help("0: original weights; 4 or 8: load-time groupwise packing");
    generate_parser.add_argument("--group-size").dest("group_size").default_value<std::int32_t>(128);
    generate_parser.add_argument("--packed-prefill")
        .dest("packed_prefill")
        .action(kidi::cli::Action::STORE_TRUE)
        .help("use packed GEMM for prefill; default may cache FP16 matrices for native QAT");
    generate_parser.add_argument("--runs").default_value<std::int32_t>(1);
    generate_parser.add_argument("--warmups").default_value<std::int32_t>(0);

    auto& inspect_parser = commands.add_parser("inspect", "inspect a model package");
    inspect_parser.description("Inspect an RTG or Gemma 4 model package.");
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
        if (command == "generate") return generate(arguments);
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