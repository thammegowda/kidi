#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "kidi/cli/argparse.h"
#include "kidi/core/version.h"
#include "kidi/rtg/package.h"
#include "kidi/rtg/translator.h"
#include "kidi/runtime/ynn.h"

namespace {

int inspect(const kidi::cli::Namespace& arguments) {
    auto package = kidi::rtg::Package::load(arguments.get<std::filesystem::path>("model"));
    if (!package) {
        std::cerr << "kidi: " << package.error().message << '\n';
        return 1;
    }
    const auto& manifest = package->manifest();
    std::cout << "format: " << manifest.format_version << '\n'
              << "model: " << manifest.model_type << '\n'
              << "weights: " << manifest.weights_file.string() << '\n'
              << "weight encoding: " << kidi::model::to_string(manifest.weights.encoding) << '\n'
              << "weight tensors: " << package->weights().size() << '\n'
              << "vocabularies: " << package->source_tokenizer().vocabulary_size() << ", "
              << package->target_tokenizer().vocabulary_size() << '\n'
              << "beam: " << manifest.decode_defaults.beam_size << '\n'
              << "runtime: ynnpack_cpu\n"
              << "cpu features: " << kidi::runtime::ynn_supported_arch_names() << '\n';
    return 0;
}

int predict(const kidi::cli::Namespace& arguments) {
    const auto& input_type = arguments.get<std::string>("inp_type");
    if (input_type != "text") {
        std::cerr << "kidi: input type '" << input_type << "' is not supported yet\n";
        return 2;
    }
    if (arguments.contains("threads")) {
        const auto threads = arguments.get<std::int32_t>("threads");
        if (threads <= 0) {
            std::cerr << "kidi: thread count must be positive\n";
            return 2;
        }
        kidi::runtime::set_ynn_thread_count(static_cast<std::size_t>(threads));
    }

    const auto& input_path = arguments.get<std::string>("input");
    std::ifstream input_file;
    std::istream* input = &std::cin;
    if (input_path != "-") {
        input_file.open(input_path);
        if (!input_file) {
            std::cerr << "kidi: cannot open input file: " << input_path << '\n';
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
            std::cerr << "kidi: cannot open output file: " << output_path << '\n';
            return 1;
        }
        output = &output_file;
    }

    kidi::rtg::InferenceStats profile;
    auto* profile_ptr = arguments.get<bool>("profile") ? &profile : nullptr;
    auto translator = kidi::rtg::Translator::load(arguments.get<std::filesystem::path>("model"), profile_ptr);
    if (!translator) {
        std::cerr << "kidi: " << translator.error().message << '\n';
        return 1;
    }

    kidi::rtg::DecodeOptions options;
    if (arguments.contains("beam_size")) options.beam_size = arguments.get<std::int32_t>("beam_size");
    if (arguments.contains("maximum_extra_tokens")) {
        options.maximum_extra_tokens = arguments.get<std::int32_t>("maximum_extra_tokens");
    }
    if (arguments.contains("length_penalty")) options.length_penalty = arguments.get<float>("length_penalty");
    options.compute_score = arguments.get<bool>("score");

    std::size_t line_number = 0;
    std::size_t translated_items = 0;
    std::size_t target_tokens = 0;
    for (std::string line; std::getline(*input, line);) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) {
            *output << '\n';
            continue;
        }
        auto translation = translator->translate(line, options, profile_ptr);
        if (!translation) {
            std::cerr << "kidi: input line " << line_number << ": " << translation.error().message << '\n';
            return 1;
        }
        *output << translation->text;
        if (arguments.get<bool>("score")) *output << '\t' << translation->score;
        *output << '\n';
        if (!*output) {
            std::cerr << "kidi: cannot write output: " << output_path << '\n';
            return 1;
        }
        ++translated_items;
        target_tokens += translation->token_ids.size();
    }
    if (input->bad()) {
        std::cerr << "kidi: cannot read input: " << input_path << '\n';
        return 1;
    }
    if (arguments.get<bool>("stats")) {
        std::cerr << "kidi_metrics|input_lines=" << line_number << "|translated_items=" << translated_items
                  << "|target_tokens=" << target_tokens << '\n';
    }
    if (profile_ptr) {
        std::cerr << "kidi_profile|backend=ynnpack_cpu|threads=" << kidi::runtime::ynn_thread_count()
                  << "|ynn_arch_flags=" << kidi::runtime::ynn_supported_arch_flags()
                  << "|ynn_arch=" << kidi::runtime::ynn_supported_arch_names()
                  << "|package_load_ns=" << profile.package_load_ns << "|graph_compile_ns=" << profile.graph_compile_ns
                  << "|source_tokenize_ns=" << profile.source_tokenize_ns << "|encoder_ns=" << profile.encoder_ns
                  << "|source_embedding_ns=" << profile.source_embedding_ns
                  << "|encoder_max_concurrency=" << profile.encoder_graph.max_concurrency
                  << "|encoder_prepare_ns=" << profile.encoder_graph.prepare_ns
                  << "|encoder_reshape_ns=" << profile.encoder_graph.reshape_ns
                  << "|encoder_bind_ns=" << profile.encoder_graph.bind_ns
                  << "|encoder_invoke_ns=" << profile.encoder_graph.invoke_ns << "|decoder_ns=" << profile.decoder_ns
                  << "|source_projection_ns=" << profile.source_projection_ns
                  << "|source_projection_max_concurrency=" << profile.source_projection_graph.max_concurrency
                  << "|source_projection_prepare_ns=" << profile.source_projection_graph.prepare_ns
                  << "|source_projection_reshape_ns=" << profile.source_projection_graph.reshape_ns
                  << "|source_projection_bind_ns=" << profile.source_projection_graph.bind_ns
                  << "|source_projection_invoke_ns=" << profile.source_projection_graph.invoke_ns
                  << "|target_embedding_ns=" << profile.target_embedding_ns
                  << "|decoder_max_concurrency=" << profile.decoder_graph.max_concurrency
                  << "|decoder_prepare_ns=" << profile.decoder_graph.prepare_ns
                  << "|decoder_reshape_ns=" << profile.decoder_graph.reshape_ns
                  << "|decoder_bind_ns=" << profile.decoder_graph.bind_ns
                  << "|decoder_invoke_ns=" << profile.decoder_graph.invoke_ns
                  << "|last_hidden_ns=" << profile.last_hidden_ns
                  << "|generator_max_concurrency=" << profile.generator_graph.max_concurrency
                  << "|generator_prepare_ns=" << profile.generator_graph.prepare_ns
                  << "|generator_reshape_ns=" << profile.generator_graph.reshape_ns
                  << "|generator_bind_ns=" << profile.generator_graph.bind_ns
                  << "|generator_invoke_ns=" << profile.generator_graph.invoke_ns
                  << "|host_search_ns=" << profile.host_search_ns << "|target_decode_ns=" << profile.target_decode_ns
                  << "|translate_ns=" << profile.translate_ns << "|decoder_steps=" << profile.decoder_steps << '\n';
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    kidi::cli::ArgumentParser parser("kidi", "Run RTG translation models with YNNPACK.");
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
        std::cerr << error.usage() << "kidi: error: " << error.what() << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "kidi: " << error.what() << '\n';
        return 1;
    }
}