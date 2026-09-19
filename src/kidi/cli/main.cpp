#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "kidi/cli/argparse.h"
#include "kidi/core/version.h"
#include "kidi/rtg/package.h"
#include "kidi/rtg/translator.h"

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
              << "linear layout: " << kidi::model::to_string(manifest.weights.linear_layout) << '\n'
              << "weight tensors: " << package->weights().size() << '\n'
              << "vocabularies: " << package->source_tokenizer().vocabulary_size() << ", "
              << package->target_tokenizer().vocabulary_size() << '\n'
              << "beam: " << manifest.decode_defaults.beam_size << '\n';
    return 0;
}

int predict(const kidi::cli::Namespace& arguments) {
    const auto& input_type = arguments.get<std::string>("inp_type");
    if (input_type != "text") {
        std::cerr << "kidi: input type '" << input_type << "' is not supported yet\n";
        return 2;
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

    auto translator = kidi::rtg::Translator::load(arguments.get<std::filesystem::path>("model"));
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

    std::size_t line_number = 0;
    for (std::string line; std::getline(*input, line);) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) {
            *output << '\n';
            continue;
        }
        auto translation = translator->translate(line, options);
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
    }
    if (input->bad()) {
        std::cerr << "kidi: cannot read input: " << input_path << '\n';
        return 1;
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