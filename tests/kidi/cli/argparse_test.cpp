#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

#include "kidi/cli/argparse.h"

namespace {

void configure(kidi::cli::ArgumentParser& parser) {
    parser.version("kidi 1.2.3");
    auto& commands = parser.add_subparsers().required();

    auto& inspect = commands.add_parser("inspect", "inspect a model package");
    inspect.add_argument("-m", "--model").type<std::filesystem::path>().required().metavar("DIR");
    inspect.add_argument("--format").choices({"text", "json"}).default_value(std::string("text"));

    auto& predict = commands.add_parser("predict", "translate tokenized text");
    predict.add_argument("-b", "--beam-size").type<std::int32_t>().metavar("N");
    predict.add_argument("-a", "--length-penalty").default_value(0.6F).metavar("ALPHA");
    predict.add_argument("--score").action(kidi::cli::Action::STORE_TRUE);
    predict.add_argument("--stats").action(kidi::cli::Action::STORE_TRUE);
    predict.add_argument("--profile").action(kidi::cli::Action::STORE_TRUE);
    predict.add_argument("-j", "--threads").type<std::int32_t>().metavar("N");
    predict.add_argument("-i", "--in").dest("input").default_value(std::string("-")).metavar("FILE");
    predict.add_argument("-o", "--out").dest("output").default_value(std::string("-")).metavar("FILE");
    predict.add_argument("-it", "--inp-type").choices({"text", "jsonl"}).default_value(std::string("text"));
    predict.add_argument("-m", "--model").type<std::filesystem::path>().required().metavar("DIR");
}

bool contains(std::string_view value, std::string_view part) { return value.find(part) != std::string_view::npos; }

} // namespace

int main() {
    kidi::cli::ArgumentParser parser("kidi", "small inference toolkit");
    configure(parser);

    constexpr std::array<std::string_view, 16> ARGUMENTS = {
        "predict", "--beam-size=3", "--score", "--stats", "--profile", "-i",  "input.txt", "-o=output.txt",
        "-it",     "jsonl",         "-m",      "model",   "-a",        "0.7", "-j",        "3",
    };
    const auto arguments = parser.parse_args(ARGUMENTS);
    if (arguments.get<std::string>("command") != "predict" || arguments.get<std::int32_t>("beam_size") != 3 ||
        !arguments.get<bool>("score") || !arguments.get<bool>("stats") || !arguments.get<bool>("profile") ||
        arguments.get<std::int32_t>("threads") != 3 || arguments.get<std::filesystem::path>("model") != "model" ||
        arguments.get<std::string>("input") != "input.txt" || arguments.get<std::string>("output") != "output.txt" ||
        arguments.get<std::string>("inp_type") != "jsonl" || arguments.get<float>("length_penalty") != 0.7F) {
        std::cerr << "typed subcommand parse failed\n";
        return 1;
    }

    constexpr std::array<std::string_view, 2> HELP_ARGUMENTS = {"predict", "--help"};
    const auto help = parser.parse_args(HELP_ARGUMENTS);
    if (help.status() != kidi::cli::ParseStatus::HELP || !contains(help.output(), "usage: kidi predict") ||
        !contains(help.output(), "--beam-size N") || !contains(help.output(), "--stats") ||
        !contains(help.output(), "--profile") || !contains(help.output(), "--threads N") ||
        !contains(help.output(), "--in FILE") || !contains(help.output(), "--out FILE") ||
        !contains(help.output(), "--inp-type {text,jsonl}") || !contains(help.output(), "--model DIR")) {
        std::cerr << "subcommand help failed\n";
        return 1;
    }

    constexpr std::array<std::string_view, 3> DEFAULT_ARGUMENTS = {"predict", "-m", "model"};
    const auto defaults = parser.parse_args(DEFAULT_ARGUMENTS);
    if (defaults.get<std::string>("input") != "-" || defaults.get<std::string>("output") != "-" ||
        defaults.get<std::string>("inp_type") != "text" || defaults.get<bool>("score") || defaults.get<bool>("stats") ||
        defaults.get<bool>("profile") || defaults.contains("threads")) {
        std::cerr << "predict defaults failed\n";
        return 1;
    }

    constexpr std::array<std::string_view, 3> INSPECT_ARGUMENTS = {"inspect", "--model", "model"};
    const auto inspect = parser.parse_args(INSPECT_ARGUMENTS);
    if (inspect.get<std::string>("format") != "text") {
        std::cerr << "default value failed\n";
        return 1;
    }

    try {
        constexpr std::array<std::string_view, 2> INVALID_ARGUMENTS = {"predict", "--beam-size=nope"};
        static_cast<void>(parser.parse_args(INVALID_ARGUMENTS));
        std::cerr << "invalid integer was accepted\n";
        return 1;
    } catch (const kidi::cli::ParseError& error) {
        if (!contains(error.what(), "expected an integer") || !contains(error.usage(), "kidi predict")) {
            std::cerr << "invalid integer diagnostic failed\n";
            return 1;
        }
    }

    try {
        constexpr std::array<std::string_view, 1> INCOMPLETE_ARGUMENTS = {"predict"};
        static_cast<void>(parser.parse_args(INCOMPLETE_ARGUMENTS));
        std::cerr << "missing positional was accepted\n";
        return 1;
    } catch (const kidi::cli::ParseError& error) {
        if (!contains(error.what(), "--model")) {
            std::cerr << "missing positional diagnostic failed\n";
            return 1;
        }
    }

    return 0;
}