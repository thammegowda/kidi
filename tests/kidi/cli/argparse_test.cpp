#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "kidi/cli/argparse.h"

namespace {

auto configure(kidi::cli::ArgumentParser& parser) -> void {
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
    predict.add_argument("--option").action(kidi::cli::Action::APPEND).metavar("KEY=VALUE");
    predict.add_argument("-j", "--threads").type<std::int32_t>().metavar("N");
    predict.add_argument("-i", "--in").dest("input").default_value(std::string("-")).metavar("FILE");
    predict.add_argument("-o", "--out").dest("output").default_value(std::string("-")).metavar("FILE");
    predict.add_argument("-it", "--inp-type").choices({"text", "jsonl"}).default_value(std::string("text"));
    predict.add_argument("-m", "--model").type<std::filesystem::path>().required().metavar("DIR");
}

auto contains(std::string_view value, std::string_view part) -> bool {
    return value.find(part) != std::string_view::npos;
}

class ArgparseTest : public ::testing::Test {
protected:
    void SetUp() override { configure(parser); }

    kidi::cli::ArgumentParser parser{"kidi", "small inference toolkit"};
};

TEST_F(ArgparseTest, ParsesTypedSubcommand) {
    constexpr std::array<std::string_view, 20> ARGUMENTS = {
        "predict",
        "--beam-size=3",
        "--score",
        "--stats",
        "--profile",
        "-i",
        "input.txt",
        "-o=output.txt",
        "-it",
        "jsonl",
        "-m",
        "model",
        "-a",
        "0.7",
        "-j",
        "3",
        "--option=gender=female",
        "--option",
        "age=young adult",
        "--option=pitch=high",
    };
    const auto arguments = parser.parse_args(ARGUMENTS);
    EXPECT_EQ(arguments.get<std::string>("command"), "predict") << "typed subcommand parse failed";
    EXPECT_EQ(arguments.get<std::int32_t>("beam_size"), 3) << "typed subcommand parse failed";
    EXPECT_TRUE(arguments.get<bool>("score")) << "typed subcommand parse failed";
    EXPECT_TRUE(arguments.get<bool>("stats")) << "typed subcommand parse failed";
    EXPECT_TRUE(arguments.get<bool>("profile")) << "typed subcommand parse failed";
    EXPECT_EQ(arguments.get<std::int32_t>("threads"), 3) << "typed subcommand parse failed";
    EXPECT_EQ(arguments.get<std::filesystem::path>("model"), "model") << "typed subcommand parse failed";
    EXPECT_EQ(arguments.get<std::string>("input"), "input.txt") << "typed subcommand parse failed";
    EXPECT_EQ(arguments.get<std::string>("output"), "output.txt") << "typed subcommand parse failed";
    EXPECT_EQ(arguments.get<std::string>("inp_type"), "jsonl") << "typed subcommand parse failed";
    EXPECT_EQ(arguments.get<float>("length_penalty"), 0.7F) << "typed subcommand parse failed";
    EXPECT_EQ(arguments.get<std::vector<std::string>>("option"),
              (std::vector<std::string>{"gender=female", "age=young adult", "pitch=high"}))
        << "typed subcommand parse failed";
}

TEST_F(ArgparseTest, PrintsSubcommandHelp) {
    constexpr std::array<std::string_view, 2> HELP_ARGUMENTS = {"predict", "--help"};
    const auto help = parser.parse_args(HELP_ARGUMENTS);
    EXPECT_EQ(help.status(), kidi::cli::ParseStatus::HELP) << "subcommand help failed";
    for (const auto* part : {"usage: kidi predict", "--beam-size N", "--stats", "--profile", "--option KEY=VALUE",
                             "--threads N", "--in FILE", "--out FILE", "--inp-type {text,jsonl}", "--model DIR"})
        EXPECT_TRUE(contains(help.output(), part)) << "subcommand help failed: missing " << part;
}

TEST_F(ArgparseTest, AppliesPredictDefaults) {
    constexpr std::array<std::string_view, 3> DEFAULT_ARGUMENTS = {"predict", "-m", "model"};
    const auto defaults = parser.parse_args(DEFAULT_ARGUMENTS);
    EXPECT_EQ(defaults.get<std::string>("input"), "-") << "predict defaults failed";
    EXPECT_EQ(defaults.get<std::string>("output"), "-") << "predict defaults failed";
    EXPECT_EQ(defaults.get<std::string>("inp_type"), "text") << "predict defaults failed";
    EXPECT_FALSE(defaults.get<bool>("score")) << "predict defaults failed";
    EXPECT_FALSE(defaults.get<bool>("stats")) << "predict defaults failed";
    EXPECT_FALSE(defaults.get<bool>("profile")) << "predict defaults failed";
    EXPECT_TRUE(defaults.get<std::vector<std::string>>("option").empty()) << "predict defaults failed";
    EXPECT_FALSE(defaults.contains("threads")) << "predict defaults failed";
}

TEST_F(ArgparseTest, AppliesDefaultValue) {
    constexpr std::array<std::string_view, 3> INSPECT_ARGUMENTS = {"inspect", "--model", "model"};
    const auto inspect = parser.parse_args(INSPECT_ARGUMENTS);
    EXPECT_EQ(inspect.get<std::string>("format"), "text") << "default value failed";
}

TEST_F(ArgparseTest, RejectsInvalidInteger) {
    constexpr std::array<std::string_view, 2> INVALID_ARGUMENTS = {"predict", "--beam-size=nope"};
    try {
        static_cast<void>(parser.parse_args(INVALID_ARGUMENTS));
        FAIL() << "invalid integer was accepted";
    } catch (const kidi::cli::ParseError& error) {
        EXPECT_TRUE(contains(error.what(), "expected an integer")) << "invalid integer diagnostic failed";
        EXPECT_TRUE(contains(error.usage(), "kidi predict")) << "invalid integer diagnostic failed";
    }
}

TEST_F(ArgparseTest, RejectsMissingRequired) {
    constexpr std::array<std::string_view, 1> INCOMPLETE_ARGUMENTS = {"predict"};
    try {
        static_cast<void>(parser.parse_args(INCOMPLETE_ARGUMENTS));
        FAIL() << "missing positional was accepted";
    } catch (const kidi::cli::ParseError& error) {
        EXPECT_TRUE(contains(error.what(), "--model")) << "missing positional diagnostic failed";
    }
}

} // namespace
