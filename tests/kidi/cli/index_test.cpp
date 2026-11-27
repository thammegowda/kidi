#include "kidi/cli/main.h"
#include "kidi/inference/generator.h"
#include "kidi/ops/context.h"
#include "kidi/search/hnsw.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <thread>

#include <gtest/gtest.h>

namespace {
using kidi::ops::require;

auto run(std::vector<std::string> arguments, std::string* progress = nullptr) -> std::pair<int, std::string> {
    std::vector<const char*> argv;
    for (const auto& argument : arguments) argv.push_back(argument.c_str());
    std::ostringstream output;
    std::ostringstream diagnostics;
    struct Redirect {
        std::streambuf* previous;
        std::streambuf* previous_errors;
        ~Redirect() {
            std::cout.rdbuf(previous);
            std::cerr.rdbuf(previous_errors);
        }
    } restore{std::cout.rdbuf(output.rdbuf()), std::cerr.rdbuf(diagnostics.rdbuf())};
    const auto status = kidi::cli::main(static_cast<int>(argv.size()), argv.data());
    if (progress) *progress = diagnostics.str();
    return {status, output.str()};
}

auto read(const std::filesystem::path& path) -> std::string {
    std::ifstream input(path, std::ios::binary);
    EXPECT_TRUE(input.good()) << "cannot read index test output: " << path;
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

auto prepare(const std::filesystem::path& directory) -> void {
    const std::filesystem::path fixture(KIDI_GEMMA4_FIXTURE);
    ASSERT_TRUE(std::filesystem::create_directory(directory)) << "cannot create index CLI fixture";
    std::filesystem::copy_file(fixture / "model.safetensors", directory / "model.safetensors");
    auto manifest = YAML::LoadFile((fixture / "config.json").string());
    manifest["decode"]["maximum_new_tokens"] = 4;
    manifest["decode"]["context_size"] = 16;
    std::ofstream(directory / "config.json") << manifest;
    std::ofstream(directory / "tokenizer.json") << R"({
      "version":"1.0", "pre_tokenizer":{"type":"WhitespaceSplit"},
      "decoder":{"type":"WordPiece","prefix":"##","cleanup":false},
      "model":{"type":"WordLevel","unk_token":"<unk>","vocab":{
        "<pad>":0,"<eos>":1,"<bos>":2,"<turn|>":3,"<|turn>":4,"<unk>":5,
        "alpha":6,"beta":7,"gamma":8,"delta":9,"theta":10,"zeta":11,"eta":12,"iota":13,"kappa":14,"lambda":15}}
    })";
}

class IndexCliTest : public ::testing::Test {
protected:
    void SetUp() override { ASSERT_NO_FATAL_FAILURE(prepare(root)); }
    void TearDown() override {
        std::error_code cleanup;
        std::filesystem::remove_all(root, cleanup);
        if (cleanup) ADD_FAILURE() << "index CLI test cleanup: " << cleanup.message();
    }

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        ("kidi-index-cli-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
};

TEST_F(IndexCliTest, BuildsIndex) {
    using namespace kidi;
    auto dense = require(inference::Generator::load(
        root, inference::ModelLoadOptions{.device = tensor::Device::cpu(), .dense_lm_head = true}));
    const auto threads = std::min<std::size_t>(4, std::max(1U, std::thread::hardware_concurrency()));
    std::vector<std::string> command{
        "kidi",         "index", "--model",           root.string(),
        "--m",          "4",     "--ef-construction", "32",
        "--candidates", "4",     "--ef-search",       "16",
        "--builder",    "cpu",   "--threads",         std::to_string(threads),
    };
    std::string progress;
    const auto [status, output] = run(command, &progress);
    ASSERT_EQ(status, 0) << "native CLI failed to build a complete index";
    const auto report = nlohmann::json::parse(output);
    const auto path = root / "lm_head.hnsw";
    EXPECT_EQ(report["nodes"], 16) << "index CLI did not report the requested construction parameters";
    EXPECT_EQ(report["candidates"], 4) << "index CLI did not report the requested construction parameters";
    EXPECT_EQ(report["threads"], threads) << "index CLI did not report the requested construction parameters";
    EXPECT_EQ(report["bytes"], std::filesystem::file_size(path))
        << "index CLI did not report the requested construction parameters";
    EXPECT_NE(progress.find("100.00%"), std::string::npos) << "index progress omitted percentage or ETA";
    EXPECT_NE(progress.find("ETA"), std::string::npos) << "index progress omitted percentage or ETA";
    const auto original = read(path);
    EXPECT_EQ(run(command).first, 2) << "CLI replaced an existing index";
    EXPECT_EQ(read(path), original) << "CLI replaced an existing index";
    auto serial_command = command;
    serial_command.back() = "1";
    serial_command.insert(serial_command.end(), {"--output", (root / "serial.hnsw").string()});
    EXPECT_EQ(run(serial_command).first, 0) << "CLI graph depends on thread count";
    EXPECT_EQ(read(root / "serial.hnsw"), original) << "CLI graph depends on thread count";
#if !defined(KIDI_HAS_CUDA)
    auto automatic_command = serial_command;
    const auto builder = std::ranges::find(automatic_command, "--builder");
    automatic_command.erase(builder, builder + 2);
    automatic_command.back() = (root / "automatic.hnsw").string();
    const auto [automatic_status, automatic_output] = run(automatic_command);
    ASSERT_EQ(automatic_status, 0) << "CPU-only build did not automatically select native CPU indexing";
    EXPECT_EQ(nlohmann::json::parse(automatic_output)["builder"], "cpu")
        << "CPU-only build did not automatically select native CPU indexing";
    EXPECT_EQ(read(root / "automatic.hnsw"), original)
        << "CPU-only build did not automatically select native CPU indexing";
    automatic_command.insert(automatic_command.end(), {"--builder", "cuda"});
    EXPECT_EQ(run(automatic_command).first, 2) << "CPU-only build accepted CUDA indexing";
#endif
    auto invalid_threads = command;
    invalid_threads.back() = "0";
    std::string error_output;
    EXPECT_EQ(run(invalid_threads, &error_output).first, 2)
        << "zero indexing threads were accepted or missing formatted error";
    EXPECT_NE(error_output.find("[kidi] [error] HNSW thread count"), std::string::npos)
        << "zero indexing threads were accepted or missing formatted error";
    auto quiet = invalid_threads;
    quiet.insert(quiet.begin() + 1, {"--log-level", "off"});
    std::string suppressed_output;
    EXPECT_EQ(run(quiet, &suppressed_output).first, 2) << "--log-level off did not suppress CLI error logging";
    EXPECT_TRUE(suppressed_output.empty()) << "--log-level off did not suppress CLI error logging";
    auto sparse = require(inference::Generator::load(root, tensor::Device::cpu()));
    inference::GenerationOptions options{
        .maximum_new_tokens = 4,
        .context_size = 16,
        .prefill_chunk_size = 2,
        .raw_prompt = true,
        .ignore_eos = true,
    };
    for (const auto* prompt : {"alpha beta gamma delta", "zeta eta", "theta iota"}) {
        SCOPED_TRACE(prompt);
        const auto expected = require(dense.generate(prompt, options));
        const auto actual = require(sparse.generate(prompt, options));
        EXPECT_EQ(actual.generation.token_ids, expected.generation.token_ids)
            << "checkpoint-autoloaded candidate subset changed fixture generation";
        EXPECT_EQ(actual.text, expected.text) << "checkpoint-autoloaded candidate subset changed fixture generation";
    }
    std::ofstream(path, std::ios::binary | std::ios::trunc) << "broken";
    EXPECT_FALSE(inference::Generator::load(root, tensor::Device::cpu()))
        << "corrupt checkpoint index was silently ignored";
    EXPECT_TRUE(inference::Generator::load(
                    root, inference::ModelLoadOptions{.device = tensor::Device::cpu(), .dense_lm_head = true})
                    .has_value())
        << "explicit dense override did not bypass the checkpoint index";
}
} // namespace
