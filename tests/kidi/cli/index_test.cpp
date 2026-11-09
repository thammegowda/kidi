#include "kidi/cli/main.h"
#include "kidi/core/hnsw.h"
#include "kidi/inference/generator.h"
#include "kidi/ops/context.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace {
using kidi::ops::require;

auto check(bool valid, const char* message) -> void {
    if (!valid) throw std::runtime_error(message);
}

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
    check(input.good(), "cannot read index test output");
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

auto prepare(const std::filesystem::path& directory) -> void {
    const std::filesystem::path fixture(KIDI_GEMMA4_FIXTURE);
    check(std::filesystem::create_directory(directory), "cannot create index CLI fixture");
    std::filesystem::copy_file(fixture / "model.safetensors", directory / "model.safetensors");
    YAML::Node manifest;
    manifest["format_version"] = 1;
    manifest["weights_file"] = "model.safetensors";
    manifest["tokenizer_file"] = "tokenizer.json";
    manifest["model"] = YAML::LoadFile((fixture / "model.yaml").string())["model"];
    manifest["decode"]["maximum_new_tokens"] = 4;
    manifest["decode"]["context_size"] = 16;
    std::ofstream(directory / "model.yaml") << manifest;
    std::ofstream(directory / "tokenizer.json") << R"({
      "version":"1.0", "pre_tokenizer":{"type":"WhitespaceSplit"},
      "decoder":{"type":"WordPiece","prefix":"##","cleanup":false},
      "model":{"type":"WordLevel","unk_token":"<unk>","vocab":{
        "<pad>":0,"<eos>":1,"<bos>":2,"<turn|>":3,"<|turn>":4,"<unk>":5,
        "alpha":6,"beta":7,"gamma":8,"delta":9,"theta":10,"zeta":11,"eta":12,"iota":13,"kappa":14,"lambda":15}}
    })";
}
} // namespace

auto main() -> int {
    using namespace kidi;
    const auto root = std::filesystem::temp_directory_path() /
                      ("kidi-index-cli-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        prepare(root);
        auto dense = require(inference::Generator::load(
            root, inference::ModelLoadOptions{.device = tensor::Device::cpu(), .dense_lm_head = true}));
        std::vector<std::string> command{
            "kidi",         "index", "--model",     root.string(), "--m",       "4",   "--ef-construction", "32",
            "--candidates", "4",     "--ef-search", "16",          "--builder", "cpu", "--threads",         "4",
        };
        std::string progress;
        const auto [status, output] = run(command, &progress);
        check(status == 0, "native CLI failed to build a complete index");
        const auto report = nlohmann::json::parse(output);
        const auto path = root / "lm_head.hnsw";
        check(report["nodes"] == 16 && report["candidates"] == 4 && report["threads"] == 4 &&
                  report["bytes"] == std::filesystem::file_size(path),
              "index CLI did not report the requested construction parameters");
        check(progress.find("100.00%") != std::string::npos && progress.find("ETA") != std::string::npos,
              "index progress omitted percentage or ETA");
        const auto original = read(path);
        check(run(command).first == 2 && read(path) == original, "CLI replaced an existing index");
        auto serial_command = command;
        serial_command.back() = "1";
        serial_command.insert(serial_command.end(), {"--output", (root / "serial.hnsw").string()});
        check(run(serial_command).first == 0 && read(root / "serial.hnsw") == original,
              "CLI graph depends on thread count");
#if !defined(KIDI_HAS_CUDA)
        auto automatic_command = serial_command;
        const auto builder = std::ranges::find(automatic_command, "--builder");
        automatic_command.erase(builder, builder + 2);
        automatic_command.back() = (root / "automatic.hnsw").string();
        const auto [automatic_status, automatic_output] = run(automatic_command);
        check(automatic_status == 0 && nlohmann::json::parse(automatic_output)["builder"] == "cpu" &&
                  read(root / "automatic.hnsw") == original,
              "CPU-only build did not automatically select native CPU indexing");
        automatic_command.insert(automatic_command.end(), {"--builder", "cuda"});
        check(run(automatic_command).first == 2, "CPU-only build accepted CUDA indexing");
#endif
        auto invalid_threads = command;
        invalid_threads.back() = "0";
        check(run(invalid_threads).first == 2, "zero indexing threads were accepted");
        auto sparse = require(inference::Generator::load(root, tensor::Device::cpu()));
        inference::GenerationOptions options{
            .maximum_new_tokens = 4,
            .context_size = 16,
            .prefill_chunk_size = 2,
            .raw_prompt = true,
            .ignore_eos = true,
        };
        for (const auto* prompt : {"alpha beta gamma delta", "zeta eta", "theta iota"}) {
            const auto expected = require(dense.generate(prompt, options));
            const auto actual = require(sparse.generate(prompt, options));
            check(actual.generation.token_ids == expected.generation.token_ids && actual.text == expected.text,
                  "checkpoint-autoloaded candidate subset changed fixture generation");
        }
        std::ofstream(path, std::ios::binary | std::ios::trunc) << "broken";
        check(!inference::Generator::load(root, tensor::Device::cpu()),
              "corrupt checkpoint index was silently ignored");
        check(inference::Generator::load(
                  root, inference::ModelLoadOptions{.device = tensor::Device::cpu(), .dense_lm_head = true})
                  .has_value(),
              "explicit dense override did not bypass the checkpoint index");
        std::filesystem::remove_all(root);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "index CLI test: " << error.what() << '\n';
        std::error_code cleanup;
        std::filesystem::remove_all(root, cleanup);
        if (cleanup) std::cerr << "index CLI test cleanup: " << cleanup.message() << '\n';
        return 1;
    }
}
