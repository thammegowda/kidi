#include "kidi/cli/main.h"
#include "kidi/inference/generator.h"
#include "kidi/ops/context.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
using kidi::ops::require;
using Json = nlohmann::ordered_json;

auto check(bool valid, const char* message) -> void {
    if (!valid) throw std::runtime_error(message);
}

auto run_cli(const std::vector<std::string>& arguments) -> int {
    std::vector<const char*> argv;
    for (const auto& argument : arguments) argv.push_back(argument.c_str());
    return kidi::cli::main(static_cast<int>(argv.size()), argv.data());
}

auto complete(kidi::inference::Generator& generator) -> kidi::inference::TextGeneration {
    while (generator.pending_requests())
        for (auto& event : require(generator.step()).events)
            if (event.completed) return std::move(*event.completed);
    throw std::runtime_error("missing completion");
}
} // namespace

auto main() -> int {
    const auto root = std::filesystem::temp_directory_path() /
                      ("kidi-chat-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        std::filesystem::create_directory(root);
        const std::filesystem::path fixture(KIDI_GEMMA4_FIXTURE);
        std::filesystem::copy_file(fixture / "model.safetensors", root / "model.safetensors");
        auto config = YAML::LoadFile((fixture / "config.json").string());
        config["decode"]["maximum_new_tokens"] = 2;
        config["decode"]["context_size"] = 16;
        std::ofstream(root / "config.json") << config;
        std::ofstream(root / "tokenizer.json") << R"({
          "version":"1.0", "pre_tokenizer":{"type":"WhitespaceSplit"},
          "decoder":{"type":"WordPiece","prefix":"##","cleanup":false},
          "model":{"type":"WordLevel","unk_token":"<unk>","vocab":{
            "<pad>":0,"<eos>":1,"<bos>":2,"<turn|>":3,"<|turn>":4,"<unk>":5,
            "alpha":6,"beta":7,"gamma":8,"delta":9,"theta":10,"zeta":11,"eta":12,"iota":13,"kappa":14,"lambda":15}}
        })";
        std::ofstream(root / "tokenizer_config.json") << "{}";
        std::ofstream(root / "chat_template.jinja")
            << "{% for m in messages %}{% if m.content is string %}{{ m.content }} {% endif %}"
               "{% if m.tool_calls %}{% for tc in m.tool_calls %}{{ tc.function.name }} {% endfor %}{% endif %}"
               "{% endfor %}{% if tools %}{% for key in tools[0].function.parameters.properties %}"
               "{% if loop.first and key != 'delta' %}{{ raise_exception('tool order changed') }}{% endif %}"
               "{{ key }} {% endfor %}{% endif %}{% if enable_thinking %}theta eta{% else %}zeta{% endif %}";

        auto generator = require(kidi::inference::Generator::load(root, kidi::tensor::Device::cpu()));
        require(generator.configure_serving({1, 4, 16, 2}));
        kidi::inference::GenerationOptions options;
        options.maximum_new_tokens = 2;
        options.context_size = 16;
        options.prefill_chunk_size = 2;
        options.ignore_eos = true;
        const auto messages = Json::parse(R"([
          {"role":"user","content":"alpha"},
          {"role":"assistant","content":null,"tool_calls":[{"function":{"name":"gamma","arguments":{}}}]},
          {"role":"tool","content":"beta"}])");
        const auto args = Json::parse(R"({"enable_thinking":true,"tools":[{"function":{"parameters":{
          "properties":{"delta":{},"alpha":{}}}}}]})");
        const auto count = require(generator.chat_prompt_tokens_json(messages, args));
        check(count == 7, "structured prompt token count is incorrect");
        require(generator.enqueue_chat_json(messages, options, args));
        const auto structured = complete(generator);
        check(structured.stats.prompt_tokens == count, "generation and counting rendered different prompts");
        auto raw_options = options;
        raw_options.raw_prompt = true;
        require(generator.enqueue("alpha gamma beta delta alpha theta eta", raw_options));
        const auto raw = complete(generator);
        check(structured.generation.token_ids == raw.generation.token_ids, "structured generation changed prompt");
        require(generator.enqueue("alpha zeta", raw_options));
        const auto legacy = complete(generator);
        auto image = Json::parse(R"([{"role":"user","content":[{"type":"image_url","image_url":"x"}]}])");
        check(!generator.chat_prompt_tokens_json(image) && !generator.enqueue_chat_json(image),
              "structured generation accepted an image without embeddings");
        check(!generator.enqueue_chat_json(messages, options, {{"messages", "reserved"}}),
              "reserved template arguments were accepted");

        const auto input = root / "input.jsonl";
        const auto output = root / "output.jsonl";
        {
            std::ofstream requests(input);
            requests << Json{{"messages", messages}, {"template_args", args}, {"max_tokens", 2}}.dump() << '\n';
            requests << R"({"messages":[{"role":"user","content":"alpha"}],"max_tokens":2})" << '\n';
        }
        const std::vector<std::string> command{"kidi", "generate", "--model", root.string(), "--backend", "ynnpack",
                                               "--context-size", "16", "--cache-tokens", "16",
                                               "--max-active", "1", "--prefill-chunk-size", "2", "--ignore-eos",
                                               "--in", input.string(), "--out", output.string()};
        check(run_cli(command) == 0, "structured JSONL request failed");
        std::ifstream records(output);
        Json response;
        records >> response;
        check(response["token_ids"].get<std::vector<std::int32_t>>() == raw.generation.token_ids,
              "JSONL structured request changed generation");
        records >> response;
        check(response["token_ids"].get<std::vector<std::int32_t>>() == legacy.generation.token_ids,
              "JSONL simple request changed generation");
        records.close();
        std::ofstream(input) << R"({"messages":[{"role":"user","content":"alpha"}],"template_args":[]})" << '\n';
        check(run_cli(command) == 2, "JSONL accepted non-object template arguments");
        std::filesystem::remove_all(root);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "chat test: " << error.what() << '\n';
        std::filesystem::remove_all(root);
        return 1;
    }
}
