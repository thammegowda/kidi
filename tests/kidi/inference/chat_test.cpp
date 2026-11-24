#include "kidi/cli/main.h"
#include "kidi/inference/generator.h"
#include "kidi/ops/context.h"
#include "kidi/server/engine.h"
#include "kidi/server/routes.h"

#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

#include <gtest/gtest.h>
#include <httplib.h>

namespace {
using kidi::ops::require;
using Json = nlohmann::ordered_json;

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

class ChatTest : public ::testing::Test {
protected:
    void SetUp() override { std::filesystem::create_directory(root); }
    void TearDown() override { std::filesystem::remove_all(root); }

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        ("kidi-chat-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
};

TEST_F(ChatTest, StreamsHeartbeatWhileModelHasNoText) {
    const std::filesystem::path fixture(KIDI_GEMMA4_FIXTURE);
    for (const auto* name : {"model.safetensors", "config.json"})
        std::filesystem::copy_file(fixture / name, root / name);
    Json vocabulary;
    for (int i = 0; i < 16; ++i) vocabulary["t" + std::to_string(i)] = i;
    std::ofstream(root / "tokenizer.json")
        << Json{{"version", "1.0"},
                {"pre_tokenizer", {{"type", "WhitespaceSplit"}}},
                {"model", {{"type", "WordLevel"}, {"unk_token", "t0"}, {"vocab", vocabulary}}}};
    std::ofstream(root / "tokenizer_config.json") << "{}";
    std::ofstream(root / "chat_template.jinja") << "{{ messages[0].content }}";
    auto generator = require(kidi::inference::Generator::load(root, kidi::tensor::Device::cpu()));
    require(generator.configure_serving({1, 4, 16, 2}));
    kidi::server::ChatEngine engine(generator);
    auto created = liteapp::HttpServer::create();
    ASSERT_TRUE(created);
    auto server = std::move(*created);
    kidi::server::mount_chat_routes(server, engine, {.model_name = "test",
                                                     .defaults = {.maximum_new_tokens = 1, .context_size = 16}});
    const auto bound = server.bind();
    ASSERT_TRUE(bound);
    std::jthread listener([&] { (void)server.listen_after_bind(); });
    server.wait_until_ready();

    std::promise<void> entered, release;
    const auto gate = release.get_future().share();
    auto blocked = std::async(std::launch::async, [&] {
        return engine.run({.enqueue = [&](auto&) -> kidi::Result<std::uint64_t> {
            entered.set_value();
            gate.wait();
            return std::unexpected(kidi::Error{kidi::ErrorCode::RUNTIME, "test work released"});
        }});
    });
    entered.get_future().wait();
    httplib::Client client("127.0.0.1", *bound);
    client.set_read_timeout(std::chrono::seconds(3));
    std::string wire;
    const auto response = client.Post("/v1/chat/completions", {},
                                      R"({"messages":[{"role":"user","content":"t1"}],"stream":true})",
                                      "application/json", [&](const char* chunk, std::size_t bytes) {
                                          wire.append(chunk, bytes);
                                          return wire.find(": keep-alive\n\n") == std::string::npos;
                                      }, {});
    release.set_value();
    EXPECT_FALSE(blocked.get());
    server.stop();
    EXPECT_FALSE(response);
    EXPECT_NE(wire.find(": keep-alive\n\n"), std::string::npos);
}

TEST_F(ChatTest, PromptsAndStopTokens) {
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
           "{{ key }} {% endfor %}{% endif %}{% if enable_thinking | default(true) %}"
           "theta eta{% else %}zeta{% endif %}";

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
    const auto count = require(generator.chat_prompt_tokens(messages, args));
    EXPECT_EQ(count, 7) << "structured prompt token count is incorrect";
    {
        kidi::server::ChatEngine engine(generator);
        const auto context = require(engine.context(messages, args, options));
        EXPECT_EQ(context.prompt_tokens, count);
        EXPECT_EQ(context.context_tokens, 16);
        EXPECT_EQ(context.prompt_limit_tokens, 15);
        auto limited = options;
        limited.maximum_prompt_tokens = 5;
        EXPECT_EQ(require(engine.context(messages, args, limited)).prompt_limit_tokens, 5);
        limited.context_size = 17;
        EXPECT_FALSE(engine.context(messages, args, limited));
        const auto overfull =
            Json::array({{{"role", "user"},
                          {"content", std::string(40, ' ') + "alpha alpha alpha alpha alpha alpha alpha alpha alpha "
                                                             "alpha alpha alpha alpha alpha alpha alpha alpha"}}});
        EXPECT_GT(require(engine.context(overfull, {}, options)).prompt_tokens, 16);
        const auto generated = require(
            engine.run({.enqueue = [&](auto& current) { return current.enqueue_chat(messages, options, args); }}));
        EXPECT_EQ(generated.stats.prompt_tokens, context.prompt_tokens);
    }
    require(generator.enqueue_chat(messages, options, args));
    const auto structured = complete(generator);
    EXPECT_EQ(structured.stats.prompt_tokens, count) << "generation and counting rendered different prompts";
    auto raw_options = options;
    raw_options.raw_prompt = true;
    require(generator.enqueue("alpha gamma beta delta alpha theta eta", raw_options));
    const auto raw = complete(generator);
    EXPECT_EQ(structured.generation.token_ids, raw.generation.token_ids) << "structured generation changed prompt";
    require(generator.enqueue("alpha zeta", raw_options));
    const auto legacy = complete(generator);
    auto image = Json::parse(R"([{"role":"user","content":[{"type":"image_url","image_url":"x"}]}])");
    EXPECT_FALSE(generator.chat_prompt_tokens(image)) << "structured generation accepted an image without embeddings";
    EXPECT_FALSE(generator.enqueue_chat(image)) << "structured generation accepted an image without embeddings";
    EXPECT_FALSE(generator.enqueue_chat(messages, options, {{"messages", "reserved"}}))
        << "reserved template arguments were accepted";

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
    ASSERT_EQ(run_cli(command), 0) << "structured JSONL request failed";
    std::ifstream records(output);
    Json response;
    records >> response;
    EXPECT_EQ(response["token_ids"].get<std::vector<std::int32_t>>(), raw.generation.token_ids)
        << "JSONL structured request changed generation";
    records >> response;
    EXPECT_EQ(response["token_ids"].get<std::vector<std::int32_t>>(), legacy.generation.token_ids)
        << "JSONL simple request changed generation";
    records.close();
    std::ofstream(input) << R"({"messages":[{"role":"user","content":"alpha"}],"template_args":[]})" << '\n';
    EXPECT_EQ(run_cli(command), 2) << "JSONL accepted non-object template arguments";

    std::cerr << "ChatTest: checking template defaults\n";
    const auto simple = require(generator.generate("alpha", options));
    const auto template_default = require(generator.generate("alpha theta eta", raw_options));
    EXPECT_EQ(simple.stats.prompt_tokens, 3) << "plain generation bypassed the checkpoint chat template or its defaults";
    EXPECT_EQ(simple.generation.token_ids, template_default.generation.token_ids)
        << "plain generation bypassed the checkpoint chat template or its defaults";
    const std::array prompts{std::string("alpha")};
    EXPECT_EQ(require(generator.generate_batch(prompts, options)).rows.front().generation.token_ids,
              template_default.generation.token_ids)
        << "batch generation bypassed the checkpoint chat template";

    std::cerr << "ChatTest: checking tokenizer metadata\n";
    // Keep the model weights but remove every Gemma token name and use tokenizer metadata instead.
    std::ofstream(root / "tokenizer.json") << R"({
      "version":"1.0", "pre_tokenizer":{"type":"WhitespaceSplit"},
      "decoder":{"type":"WordPiece","prefix":"##","cleanup":false},
      "model":{"type":"WordLevel","unk_token":"unknown","vocab":{
        "padding":0,"finish":1,"start":2,"end_turn":3,"turn":4,"unknown":5,
        "alpha":6,"beta":7,"gamma":8,"delta":9,"theta":10,"zeta":11,"eta":12,"iota":13,"kappa":14,"lambda":15}}
    })";
    config.remove("bos_token_id");
    config.remove("eos_token_id");
    config.remove("pad_token_id");
    std::ofstream(root / "config.json") << config;
    std::ofstream(root / "tokenizer_config.json") << R"({"eos_token":"finish","pad_token":"padding"})";
    auto generic = require(kidi::inference::Generator::load(root, kidi::tensor::Device::cpu()));
    require(generic.warm_up(2, 16));
    const auto expected = require(generic.generate("alpha", options));
    EXPECT_EQ(expected.generation.token_ids, template_default.generation.token_ids)
        << "tokenizer metadata or non-Gemma warm-up changed generation";
    ASSERT_FALSE(expected.generation.token_ids.empty()) << "missing fixture output for stop-token test";
    const auto stop = expected.generation.token_ids.front();
    std::cerr << "ChatTest: checking stop tokens\n";
    std::ofstream(root / "generation_config.json")
        << Json{{"eos_token_id", {(stop + 1) % 16, stop, stop}}, {"pad_token_id", 0}};
    auto multiple_stops = require(kidi::inference::Generator::load(root, kidi::tensor::Device::cpu()));
    auto stop_options = options;
    stop_options.ignore_eos = false;
    const auto stopped = require(multiple_stops.generate("alpha", stop_options));
    EXPECT_TRUE(stopped.generation.token_ids.empty()) << "serial generation ignored an additional EOS token";
    EXPECT_EQ(stopped.generation.decoder_steps, 1) << "serial generation ignored an additional EOS token";
    EXPECT_FALSE(stopped.generation.truncated) << "serial generation ignored an additional EOS token";
    const auto batch_stopped = require(multiple_stops.generate_batch(prompts, stop_options));
    EXPECT_TRUE(batch_stopped.rows.front().generation.token_ids.empty())
        << "batch generation ignored an additional EOS token";
    EXPECT_EQ(batch_stopped.rows.front().generation.decoder_steps, 1)
        << "batch generation ignored an additional EOS token";
    require(multiple_stops.configure_serving({1, 4, 16, 2}));
    require(multiple_stops.enqueue("alpha", stop_options));
    const auto serving_stopped = complete(multiple_stops);
    EXPECT_TRUE(serving_stopped.generation.token_ids.empty()) << "serving ignored an additional EOS token";
    EXPECT_EQ(serving_stopped.generation.decoder_steps, 1) << "serving ignored an additional EOS token";
    EXPECT_EQ(require(multiple_stops.generate("alpha", options)).generation.token_ids, expected.generation.token_ids)
        << "ignore_eos did not bypass the checkpoint stop list";
    std::cerr << "ChatTest: checking invalid metadata\n";
    for (const auto* invalid : {R"({"eos_token_id":[]})", R"({"eos_token_id":16})",
                               R"({"eos_token_id":-1})", R"({"eos_token_id":[1],"pad_token_id":[0]})"}) {
        std::ofstream(root / "generation_config.json") << invalid;
        EXPECT_FALSE(kidi::inference::Generator::load(root, kidi::tensor::Device::cpu()))
            << "invalid generation token metadata was accepted: " << invalid;
    }
    std::filesystem::remove(root / "generation_config.json");
    config["dtype"] = "not-a-dtype";
    std::ofstream(root / "config.json") << config;
    EXPECT_FALSE(kidi::inference::Generator::load(root, kidi::tensor::Device::cpu()))
        << "invalid parameter dtype was accepted";
    config["dtype"] = "float32";
    std::ofstream(root / "config.json") << config;
    std::ofstream(root / "tokenizer_config.json") << R"({"eos_token":"missing"})";
    EXPECT_FALSE(kidi::inference::Generator::load(root, kidi::tensor::Device::cpu()))
        << "missing tokenizer EOS token was accepted";
    std::ofstream(root / "tokenizer_config.json") << R"({"eos_token":"finish"})";
    std::filesystem::remove(root / "chat_template.jinja");
    std::cerr << "ChatTest: checking missing template\n";
    auto without_template = require(kidi::inference::Generator::load(root, kidi::tensor::Device::cpu()));
    EXPECT_FALSE(without_template.generate("alpha", options)) << "chat without a checkpoint template was accepted";
    EXPECT_EQ(require(without_template.generate("alpha gamma beta delta alpha theta eta", raw_options))
                  .generation.token_ids,
              raw.generation.token_ids)
        << "raw generation requires a chat template";

    std::cerr << "ChatTest: checking context limits\n";
    auto clamped = raw_options;
    clamped.context_size = 16;
    clamped.maximum_new_tokens = 20;
    require(multiple_stops.enqueue("alpha theta eta", clamped));
    const auto clamped_out = complete(multiple_stops);
    EXPECT_LE(clamped_out.generation.token_ids.size(), 13U) << "serving enqueue should clamp generation to remaining context";
    EXPECT_FALSE(multiple_stops.enqueue("a b c d e f g h i j k l m n o p", clamped))
        << "prompt at or beyond context capacity must still be rejected";
}
} // namespace
