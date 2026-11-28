#include "kidi/agent/config.h"

#include <gtest/gtest.h>

namespace kidi::agent {
namespace {

constexpr std::string_view CUSTOM_CONFIG = R"(
version: 1
prompts:
  system: |
    You are a review assistant.
    Never modify files.
tools:
  - name: inspect
    handler: native_read
    description: Read a source file.
    parameters:
      type: object
      properties:
        path: {type: string}
        limit: {type: integer, minimum: 1}
        label: {type: string, enum: ["true", "42"]}
      required: [path]
      additionalProperties: false
)";

TEST(AgentConfig, ParsesYamlRegistry) {
    const auto config = parse_agent_config(CUSTOM_CONFIG);
    ASSERT_TRUE(config) << config.error().message;
    EXPECT_EQ(config->system_prompt, "You are a review assistant.\nNever modify files.\n");
    EXPECT_TRUE(config->continue_prompt.empty());
    EXPECT_TRUE(config->compact_prompt.empty());
    EXPECT_EQ(parse_agent_config("version: 1\nprompts: {system: Hi, continue: Go}\ntools: []")->continue_prompt, "Go");
    EXPECT_EQ(parse_agent_config("version: 1\nprompts: {system: Hi, tools: Use tools}\ntools: []")->tools_prompt,
              "Use tools");
    EXPECT_EQ(parse_agent_config("version: 1\nprompts: {system: Hi, compact: Summarize}\ntools: []")->compact_prompt,
              "Summarize");
    EXPECT_EQ(config->sampling.temperature, 0);
    const auto sampled =
        parse_agent_config("version: 1\nprompts: {system: Hi}\nsampling: {temperature: 0.6, top_p: 0.9, top_k: 20}\n"
                           "tools: []");
    ASSERT_TRUE(sampled) << sampled.error().message;
    EXPECT_FLOAT_EQ(sampled->sampling.temperature, 0.6F);
    EXPECT_FLOAT_EQ(sampled->sampling.top_p, 0.9F);
    EXPECT_EQ(sampled->sampling.top_k, 20);
    for (const auto* bad : {"{temperature: 3}", "{top_p: 0}", "{top_k: -1}", "{seed: 1}"})
        EXPECT_FALSE(parse_agent_config(std::string("version: 1\nprompts: {system: Hi}\nsampling: ") + bad +
                                        "\ntools: []"))
            << bad;
    EXPECT_TRUE(config->reasoning_effort.empty());
    const auto effort = parse_agent_config("version: 1\nprompts: {system: Hi}\nreasoning_effort: low\ntools: []");
    ASSERT_TRUE(effort) << effort.error().message;
    EXPECT_EQ(effort->reasoning_effort, "low");
    EXPECT_FALSE(parse_agent_config("version: 1\nprompts: {system: Hi}\nreasoning_effort: 1\ntools: []"));
    EXPECT_TRUE(config->speech.transcribe_model.empty() && config->speech.speak_model.empty());
    const auto speech = parse_agent_config(
        "version: 1\nprompts: {system: Hi}\ntools: []\nspeech:\n  transcribe: {model: a/asr}\n"
        "  speak: {model: b/tts, reference_text: Hello there., steps: 12, pace: 0.8, voice: {age: young adult}}");
    ASSERT_TRUE(speech) << speech.error().message;
    EXPECT_EQ(speech->speech.transcribe_model, "a/asr");
    EXPECT_EQ(speech->speech.speak_model, "b/tts");
    EXPECT_EQ(speech->speech.reference_text, "Hello there.");
    EXPECT_EQ(speech->speech.steps, 12);
    EXPECT_FLOAT_EQ(speech->speech.pace, 0.8F);
    EXPECT_EQ(speech->speech.voice.at("age"), "young adult");
    for (const auto* bad : {"{speak: {model: b/tts}}", "{speak: {model: b/tts, reference_text: Hi, steps: 0}}",
                            "{transcribe: {model: a/asr, language: en}}", "{listen: {model: a/asr}}"})
        EXPECT_FALSE(parse_agent_config(std::string("version: 1\nprompts: {system: Hi}\ntools: []\nspeech: ") + bad))
            << bad;
    const auto shipped = default_agent_config();
    ASSERT_TRUE(shipped) << shipped.error().message;
    EXPECT_FALSE(shipped->speech.transcribe_model.empty());
    EXPECT_FALSE(shipped->speech.speak_model.empty());
    const auto search = std::ranges::find(shipped->tools, "web_search", &ToolRegistration::handler);
    ASSERT_NE(search, shipped->tools.end());
    EXPECT_EQ(search->config, (Json{{"server", "tinyfish"}})) << "signing in needs no key";
    ASSERT_TRUE(shipped->mcp_servers.contains("tinyfish"));
    EXPECT_EQ(shipped->mcp_servers.at("tinyfish").url, "https://agent.tinyfish.ai/mcp");
    EXPECT_EQ(shipped->mcp_servers.at("tinyfish").name, "TinyFish");
    const auto servers = parse_agent_config("version: 1\nprompts: {system: Hi}\ntools: []\nmcp_servers:\n"
                                            "  docs: {url: 'https://docs.test/mcp'}");
    ASSERT_TRUE(servers) << servers.error().message;
    EXPECT_EQ(servers->mcp_servers.at("docs").name, "docs") << "the key names an unnamed server";
    EXPECT_TRUE(servers->mcp_servers.at("docs").include.empty() && !servers->mcp_servers.at("docs").include_all);
    const auto included = parse_agent_config("version: 1\nprompts: {system: Hi}\ntools: []\nmcp_servers:\n"
                                             "  a: {url: 'https://a.test/mcp', include: [search, get_file]}\n"
                                             "  b: {url: 'https://b.test/mcp', include: all}");
    ASSERT_TRUE(included) << included.error().message;
    EXPECT_EQ(included->mcp_servers.at("a").include, (std::vector<std::string>{"search", "get_file"}));
    EXPECT_TRUE(included->mcp_servers.at("b").include_all);
    for (const auto* bad : {"{docs: {url: 'http://docs.test/mcp'}}", "{docs: {name: Docs}}",
                            "{docs: {url: 'https://docs.test/mcp', api_key_env: KEY}}",
                            "{'bad name': {url: 'https://x.test'}}",
                            "[https://docs.test/mcp]", "{docs: {url: 'https://docs.test/mcp', include: some}}",
                            "{docs: {url: 'https://docs.test/mcp', include: ['bad name']}}"})
        EXPECT_FALSE(parse_agent_config(std::string("version: 1\nprompts: {system: Hi}\ntools: []\nmcp_servers: ") +
                                        bad))
            << bad;
    EXPECT_EQ(search->spec.parameters["additionalProperties"], false) << "arguments are strict by default";
    EXPECT_FALSE(parse_agent_config("version: 1\nprompts: {system: Hi}\ntools:\n  - {name: t, handler: h, "
                                    "description: d, parameters: {type: object}, config: [1]}"));
    ASSERT_EQ(config->tools.size(), 1);
    const auto& tool = config->tools.front();
    EXPECT_EQ(tool.handler, "native_read");
    EXPECT_EQ(tool.spec.name, "inspect");
    EXPECT_EQ(tool.spec.parameters["properties"]["limit"]["minimum"], 1);
    EXPECT_EQ(tool.spec.parameters["additionalProperties"], false);
    EXPECT_EQ(tool.spec.parameters["properties"]["label"]["enum"], Json::array({"true", "42"}));
}

TEST(AgentConfig, BindsOnlyListedHandlers) {
    auto config = parse_agent_config(CUSTOM_CONFIG);
    ASSERT_TRUE(config);
    int calls = 0;
    std::vector<Tool> handlers{
        {{.name = "native_read"},
         Capability::READ,
         [&](const Json& args, std::stop_token) -> Result<std::string> {
             ++calls;
             EXPECT_EQ(args["path"], "file");
             return std::string("contents");
         }},
        {{.name = "unlisted"},
         Capability::WRITE,
         [](const Json&, std::stop_token) -> Result<std::string> {
             ADD_FAILURE() << "unlisted handler executed";
             return std::string{};
         }},
    };
    auto tools = configure_tools(*config, handlers);
    ASSERT_TRUE(tools) << tools.error().message;
    ASSERT_EQ(tools->size(), 1);
    EXPECT_EQ(tools->front().spec.name, "inspect");
    EXPECT_EQ(tools->front().capability, Capability::READ);
    auto output = tools->front().invoke({{"path", "file"}}, {});
    ASSERT_TRUE(output);
    EXPECT_EQ(*output, "contents");
    EXPECT_EQ(calls, 1);
    config->tools.front().config = {{"mode", "fast"}};
    auto configured = configure_tools(*config, handlers);
    ASSERT_FALSE(configured) << "a handler without config rejects one";
    EXPECT_NE(configured.error().message.find("takes no config"), std::string::npos);
    const std::vector<ToolHandler> factories{{"native_read", [](const Json& settings) -> Result<Tool> {
        return Tool{{.name = "native_read"}, Capability::READ,
                    [mode = settings.value("mode", "")](const Json&, std::stop_token) -> Result<std::string> {
                        return mode;
                    }};
    }}};
    configured = configure_tools(*config, factories);
    ASSERT_TRUE(configured) << configured.error().message;
    EXPECT_EQ(configured->front().spec.name, "inspect");
    EXPECT_EQ(*configured->front().invoke({}, {}), "fast");
    config->tools.clear();
    tools = configure_tools(*config, handlers);
    ASSERT_TRUE(tools);
    EXPECT_TRUE(tools->empty());
}

TEST(AgentConfig, RejectsInvalidRegistry) {
    const std::vector<std::string> bad{
        "",
        "version: [",
        "[]",
        "version: 2\nprompts: {system: Hi}\ntools: []",
        "version: 1\nprompts: {system: ''}\ntools: []",
        "version: 1\nprompts: {system: Hi}\ntools: {}",
        "version: 1\nprompts: {system: Hi, typo: bad}\ntools: []",
        "version: 1\nprompts: {continue: Go}\ntools: []",
        "version: 1\nprompts: {system: Hi, continue: ''}\ntools: []",
        "version: 1\nprompts: {system: Hi, compact: ''}\ntools: []",
        "version: 1\nprompts: {system: Hi, compact: 42}\ntools: []",
        "version: 1\nversion: 1\nprompts: {system: Hi}\ntools: []",
        "version: 1\nprompts: {system: Hi}\ntools: []\n---\nversion: 1",
        "version: 1\nprompts: {system: Hi}\ntools:\n"
        "  - {name: x, handler: x, description: Test, parameters: {type: array}}",
        "version: 1\nprompts: {system: Hi}\ntools:\n"
        "  - {name: x, handler: x, capability: read, description: Test, parameters: {type: object}}",
        "version: 1\nprompts: {system: Hi}\ntools:\n"
        "  - {name: x, handler: x, description: Test, parameters: {type: object}}\n"
        "  - {name: x, handler: x, description: Test, parameters: {type: object}}",
        "version: 1\nprompts: {system: Hi}\ntools:\n"
        "  - {name: 'bad name', handler: x, description: Test, parameters: {type: object}}",
        "version: 1\nprompts: {system: Hi}\ntools:\n"
        "  - {name: x, handler: x, description: Test, parameters: &cycle {type: object, cycle: *cycle}}",
    };
    for (const auto& text : bad) {
        SCOPED_TRACE(text);
        const auto result = parse_agent_config(text);
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error().code, ErrorCode::INVALID_MANIFEST);
    }
    EXPECT_FALSE(parse_agent_config(std::string(256 * 1024 + 1, ' ')));
}

TEST(AgentConfig, RejectsUnboundHandlers) {
    auto config = parse_agent_config(CUSTOM_CONFIG);
    ASSERT_TRUE(config);
    EXPECT_FALSE(configure_tools(*config, std::span<const Tool>{}));
    Tool handler{{.name = "native_read"}, Capability::WRITE, [](const Json&, std::stop_token) -> Result<std::string> {
                     return std::string("write");
                 }};
    const std::vector<Tool> duplicates{handler, handler};
    EXPECT_FALSE(configure_tools(*config, duplicates));
    // A YAML alias named "inspect" cannot downgrade a write handler's capability.
    auto tools = configure_tools(*config, std::span(&handler, 1));
    ASSERT_TRUE(tools);
    EXPECT_EQ(tools->front().capability, Capability::WRITE);
}

TEST(AgentConfig, MissingFileIsAnError) {
    const auto result = load_agent_config("/nonexistent/kidi-agent-agent-config.yaml");
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::IO);
}

} // namespace
} // namespace kidi::agent
