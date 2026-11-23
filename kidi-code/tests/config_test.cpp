#include "kidi/code/config.h"

#include <gtest/gtest.h>

namespace kidi::code {
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

TEST(CodeConfig, ParsesYamlRegistry) {
    const auto config = parse_agent_config(CUSTOM_CONFIG);
    ASSERT_TRUE(config) << config.error().message;
    EXPECT_EQ(config->system_prompt, "You are a review assistant.\nNever modify files.\n");
    EXPECT_TRUE(config->continue_prompt.empty());
    EXPECT_TRUE(config->compact_prompt.empty());
    EXPECT_EQ(parse_agent_config("version: 1\nprompts: {system: Hi, continue: Go}\ntools: []")->continue_prompt, "Go");
    EXPECT_EQ(parse_agent_config("version: 1\nprompts: {system: Hi, compact: Summarize}\ntools: []")->compact_prompt,
              "Summarize");
    ASSERT_EQ(config->tools.size(), 1);
    const auto& tool = config->tools.front();
    EXPECT_EQ(tool.handler, "native_read");
    EXPECT_EQ(tool.spec.name, "inspect");
    EXPECT_EQ(tool.spec.parameters["properties"]["limit"]["minimum"], 1);
    EXPECT_EQ(tool.spec.parameters["additionalProperties"], false);
    EXPECT_EQ(tool.spec.parameters["properties"]["label"]["enum"], Json::array({"true", "42"}));
}

TEST(CodeConfig, BindsOnlyListedHandlers) {
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
    config->tools.clear();
    tools = configure_tools(*config, handlers);
    ASSERT_TRUE(tools);
    EXPECT_TRUE(tools->empty());
}

TEST(CodeConfig, RejectsInvalidRegistry) {
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

TEST(CodeConfig, RejectsUnboundHandlers) {
    auto config = parse_agent_config(CUSTOM_CONFIG);
    ASSERT_TRUE(config);
    EXPECT_FALSE(configure_tools(*config, {}));
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

TEST(CodeConfig, MissingFileIsAnError) {
    const auto result = load_agent_config("/nonexistent/kidi-code-agent-config.yaml");
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::IO);
}

} // namespace
} // namespace kidi::code
