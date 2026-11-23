#include "kidi/code/agent.h"

#include <gtest/gtest.h>
#include <algorithm>

namespace kidi::code {
namespace {

auto test_tool(int& calls, Capability capability = Capability::WRITE) -> Tool {
    return {{"edit", "test edit", {{"type", "object"}}},
            capability,
            [&calls](const Json&, std::stop_token) -> Result<std::string> {
                ++calls;
                return std::string("edited");
            }};
}

auto tool_reply(std::string name = "edit") -> Message {
    return {.role = Role::ASSISTANT, .tool_calls = {{"call-1", std::move(name), Json::object()}}};
}

TEST(CodeAgent, RoundTrip) {
    int calls = 0;
    std::vector<Tool> tools{test_tool(calls)};
    std::vector<Message> history;
    RunOptions options;
    options.permissions.write = Access::ALLOW;
    int requests = 0;
    Provider provider = [&](std::span<const Message> messages, std::span<const ToolSpec> specs,
                            std::stop_token) -> Result<Message> {
        EXPECT_EQ(specs.size(), 1);
        if (++requests == 1) {
            EXPECT_EQ(messages.back().content, "fix it");
            return tool_reply();
        }
        if (requests == 2) {
            EXPECT_EQ(messages.back().role, Role::TOOL);
            EXPECT_EQ(messages.back().tool_call_id, "call-1");
            EXPECT_EQ(messages.back().content, "edited");
        } else {
            EXPECT_EQ(messages.back().role, Role::USER);
            EXPECT_EQ(messages.back().content, "next");
            EXPECT_EQ(messages[messages.size() - 2].content, "done");
        }
        return Message{.role = Role::ASSISTANT, .content = "done"};
    };
    const auto result = run_turn(provider, tools, history, "fix it", options);
    ASSERT_TRUE(result) << result.error().message;
    EXPECT_EQ(result->text, "done");
    EXPECT_EQ(result->model_calls, 2);
    EXPECT_EQ(result->tool_calls, 1);
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(history.size(), 4);
    EXPECT_TRUE(run_turn(provider, tools, history, "next", options));
}

TEST(CodeAgent, ContinuesOnceAfterSilentTools) {
    int calls = 0;
    std::vector<Tool> tools{test_tool(calls)};
    std::vector<Message> history;
    RunOptions options{.continue_prompt = "go on"};
    options.permissions.write = Access::ALLOW;
    std::vector<std::string> last;
    Provider provider = [&](std::span<const Message> messages, std::span<const ToolSpec>,
                            std::stop_token) -> Result<Message> {
        last.push_back(messages.back().content);
        if (last.size() == 1) return tool_reply();
        if (last.size() == 3) return Message{.role = Role::ASSISTANT, .content = "done"};
        return Message{.role = Role::ASSISTANT};
    };
    const auto result = run_turn(provider, tools, history, "fix it", options);
    ASSERT_TRUE(result) << result.error().message;
    EXPECT_EQ(result->text, "done");
    EXPECT_EQ(last, (std::vector<std::string>{"fix it", "edited", "go on"}));
    ASSERT_EQ(history.size(), 4);
    EXPECT_EQ(history[2].role, Role::TOOL);

    int silent = 0;
    Provider stubborn = [&](std::span<const Message>, std::span<const ToolSpec>, std::stop_token) -> Result<Message> {
        return ++silent == 1 ? tool_reply() : Message{.role = Role::ASSISTANT};
    };
    EXPECT_FALSE(run_turn(stubborn, tools, history, "again", options)) << "only one continuation per turn";
    EXPECT_EQ(silent, 3);
    options.continue_prompt.clear();
    silent = 0;
    EXPECT_FALSE(run_turn(stubborn, tools, history, "again", options)) << "disabled without a prompt";
    EXPECT_EQ(silent, 2);
}

TEST(CodeAgent, DeniedToolsStayDenied) {
    int calls = 0;
    std::vector<Tool> tools{test_tool(calls, Capability::EXECUTE)};
    std::vector<Message> history;
    int requests = 0;
    int approvals = 0;
    Callbacks callbacks{.approve = [&](const ToolSpec&, const ToolCall&) -> Result<bool> {
        ++approvals;
        return true;
    }};
    Provider provider = [&](std::span<const Message> messages, std::span<const ToolSpec> specs,
                            std::stop_token) -> Result<Message> {
        EXPECT_TRUE(specs.empty());
        if (++requests == 1) return tool_reply();
        EXPECT_NE(messages.back().content.find("permission denied"), std::string::npos);
        return Message{.role = Role::ASSISTANT, .content = "denied"};
    };
    EXPECT_TRUE(run_turn(provider, tools, history, "run", {}, callbacks));
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(approvals, 0);
}

TEST(CodeAgent, ApprovalPerCall) {
    for (const bool approve : {false, true}) {
        int calls = 0;
        int approvals = 0;
        int requests = 0;
        std::vector<Tool> tools{test_tool(calls)};
        std::vector<Message> history;
        Provider provider = [&](auto, auto, std::stop_token) -> Result<Message> {
            if (++requests < 3) return tool_reply();
            return Message{.role = Role::ASSISTANT, .content = "done"};
        };
        Callbacks callbacks{.approve = [&](const ToolSpec&, const ToolCall&) -> Result<bool> {
            ++approvals;
            return approve;
        }};
        EXPECT_TRUE(run_turn(provider, tools, history, "edit", {}, callbacks));
        EXPECT_EQ(approvals, 2);
        EXPECT_EQ(calls, approve ? 2 : 0);
    }
}

TEST(CodeAgent, MissingApprovalFailsClosed) {
    int calls = 0;
    std::vector<Tool> tools{test_tool(calls)};
    std::vector<Message> history;
    int requests = 0;
    Provider provider = [&](std::span<const Message> messages, auto, std::stop_token) -> Result<Message> {
        if (++requests == 1) return tool_reply();
        EXPECT_NE(messages.back().content.find("no approval callback"), std::string::npos);
        return Message{.role = Role::ASSISTANT, .content = "not edited"};
    };
    EXPECT_TRUE(run_turn(provider, tools, history, "edit"));
    EXPECT_EQ(calls, 0);
}

TEST(CodeAgent, BadCallsNeverExecute) {
    for (const bool duplicate : {false, true}) {
        int calls = 0;
        std::vector<Tool> tools{test_tool(calls)};
        std::vector<Message> history;
        Provider provider = [&](auto, auto, std::stop_token) -> Result<Message> {
            auto message = tool_reply();
            if (duplicate)
                message.tool_calls.push_back(message.tool_calls.front());
            else
                message.tool_calls.front().arguments = Json::array();
            return message;
        };
        RunOptions options;
        options.permissions.write = Access::ALLOW;
        EXPECT_FALSE(run_turn(provider, tools, history, "edit", options));
        EXPECT_EQ(calls, 0);
        EXPECT_EQ(history.size(), 1);
    }
}

TEST(CodeAgent, UnknownAndFailedTools) {
    int requests = 0;
    std::vector<Message> history;
    std::vector<Event> events;
    Tool tool{
        {"fail", "fails", Json::object()}, Capability::READ, [](const Json&, std::stop_token) -> Result<std::string> {
            return std::unexpected(Error{ErrorCode::IO, "disk error"});
        }};
    Provider provider = [&](std::span<const Message> messages, auto, std::stop_token) -> Result<Message> {
        if (++requests == 1)
            return Message{.role = Role::ASSISTANT,
                           .tool_calls = {{"a", "missing", Json::object()}, {"b", "fail", Json::object()}}};
        EXPECT_NE(messages[messages.size() - 2].content.find("unknown tool"), std::string::npos);
        EXPECT_NE(messages.back().content.find("disk error"), std::string::npos);
        return Message{.role = Role::ASSISTANT, .content = "reported errors"};
    };
    Callbacks callbacks{.observe = [&](const Event& event) { events.push_back(event); }};
    EXPECT_TRUE(run_turn(provider, std::span(&tool, 1), history, "try", {}, callbacks));
    EXPECT_EQ(std::ranges::count_if(
                  events, [](const Event& event) { return event.kind == EventKind::TOOL_FINISH && event.is_error; }),
              2);
}

TEST(CodeAgent, LimitsBoundExecution) {
    int calls = 0;
    std::vector<Tool> tools{test_tool(calls)};
    std::vector<Message> history;
    Provider provider = [](auto, auto, std::stop_token) -> Result<Message> { return tool_reply(); };
    RunOptions options;
    options.permissions.write = Access::ALLOW;
    options.max_model_calls = 2;
    auto result = run_turn(provider, tools, history, "loop", options);
    ASSERT_FALSE(result);
    EXPECT_NE(result.error().message.find("model call limit"), std::string::npos);
    EXPECT_EQ(calls, 2);
    options.max_tool_calls = 1;
    history.clear();
    calls = 0;
    result = run_turn(provider, tools, history, "loop", options);
    ASSERT_FALSE(result);
    EXPECT_NE(result.error().message.find("tool call limit"), std::string::npos);
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(history.back().role, Role::TOOL);
}

TEST(CodeAgent, CancelPairsWholeBatch) {
    std::stop_source source;
    int calls = 0;
    Tool tool{{"cancel", "cancel", Json::object()},
              Capability::READ,
              [&](const Json&, std::stop_token) -> Result<std::string> {
                  ++calls;
                  source.request_stop();
                  return std::string("cancelled after effect");
              }};
    Provider provider = [](auto, auto, std::stop_token) -> Result<Message> {
        return Message{.role = Role::ASSISTANT,
                       .tool_calls = {{"a", "cancel", Json::object()}, {"b", "cancel", Json::object()}}};
    };
    std::vector<Message> history;
    RunOptions options{.max_model_calls = 1};
    auto result = run_turn(provider, std::span(&tool, 1), history, "stop", options, {}, source.get_token());
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().message, "cancelled");
    EXPECT_EQ(calls, 1);
    ASSERT_EQ(history.size(), 4);
    EXPECT_EQ(history.back().tool_call_id, "b");
    EXPECT_NE(history.back().content.find("cancelled"), std::string::npos);
    source = std::stop_source{};
    Provider finish = [](auto, auto, std::stop_token) -> Result<Message> {
        return Message{.role = Role::ASSISTANT, .content = "recovered"};
    };
    EXPECT_TRUE(run_turn(finish, {}, history, "continue"));
}

TEST(CodeAgent, HistoryAndOutputBudgets) {
    std::vector<Message> history;
    int calls = 0;
    Provider provider = [&](auto, auto, std::stop_token) -> Result<Message> {
        ++calls;
        return Message{.role = Role::ASSISTANT, .content = "done"};
    };
    RunOptions options{.max_history_bytes = 40};
    EXPECT_FALSE(run_turn(provider, {}, history, std::string(100, 'x'), options));
    EXPECT_EQ(calls, 0);
    EXPECT_TRUE(history.empty());
    options = RunOptions{};
    options.max_tool_output_bytes = 8;
    Tool tool{{"large", "large output", Json::object()},
              Capability::READ,
              [](const Json&, std::stop_token) -> Result<std::string> { return std::string(1000, 'x'); }};
    std::string expected_error = "max_tool_output_bytes";
    calls = 0;
    provider = [&](std::span<const Message> messages, auto, std::stop_token) -> Result<Message> {
        if (++calls == 1) return tool_reply("large");
        EXPECT_NE(messages.back().content.find(expected_error), std::string::npos);
        return Message{.role = Role::ASSISTANT, .content = "done"};
    };
    EXPECT_TRUE(run_turn(provider, std::span(&tool, 1), history, "read", options));
    history.clear();
    calls = 0;
    options.max_tool_output_bytes = 1024;
    options.max_history_bytes = 512;
    expected_error = "history budget";
    EXPECT_TRUE(run_turn(provider, std::span(&tool, 1), history, "read", options));
}

TEST(CodeAgent, RejectsInvalidSetup) {
    std::vector<Message> history{{.role = Role::TOOL, .content = "orphan", .tool_call_id = "missing"}};
    Provider provider = [](auto, auto, std::stop_token) -> Result<Message> {
        ADD_FAILURE() << "provider must not be called";
        return Message{};
    };
    EXPECT_FALSE(run_turn(provider, {}, history, "task"));
    history.clear();
    int calls = 0;
    std::vector<Tool> tools{test_tool(calls), test_tool(calls)};
    EXPECT_FALSE(run_turn(provider, tools, history, "task"));
    EXPECT_TRUE(history.empty());
    RunOptions options{.max_model_calls = 0};
    EXPECT_FALSE(run_turn(provider, {}, history, "task", options));
    std::stop_source source;
    source.request_stop();
    EXPECT_FALSE(run_turn(provider, {}, history, "task", {}, {}, source.get_token()));
    EXPECT_TRUE(history.empty());
}

} // namespace
} // namespace kidi::code
