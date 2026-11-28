#include "kidi/agent/agent.h"

#include <gtest/gtest.h>
#include <algorithm>

namespace kidi::agent {
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

TEST(Agent, RoundTrip) {
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

TEST(Agent, ContinuesOnceAfterSilentTools) {
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

TEST(Agent, DeniedToolsStayDenied) {
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

TEST(Agent, ApprovalPerCall) {
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

TEST(Agent, SetupRunsOnceTheUserAgrees) {
    int calls = 0, asked = 0, runs = 0;
    bool ready = false;
    auto tool = test_tool(calls, Capability::NETWORK);
    tool.setup = std::make_shared<const ToolSetup>(ToolSetup{
        "Example",
        [&]() -> std::optional<std::string> {
            if (ready) return std::nullopt;
            return "sign in to Example";
        },
        [&](const BrowserSignIn& browser, std::stop_token) -> Result<void> {
            ++runs;
            EXPECT_EQ(browser.redirect_uri, "http://127.0.0.1:1/oauth/callback");
            ready = true;
            return {};
        }});
    std::vector<Tool> tools{tool};
    RunOptions options;
    options.permissions.network = Access::ALLOW;
    std::vector<std::string> results;
    const auto provider = [&](std::span<const Message> messages, auto, std::stop_token) -> Result<Message> {
        if (messages.back().role == Role::TOOL) {
            results.push_back(messages.back().content);
            return Message{.role = Role::ASSISTANT, .content = "done"};
        }
        return tool_reply();
    };
    const auto turn = [&](Callbacks callbacks) {
        std::vector<Message> history;
        EXPECT_TRUE(run_turn(provider, tools, history, "go", options, callbacks));
    };
    turn({});
    EXPECT_NE(results.back().find("needs Example to be connected (sign in to Example)"), std::string::npos)
        << "without a front end that can ask, the model learns how to get set up";
    Callbacks declining{.setup = [&](const ToolSpec& spec, const ToolSetup& setup, const std::string& reason)
                            -> Result<std::optional<BrowserSignIn>> {
        ++asked;
        EXPECT_EQ(spec.name, "edit");
        EXPECT_EQ(setup.service, "Example");
        EXPECT_EQ(reason, "sign in to Example");
        return std::optional<BrowserSignIn>{};
    }};
    turn(declining);
    EXPECT_NE(results.back().find("The user chose not to connect Example now"), std::string::npos);
    EXPECT_EQ(runs, 0);
    Callbacks accepting{.setup = [&](const ToolSpec&, const ToolSetup&, const std::string&)
                            -> Result<std::optional<BrowserSignIn>> {
        ++asked;
        return std::optional<BrowserSignIn>(BrowserSignIn{"http://127.0.0.1:1/oauth/callback", {}});
    }};
    turn(accepting);
    EXPECT_EQ(results.back(), "edited");
    EXPECT_EQ(runs, 1);
    turn(accepting);
    EXPECT_EQ(asked, 2) << "a ready tool asks nothing";
    EXPECT_EQ(calls, 2);
    options.permissions.network = Access::DENY;
    turn(accepting);
    EXPECT_EQ(asked, 2) << "a denied tool never asks for setup";
}

TEST(Agent, FailedCheckSkipsApprovalAndCall) {
    int calls = 0, approvals = 0, requests = 0;
    auto tool = test_tool(calls);
    tool.check = [](const ToolCall&, std::span<const Message> history) -> Result<void> {
        EXPECT_EQ(history.back().role, Role::ASSISTANT) << "the check sees the conversation up to the call";
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "not allowed here"});
    };
    std::vector<Tool> tools{tool};
    std::vector<Message> history;
    Provider provider = [&](std::span<const Message> messages, auto, std::stop_token) -> Result<Message> {
        if (++requests == 1) return tool_reply();
        EXPECT_EQ(messages.back().content, R"({"error":"not allowed here"})");
        return Message{.role = Role::ASSISTANT, .content = "done"};
    };
    Callbacks callbacks{.approve = [&](const ToolSpec&, const ToolCall&) -> Result<bool> {
        ++approvals;
        return true;
    }};
    EXPECT_TRUE(run_turn(provider, tools, history, "edit", {}, callbacks));
    EXPECT_EQ(approvals, 0);
    EXPECT_EQ(calls, 0);
}

TEST(Agent, MissingApprovalFailsClosed) {
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

TEST(Agent, BadCallsNeverExecute) {
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

TEST(Agent, UnknownAndFailedTools) {
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

TEST(Agent, LimitsBoundExecution) {
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

TEST(Agent, CancelPairsWholeBatch) {
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

TEST(Agent, HistoryAndOutputBudgets) {
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

TEST(Agent, RejectsInvalidSetup) {
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

auto counted(std::span<const Message> messages, std::span<const ToolSpec> tools, std::stop_token)
    -> Result<ContextUsage> {
    std::size_t bytes = tools.size() * 80;
    for (const auto& message : messages) {
        bytes += message.content.size() + message.reasoning_content.size() + 32;
        for (const auto& call : message.tool_calls) bytes += call.arguments.dump().size() + 32;
    }
    return ContextUsage{bytes / 4 + 1, 4096, 512};
}

auto compactable_history() -> std::vector<Message> {
    return {{.role = Role::SYSTEM, .content = "Original trusted rules"},
            {.role = Role::USER, .content = "Inspect the project"},
            {.role = Role::ASSISTANT, .tool_calls = {{"old", "read_file", {{"path", "source"}}}}},
            {.role = Role::TOOL, .content = std::string(26000, 'x'), .tool_call_id = "old"},
            {.role = Role::ASSISTANT, .tool_calls = {{"recent", "read_file", {{"path", "README"}}}}},
            {.role = Role::TOOL, .content = "README inspected", .tool_call_id = "recent"}};
}

TEST(Agent, CompactsWithoutReplayingTools) {
    int calls = 0, executions = 0, summaries = 0, compactions = 0;
    bool saw_calls = false;
    Provider provider = [&](std::span<const Message> messages, auto tools, std::stop_token stop) -> Result<Message> {
        EXPECT_LT(counted(messages, tools, stop)->prompt_tokens, 4096);
        EXPECT_EQ(messages.front().content, "Trusted rules");
        if (++calls <= 3) return tool_reply("large");
        EXPECT_TRUE(messages[1].is_summary);
        EXPECT_EQ(messages[2].content, "Read source files");
        return Message{.role = Role::ASSISTANT, .content = "done"};
    };
    provider.context = counted;
    provider.summarize = [&](std::span<const Message> messages, std::size_t maximum,
                             std::stop_token stop) -> Result<Message> {
        ++summaries;
        EXPECT_LT(counted(messages, {}, stop)->prompt_tokens + maximum, 4096);
        EXPECT_EQ(messages.front().content, "Summarize factual progress");
        saw_calls |= messages.back().content.find("tool_calls") != std::string::npos;
        return Message{.role = Role::ASSISTANT, .content = "The requested source read completed. Continue inspection."};
    };
    Tool tool{{"large", "large output", Json::object()},
              Capability::READ,
              [&](const Json&, std::stop_token) -> Result<std::string> {
                  ++executions;
                  return std::string(16000, 'x');
              }};
    std::vector<Message> history{{.role = Role::SYSTEM, .content = "Trusted rules"}};
    RunOptions options{.compact_prompt = "Summarize factual progress"};
    Callbacks callbacks{.observe = [&](const Event& event) {
        if (event.kind == EventKind::COMPACT_FINISH) ++compactions;
    }};
    auto result = run_turn(provider, std::span(&tool, 1), history, "Read source files", options, callbacks);
    ASSERT_TRUE(result) << result.error().message;
    EXPECT_EQ(executions, 3);
    EXPECT_EQ(result->model_calls, 4);
    EXPECT_EQ(result->tool_calls, 3);
    EXPECT_EQ(result->compactions, 3);
    EXPECT_EQ(compactions, 3);
    EXPECT_GE(summaries, 3);
    EXPECT_TRUE(saw_calls);
}

TEST(Agent, OverfullHistoryAndRecentPair) {
    auto history = compactable_history();
    int summaries = 0;
    Provider provider = [](auto, auto, std::stop_token) -> Result<Message> {
        return Message{.role = Role::ASSISTANT, .content = "continued"};
    };
    provider.context = counted;
    provider.summarize = [&](auto messages, std::size_t maximum, std::stop_token stop) -> Result<Message> {
        ++summaries;
        EXPECT_LT(counted(messages, {}, stop)->prompt_tokens + maximum, 4096);
        return Message{.role = Role::ASSISTANT, .content = "Source inspected; README remains in recent context."};
    };
    RunOptions options{.compact_prompt = "Summarize"};
    auto result = compact_history(provider, {}, history, options);
    ASSERT_TRUE(result) << result.error().message;
    EXPECT_GE(summaries, 2);
    ASSERT_EQ(history.size(), 5);
    EXPECT_EQ(history[0].content, "Original trusted rules");
    EXPECT_TRUE(history[1].is_summary);
    EXPECT_EQ(history[2].content, "Inspect the project");
    EXPECT_EQ(history[3].tool_calls[0].id, "recent");
    EXPECT_EQ(history[4].tool_call_id, "recent");
    EXPECT_TRUE(run_turn(provider, {}, history, "Continue", options));
}

TEST(Agent, FailedSummaryDropsOlderHistory) {
    for (int failure = 0; failure < 4; ++failure) {
        SCOPED_TRACE(failure);
        auto history = compactable_history();
        Provider provider = [](auto, auto, std::stop_token) -> Result<Message> { return Message{}; };
        provider.context = counted;
        provider.summarize = [&](auto, std::size_t, std::stop_token) -> Result<Message> {
            if (failure == 0) return std::unexpected(Error{ErrorCode::RUNTIME, "did not complete: length"});
            if (failure == 1) return Message{.role = Role::ASSISTANT, .content = "   "};
            if (failure == 2) return tool_reply();
            return Message{.role = Role::ASSISTANT, .content = "first fragment"};
        };
        RunOptions options{.compact_prompt = "Summarize", .max_summary_calls = failure == 3 ? 1u : 16u};
        ASSERT_TRUE(compact_history(provider, {}, history, options));
        ASSERT_EQ(history.size(), 5);
        EXPECT_TRUE(history[1].is_summary);
        EXPECT_NE(history[1].content.find("dropped"), std::string::npos);
        EXPECT_EQ(history[1].content.find("first fragment") != std::string::npos, failure == 3);
        EXPECT_EQ(history[2].content, "Inspect the project");
        EXPECT_EQ(history.back().tool_call_id, "recent");
    }
}

TEST(Agent, CancelledCompactionKeepsHistory) {
    auto history = compactable_history();
    std::stop_source source;
    Provider provider = [](auto, auto, std::stop_token) -> Result<Message> { return Message{}; };
    provider.context = counted;
    provider.summarize = [&](auto, std::size_t, std::stop_token) -> Result<Message> {
        source.request_stop();
        return Message{.role = Role::ASSISTANT, .content = "summary"};
    };
    ASSERT_FALSE(compact_history(provider, {}, history, {.compact_prompt = "Summarize"}, {}, source.get_token()));
    ASSERT_EQ(history.size(), 6);
    EXPECT_EQ(history[3].content, std::string(26000, 'x'));
}

TEST(Agent, CompactionLimitsAndLargeTask) {
    int executions = 0;
    Provider provider = [](auto, auto, std::stop_token) -> Result<Message> { return tool_reply("large"); };
    provider.context = counted;
    provider.summarize = [](auto, std::size_t, std::stop_token) -> Result<Message> {
        return Message{.role = Role::ASSISTANT, .content = "Read completed"};
    };
    Tool tool{{"large", "large output", Json::object()},
              Capability::READ,
              [&](const Json&, std::stop_token) -> Result<std::string> {
                  ++executions;
                  return std::string(16000, 'x');
              }};
    RunOptions options{.compact_prompt = "Summarize", .max_compactions = 1};
    std::vector<Message> history;
    auto result = run_turn(provider, std::span(&tool, 1), history, "Read", options);
    ASSERT_FALSE(result);
    EXPECT_TRUE(result.error().message.starts_with("context is too small for this task"));
    EXPECT_EQ(executions, 2);
    EXPECT_EQ(history.back().role, Role::TOOL);
    history.clear();
    result = run_turn(provider, {}, history, std::string(20000, 'x'), options);
    ASSERT_FALSE(result);
    EXPECT_NE(result.error().message.find("shorten the request"), std::string::npos);
    EXPECT_EQ(executions, 2);
}

TEST(Agent, ByteBudgetPinsIncomingTask) {
    auto history = compactable_history();
    Provider provider = [](auto, auto, std::stop_token) -> Result<Message> {
        return Message{.role = Role::ASSISTANT, .content = "done"};
    };
    provider.context = counted;
    provider.summarize = [](auto, std::size_t, std::stop_token) -> Result<Message> {
        return Message{.role = Role::ASSISTANT, .content = "Old work complete"};
    };
    RunOptions options{.max_history_bytes = 27000, .compact_prompt = "Summarize"};
    const std::string incoming = "NEW TASK " + std::string(1500, 'z');
    auto result = run_turn(provider, {}, history, incoming, options);
    ASSERT_TRUE(result) << result.error().message;
    EXPECT_EQ(result->compactions, 1);
    EXPECT_EQ(history[2].content, incoming);
}

TEST(Agent, NoUsefulReductionKeepsHistory) {
    std::vector<Message> history{{.role = Role::USER, .content = "Task"},
                                 {.role = Role::ASSISTANT, .content = std::string(200, 'x')}};
    Provider provider = [](auto, auto, std::stop_token) -> Result<Message> { return Message{}; };
    provider.context = counted;
    provider.summarize = [](auto, std::size_t, std::stop_token) -> Result<Message> {
        return Message{.role = Role::ASSISTANT, .content = std::string(700, 'y')};
    };
    auto result = compact_history(provider, {}, history, {.compact_prompt = "Summarize"});
    ASSERT_FALSE(result);
    EXPECT_NE(result.error().message.find("did not free enough"), std::string::npos);
    EXPECT_EQ(history.size(), 2);
    EXPECT_EQ(history.back().content, std::string(200, 'x'));
}

TEST(Agent, LargeFirstRequestStillFits) {
    Provider provider = [](auto, auto, std::stop_token) -> Result<Message> {
        return Message{.role = Role::ASSISTANT, .content = "done"};
    };
    provider.context = counted;
    provider.summarize = [](auto, std::size_t, std::stop_token) -> Result<Message> {
        ADD_FAILURE() << "there is no older history to summarize";
        return Message{};
    };
    std::vector<Message> history;
    auto copied = provider;
    auto result = run_turn(copied, {}, history, std::string(13000, 'x'), {.compact_prompt = "Summarize"});
    ASSERT_TRUE(result) << result.error().message;
    EXPECT_EQ(result->compactions, 0);
}

} // namespace
} // namespace kidi::agent
