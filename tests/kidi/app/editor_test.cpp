#include <string>

#include <gtest/gtest.h>

#include "kidi/app/editor.h"

namespace {

using kidi::app::EditorRequest;
using kidi::app::EditorTask;
using kidi::app::EditorTurn;
using kidi::app::LengthPreference;

const std::string invalid_utf8{"\xff", 1};

TEST(EditorTest, ParsesTasksAndPreferences) {
    auto parsed = kidi::app::parse_editor_task("polish");
    ASSERT_TRUE(parsed) << "cannot parse polish task";
    EXPECT_EQ(*parsed, EditorTask::POLISH) << "cannot parse polish task";
    EXPECT_FALSE(kidi::app::parse_editor_task("unknown")) << "accepted an unknown task";
    auto shorter = kidi::app::parse_length_preference("shorter");
    ASSERT_TRUE(shorter) << "cannot parse shorter preference";
    EXPECT_EQ(*shorter, LengthPreference::SHORTER) << "cannot parse shorter preference";
    for (const auto task : {"chat", "email_compose", "email_polish", "email_reply"})
        EXPECT_TRUE(kidi::app::parse_editor_task(task)) << "cannot parse a chat/email action: " << task;
}

TEST(EditorTest, PolishPromptHasSafeguards) {
    auto valid = kidi::app::validate_editor_request(EditorRequest{
        .task = EditorTask::POLISH,
        .text = "thanks for you're help",
        .tone = "professional",
        .length = LengthPreference::PRESERVE,
    });
    ASSERT_TRUE(valid) << valid.error().message;
    const auto messages = kidi::app::editor_messages(*valid);
    ASSERT_EQ(messages.size(), 2) << "editor messages have invalid roles";
    EXPECT_EQ(messages[0].role, "system") << "editor messages have invalid roles";
    EXPECT_EQ(messages[1].role, "user") << "editor messages have invalid roles";
    EXPECT_TRUE(messages[0].content.contains("untrusted data")) << "polish prompt is missing safeguards";
    EXPECT_TRUE(messages[0].content.contains("Return only the resulting text")) << "polish prompt is missing safeguards";
    EXPECT_TRUE(messages[1].content.contains("<text_to_edit>")) << "polish prompt is missing safeguards";
}

TEST(EditorTest, CustomTaskRequiresInstruction) {
    auto custom = kidi::app::validate_editor_request(EditorRequest{
        .task = EditorTask::CUSTOM,
        .text = "Hello",
    });
    EXPECT_FALSE(custom) << "accepted a custom task without an instruction";
}

TEST(EditorTest, ExplainPrompt) {
    auto explained = kidi::app::validate_editor_request(EditorRequest{
        .task = EditorTask::EXPLAIN,
        .text = "The agreement terminates upon breach.",
    });
    ASSERT_TRUE(explained) << explained.error().message;
    const auto explanation = kidi::app::editor_messages(*explained);
    EXPECT_TRUE(explanation[0].content.contains("Return only the explanation")) << "explain prompt is invalid";
    EXPECT_TRUE(explanation[1].content.contains("<text_to_explain>")) << "explain prompt is invalid";
}

TEST(EditorTest, SystemPromptOverride) {
    auto overridden = kidi::app::validate_editor_request(EditorRequest{
        .task = EditorTask::POLISH,
        .text = "Hello",
        .system_prompt = "Use the configured house style. Return only the edited text.",
    });
    ASSERT_TRUE(overridden) << overridden.error().message;
    const auto override_messages = kidi::app::editor_messages(*overridden);
    EXPECT_TRUE(override_messages[0].content.starts_with("Use the configured house style."))
        << "system prompt override was not applied";
    EXPECT_TRUE(override_messages[0].content.contains("Keep approximately the same length"))
        << "system prompt override was not applied";
    EXPECT_FALSE(kidi::app::validate_editor_request(
        EditorRequest{.task = EditorTask::POLISH, .text = "Hello", .system_prompt = "12345"},
        {.maximum_system_prompt_bytes = 4}))
        << "accepted a system prompt beyond the configured limit";
}

TEST(EditorTest, RejectsInvalidText) {
    EXPECT_FALSE(kidi::app::validate_editor_request(EditorRequest{
        .task = EditorTask::POLISH,
        .text = invalid_utf8,
    })) << "accepted invalid UTF-8";
    EXPECT_FALSE(kidi::app::validate_editor_request(EditorRequest{.task = EditorTask::POLISH, .text = "12345"},
                                                    {.maximum_text_bytes = 4}))
        << "accepted text beyond the configured limit";
}

TEST(EditorTest, ChatKeepsHistory) {
    auto chat = kidi::app::validate_editor_request(EditorRequest{
        .task = EditorTask::CHAT,
        .text = "",
        .instruction = "What is a polite way to ask for feedback?",
        .history =
            {
                {EditorTurn::Role::USER, "Hello"},
                {EditorTurn::Role::ASSISTANT, "How can I help?"},
            },
    });
    ASSERT_TRUE(chat) << chat.error().message;
    const auto chat_messages = kidi::app::editor_messages(*chat);
    ASSERT_EQ(chat_messages.size(), 4) << "chat history or latest user instruction was lost";
    EXPECT_EQ(chat_messages[1].role, "user") << "chat history or latest user instruction was lost";
    EXPECT_EQ(chat_messages[2].role, "assistant") << "chat history or latest user instruction was lost";
    EXPECT_EQ(chat_messages[3].content, *chat->instruction) << "chat history or latest user instruction was lost";
}

TEST(EditorTest, EmailReply) {
    auto reply = kidi::app::validate_editor_request(EditorRequest{
        .task = EditorTask::EMAIL_REPLY,
        .text = "Can we meet on Friday?",
        .instruction = "Ask for Monday instead.",
    });
    ASSERT_TRUE(reply) << reply.error().message;
    const auto reply_messages = kidi::app::editor_messages(*reply);
    EXPECT_TRUE(reply_messages[0].content.contains("Do not invent facts"))
        << "email reply instruction is not separated from source data";
    EXPECT_TRUE(reply_messages[1].content.contains("<email_to_reply>"))
        << "email reply instruction is not separated from source data";
    EXPECT_TRUE(reply_messages[1].content.ends_with("User instruction:\nAsk for Monday instead."))
        << "email reply instruction is not separated from source data";
}

TEST(EditorTest, EmailComposeWithoutSource) {
    auto compose = kidi::app::validate_editor_request(EditorRequest{
        .task = EditorTask::EMAIL_COMPOSE,
        .text = "",
        .instruction = "Tell Sam the report is ready.",
    });
    EXPECT_TRUE(compose) << "cannot compose an email from notes without a source draft";
}

TEST(EditorTest, RejectsInvalidRequests) {
    EXPECT_FALSE(kidi::app::validate_editor_request(EditorRequest{.task = EditorTask::CHAT, .text = ""}))
        << "accepted empty chat";
    EXPECT_FALSE(kidi::app::validate_editor_request(EditorRequest{.task = EditorTask::EMAIL_REPLY, .text = ""}))
        << "accepted reply without a source email";
    EXPECT_FALSE(kidi::app::validate_editor_request(EditorRequest{
        .task = EditorTask::CHAT,
        .text = "Hi",
        .history = {{EditorTurn::Role::ASSISTANT, "Unmatched turn"}},
    })) << "accepted incomplete chat history";
    EXPECT_FALSE(kidi::app::validate_editor_request(EditorRequest{
        .task = EditorTask::CHAT,
        .text = "Hi",
        .history = {{EditorTurn::Role::ASSISTANT, "Wrong"}, {EditorTurn::Role::USER, "Order"}},
    })) << "accepted role-spoofed chat history";
    EXPECT_FALSE(kidi::app::validate_editor_request(
        EditorRequest{.task = EditorTask::CHAT,
                      .text = "Hi",
                      .history = {{EditorTurn::Role::USER, "123"}, {EditorTurn::Role::ASSISTANT, "456"}}},
        {.maximum_history_bytes = 5}))
        << "accepted chat history beyond its byte limit";
    EXPECT_FALSE(kidi::app::validate_editor_request(EditorRequest{
        .task = EditorTask::CHAT,
        .text = "Hi",
        .history = {{EditorTurn::Role::USER, invalid_utf8}, {EditorTurn::Role::ASSISTANT, "Hello"}},
    })) << "accepted invalid UTF-8 in chat history";
}

} // namespace
