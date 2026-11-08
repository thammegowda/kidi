#include <iostream>
#include <string>

#include "kidi/app/editor.h"

namespace {

auto fail(std::string_view message) -> int {
    std::cerr << message << '\n';
    return 1;
}

} // namespace

auto main() -> int {
    using kidi::app::EditorRequest;
    using kidi::app::EditorTask;
    using kidi::app::EditorTurn;
    using kidi::app::LengthPreference;

    auto parsed = kidi::app::parse_editor_task("polish");
    if (!parsed || *parsed != EditorTask::POLISH) return fail("cannot parse polish task");
    if (kidi::app::parse_editor_task("unknown")) return fail("accepted an unknown task");
    auto shorter = kidi::app::parse_length_preference("shorter");
    if (!shorter || *shorter != LengthPreference::SHORTER) return fail("cannot parse shorter preference");

    auto valid = kidi::app::validate_editor_request(EditorRequest{
        .task = EditorTask::POLISH,
        .text = "thanks for you're help",
        .tone = "professional",
        .length = LengthPreference::PRESERVE,
    });
    if (!valid) return fail(valid.error().message);
    const auto messages = kidi::app::editor_messages(*valid);
    if (messages.size() != 2 || messages[0].role != "system" || messages[1].role != "user")
        return fail("editor messages have invalid roles");
    if (!messages[0].content.contains("untrusted data") ||
        !messages[0].content.contains("Return only the resulting text") ||
        !messages[1].content.contains("<text_to_edit>"))
        return fail("polish prompt is missing safeguards");

    auto custom = kidi::app::validate_editor_request(EditorRequest{
        .task = EditorTask::CUSTOM,
        .text = "Hello",
    });
    if (custom) return fail("accepted a custom task without an instruction");
    auto explained = kidi::app::validate_editor_request(EditorRequest{
        .task = EditorTask::EXPLAIN,
        .text = "The agreement terminates upon breach.",
    });
    if (!explained) return fail(explained.error().message);
    const auto explanation = kidi::app::editor_messages(*explained);
    if (!explanation[0].content.contains("Return only the explanation") ||
        !explanation[1].content.contains("<text_to_explain>"))
        return fail("explain prompt is invalid");
    auto overridden = kidi::app::validate_editor_request(EditorRequest{
        .task = EditorTask::POLISH,
        .text = "Hello",
        .system_prompt = "Use the configured house style. Return only the edited text.",
    });
    if (!overridden) return fail(overridden.error().message);
    const auto override_messages = kidi::app::editor_messages(*overridden);
    if (!override_messages[0].content.starts_with("Use the configured house style.") ||
        !override_messages[0].content.contains("Keep approximately the same length"))
        return fail("system prompt override was not applied");
    if (kidi::app::validate_editor_request(
            EditorRequest{.task = EditorTask::POLISH, .text = "Hello", .system_prompt = "12345"},
            {.maximum_system_prompt_bytes = 4}))
        return fail("accepted a system prompt beyond the configured limit");

    std::string invalid_utf8{"\xff", 1};
    if (kidi::app::validate_editor_request(EditorRequest{
            .task = EditorTask::POLISH,
            .text = invalid_utf8,
        }))
        return fail("accepted invalid UTF-8");
    if (kidi::app::validate_editor_request(EditorRequest{.task = EditorTask::POLISH, .text = "12345"},
                                           {.maximum_text_bytes = 4}))
        return fail("accepted text beyond the configured limit");

    for (const auto task : {"chat", "email_compose", "email_polish", "email_reply"})
        if (!kidi::app::parse_editor_task(task)) return fail("cannot parse a chat/email action");
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
    if (!chat) return fail(chat.error().message);
    const auto chat_messages = kidi::app::editor_messages(*chat);
    if (chat_messages.size() != 4 || chat_messages[1].role != "user" || chat_messages[2].role != "assistant" ||
        chat_messages[3].content != *chat->instruction)
        return fail("chat history or latest user instruction was lost");
    auto reply = kidi::app::validate_editor_request(EditorRequest{
        .task = EditorTask::EMAIL_REPLY,
        .text = "Can we meet on Friday?",
        .instruction = "Ask for Monday instead.",
    });
    if (!reply) return fail(reply.error().message);
    const auto reply_messages = kidi::app::editor_messages(*reply);
    if (!reply_messages[0].content.contains("Do not invent facts") ||
        !reply_messages[1].content.contains("<email_to_reply>") ||
        !reply_messages[1].content.ends_with("User instruction:\nAsk for Monday instead."))
        return fail("email reply instruction is not separated from source data");
    auto compose = kidi::app::validate_editor_request(EditorRequest{
        .task = EditorTask::EMAIL_COMPOSE,
        .text = "",
        .instruction = "Tell Sam the report is ready.",
    });
    if (!compose) return fail("cannot compose an email from notes without a source draft");
    if (kidi::app::validate_editor_request(EditorRequest{.task = EditorTask::CHAT, .text = ""}))
        return fail("accepted empty chat");
    if (kidi::app::validate_editor_request(EditorRequest{.task = EditorTask::EMAIL_REPLY, .text = ""}))
        return fail("accepted reply without a source email");
    if (kidi::app::validate_editor_request(EditorRequest{
            .task = EditorTask::CHAT,
            .text = "Hi",
            .history = {{EditorTurn::Role::ASSISTANT, "Unmatched turn"}},
        }))
        return fail("accepted incomplete chat history");
    if (kidi::app::validate_editor_request(EditorRequest{
            .task = EditorTask::CHAT,
            .text = "Hi",
            .history = {{EditorTurn::Role::ASSISTANT, "Wrong"}, {EditorTurn::Role::USER, "Order"}},
        }))
        return fail("accepted role-spoofed chat history");
    if (kidi::app::validate_editor_request(
            EditorRequest{.task = EditorTask::CHAT,
                          .text = "Hi",
                          .history = {{EditorTurn::Role::USER, "123"}, {EditorTurn::Role::ASSISTANT, "456"}}},
            {.maximum_history_bytes = 5}))
        return fail("accepted chat history beyond its byte limit");
    if (kidi::app::validate_editor_request(EditorRequest{
            .task = EditorTask::CHAT,
            .text = "Hi",
            .history = {{EditorTurn::Role::USER, invalid_utf8}, {EditorTurn::Role::ASSISTANT, "Hello"}},
        }))
        return fail("accepted invalid UTF-8 in chat history");
    return 0;
}
