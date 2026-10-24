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
    if (kidi::app::validate_editor_request(
            EditorRequest{.task = EditorTask::POLISH, .text = "12345"},
            {.maximum_text_bytes = 4}))
        return fail("accepted text beyond the configured limit");

    return 0;
}
