#include <iostream>
#include <string>

#include "kidi/app/document.h"

namespace {
auto fail(std::string_view message) -> int {
    std::cerr << message << '\n';
    return 1;
}
} // namespace

auto main() -> int {
    auto paragraphs = kidi::app::document_parts("First paragraph.\n\nSecond paragraph.\n\nThird.", 1200, true);
    if (!paragraphs || paragraphs->size() != 3 || (*paragraphs)[1].text != "Second paragraph.")
        return fail("paragraph order changed");
    for (const auto& part : *paragraphs)
        if (!part.paragraph_end) return fail("lost a paragraph boundary");

    auto split = kidi::app::document_parts("One sentence. Two sentences follow here.\n\nEnd.", 20, true);
    if (!split || split->size() < 3 || (*split)[0].text != "One sentence." || (*split)[0].paragraph_end ||
        split->back().text != "End." || !split->back().paragraph_end)
        return fail("long paragraph was not split at a natural boundary");
    auto unicode = kidi::app::document_parts("\xF0\x9F\x98\x80\xF0\x9F\x98\x80\xF0\x9F\x98\x80", 2, true);
    if (!unicode || unicode->size() != 2 || (*unicode)[0].text.size() != 8 || (*unicode)[1].text.size() != 4)
        return fail("Unicode codepoints were split");
    if (kidi::app::document_parts("", 1200, true) || kidi::app::document_parts(" \n\n\t ", 1200, true) ||
        kidi::app::document_parts("text", 0, true) ||
        kidi::app::document_parts(std::string(64 * 1024 + 1, 'x'), 1200, true) ||
        kidi::app::document_parts(std::string("\xff", 1), 1200, true))
        return fail("accepted invalid document input");
    std::string many;
    for (std::size_t index = 0; index < 300; ++index) many += "Paragraph.\n\n";
    auto many_parts = kidi::app::document_parts(many, 1200, true);
    if (!many_parts || many_parts->size() != 300) return fail("raw paragraph segmentation lost document blocks");

    auto request = kidi::app::validate_editor_request({
        .task = kidi::app::EditorTask::TRANSLATE,
        .text = "Hello.",
        .system_prompt = "Translate into {target_language}.",
        .target_language = "Kannada",
    });
    if (!request) return fail(request.error().message);
    auto messages = kidi::app::editor_messages(*request);
    if (!messages[0].content.contains("Kannada") || messages[0].content.contains("{target_language}") ||
        !messages[1].content.contains("<text_to_translate>"))
        return fail("translation prompt was not constructed in C++");
    if (kidi::app::validate_editor_request({
            .task = kidi::app::EditorTask::TRANSLATE,
            .text = "Hello.",
        }))
        return fail("accepted translation without a target");
    return 0;
}
