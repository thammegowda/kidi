#include <string>

#include <gtest/gtest.h>

#include "kidi/app/document.h"

namespace {

TEST(DocumentTest, KeepsParagraphOrder) {
    auto paragraphs = kidi::app::document_parts("First paragraph.\n\nSecond paragraph.\n\nThird.", 1200, true);
    ASSERT_TRUE(paragraphs) << "paragraph order changed";
    ASSERT_EQ(paragraphs->size(), 3) << "paragraph order changed";
    EXPECT_EQ((*paragraphs)[1].text, "Second paragraph.") << "paragraph order changed";
    for (const auto& part : *paragraphs) EXPECT_TRUE(part.paragraph_end) << "lost a paragraph boundary";
}

TEST(DocumentTest, SplitsLongParagraph) {
    auto split = kidi::app::document_parts("One sentence. Two sentences follow here.\n\nEnd.", 20, true);
    ASSERT_TRUE(split) << "long paragraph was not split at a natural boundary";
    ASSERT_GE(split->size(), 3) << "long paragraph was not split at a natural boundary";
    EXPECT_EQ((*split)[0].text, "One sentence.") << "long paragraph was not split at a natural boundary";
    EXPECT_FALSE((*split)[0].paragraph_end) << "long paragraph was not split at a natural boundary";
    EXPECT_EQ(split->back().text, "End.") << "long paragraph was not split at a natural boundary";
    EXPECT_TRUE(split->back().paragraph_end) << "long paragraph was not split at a natural boundary";
}

TEST(DocumentTest, KeepsCodepointsWhole) {
    auto unicode = kidi::app::document_parts("\xF0\x9F\x98\x80\xF0\x9F\x98\x80\xF0\x9F\x98\x80", 2, true);
    ASSERT_TRUE(unicode) << "Unicode codepoints were split";
    ASSERT_EQ(unicode->size(), 2) << "Unicode codepoints were split";
    EXPECT_EQ((*unicode)[0].text.size(), 8) << "Unicode codepoints were split";
    EXPECT_EQ((*unicode)[1].text.size(), 4) << "Unicode codepoints were split";
}

TEST(DocumentTest, RejectsInvalidInput) {
    EXPECT_FALSE(kidi::app::document_parts("", 1200, true)) << "accepted invalid document input";
    EXPECT_FALSE(kidi::app::document_parts(" \n\n\t ", 1200, true)) << "accepted invalid document input";
    EXPECT_FALSE(kidi::app::document_parts("text", 0, true)) << "accepted invalid document input";
    EXPECT_FALSE(kidi::app::document_parts(std::string(64 * 1024 + 1, 'x'), 1200, true))
        << "accepted invalid document input";
    EXPECT_FALSE(kidi::app::document_parts(std::string("\xff", 1), 1200, true)) << "accepted invalid document input";
}

TEST(DocumentTest, SegmentsManyRawParagraphs) {
    std::string many;
    for (std::size_t index = 0; index < 300; ++index) many += "Paragraph.\n\n";
    auto many_parts = kidi::app::document_parts(many, 1200, true);
    ASSERT_TRUE(many_parts) << "raw paragraph segmentation lost document blocks";
    EXPECT_EQ(many_parts->size(), 300) << "raw paragraph segmentation lost document blocks";
}

TEST(DocumentTest, BuildsTranslationPrompt) {
    auto request = kidi::app::validate_editor_request({
        .task = kidi::app::EditorTask::TRANSLATE,
        .text = "Hello.",
        .system_prompt = "Translate into {target_language}.",
        .target_language = "Kannada",
    });
    ASSERT_TRUE(request) << request.error().message;
    auto messages = kidi::app::editor_messages(*request);
    EXPECT_TRUE(messages[0].content.contains("Kannada")) << "translation prompt was not constructed in C++";
    EXPECT_FALSE(messages[0].content.contains("{target_language}")) << "translation prompt was not constructed in C++";
    EXPECT_TRUE(messages[1].content.contains("<text_to_translate>")) << "translation prompt was not constructed in C++";
    EXPECT_FALSE(kidi::app::validate_editor_request({
        .task = kidi::app::EditorTask::TRANSLATE,
        .text = "Hello.",
    })) << "accepted translation without a target";
}

} // namespace
