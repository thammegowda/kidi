#include "kidi/text/sentence.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>

#include <gtest/gtest.h>

namespace {

auto expect_split(const kidi::text::SentenceSegmenter& segmenter, std::string_view input,
                  std::initializer_list<std::string_view> expected) -> ::testing::AssertionResult {
    auto actual = segmenter.split(input);
    if (!actual) return ::testing::AssertionFailure() << "split failed: " << actual.error().message;
    if (actual->size() != expected.size() || !std::ranges::equal(*actual, expected))
        return ::testing::AssertionFailure() << "unexpected sentences for: " << input;
    return ::testing::AssertionSuccess();
}

class SentenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto path = std::filesystem::temp_directory_path() / "kidi-sentence-prefixes.tsv";
        std::ofstream(path) << "*\talways\tDr\n*\talways\tU.S\nen\tnumeric\tNo\n";
        auto loaded = kidi::text::SentenceSegmenter::load(path);
        std::filesystem::remove(path);
        ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
        segmenter.emplace(std::move(*loaded));
    }

    std::optional<kidi::text::SentenceSegmenter> segmenter;
};

TEST_F(SentenceTest, SegmentsLinguistically) {
    const auto& s = *segmenter;
    EXPECT_TRUE(expect_split(s, "Dr. Smith paid 3.14 dollars. He left at 5 p.m. Then slept.",
                             {"Dr. Smith paid 3.14 dollars.", "He left at 5 p.m.", "Then slept."}))
        << "linguistic sentence segmentation failed";
    EXPECT_TRUE(expect_split(s, "J. R. R. Tolkien wrote it. The U.S. Army read it.",
                             {"J. R. R. Tolkien wrote it.", "The U.S. Army read it."}))
        << "linguistic sentence segmentation failed";
    EXPECT_TRUE(expect_split(s, "She asked, \"Ready?\" He nodded... Then left.",
                             {"She asked, \"Ready?\"", "He nodded...", "Then left."}))
        << "linguistic sentence segmentation failed";
    EXPECT_TRUE(expect_split(s, "She shouted \"Stop!\" then left.", {"She shouted \"Stop!\" then left."}))
        << "linguistic sentence segmentation failed";
    EXPECT_TRUE(expect_split(s, "No. 5 arrived. No. Thanks.", {"No. 5 arrived.", "No.", "Thanks."}))
        << "linguistic sentence segmentation failed";
    EXPECT_TRUE(expect_split(s, "第一句。第二句！第三句？", {"第一句。", "第二句！", "第三句？"}))
        << "linguistic sentence segmentation failed";
    EXPECT_TRUE(expect_split(s, "Chapter heading\n\nAn unpunctuated paragraph",
                             {"Chapter heading", "An unpunctuated paragraph"}))
        << "linguistic sentence segmentation failed";
}

TEST_F(SentenceTest, RejectsInvalidUtf8) {
    std::string invalid{"bad"};
    invalid.push_back(static_cast<char>(0xFF));
    invalid += "text";
    EXPECT_FALSE(segmenter->split(invalid)) << "invalid sentence UTF-8 was accepted";
}

} // namespace
