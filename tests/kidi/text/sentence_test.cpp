#include "kidi/text/sentence.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
auto expect(const kidi::text::SentenceSegmenter& segmenter, std::string_view input,
            std::initializer_list<std::string_view> expected) -> bool {
    auto actual = segmenter.split(input);
    if (!actual || actual->size() != expected.size()) return false;
    return std::ranges::equal(*actual, expected);
}
} // namespace

auto main() -> int {
    const auto path = std::filesystem::temp_directory_path() / "kidi-sentence-prefixes.tsv";
    std::ofstream(path) << "*\talways\tDr\n*\talways\tU.S\nen\tnumeric\tNo\n";
    auto segmenter = kidi::text::SentenceSegmenter::load(path);
    std::filesystem::remove(path);
    if (!segmenter) {
        std::cerr << segmenter.error().message << '\n';
        return 1;
    }
    if (!expect(*segmenter, "Dr. Smith paid 3.14 dollars. He left at 5 p.m. Then slept.",
                {"Dr. Smith paid 3.14 dollars.", "He left at 5 p.m.", "Then slept."}) ||
        !expect(*segmenter, "J. R. R. Tolkien wrote it. The U.S. Army read it.",
                {"J. R. R. Tolkien wrote it.", "The U.S. Army read it."}) ||
        !expect(*segmenter, "She asked, \"Ready?\" He nodded... Then left.",
                {"She asked, \"Ready?\"", "He nodded...", "Then left."}) ||
        !expect(*segmenter, "She shouted \"Stop!\" then left.", {"She shouted \"Stop!\" then left."}) ||
        !expect(*segmenter, "No. 5 arrived. No. Thanks.", {"No. 5 arrived.", "No.", "Thanks."}) ||
        !expect(*segmenter, "第一句。第二句！第三句？", {"第一句。", "第二句！", "第三句？"}) ||
        !expect(*segmenter, "Chapter heading\n\nAn unpunctuated paragraph",
                {"Chapter heading", "An unpunctuated paragraph"})) {
        std::cerr << "linguistic sentence segmentation failed\n";
        return 1;
    }
    std::string invalid{"bad"};
    invalid.push_back(static_cast<char>(0xFF));
    invalid += "text";
    if (segmenter->split(invalid)) {
        std::cerr << "invalid sentence UTF-8 was accepted\n";
        return 1;
    }
    return 0;
}
