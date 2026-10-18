#include "kidi/text/phonemizer.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>

auto main() -> int {
    const auto path = std::filesystem::temp_directory_path() / "kidi-pronunciation-lexicon.tsv";
    std::ofstream(path) << "hello\th\xC9\x99l\xCB\x88O\nworld\tw\xCB\x88\xC9\x9C\xC9\xB9ld\n";
    YAML::Node vocabulary;
    const std::string symbols = "helOwrd .";
    for (std::size_t index = 0; index < symbols.size(); ++index)
        vocabulary[std::string(1, symbols[index])] = static_cast<int>(index + 1);
    vocabulary["\xC9\x99"] = 20;
    vocabulary["\xCB\x88"] = 21;
    vocabulary["\xC9\x9C"] = 22;
    vocabulary["\xC9\xB9"] = 23;
    auto phonemizer = kidi::text::LexiconPhonemizer::load(path, vocabulary);
    std::filesystem::remove(path);
    if (!phonemizer) {
        std::cerr << phonemizer.error().message << '\n';
        return 1;
    }
    auto result = phonemizer->phonemize("Hello world.");
    constexpr std::array<std::int32_t, 13> IDS{1, 20, 3, 21, 4, 8, 5, 21, 22, 23, 3, 7, 9};
    if (!result || result->text != "h\xC9\x99l\xCB\x88O w\xCB\x88\xC9\x9C\xC9\xB9ld." ||
        !std::ranges::equal(result->ids, IDS)) {
        std::cerr << "lexicon phonemization failed\n";
        return 1;
    }
    if (phonemizer->phonemize("unknown")) {
        std::cerr << "unknown lexicon word was accepted\n";
        return 1;
    }
    return 0;
}
