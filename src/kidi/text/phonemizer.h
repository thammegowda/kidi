#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "kidi/core/error.h"

namespace kidi::text {

struct PhonemeSequence {
    std::string text;
    std::vector<std::int32_t> ids;
};

class LexiconPhonemizer {
public:
    static auto load(const std::filesystem::path& lexicon, const YAML::Node& vocabulary) -> Result<LexiconPhonemizer>;
    auto phonemize(std::string_view text) const -> Result<PhonemeSequence>;

private:
    LexiconPhonemizer(std::map<std::string, std::string, std::less<>> lexicon,
                      std::map<char32_t, std::int32_t> vocabulary)
        : lexicon_(std::move(lexicon)), vocabulary_(std::move(vocabulary)) {}

    std::map<std::string, std::string, std::less<>> lexicon_;
    std::map<char32_t, std::int32_t> vocabulary_;
};

} // namespace kidi::text
