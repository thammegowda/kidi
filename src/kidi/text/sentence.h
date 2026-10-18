#pragma once

#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kidi/core/error.h"

namespace kidi::text {

class SentenceSegmenter {
public:
    static auto load(const std::filesystem::path& prefixes) -> Result<SentenceSegmenter>;
    auto split(std::string_view text, std::string_view language = "en") const -> Result<std::vector<std::string>>;

private:
    struct Prefixes {
        std::set<std::string, std::less<>> always;
        std::set<std::string, std::less<>> numeric;
    };

    explicit SentenceSegmenter(std::map<std::string, Prefixes, std::less<>> prefixes)
        : prefixes_(std::move(prefixes)) {}

    auto contains(std::string_view language, std::string_view prefix, bool numeric) const -> bool;

    std::map<std::string, Prefixes, std::less<>> prefixes_;
};

} // namespace kidi::text
