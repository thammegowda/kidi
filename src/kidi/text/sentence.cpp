#include "kidi/text/sentence.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <optional>

#include <uni_algo/case.h>
#include <uni_algo/conv.h>
#include <uni_algo/prop.h>

namespace kidi::text {
namespace {

auto whitespace(char32_t value) -> bool { return una::codepoint::is_whitespace(value); }
auto newline(char32_t value) -> bool { return value == U'\n' || value == U'\r'; }
auto digit(char32_t value) -> bool { return value >= U'0' && value <= U'9'; }

auto terminator(char32_t value) -> bool {
    constexpr std::array VALUES{
        U'.',      U'!',      U'?',      U'\u0589', U'\u061F', U'\u06D4', U'\u0700', U'\u0701', U'\u0702', U'\u0964',
        U'\u0965', U'\u104A', U'\u104B', U'\u1362', U'\u166E', U'\u1803', U'\u1809', U'\u2026', U'\u2E2E', U'\u3002',
        U'\uA4FF', U'\uA60E', U'\uA60F', U'\uFE52', U'\uFE56', U'\uFE57', U'\uFF01', U'\uFF0E', U'\uFF1F',
    };
    return std::ranges::find(VALUES, value) != VALUES.end();
}

auto closer(char32_t value) -> bool {
    constexpr std::array VALUES{
        U'"',      U'\'',     U')',      U']',      U'}',      U'\u00BB', U'\u2019', U'\u201D', U'\u203A',
        U'\u3009', U'\u300B', U'\u300D', U'\u300F', U'\u3011', U'\u3015', U'\u3017', U'\u3019', U'\u301B',
    };
    return std::ranges::find(VALUES, value) != VALUES.end();
}

auto opener(char32_t value) -> bool {
    constexpr std::array VALUES{
        U'"',      U'\'',     U'(',      U'[',      U'{',      U'\u00AB', U'\u00BF', U'\u00A1', U'\u2018', U'\u201C',
        U'\u2039', U'\u3008', U'\u300A', U'\u300C', U'\u300E', U'\u3010', U'\u3014', U'\u3016', U'\u3018', U'\u301A',
    };
    return std::ranges::find(VALUES, value) != VALUES.end();
}

auto normalized_language(std::string_view language) -> std::string {
    std::string result;
    for (const auto value : language) {
        if (value == '-' || value == '_' || value == ' ') break;
        result.push_back(value >= 'A' && value <= 'Z' ? static_cast<char>(value - 'A' + 'a') : value);
    }
    if (result == "english") return "en";
    if (result == "german") return "de";
    if (result == "french") return "fr";
    if (result == "chinese" || result == "mandarin") return "zh";
    return result.empty() || result == "auto" || result == "none" ? "en" : result;
}

auto word_before(std::u32string_view text, std::size_t end) -> std::string {
    auto begin = end;
    while (begin > 0) {
        const auto value = text[begin - 1];
        if (una::codepoint::is_alphabetic(value) || una::codepoint::is_numeric(value) || value == U'.' ||
            value == U'-' || value == U'/')
            --begin;
        else
            break;
    }
    return una::cases::to_lowercase_utf8(una::utf32to8(text.substr(begin, end - begin)));
}

auto next_content(std::u32string_view text, std::size_t begin) -> std::optional<char32_t> {
    while (begin < text.size() && (whitespace(text[begin]) || closer(text[begin]) || opener(text[begin]))) ++begin;
    if (begin == text.size()) return {};
    return text[begin];
}

auto sentence_starter(char32_t value) -> bool {
    if (una::codepoint::is_uppercase(value) || una::codepoint::is_numeric(value)) return true;
    return una::codepoint::is_alphabetic(value) && !una::codepoint::is_lowercase(value);
}

auto emit(std::vector<std::string>& output, std::u32string_view text, std::size_t begin, std::size_t end) -> void {
    while (begin < end && whitespace(text[begin])) ++begin;
    while (end > begin && whitespace(text[end - 1])) --end;
    if (begin < end) output.push_back(una::utf32to8(text.substr(begin, end - begin)));
}

} // namespace

auto SentenceSegmenter::load(const std::filesystem::path& prefixes) -> Result<SentenceSegmenter> {
    std::ifstream input(prefixes);
    if (!input) return std::unexpected(Error{ErrorCode::IO, "cannot open sentence prefix data: " + prefixes.string()});
    std::map<std::string, Prefixes, std::less<>> data;
    for (std::string line; std::getline(input, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.front() == '#') continue;
        const auto first = line.find('\t'), second = first == std::string::npos ? first : line.find('\t', first + 1);
        if (first == std::string::npos || second == std::string::npos || !first || second == first + 1 ||
            second + 1 == line.size() || line.find('\t', second + 1) != std::string::npos)
            return std::unexpected(
                Error{ErrorCode::INVALID_MANIFEST, "malformed sentence prefix row: " + prefixes.string()});
        auto language = normalized_language(std::string_view(line).substr(0, first));
        auto kind = std::string_view(line).substr(first + 1, second - first - 1);
        auto prefix = una::cases::to_lowercase_utf8(std::string_view(line).substr(second + 1));
        auto& entries = data[std::move(language)];
        if (kind == "always")
            entries.always.insert(std::move(prefix));
        else if (kind == "numeric")
            entries.numeric.insert(std::move(prefix));
        else
            return std::unexpected(
                Error{ErrorCode::INVALID_MANIFEST, "unknown sentence prefix kind: " + std::string(kind)});
    }
    if (input.bad() || data.empty())
        return std::unexpected(Error{ErrorCode::IO, "cannot read sentence prefix data: " + prefixes.string()});
    return SentenceSegmenter(std::move(data));
}

auto SentenceSegmenter::contains(std::string_view language, std::string_view prefix, bool numeric) const -> bool {
    const auto present = [&](std::string_view key) {
        const auto found = prefixes_.find(key);
        if (found == prefixes_.end()) return false;
        const auto& values = numeric ? found->second.numeric : found->second.always;
        return values.contains(prefix);
    };
    return present("*") || present(language);
}

auto SentenceSegmenter::split(std::string_view text, std::string_view language) const
    -> Result<std::vector<std::string>> {
    una::error error;
    const auto codepoints = una::strict::utf8to32u(text, error);
    if (error) return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "sentence input is not valid UTF-8"});

    const auto locale = normalized_language(language);
    const auto period_boundary = [&](std::size_t index, std::size_t after) {
        if (index && index + 1 < codepoints.size() && digit(codepoints[index - 1]) && digit(codepoints[index + 1]))
            return false;
        if (index + 1 < codepoints.size() && !whitespace(codepoints[index + 1]) && !closer(codepoints[index + 1]) &&
            !terminator(codepoints[index + 1]))
            return false;
        const auto following = next_content(codepoints, after);
        if (!following) return true;
        const auto word = word_before(codepoints, index);
        if (contains(locale, word, false)) return false;
        if (word.size() == 1 && index && una::codepoint::is_uppercase(codepoints[index - 1])) return false;
        if (word.find('.') != std::string::npos && word != "a.m" && word != "p.m") return false;
        if (contains(locale, word, true) && una::codepoint::is_numeric(*following)) return false;
        return sentence_starter(*following);
    };

    std::vector<std::string> output;
    std::size_t begin = 0;
    for (std::size_t index = 0; index < codepoints.size();) {
        if (newline(codepoints[index])) {
            auto end = index;
            std::size_t breaks = 0;
            while (end < codepoints.size() && whitespace(codepoints[end])) {
                if (newline(codepoints[end])) ++breaks;
                ++end;
            }
            if (breaks >= 2) {
                emit(output, codepoints, begin, index);
                begin = end;
                index = end;
                continue;
            }
        }
        if (!terminator(codepoints[index])) {
            ++index;
            continue;
        }
        auto end = index + 1;
        while (end < codepoints.size() && terminator(codepoints[end])) ++end;
        while (end < codepoints.size() && closer(codepoints[end])) ++end;
        if (codepoints[index] == U'.' && !period_boundary(index, end)) {
            ++index;
            continue;
        }
        const auto following = next_content(codepoints, end);
        const auto cjk = codepoints[index] == U'\u3002' || codepoints[index] == U'\uFF0E' ||
                         codepoints[index] == U'\uFF1F' || codepoints[index] == U'\uFF01';
        if (!following || cjk || sentence_starter(*following)) {
            emit(output, codepoints, begin, end);
            begin = end;
        }
        index = end;
    }
    emit(output, codepoints, begin, codepoints.size());
    if (output.empty()) return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "text has no sentence content"});
    return output;
}

} // namespace kidi::text
