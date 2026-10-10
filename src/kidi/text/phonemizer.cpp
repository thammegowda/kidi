#include "kidi/text/phonemizer.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

#include <uni_algo/conv.h>

namespace kidi::text {
namespace {

auto ascii_lower(std::string value) -> std::string {
    for (auto& character : value) {
        const auto byte = static_cast<unsigned char>(character);
        if (byte < 0x80) character = static_cast<char>(std::tolower(byte));
    }
    return value;
}

auto punctuation(char character) -> bool {
    constexpr std::string_view VALUES = ";:,.!?\"()";
    return VALUES.find(character) != std::string_view::npos;
}

auto append_punctuation(std::string& output, std::string_view value) -> void {
    for (const auto character : value)
        if (punctuation(character)) output.push_back(character);
}

} // namespace

auto LexiconPhonemizer::load(const std::filesystem::path& lexicon, const YAML::Node& vocabulary)
    -> Result<LexiconPhonemizer> {
    try {
        std::ifstream input(lexicon);
        if (!input)
            return std::unexpected(Error{ErrorCode::IO, "cannot open pronunciation lexicon: " + lexicon.string()});
        std::map<std::string, std::string, std::less<>> entries;
        for (std::string line; std::getline(input, line);) {
            const auto separator = line.find('\t');
            if (separator == std::string::npos || !separator || separator + 1 == line.size())
                return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "malformed pronunciation lexicon row"});
            entries.emplace(line.substr(0, separator), line.substr(separator + 1));
        }
        if (input.bad() || entries.empty())
            return std::unexpected(Error{ErrorCode::IO, "cannot read pronunciation lexicon: " + lexicon.string()});

        if (!vocabulary.IsMap())
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "phoneme vocabulary must be a mapping"});
        std::map<char32_t, std::int32_t> ids;
        for (const auto& item : vocabulary) {
            const auto token = item.first.as<std::string>();
            una::error error;
            const auto codepoints = una::strict::utf8to32u(token, error);
            if (error || codepoints.size() != 1)
                return std::unexpected(
                    Error{ErrorCode::INVALID_MANIFEST, "phoneme vocabulary keys must be Unicode codepoints"});
            ids.emplace(codepoints[0], item.second.as<std::int32_t>());
        }
        return LexiconPhonemizer(std::move(entries), std::move(ids));
    } catch (const YAML::Exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, error.what()});
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

auto LexiconPhonemizer::phonemize(std::string_view text) const -> Result<PhonemeSequence> {
    std::istringstream input{std::string(text)};
    std::string ipa;
    for (std::string chunk; input >> chunk;) {
        std::size_t begin = 0, end = chunk.size();
        while (begin < end && punctuation(chunk[begin])) ++begin;
        while (end > begin && punctuation(chunk[end - 1])) --end;
        const auto word = ascii_lower(chunk.substr(begin, end - begin));
        if (word.empty()) continue;
        const auto found = lexicon_.find(word);
        if (found == lexicon_.end())
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, "pronunciation lexicon has no entry for: " + word});
        if (!ipa.empty()) ipa.push_back(' ');
        append_punctuation(ipa, std::string_view(chunk).substr(0, begin));
        ipa += found->second;
        append_punctuation(ipa, std::string_view(chunk).substr(end));
    }
    if (ipa.empty()) return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "text to phonemize must not be empty"});

    una::error error;
    const auto codepoints = una::strict::utf8to32u(ipa, error);
    if (error) return std::unexpected(Error{ErrorCode::RUNTIME, "phonemizer produced invalid UTF-8"});
    std::vector<std::int32_t> ids;
    ids.reserve(codepoints.size());
    for (const auto codepoint : codepoints) {
        const auto found = vocabulary_.find(codepoint);
        if (found != vocabulary_.end()) ids.push_back(found->second);
    }
    if (ids.empty()) return std::unexpected(Error{ErrorCode::RUNTIME, "phonemizer produced no model tokens"});
    return PhonemeSequence{std::move(ipa), std::move(ids)};
}

} // namespace kidi::text
