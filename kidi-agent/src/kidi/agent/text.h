#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace kidi::agent {

// ASCII lowercase; other bytes, including UTF-8, are unchanged.
inline auto lower(std::string_view text) -> std::string {
    std::string result(text);
    for (auto& character : result)
        if (character >= 'A' && character <= 'Z') character = static_cast<char>(character - 'A' + 'a');
    return result;
}

// `text` without leading and trailing spaces, tabs, line breaks and form feeds.
inline auto trim(std::string_view text) -> std::string_view {
    const auto first = text.find_first_not_of(" \t\r\n\f");
    if (first == std::string_view::npos) return {};
    return text.substr(first, text.find_last_not_of(" \t\r\n\f") - first + 1);
}

// At most `limit` bytes of `text`, cut on a UTF-8 character boundary.
inline auto utf8_prefix(std::string_view text, std::size_t limit) -> std::string_view {
    if (text.size() <= limit) return text;
    while (limit > 0 && (static_cast<unsigned char>(text[limit]) & 0xC0) == 0x80) --limit;
    return text.substr(0, limit);
}

// `text` on one line with whitespace runs collapsed, cut to `limit` bytes and marked with "…" when cut.
inline auto one_line(std::string_view text, std::size_t limit) -> std::string {
    std::string out;
    bool space = false;
    for (const auto character : text) {
        if (character == ' ' || character == '\n' || character == '\r' || character == '\t') {
            space = !out.empty();
            continue;
        }
        if (space) out += ' ';
        space = false;
        out += character;
    }
    if (out.size() <= limit) return out;
    return std::string(utf8_prefix(out, limit)) + "…";
}

} // namespace kidi::agent
