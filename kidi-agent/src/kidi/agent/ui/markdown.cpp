#include "kidi/agent/ui/markdown.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <utility>

namespace kidi::agent::ui {
namespace {

constexpr std::string_view RESET = "\033[0m";
constexpr std::string_view BOLD = "\033[1m";
constexpr std::string_view DIM = "\033[2m";
constexpr std::string_view ITALIC = "\033[3m";
constexpr std::string_view UNDERLINE = "\033[4m";
constexpr std::string_view STRIKE = "\033[9m";
constexpr std::string_view CODE = "\033[38;5;180m";
constexpr std::string_view BULLET = "\033[36m";

// Decodes one UTF-8 code point; invalid bytes decode as themselves so widths stay defined.
auto decode(std::string_view text, std::size_t& index) -> std::uint32_t {
    const auto byte = static_cast<unsigned char>(text[index]);
    const std::size_t count = byte < 0x80              ? 1
                              : (byte >> 5) == 6       ? 2
                              : (byte >> 4) == 14      ? 3
                              : (byte >> 3) == 30      ? 4
                                                       : 1;
    if (count == 1 || index + count > text.size()) {
        ++index;
        return byte;
    }
    std::uint32_t codepoint = byte & (0x7f >> count);
    for (std::size_t i = 1; i < count; ++i)
        codepoint = (codepoint << 6) | (static_cast<unsigned char>(text[index + i]) & 0x3f);
    index += count;
    return codepoint;
}

auto cell_width(std::uint32_t codepoint) -> std::size_t {
    if (codepoint < 0x20 || (codepoint >= 0x7f && codepoint < 0xa0)) return 0;
    if ((codepoint >= 0x300 && codepoint <= 0x36f) || (codepoint >= 0x200b && codepoint <= 0x200f) ||
        (codepoint >= 0xfe00 && codepoint <= 0xfe0f) || (codepoint >= 0x1ab0 && codepoint <= 0x1aff) ||
        (codepoint >= 0x20d0 && codepoint <= 0x20ff) || (codepoint >= 0xe0100 && codepoint <= 0xe01ef))
        return 0;
    if ((codepoint >= 0x1100 && codepoint <= 0x115f) || (codepoint >= 0x2e80 && codepoint <= 0xa4cf) ||
        (codepoint >= 0xac00 && codepoint <= 0xd7a3) || (codepoint >= 0xf900 && codepoint <= 0xfaff) ||
        (codepoint >= 0xfe30 && codepoint <= 0xfe4f) || (codepoint >= 0xff00 && codepoint <= 0xff60) ||
        (codepoint >= 0xffe0 && codepoint <= 0xffe6) || (codepoint >= 0x1f300 && codepoint <= 0x1f64f) ||
        (codepoint >= 0x1f900 && codepoint <= 0x1faff) || (codepoint >= 0x20000 && codepoint <= 0x3fffd))
        return 2;
    return 1;
}

// Length of the ANSI CSI sequence at `index`, or 0.
auto escape_length(std::string_view text, std::size_t index) -> std::size_t {
    if (text[index] != '\033' || index + 1 >= text.size() || text[index + 1] != '[') return 0;
    auto end = index + 2;
    while (end < text.size() && (text[end] < 0x40 || text[end] > 0x7e)) ++end;
    return end < text.size() ? end + 1 - index : text.size() - index;
}

auto trim(std::string_view text) -> std::string_view {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) text.remove_prefix(1);
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t')) text.remove_suffix(1);
    return text;
}

auto word(char c) -> bool { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }

// Appends `text` with inline Markdown styles; `active` is the style to restore after each span.
auto inline_styles(std::string_view text, const std::string& active, std::string& out) -> void {
    static constexpr std::pair<std::string_view, std::string_view> SPANS[] = {
        {"**", BOLD}, {"__", BOLD}, {"~~", STRIKE}, {"*", ITALIC}, {"_", ITALIC}};
    for (std::size_t i = 0; i < text.size();) {
        const char c = text[i];
        if (c == '\\' && i + 1 < text.size() && std::ispunct(static_cast<unsigned char>(text[i + 1]))) {
            out += text[i + 1];
            i += 2;
            continue;
        }
        if (c == '`') {
            auto ticks = i;
            while (ticks < text.size() && text[ticks] == '`') ++ticks;
            const auto marker = text.substr(i, ticks - i);
            if (const auto close = text.find(marker, ticks); close != std::string_view::npos) {
                out.append(CODE).append(text.substr(ticks, close - ticks)).append(RESET).append(active);
                i = close + marker.size();
            } else {
                out += marker;
                i = ticks;
            }
            continue;
        }
        bool styled = false;
        for (const auto& [marker, style] : SPANS) {
            if (!text.substr(i).starts_with(marker)) continue;
            const auto start = i + marker.size();
            const bool underscore = marker.front() == '_';
            if (start >= text.size() || text[start] == ' ' || (underscore && i > 0 && word(text[i - 1]))) break;
            auto close = text.find(marker, start);
            while (close != std::string_view::npos &&
                   (text[close - 1] == ' ' ||
                    (underscore && close + marker.size() < text.size() && word(text[close + marker.size()]))))
                close = text.find(marker, close + 1);
            if (close == std::string_view::npos || close == start) break;
            out += style;
            inline_styles(text.substr(start, close - start), active + std::string(style), out);
            out.append(RESET).append(active);
            i = close + marker.size();
            styled = true;
            break;
        }
        if (styled) continue;
        if (c == '[') {
            const auto label_end = text.find("](", i + 1);
            const auto url_end = label_end == std::string_view::npos ? label_end : text.find(')', label_end + 2);
            if (url_end != std::string_view::npos) {
                const auto label = text.substr(i + 1, label_end - i - 1);
                const auto url = text.substr(label_end + 2, url_end - label_end - 2);
                out += UNDERLINE;
                inline_styles(label, active + std::string(UNDERLINE), out);
                out.append(RESET).append(active);
                if (url != label) out.append(DIM).append(" (").append(url).append(")").append(RESET).append(active);
                i = url_end + 1;
                continue;
            }
        }
        // Copy a whole run of ordinary text at once, keeping multi-byte characters intact.
        auto end = i + 1;
        while (end < text.size() && std::string_view("\\`*_~[").find(text[end]) == std::string_view::npos) ++end;
        out += text.substr(i, end - i);
        i = end;
    }
}

auto styled(std::string_view text, std::string_view base = {}) -> std::string {
    std::string out(base);
    inline_styles(text, std::string(base), out);
    if (out.find('\033') != std::string::npos) out += RESET;
    return out;
}

auto table_row(std::string_view line) -> bool { return trim(line).starts_with('|'); }

auto cells(std::string_view line) -> std::vector<std::string> {
    line = trim(line);
    if (line.starts_with('|')) line.remove_prefix(1);
    if (line.ends_with('|') && !line.ends_with("\\|")) line.remove_suffix(1);
    // Pipes inside a code span belong to the cell: models rarely escape them, e.g. `"deg" | "rad"`.
    std::vector<std::string> result(1);
    std::size_t ticks = 0;
    for (std::size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '`') {
            auto run = i;
            while (run < line.size() && line[run] == '`') ++run;
            const auto count = run - i;
            if (!ticks && line.find(std::string(count, '`'), run) != std::string_view::npos) ticks = count;
            else if (ticks == count) ticks = 0;
            result.back() += line.substr(i, count);
            i = run - 1;
        } else if (line[i] == '\\' && i + 1 < line.size() && line[i + 1] == '|') {
            result.back() += '|';
            ++i;
        } else if (line[i] == '|' && !ticks) {
            result.emplace_back();
        } else {
            result.back() += line[i];
        }
    }
    for (auto& cell : result) cell = std::string(trim(cell));
    return result;
}

auto separator(const std::vector<std::string>& row) -> bool {
    return std::ranges::all_of(row, [](const std::string& cell) {
        std::string_view text = cell;
        if (text.starts_with(':')) text.remove_prefix(1);
        if (text.ends_with(':')) text.remove_suffix(1);
        return !text.empty() && text.find_first_not_of('-') == std::string_view::npos;
    });
}

auto repeat(std::string_view text, std::size_t count) -> std::string {
    std::string out;
    for (std::size_t i = 0; i < count; ++i) out += text;
    return out;
}

} // namespace

auto display_width(std::string_view text) -> std::size_t {
    std::size_t width = 0;
    for (std::size_t i = 0; i < text.size();) {
        if (const auto length = escape_length(text, i)) {
            i += length;
            continue;
        }
        width += cell_width(decode(text, i));
    }
    return width;
}

auto fit_width(std::string_view text, std::size_t width) -> std::string {
    if (display_width(text) <= width) return std::string(text);
    std::string out;
    std::size_t used = 0;
    bool escaped = false;
    for (std::size_t i = 0; i < text.size();) {
        if (const auto length = escape_length(text, i)) {
            out += text.substr(i, length);
            escaped = true;
            i += length;
            continue;
        }
        const auto start = i;
        const auto cells = cell_width(decode(text, i));
        if (used + cells + 1 > width) break;
        out += text.substr(start, i - start);
        used += cells;
    }
    if (width) out += "…";
    if (escaped) out += RESET;
    return out;
}

auto wrap_width(std::string_view text, std::size_t width) -> std::vector<std::string> {
    width = std::max<std::size_t>(width, 1);
    std::vector<std::string> rows(1);
    std::string active;
    std::size_t used = 0;
    for (std::size_t i = 0; i < text.size();) {
        if (const auto length = escape_length(text, i)) {
            const auto code = text.substr(i, length);
            if (code == RESET || code == "\033[m") active.clear();
            else active += code;
            rows.back() += code;
            i += length;
            continue;
        }
        const auto start = i;
        const auto cells = cell_width(decode(text, i));
        if (used && used + cells > width) {
            if (!active.empty()) rows.back() += RESET;
            rows.push_back(active);
            used = 0;
        }
        rows.back() += text.substr(start, i - start);
        used += cells;
    }
    return rows;
}

auto wrap_words(std::string_view text, std::size_t width) -> std::vector<std::string> {
    width = std::max<std::size_t>(width, 1);
    std::vector<std::string> rows(1);
    std::string active, space_active;
    std::size_t used = 0, space = std::string::npos, space_used = 0;
    const auto hard_break = [&] {
        if (!active.empty()) rows.back() += RESET;
        rows.push_back(active);
        used = 0;
        space = std::string::npos;
    };
    for (std::size_t i = 0; i < text.size();) {
        if (const auto length = escape_length(text, i)) {
            const auto code = text.substr(i, length);
            if (code == RESET || code == "\033[m") active.clear();
            else active += code;
            rows.back() += code;
            i += length;
            continue;
        }
        const auto start = i;
        const auto cells = cell_width(decode(text, i));
        const auto glyph = text.substr(start, i - start);
        if (glyph == " ") {
            if (used + 1 > width) {
                hard_break();
                continue;
            }
            rows.back() += glyph;
            ++used;
            space = rows.back().size();
            space_used = used;
            space_active = active;
            continue;
        }
        if (used + cells > width) {
            if (space != std::string::npos) {
                // Move the partial word after the last space to the next row, dropping that space.
                auto rest = rows.back().substr(space);
                rows.back().resize(space - 1);
                if (!space_active.empty()) rows.back() += RESET;
                rows.push_back(space_active + rest);
                used -= space_used;
                space = std::string::npos;
            }
            if (used && used + cells > width) hard_break();
        }
        rows.back() += glyph;
        used += cells;
    }
    return rows;
}

auto MarkdownStream::feed(std::string_view text, std::size_t width) -> std::vector<std::string> {
    std::vector<std::string> out;
    partial_ += text;
    std::size_t start = 0;
    for (auto end = partial_.find('\n'); end != std::string::npos; end = partial_.find('\n', start)) {
        auto row = std::string_view(partial_).substr(start, end - start);
        if (row.ends_with('\r')) row.remove_suffix(1);
        line(row, width, out);
        start = end + 1;
    }
    partial_.erase(0, start);
    return out;
}

auto MarkdownStream::preview(std::size_t width) const -> std::vector<std::string> {
    auto copy = *this;
    return copy.finish(width);
}

auto MarkdownStream::finish(std::size_t width) -> std::vector<std::string> {
    std::vector<std::string> out;
    if (!partial_.empty()) line(std::exchange(partial_, {}), width, out);
    flush_table(width, out);
    fence_.clear();
    return out;
}

auto MarkdownStream::line(std::string_view text, std::size_t width, std::vector<std::string>& out) -> void {
    if (!color_) {
        out.emplace_back(text);
        return;
    }
    const auto trimmed = trim(text);
    if (!fence_.empty()) {
        if (trimmed.starts_with(fence_) && trimmed.find_first_not_of(fence_.front()) == std::string_view::npos) {
            out.push_back(std::string(DIM) + std::string(text) + std::string(RESET));
            fence_.clear();
        } else {
            out.push_back(std::string(CODE) + std::string(text) + std::string(RESET));
        }
        return;
    }
    if (table_row(text)) {
        table_.emplace_back(text);
        return;
    }
    flush_table(width, out);
    if (trimmed.starts_with("```") || trimmed.starts_with("~~~")) {
        fence_ = std::string(trimmed.substr(0, trimmed.find_first_not_of(trimmed.front())));
        out.push_back(std::string(DIM) + std::string(text) + std::string(RESET));
        return;
    }
    if (trimmed.empty()) {
        out.emplace_back();
        return;
    }
    const auto level = trimmed.find_first_not_of('#');
    if (level >= 1 && level <= 6 && level < trimmed.size() && trimmed[level] == ' ') {
        out.push_back(styled(trim(trimmed.substr(level)), level == 1 ? "\033[1;4m" : BOLD));
        return;
    }
    if (trimmed.size() >= 3 && std::string_view("-*_").find(trimmed.front()) != std::string_view::npos &&
        trimmed.find_first_not_of(std::string{trimmed.front(), ' '}) == std::string_view::npos) {
        out.push_back(std::string(DIM) + repeat("─", std::clamp<std::size_t>(width, 3, 81) - 1) + std::string(RESET));
        return;
    }
    if (trimmed.starts_with('>')) {
        out.push_back(std::string(DIM) + "│ " + std::string(RESET) + styled(trim(trimmed.substr(1))));
        return;
    }
    const auto indent = text.substr(0, text.size() - text.substr(text.find_first_not_of(" \t")).size());
    auto marker = trimmed.substr(0, trimmed.find(' '));
    const bool unordered = marker == "-" || marker == "*" || marker == "+";
    const bool ordered = marker.size() >= 2 && marker.size() <= 10 && (marker.back() == '.' || marker.back() == ')') &&
                         marker.substr(0, marker.size() - 1).find_first_not_of("0123456789") == std::string_view::npos;
    if ((unordered || ordered) && marker.size() < trimmed.size()) {
        auto body = trim(trimmed.substr(marker.size()));
        std::string bullet = unordered ? std::string(BULLET) + "•" + std::string(RESET) : std::string(marker);
        if (unordered && (body.starts_with("[ ] ") || body.starts_with("[x] ") || body.starts_with("[X] "))) {
            bullet = body[1] == ' ' ? "☐" : "☑";
            body.remove_prefix(4);
        }
        out.push_back(std::string(indent) + bullet + " " + styled(body));
        return;
    }
    out.push_back(styled(text));
}

auto MarkdownStream::flush_table(std::size_t width, std::vector<std::string>& out) -> void {
    if (table_.empty()) return;
    std::vector<std::vector<std::string>> rows;
    for (const auto& row : table_) rows.push_back(cells(row));
    const bool aligned = rows.size() >= 2 && separator(rows[1]);
    // The header fixes the column count; a row with stray pipes keeps its extra text in the last cell.
    const std::size_t columns = rows.front().size();
    for (auto& row : rows)
        while (row.size() > columns) {
            row[row.size() - 2] += " | " + row.back();
            row.pop_back();
        }
    std::vector<std::vector<std::string>> rendered;
    std::vector<std::size_t> widths(columns, 0);
    for (std::size_t r = 0; aligned && r < rows.size(); ++r) {
        if (r == 1) continue;
        auto& cells_out = rendered.emplace_back();
        for (std::size_t c = 0; c < columns; ++c) {
            cells_out.push_back(c < rows[r].size() ? styled(rows[r][c], r == 0 ? BOLD : std::string_view{}) : "");
            widths[c] = std::max(widths[c], display_width(cells_out.back()));
        }
    }
    // Borders and padding take 3 cells per column plus one; shrink the widest columns until the box fits, then wrap
    // their cells. Only a table too narrow to hold 3 cells per column falls back to its source rows.
    constexpr std::size_t MIN_COLUMN = 3;
    const auto frame = 3 * columns + 1;
    const auto room = width > frame + 1 ? width - frame - 1 : 0;
    std::size_t total = 0;
    for (const auto cell : widths) total += cell;
    while (total > room) {
        const auto widest = std::ranges::max_element(widths);
        if (*widest <= MIN_COLUMN) break;
        --*widest;
        --total;
    }
    if (!aligned || total > room) {
        for (const auto& row : table_) out.push_back(styled(row));
        table_.clear();
        return;
    }
    std::vector<bool> right(columns, false), center(columns, false);
    for (std::size_t c = 0; c < rows[1].size() && c < columns; ++c) {
        right[c] = rows[1][c].ends_with(':');
        center[c] = right[c] && rows[1][c].starts_with(':');
    }
    const auto border = [&](std::string_view left, std::string_view middle, std::string_view end) {
        std::string line(DIM);
        line += left;
        for (std::size_t c = 0; c < columns; ++c) {
            line += repeat("─", widths[c] + 2);
            line += c + 1 < columns ? middle : end;
        }
        return line + std::string(RESET);
    };
    const std::string bar = std::string(DIM) + "│" + std::string(RESET);
    std::vector<std::vector<std::vector<std::string>>> wrapped;
    bool multiline = false;
    for (const auto& row : rendered) {
        auto& cells_out = wrapped.emplace_back();
        for (std::size_t c = 0; c < columns; ++c) {
            cells_out.push_back(wrap_words(row[c], widths[c]));
            multiline = multiline || cells_out.back().size() > 1;
        }
    }
    out.push_back(border("┌", "┬", "┐"));
    for (std::size_t r = 0; r < wrapped.size(); ++r) {
        std::size_t height = 1;
        for (const auto& cell : wrapped[r]) height = std::max(height, cell.size());
        for (std::size_t h = 0; h < height; ++h) {
            std::string line = bar;
            for (std::size_t c = 0; c < columns; ++c) {
                const auto text = h < wrapped[r][c].size() ? wrapped[r][c][h] : std::string{};
                const auto pad = widths[c] - std::min(widths[c], display_width(text));
                const auto before = center[c] ? pad / 2 : right[c] ? pad : 0;
                line += " " + std::string(before, ' ') + text + std::string(pad - before, ' ') + " " + bar;
            }
            out.push_back(std::move(line));
        }
        // Rule between rows once any cell wraps, so wrapped rows stay distinguishable.
        if (r == 0 || (multiline && r + 1 < wrapped.size())) out.push_back(border("├", "┼", "┤"));
    }
    out.push_back(border("└", "┴", "┘"));
    table_.clear();
}

auto render_markdown(std::string_view text, bool color, std::size_t width) -> std::vector<std::string> {
    MarkdownStream stream(color);
    auto lines = stream.feed(text, width);
    for (auto& line : stream.finish(width)) lines.push_back(std::move(line));
    return lines;
}

} // namespace kidi::agent::ui
