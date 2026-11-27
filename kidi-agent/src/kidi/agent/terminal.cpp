#include "kidi/agent/terminal.h"

#include "kidi/agent/markdown.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#endif

namespace kidi::agent {
namespace {

constexpr std::string_view RESET = "\033[0m";
constexpr std::string_view PLACEHOLDER = "Ask Kidi; /help lists commands";
constexpr std::size_t MAX_PASTE_BYTES = 4 * 1024 * 1024;

auto preview(std::string_view text, std::size_t lines = 6, std::size_t bytes = 1200,
             std::string_view suffix = "... (display truncated)") -> std::string {
    auto clean = terminal_text(text);
    std::size_t end = 0, count = 0;
    while (end < clean.size() && end < bytes && count < lines)
        if (clean[end++] == '\n') ++count;
    while (end < clean.size() && end && (static_cast<unsigned char>(clean[end]) & 0xc0) == 0x80) --end;
    if (end < clean.size()) {
        clean.resize(end);
        while (!clean.empty() && clean.back() == '\n') clean.pop_back();
        clean += suffix;
    }
    while (!clean.empty() && clean.back() == '\n') clean.pop_back();
    return clean;
}

auto ansi_color(TerminalTone tone) -> std::string_view {
    switch (tone) {
        case TerminalTone::USER:
            return "\033[1;32m";
        case TerminalTone::TOOL:
            return "\033[1;36m";
        case TerminalTone::THINKING:
        case TerminalTone::DIM:
        case TerminalTone::SESSION:
            return "\033[2;37m";
        case TerminalTone::ASSISTANT:
            return {};
        case TerminalTone::ERROR:
            return "\033[31m";
        case TerminalTone::CONTEXT:
            return "\033[36m";
        case TerminalTone::WARNING:
            return "\033[33m";
        case TerminalTone::DEFAULT:
            return {};
    }
    return {};
}

auto split_lines(std::string_view text) -> std::vector<std::string_view> {
    std::vector<std::string_view> lines;
    std::size_t start = 0;
    for (auto end = text.find('\n'); end != std::string_view::npos; end = text.find('\n', start)) {
        lines.push_back(text.substr(start, end - start));
        start = end + 1;
    }
    if (start < text.size() || lines.empty()) lines.push_back(text.substr(start));
    return lines;
}

auto utf8_length(unsigned char lead) -> std::size_t {
    return lead < 0x80 ? 1 : (lead >> 5) == 6 ? 2 : (lead >> 4) == 14 ? 3 : (lead >> 3) == 30 ? 4 : 1;
}

auto seconds_since(std::chrono::steady_clock::time_point start) -> std::string {
    std::ostringstream text;
    text << std::fixed << std::setprecision(1)
         << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << 's';
    return text.str();
}

#if defined(__unix__) || defined(__APPLE__)
// Signal handlers only see these: the wake pipe for resizes and the terminal state to restore on termination.
std::atomic<int> wake_fd{-1};
termios saved_terminal{};
std::atomic<bool> terminal_raw{false};

auto on_resize(int) -> void {
    const auto saved = errno;
    if (const auto fd = wake_fd.load(); fd >= 0) (void)!::write(fd, "w", 1);
    errno = saved;
}

auto on_terminate(int signal) -> void {
    if (terminal_raw.load()) {
        ::tcsetattr(STDIN_FILENO, TCSANOW, &saved_terminal);
        constexpr char RESTORE[] = "\r\n\033[?2004l\033[?25h";
        (void)!::write(STDERR_FILENO, RESTORE, sizeof(RESTORE) - 1);
    }
    ::signal(signal, SIG_DFL);
    ::raise(signal);
}

auto write_all(int fd, std::string_view text) -> void {
    while (!text.empty()) {
        const auto count = ::write(fd, text.data(), text.size());
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return;
        text.remove_prefix(static_cast<std::size_t>(count));
    }
}
#endif

} // namespace

auto terminal_text(std::string_view text) -> std::string {
    std::string result;
    for (std::size_t i = 0; i < text.size();) {
        const auto byte = static_cast<unsigned char>(text[i]);
        std::size_t count = byte < 0x80                    ? 1
                            : byte >= 0xc2 && byte <= 0xdf ? 2
                            : byte >= 0xe0 && byte <= 0xef ? 3
                            : byte >= 0xf0 && byte <= 0xf4 ? 4
                                                           : 0;
        unsigned codepoint = count == 1 ? byte : count ? byte & ((1u << (7 - count)) - 1) : 0;
        bool valid = count && count <= text.size() - i;
        for (std::size_t j = 1; valid && j < count; ++j) {
            const auto next = static_cast<unsigned char>(text[i + j]);
            valid = (next & 0xc0) == 0x80;
            codepoint = (codepoint << 6) | (next & 0x3f);
        }
        valid = valid && (count < 3 || codepoint >= (count == 3 ? 0x800u : 0x10000u)) && codepoint <= 0x10ffff &&
                !(codepoint >= 0xd800 && codepoint <= 0xdfff);
        const bool control = (codepoint < 32 && codepoint != '\n' && codepoint != '\t') ||
                             (codepoint >= 0x7f && codepoint <= 0x9f) || (codepoint >= 0x202a && codepoint <= 0x202e) ||
                             (codepoint >= 0x2066 && codepoint <= 0x2069);
        if (valid && !control) {
            result.append(text.substr(i, count));
            i += count;
        } else {
            constexpr char HEX[] = "0123456789abcdef";
            result += "\\x";
            result += HEX[byte >> 4];
            result += HEX[byte & 15];
            ++i;
        }
    }
    return result;
}

// The interactive UI draws a small live region (streaming preview, input and status bar) below the cursor and prints
// finished transcript lines above it, so the transcript lives in the terminal's own scrollback: native scrolling,
// selection and copying keep working. All UI state belongs to the thread in run_interactive; other threads post tasks.
struct Terminal::Impl {
    struct Request {
        std::mutex mutex;
        std::condition_variable ready;
        bool completed = false;
        TerminalInput result{InputState::INTERRUPTED, {}};
        std::string prompt;
        bool remember = true;
    };

    Impl(bool edit, bool color) : edit(edit), color(color) {}

    bool edit;
    bool color;
    std::atomic<bool> running{false};
    std::mutex tasks_mutex;
    std::vector<std::function<void()>> tasks;
#if defined(__unix__) || defined(__APPLE__)
    int wake[2] = {-1, -1};
    int capture[2] = {-1, -1};
    int tty = -1, saved_err = -1;
#endif

    std::vector<std::string> commits;
    std::string captured;
    std::vector<std::string> history;
    std::size_t history_index = 0;
    std::string status, activity;
    std::chrono::steady_clock::time_point activity_started;
    bool activity_running = false, tool_activity = false;
    std::optional<std::string> pending_tool;
    bool tool_committed = false;
    std::optional<MarkdownStream> markdown;
    std::string reasoning_head, reasoning_tail;
    bool reasoning = false;
    std::shared_ptr<Request> request;
    std::string input;
    std::size_t caret = 0;
    std::optional<std::string> paste;
    bool paste_truncated = false;
    std::string keys;
    std::size_t live_rows = 0, caret_row = 0;
    bool clear_screen = false, input_open = true;
    std::function<void()> interrupt;
    bool worker_done = false;
    int worker_result = 1;
    std::exception_ptr worker_exception;

    auto post(std::function<void()> task) -> void {
        {
            const std::lock_guard lock(tasks_mutex);
            tasks.push_back(std::move(task));
        }
#if defined(__unix__) || defined(__APPLE__)
        (void)!::write(wake[1], "t", 1);
#endif
    }

    auto run_tasks() -> void {
        std::vector<std::function<void()>> ready;
        {
            const std::lock_guard lock(tasks_mutex);
            ready.swap(tasks);
        }
        for (auto& task : ready) task();
    }

    auto size() const -> std::pair<std::size_t, std::size_t> {
#if defined(__unix__) || defined(__APPLE__)
        winsize window{};
        if (::ioctl(tty, TIOCGWINSZ, &window) == 0 && window.ws_col && window.ws_row)
            return {std::max<std::size_t>(window.ws_col, 20), std::max<std::size_t>(window.ws_row, 6)};
#endif
        return {80, 24};
    }

    auto paint(std::string_view text, TerminalTone tone) const -> std::string {
        const auto ansi = ansi_color(tone);
        if (!color || ansi.empty() || text.empty()) return std::string(text);
        return std::string(ansi) + std::string(text) + std::string(RESET);
    }

    // Every committed line goes through here so a running tool's summary is printed before anything it caused.
    auto commit_line(std::string line) -> void {
        flush_tool();
        commits.push_back(std::move(line));
    }

    auto flush_tool() -> void {
        if (!pending_tool || tool_committed) return;
        tool_committed = true;
        for (const auto part : split_lines(*pending_tool)) commits.push_back(paint(part, TerminalTone::TOOL));
    }

    auto commit(std::string_view text, TerminalTone tone) -> void {
        if (tone == TerminalTone::ASSISTANT) {
            for (auto& line : render_markdown(terminal_text(text), color, size().first)) commit_line(std::move(line));
            return;
        }
        const auto clean = terminal_text(text);
        for (const auto line : split_lines(clean)) commit_line(paint(line, tone));
    }

    auto commit_reasoning() -> void {
        if (!reasoning) return;
        reasoning = false;
        const auto summary = preview(reasoning_head, 2, 300, "... (thinking preview)");
        reasoning_head.clear();
        reasoning_tail.clear();
        if (!summary.empty()) commit("Thinking: " + summary, TerminalTone::THINKING);
    }

    auto complete(InputState state, std::string text = {}) -> void {
        if (!request) return;
        const auto active = request;
        if (state == InputState::LINE && active->remember && !text.empty() &&
            (history.empty() || history.back() != text)) {
            if (history.size() == 100) history.erase(history.begin());
            history.push_back(text);
        }
        if (!text.empty()) {
            if (active->remember) commit_line("");
            const auto shown = preview(text, 40, 8192, "... (input display truncated)");
            const auto lines = split_lines(shown);
            for (std::size_t i = 0; i < lines.size(); ++i)
                commit_line(paint(std::string(i ? "  " : active->remember ? "> " : "? ") + std::string(lines[i]),
                                  TerminalTone::USER));
        }
        {
            const std::lock_guard lock(active->mutex);
            active->result = {state, std::move(text)};
            active->completed = true;
        }
        active->ready.notify_one();
        request.reset();
        input.clear();
        caret = 0;
        history_index = history.size();
    }

    // Line editing on the UTF-8 input buffer; `caret` is always a byte offset at a character boundary.
    auto line_start() const -> std::size_t {
        const auto newline = caret ? input.rfind('\n', caret - 1) : std::string::npos;
        return newline == std::string::npos ? 0 : newline + 1;
    }
    auto line_end() const -> std::size_t {
        const auto newline = input.find('\n', caret);
        return newline == std::string::npos ? input.size() : newline;
    }
    auto previous(std::size_t index) const -> std::size_t {
        if (!index) return 0;
        do --index;
        while (index && (static_cast<unsigned char>(input[index]) & 0xc0) == 0x80);
        return index;
    }
    auto next(std::size_t index) const -> std::size_t {
        if (index >= input.size()) return input.size();
        return std::min(input.size(), index + utf8_length(static_cast<unsigned char>(input[index])));
    }
    auto insert(std::string_view text) -> void {
        std::string clean;
        for (const char c : text)
            if (static_cast<unsigned char>(c) >= 0x20 || c == '\n' || c == '\t') clean += c;
        input.insert(caret, clean);
        caret += clean.size();
    }
    // Moves to the same character column on the previous or next line.
    auto vertical(bool up) -> void {
        const auto start = line_start();
        std::size_t column = 0;
        for (auto index = start; index < caret; index = next(index)) ++column;
        std::size_t target = 0;
        if (up) {
            if (!start) return;
            const auto newline = start >= 2 ? input.rfind('\n', start - 2) : std::string::npos;
            target = newline == std::string::npos ? 0 : newline + 1;
        } else {
            const auto end = line_end();
            if (end >= input.size()) return;
            target = end + 1;
        }
        auto stop = input.find('\n', target);
        if (stop == std::string::npos) stop = input.size();
        caret = target;
        while (column-- && caret < stop) caret = next(caret);
    }
    auto browsing() const -> bool { return history_index < history.size() && input == history[history_index]; }
    auto recall(bool older) -> void {
        if (older && history_index) --history_index;
        if (!older && history_index < history.size()) ++history_index;
        input = history_index < history.size() ? history[history_index] : std::string{};
        caret = input.size();
    }

    auto submit_paste() -> void {
        auto text = std::move(*paste);
        paste.reset();
        std::string normalized;
        for (std::size_t i = 0; i < text.size(); ++i) {
            if (text[i] == '\r') {
                normalized += '\n';
                if (i + 1 < text.size() && text[i + 1] == '\n') ++i;
            } else {
                normalized += text[i];
            }
        }
        if (paste_truncated) normalized += "\n[paste truncated at 4 MiB]\n";
        if (request) insert(normalized);
    }

    auto control(char key) -> void {
        if (key == 0x03) {
            if (request) {
                input.clear();
                caret = 0;
                complete(InputState::INTERRUPTED);
            } else if (interrupt) {
                commit("Cancelling current operation...", TerminalTone::WARNING);
                interrupt();
            }
            return;
        }
        if (key == 0x0c) {
            clear_screen = true;
            return;
        }
        if (!request) return;
        switch (key) {
            case 0x01:
                caret = line_start();
                break;
            case 0x05:
                caret = line_end();
                break;
            case 0x02:
                caret = previous(caret);
                break;
            case 0x06:
                caret = next(caret);
                break;
            case 0x04:
                if (input.empty()) complete(InputState::END);
                else input.erase(caret, next(caret) - caret);
                break;
            case 0x0b:
                input.erase(caret, line_end() - caret);
                break;
            case 0x15: {
                const auto start = line_start();
                input.erase(start, caret - start);
                caret = start;
                break;
            }
            case 0x17: {
                auto start = caret;
                while (start && input[start - 1] == ' ') --start;
                while (start && input[start - 1] != ' ' && input[start - 1] != '\n') start = previous(start);
                input.erase(start, caret - start);
                caret = start;
                break;
            }
            case 0x08:
            case 0x7f: {
                const auto start = previous(caret);
                input.erase(start, caret - start);
                caret = start;
                break;
            }
            case 0x09:
                if (request->remember && input.starts_with('/')) {
                    static constexpr std::string_view COMMANDS[] = {
                        "/help", "/exit", "/reset", "/permissions", "/permissions reset", "/context", "/compact"};
                    const auto match =
                        std::ranges::find_if(COMMANDS, [&](auto command) { return command.starts_with(input); });
                    if (match != std::end(COMMANDS)) {
                        input = *match;
                        caret = input.size();
                    }
                } else {
                    insert("\t");
                }
                break;
            case '\r':
                complete(InputState::LINE, std::exchange(input, {}));
                break;
            case '\n':
                insert("\n");
                break;
            default:
                break;
        }
    }

    auto sequence(std::string_view code) -> void {
        if (code == "\033[200~") {
            paste.emplace();
            paste_truncated = false;
            return;
        }
        if (!request) return;
        const char final = code.back();
        const auto parameters = code.substr(2, code.size() - 3);
        const bool word = parameters.ends_with(";5") || parameters.ends_with(";3");
        if (final == 'A' || final == 'B') {
            const bool up = final == 'A';
            const bool edge = up ? line_start() == 0 : line_end() == input.size();
            if (!edge) vertical(up);
            else if (input.empty() || browsing()) recall(up);
        } else if (final == 'C' || final == 'D') {
            const bool left = final == 'D';
            do caret = left ? previous(caret) : next(caret);
            while (word && caret && caret < input.size() && input[left ? caret - 1 : caret] != ' ');
        } else if (final == 'H' || (final == '~' && (parameters == "1" || parameters == "7"))) {
            caret = line_start();
        } else if (final == 'F' || (final == '~' && (parameters == "4" || parameters == "8"))) {
            caret = line_end();
        } else if (final == '~' && parameters == "3") {
            input.erase(caret, next(caret) - caret);
        } else if (final == 'u' && parameters == "13;2") {
            insert("\n");
        }
    }

    // Parses buffered keyboard bytes. An incomplete escape or character waits for more bytes until `flush`.
    auto handle_keys(bool flush) -> void {
        std::size_t i = 0;
        while (i < keys.size()) {
            if (paste) {
                static constexpr std::string_view END = "\033[201~";
                const auto end = keys.find(END, i);
                const auto stop = end == std::string::npos ? keys.size() - std::min(keys.size() - i, END.size() - 1)
                                                           : end;
                const auto room = MAX_PASTE_BYTES - std::min(MAX_PASTE_BYTES, paste->size());
                paste->append(keys, i, std::min(room, stop - i));
                paste_truncated = paste_truncated || stop - i > room;
                i = stop;
                if (end == std::string::npos) break;
                i += END.size();
                submit_paste();
                continue;
            }
            const auto byte = static_cast<unsigned char>(keys[i]);
            if (byte == 0x1b) {
                if (i + 1 >= keys.size()) {
                    if (!flush) break;
                    ++i;
                    continue;
                }
                const char kind = keys[i + 1];
                if (kind == '[' || kind == 'O') {
                    auto end = i + 2;
                    while (end < keys.size() && keys[end] >= 0x20 && keys[end] <= 0x3f) ++end;
                    if (end >= keys.size()) {
                        if (!flush) break;
                        i = keys.size();
                        continue;
                    }
                    auto code = keys.substr(i, end + 1 - i);
                    if (kind == 'O') code[1] = '[';
                    i = end + 1;
                    sequence(code);
                    continue;
                }
                if (request && (kind == '\r' || kind == '\n')) insert("\n");
                i += 2;
                continue;
            }
            if (byte < 0x20 || byte == 0x7f) {
                control(keys[i++]);
                continue;
            }
            auto end = i;
            while (end < keys.size() && static_cast<unsigned char>(keys[end]) >= 0x20 && keys[end] != 0x7f) {
                const auto length = utf8_length(static_cast<unsigned char>(keys[end]));
                if (end + length > keys.size()) break;
                end += length;
            }
            if (end == i) {
                if (!flush) break;
                ++i;
                continue;
            }
            if (request && !request->remember && input.empty() &&
                (keys[i] == 'y' || keys[i] == 'a' || keys[i] == 'n')) {
                complete(InputState::LINE, std::string(1, keys[i]));
                i += 1;
                continue;
            }
            if (request) insert(std::string_view(keys).substr(i, end - i));
            i = end;
        }
        keys.erase(0, i);
    }

    struct Live {
        std::vector<std::string> lines;
        std::size_t caret_line = 0, caret_column = 0;
        bool cursor = false;
    };

    // Builds the live region. Every line fits in width - 1 cells, so the region's row count is exact.
    auto live(std::size_t width, std::size_t height) const -> Live {
        const auto usable = width - 1;
        Live result;
        std::vector<std::string> previews;
        if (pending_tool && !tool_committed)
            for (const auto line : split_lines(*pending_tool)) previews.push_back(paint(line, TerminalTone::TOOL));
        if (markdown) {
            for (auto& line : markdown->preview(width)) previews.push_back(std::move(line));
        } else if (reasoning) {
            auto lines = split_lines(reasoning_tail);
            if (lines.size() > 3) lines.erase(lines.begin(), lines.end() - 3);
            for (const auto line : lines) previews.push_back(paint(line, TerminalTone::THINKING));
        }
        const auto preview_rows = std::min<std::size_t>(8, std::max<std::size_t>(1, height / 3));
        std::vector<std::string> rows;
        for (auto& line : previews) {
            std::ranges::replace(line, '\t', ' ');
            for (auto& row : wrap_width(line, usable)) rows.push_back(std::move(row));
        }
        if (rows.size() > preview_rows) rows.erase(rows.begin(), rows.end() - preview_rows);
        for (auto& row : rows) result.lines.push_back(std::move(row));

        std::string rule;
        for (std::size_t i = 0; i < usable; ++i) rule += "─";
        result.lines.push_back(paint(rule, TerminalTone::DIM));
        if (request && !request->remember) {
            constexpr std::string_view HINT = "Approval required: y = allow once | a = allow tool for session | n = deny";
            result.lines.push_back(fit_width(paint(HINT, TerminalTone::WARNING), usable));
        }

        if (request) {
            // Wrap the input by cells, remembering where the caret lands.
            const auto prompt = request->prompt;
            const auto indent = std::max<std::size_t>(2, display_width(prompt));
            const auto room = std::max<std::size_t>(8, usable > indent ? usable - indent : 8);
            std::vector<std::string> rows(1);
            std::size_t used = 0, caret_row = 0, caret_column = 0;
            for (std::size_t index = 0;;) {
                if (index == caret) {
                    if (used >= room) {
                        rows.emplace_back();
                        used = 0;
                    }
                    caret_row = rows.size() - 1;
                    caret_column = used;
                }
                if (index >= input.size()) break;
                if (input[index] == '\n') {
                    rows.emplace_back();
                    used = 0;
                    ++index;
                    continue;
                }
                const auto end = next(index);
                auto glyph = input.substr(index, end - index);
                if (glyph == "\t") glyph = "    ";
                const auto cells = display_width(glyph);
                if (used + cells > room) {
                    rows.emplace_back();
                    used = 0;
                }
                rows.back() += glyph;
                used += cells;
                index = end;
            }
            const auto budget = height - 1 > result.lines.size() + 1 ? height - 1 - result.lines.size() - 1 : 1;
            const auto shown = std::min<std::size_t>(rows.size(), std::clamp<std::size_t>(budget, 1, 12));
            const auto first = caret_row + 1 > shown ? caret_row + 1 - shown : 0;
            for (std::size_t row = first; row < first + shown && row < rows.size(); ++row) {
                std::string line(indent, ' ');
                if (!row)
                    line = (color ? "\033[32m" + prompt + std::string(RESET) : prompt) +
                           std::string(indent - display_width(prompt), ' ');
                if (row == 0 && input.empty() && request->remember)
                    line += paint(fit_width(PLACEHOLDER, room), TerminalTone::DIM);
                else
                    line += rows[row];
                if (row == caret_row) {
                    result.caret_line = result.lines.size();
                    result.caret_column = indent + caret_column;
                }
                result.lines.push_back(std::move(line));
            }
            result.cursor = true;
        }

        std::string metrics = status.empty() ? "Ready" : status, phase;
        if (const auto split = metrics.rfind(" | "); split != std::string::npos) {
            phase = metrics.substr(split + 3);
            metrics.resize(split);
        }
        if (activity_running) {
            static constexpr std::string_view SPINNER[] = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};
            const auto tick = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::steady_clock::now() - activity_started)
                                  .count() /
                              100;
            phase = std::string(SPINNER[tick % 10]) + ' ' +
                    (request ? std::string("Awaiting approval") : activity) + ' ' + seconds_since(activity_started);
        }
        auto bar = " " + metrics + (phase.empty() ? "" : " | " + phase);
        bar = fit_width(bar, usable);
        bar += std::string(usable - display_width(bar), ' ');
        result.lines.push_back(color ? "\033[48;5;238m\033[97m" + bar + std::string(RESET) : bar);
        if (!result.cursor) result.caret_line = result.lines.size() - 1;
        return result;
    }

#if defined(__unix__) || defined(__APPLE__)
    // Replaces the live region: committed lines go above it into scrollback, then the region is drawn again.
    auto render(bool final = false) -> void {
        const auto [width, height] = size();
        std::string out = "\033[?25l";
        if (clear_screen) {
            out += "\033[H\033[2J";
            clear_screen = false;
        } else if (live_rows) {
            if (caret_row) out += "\033[" + std::to_string(caret_row) + "A";
            out += "\r\033[J";
        }
        for (const auto& line : commits) out += line + "\r\n";
        commits.clear();
        if (final) {
            live_rows = caret_row = 0;
            out += "\033[?25h";
            write_all(tty, out);
            return;
        }
        const auto region = live(width, height);
        for (std::size_t i = 0; i < region.lines.size(); ++i) {
            out += region.lines[i];
            if (i + 1 < region.lines.size()) out += "\r\n";
        }
        if (const auto up = region.lines.size() - 1 - region.caret_line) out += "\033[" + std::to_string(up) + "A";
        out += "\r";
        if (region.caret_column) out += "\033[" + std::to_string(region.caret_column) + "C";
        if (region.cursor) out += "\033[?25h";
        live_rows = region.lines.size();
        caret_row = region.caret_line;
        write_all(tty, out);
    }

    // Output written to stdout or stderr while the UI runs (logs, libraries) becomes dim transcript lines.
    auto drain_capture(bool flush) -> void {
        char buffer[4096];
        for (;;) {
            pollfd ready{capture[0], POLLIN, 0};
            if (::poll(&ready, 1, 0) <= 0) break;
            const auto count = ::read(capture[0], buffer, sizeof(buffer));
            if (count <= 0) break;
            captured.append(buffer, static_cast<std::size_t>(count));
        }
        std::size_t start = 0;
        for (auto end = captured.find('\n'); end != std::string::npos; end = captured.find('\n', start)) {
            commit(std::string_view(captured).substr(start, end - start), TerminalTone::DIM);
            start = end + 1;
        }
        captured.erase(0, start);
        if (flush && !captured.empty()) commit(std::exchange(captured, {}), TerminalTone::DIM);
    }
#endif
};

Terminal::Terminal(std::string_view color) {
    const auto* term = std::getenv("TERM");
    const bool dumb = term && (std::string_view(term) == "dumb" || std::string_view(term) == "emacs" ||
                               std::string_view(term) == "cons25");
#if defined(__unix__) || defined(__APPLE__)
    const bool edit = ::isatty(STDIN_FILENO) && ::isatty(STDOUT_FILENO) && !dumb;
    const bool use_color = color == "always" || (color == "auto" && edit && !std::getenv("NO_COLOR"));
#else
    const bool edit = false;
    const bool use_color = color == "always";
#endif
    impl_ = std::make_unique<Impl>(edit, use_color);
}

Terminal::~Terminal() = default;

auto Terminal::interactive() const -> bool { return impl_->edit; }

auto Terminal::run_interactive(std::function<int()> worker, std::function<void()> interrupt) -> int {
#if defined(__unix__) || defined(__APPLE__)
    if (!impl_->edit) return worker();
    auto& state = *impl_;
    state.interrupt = std::move(interrupt);
    if (::tcgetattr(STDIN_FILENO, &saved_terminal) < 0 || ::pipe(state.wake) < 0) return worker();
    if (::pipe(state.capture) < 0) {
        ::close(state.wake[0]);
        ::close(state.wake[1]);
        return worker();
    }
    for (const auto fd : {state.wake[0], state.wake[1], state.capture[0]}) ::fcntl(fd, F_SETFD, FD_CLOEXEC);
    for (const auto fd : {state.wake[0], state.wake[1]}) ::fcntl(fd, F_SETFL, ::fcntl(fd, F_GETFL) | O_NONBLOCK);
    std::cout.flush();
    std::cerr.flush();
    state.tty = ::fcntl(STDOUT_FILENO, F_DUPFD_CLOEXEC, 3);
    state.saved_err = ::fcntl(STDERR_FILENO, F_DUPFD_CLOEXEC, 3);
    ::dup2(state.capture[1], STDOUT_FILENO);
    ::dup2(state.capture[1], STDERR_FILENO);
    ::close(state.capture[1]);

    auto raw = saved_terminal;
    raw.c_lflag &= ~static_cast<tcflag_t>(ICANON | ECHO | ISIG | IEXTEN);
    raw.c_iflag &= ~static_cast<tcflag_t>(IXON | ICRNL | INLCR);
    raw.c_cc[VMIN] = 1;
    raw.c_cc[VTIME] = 0;
    ::tcsetattr(STDIN_FILENO, TCSADRAIN, &raw);
    terminal_raw = true;
    wake_fd = state.wake[1];
    struct sigaction resize{}, terminate{}, old_resize{}, old_term{}, old_hup{}, old_quit{};
    resize.sa_handler = on_resize;
    terminate.sa_handler = on_terminate;
    ::sigaction(SIGWINCH, &resize, &old_resize);
    ::sigaction(SIGTERM, &terminate, &old_term);
    ::sigaction(SIGHUP, &terminate, &old_hup);
    ::sigaction(SIGQUIT, &terminate, &old_quit);
    write_all(state.tty, "\033[?2004h");

    state.running = true;
    state.render();
    std::jthread background([&] {
        try {
            state.worker_result = worker();
        } catch (...) {
            state.worker_exception = std::current_exception();
        }
        state.post([&] { state.worker_done = true; });
    });
    while (!state.worker_done) {
        pollfd fds[] = {{STDIN_FILENO, static_cast<short>(state.input_open ? POLLIN : 0), 0},
                        {state.wake[0], POLLIN, 0},
                        {state.capture[0], POLLIN, 0}};
        const int timeout = !state.keys.empty() && !state.paste ? 30 : state.activity_running ? 100 : -1;
        const int ready = ::poll(fds, 3, timeout);
        if (ready < 0 && errno != EINTR) break;
        if (ready > 0 && (fds[1].revents & POLLIN)) {
            char drain[256];
            while (::read(state.wake[0], drain, sizeof(drain)) > 0) {
            }
        }
        state.run_tasks();
        if (ready > 0 && (fds[2].revents & POLLIN)) state.drain_capture(false);
        if (ready > 0 && (fds[0].revents & (POLLIN | POLLHUP | POLLERR))) {
            char buffer[4096];
            const auto count = ::read(STDIN_FILENO, buffer, sizeof(buffer));
            if (count > 0) {
                state.keys.append(buffer, static_cast<std::size_t>(count));
            } else if (count == 0 || errno != EINTR) {
                state.input_open = false;
                if (state.request) state.complete(InputState::END);
            }
        }
        state.handle_keys(ready == 0);
        state.render();
    }
    background.join();
    state.run_tasks();
    std::cout.flush();
    std::cerr.flush();
    state.drain_capture(true);
    state.flush_tool();
    state.render(true);
    write_all(state.tty, "\033[?2004l");

    ::tcsetattr(STDIN_FILENO, TCSADRAIN, &saved_terminal);
    terminal_raw = false;
    wake_fd = -1;
    ::sigaction(SIGWINCH, &old_resize, nullptr);
    ::sigaction(SIGTERM, &old_term, nullptr);
    ::sigaction(SIGHUP, &old_hup, nullptr);
    ::sigaction(SIGQUIT, &old_quit, nullptr);
    ::dup2(state.tty, STDOUT_FILENO);
    ::dup2(state.saved_err, STDERR_FILENO);
    for (const auto fd : {state.tty, state.saved_err, state.capture[0], state.wake[0], state.wake[1]}) ::close(fd);
    state.running = false;
    if (state.worker_exception) std::rethrow_exception(state.worker_exception);
    return state.worker_result;
#else
    (void)interrupt;
    return worker();
#endif
}

auto Terminal::read(std::string_view prompt, bool remember) -> Result<TerminalInput> {
    if (!impl_->edit || !impl_->running) {
        std::cerr << prompt << std::flush;
        std::string line;
        errno = 0;
        if (!std::getline(std::cin, line)) {
            if (errno == EINTR) {
                std::cin.clear();
                return TerminalInput{InputState::INTERRUPTED, {}};
            }
            if (std::cin.eof()) return TerminalInput{InputState::END, {}};
            return std::unexpected(Error{ErrorCode::IO, "cannot read terminal input"});
        }
        return TerminalInput{InputState::LINE, std::move(line)};
    }
    auto request = std::make_shared<Impl::Request>();
    request->prompt = terminal_text(prompt);
    request->remember = remember;
    impl_->post([state = impl_.get(), request] {
        state->request = request;
        state->input.clear();
        state->caret = 0;
        state->history_index = state->history.size();
        if (!state->input_open) state->complete(InputState::END);
    });
    std::unique_lock lock(request->mutex);
    request->ready.wait(lock, [&] { return request->completed; });
    return std::move(request->result);
}

auto Terminal::label(std::string_view text, std::string_view color) const -> std::string {
    if (!impl_->color || color.empty()) return std::string(text);
    return std::string(color) + std::string(text) + std::string(RESET);
}

auto Terminal::write(std::string_view text, TerminalTone tone) -> void {
    if (impl_->running) {
        auto line = std::string(text);
        if (line.ends_with('\n')) line.pop_back();
        impl_->post([state = impl_.get(), line = std::move(line), tone] { state->commit(line, tone); });
        return;
    }
    const auto clean = terminal_text(text);
    const auto color = ansi_color(tone);
    if (impl_->color && !color.empty()) std::cerr << color;
    std::cerr << clean;
    if (impl_->color && !color.empty()) std::cerr << RESET;
    std::cerr.flush();
}

auto Terminal::tool(std::string_view name, const Json& arguments, bool full) -> void {
    std::string summary = "[tool] " + terminal_text(name);
    if (full) write(summary + "\n", TerminalTone::TOOL);
    for (const auto& [key, value] : arguments.items()) {
        const auto text =
            value.is_string() ? value.get<std::string>() : value.dump(2, ' ', false, Json::error_handler_t::replace);
        const auto prefix = name == "run_command" && key == "command" ? "  $ " : "  " + terminal_text(key) + ": ";
        if (full)
            write(prefix + terminal_text(text) + "\n", TerminalTone::DIM);
        else
            summary += prefix + text;
    }
    if (full) return;
    summary = preview(summary, 2, 300);
    if (!impl_->running) {
        write(summary + "\n", TerminalTone::TOOL);
        return;
    }
    // A running tool's summary stays in the live region until it finishes, then lands with its timing.
    impl_->post([state = impl_.get(), summary = std::move(summary)] {
        state->flush_tool();
        state->pending_tool = summary;
        state->tool_committed = false;
    });
}

auto Terminal::thinking(std::string_view text) -> void {
    if (text.empty()) return;
    write("Thinking: " + preview(text, 2, 300) + "\n", TerminalTone::THINKING);
}

auto Terminal::answer(std::string_view text) -> void {
    if (!impl_->edit) {
        std::cout << terminal_text(text) << '\n' << std::flush;
        return;
    }
    write("\n" + std::string(text), TerminalTone::ASSISTANT);
}

auto Terminal::status(std::string_view text) -> void {
    if (!impl_->running) return;
    impl_->post([state = impl_.get(), text = terminal_text(text)] { state->status = text; });
}

auto Terminal::activity(std::string_view text, bool tool) -> void {
    if (!impl_->running) return;
    impl_->post([state = impl_.get(), text = terminal_text(text), tool] {
        state->activity = text;
        state->activity_running = true;
        state->activity_started = std::chrono::steady_clock::now();
        state->tool_activity = tool && state->pending_tool.has_value();
    });
}

auto Terminal::finish_activity(bool failed) -> void {
    if (!impl_->running) return;
    impl_->post([state = impl_.get(), failed] {
        if (state->tool_activity && state->pending_tool) {
            const auto timing =
                std::string(failed ? "[failed, " : "[done, ") + seconds_since(state->activity_started) + "]";
            if (state->tool_committed) {
                state->commit_line(state->paint("  " + timing, TerminalTone::DIM));
            } else {
                auto text = *state->pending_tool;
                const auto arguments = text.find("  ");
                text.insert(arguments == std::string::npos ? text.size() : arguments, " " + timing);
                state->tool_committed = true;
                for (const auto part : split_lines(text)) state->commit_line(state->paint(part, TerminalTone::TOOL));
            }
            state->pending_tool.reset();
        }
        state->tool_activity = false;
        state->activity.clear();
        state->activity_running = false;
    });
}

auto Terminal::stream(std::string_view text, bool thinking) -> void {
    if (text.empty() || !impl_->running) return;
    impl_->post([state = impl_.get(), text = terminal_text(text), thinking] {
        if (thinking && !state->markdown) {
            state->reasoning = true;
            if (state->reasoning_head.size() < 400) state->reasoning_head += text.substr(0, 400);
            state->reasoning_tail += text;
            if (state->reasoning_tail.size() > 4096) {
                auto cut = state->reasoning_tail.size() - 2048;
                while (cut < state->reasoning_tail.size() &&
                       (static_cast<unsigned char>(state->reasoning_tail[cut]) & 0xc0) == 0x80)
                    ++cut;
                state->reasoning_tail.erase(0, cut);
            }
            return;
        }
        if (thinking) return;
        if (!state->markdown) {
            state->commit_reasoning();
            state->commit_line({});
            state->markdown.emplace(state->color);
        }
        for (auto& line : state->markdown->feed(text, state->size().first)) state->commit_line(std::move(line));
    });
}

auto Terminal::end_stream() -> void {
    if (!impl_->running) return;
    impl_->post([state = impl_.get()] {
        state->commit_reasoning();
        if (state->markdown) {
            for (auto& line : state->markdown->finish(state->size().first)) state->commit_line(std::move(line));
            state->markdown.reset();
        }
    });
}

} // namespace kidi::agent
