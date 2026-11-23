#include "kidi/code/terminal.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>

#if defined(__unix__) || defined(__APPLE__)
#include <linenoise.h>
#include <unistd.h>
#endif

namespace kidi::code {
namespace {

#if defined(__unix__) || defined(__APPLE__)
auto complete(const char* text, linenoiseCompletions* completions) -> void {
    for (const auto* command : {"/help", "/exit", "/reset", "/permissions", "/context", "/compact"})
        if (std::string_view(command).starts_with(text)) linenoiseAddCompletion(completions, command);
}

auto add_history(const char* text) -> bool {
    errno = 0;
    // linenoise also returns zero for duplicates; POSIX allocation failure sets errno.
    return linenoiseHistoryAdd(text) || errno == 0;
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

Terminal::Terminal(std::string_view color) {
    const auto* term = std::getenv("TERM");
    const bool dumb = term && (std::string_view(term) == "dumb" || std::string_view(term) == "emacs" ||
                               std::string_view(term) == "cons25");
#if defined(__unix__) || defined(__APPLE__)
    edit_ = ::isatty(STDIN_FILENO) && ::isatty(STDOUT_FILENO) && !dumb;
    color_ = color == "always" || (color == "auto" && ::isatty(STDERR_FILENO) && !dumb && !std::getenv("NO_COLOR"));
    linenoiseSetMultiLine(1);
    linenoiseSetCompletionCallback(complete);
#else
    color_ = color == "always";
#endif
}

auto Terminal::label(std::string_view text, std::string_view color) const -> std::string {
    return color_ ? std::string(color) + std::string(text) + "\033[0m" : std::string(text);
}

auto Terminal::read(std::string_view prompt, bool remember) -> Result<TerminalInput> {
    std::cout.flush();
    std::cerr.flush();
    std::string line;
#if defined(__unix__) || defined(__APPLE__)
    if (edit_) {
        // Rebuild history to isolate approvals and discard linenoise's temporary entry after Ctrl-C.
        if (!linenoiseHistorySetMaxLen(1) || !add_history("") || !linenoiseHistorySetMaxLen(101))
            return std::unexpected(Error{ErrorCode::IO, "cannot allocate terminal history"});
        if (remember)
            for (const auto& entry : history_)
                if (!add_history(entry.c_str()))
                    return std::unexpected(Error{ErrorCode::IO, "cannot allocate terminal history"});
        linenoiseSetCompletionCallback(remember ? complete : nullptr);
        errno = 0;
        const std::string owned_prompt(prompt);
        std::unique_ptr<char, decltype(&linenoiseFree)> input(linenoise(owned_prompt.c_str()), linenoiseFree);
        const auto error = errno;
        std::cout.flush();
        if (!input) {
            if (error == EAGAIN || error == EINTR) return TerminalInput{InputState::INTERRUPTED, {}};
            if (error == ENOENT || error == 0) return TerminalInput{InputState::END, {}};
            return std::unexpected(Error{ErrorCode::IO, "read terminal: " + std::string(std::strerror(error))});
        }
        line = input.get();
    } else
#endif
    {
        std::cerr << prompt << std::flush;
        errno = 0;
        if (!std::getline(std::cin, line)) {
            if (errno == EINTR) {
                std::cin.clear();
                return TerminalInput{InputState::INTERRUPTED, {}};
            }
            if (std::cin.eof()) return TerminalInput{InputState::END, {}};
            return std::unexpected(Error{ErrorCode::IO, "cannot read terminal input"});
        }
    }
    if (remember && !line.empty() && (history_.empty() || history_.back() != line)) {
        constexpr std::size_t MAX_HISTORY_BYTES = 512 * 1024;
        while (!history_.empty() && (history_.size() == 100 || history_bytes_ + line.size() > MAX_HISTORY_BYTES)) {
            history_bytes_ -= history_.front().size();
            history_.erase(history_.begin());
        }
        if (line.size() <= MAX_HISTORY_BYTES) {
            history_.push_back(line);
            history_bytes_ += line.size();
        }
    }
    return TerminalInput{InputState::LINE, std::move(line)};
}

} // namespace kidi::code
