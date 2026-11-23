#pragma once

#include "kidi/core/error.h"

#include <string>
#include <string_view>
#include <vector>

namespace kidi::code {

auto terminal_text(std::string_view text) -> std::string;

enum class InputState { LINE, INTERRUPTED, END };

struct TerminalInput {
    InputState state;
    std::string text;
};

// One terminal per CLI process; linenoise has process-global editing state.
class Terminal {
public:
    explicit Terminal(std::string_view color);
    auto read(std::string_view prompt, bool remember = true) -> Result<TerminalInput>;
    auto label(std::string_view text, std::string_view color) const -> std::string;

private:
    bool edit_ = false;
    bool color_ = false;
    std::vector<std::string> history_;
    std::size_t history_bytes_ = 0;
};

} // namespace kidi::code
