#pragma once

#include "kidi/code/agent.h"
#include "kidi/core/error.h"

#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace kidi::code {

auto terminal_text(std::string_view text) -> std::string;

enum class InputState { LINE, INTERRUPTED, END };

struct TerminalInput {
    InputState state;
    std::string text;
};

enum class TerminalTone { DEFAULT, USER, TOOL, THINKING, ASSISTANT, ERROR, CONTEXT, WARNING, DIM, SESSION };

class Terminal {
public:
    explicit Terminal(std::string_view color);
    ~Terminal();
    Terminal(const Terminal&) = delete;
    auto operator=(const Terminal&) -> Terminal& = delete;
    auto interactive() const -> bool;
    auto run_interactive(std::function<int()> worker, std::function<void()> interrupt) -> int;
    auto read(std::string_view prompt, bool remember = true) -> Result<TerminalInput>;
    auto label(std::string_view text, std::string_view color) const -> std::string;
    auto write(std::string_view text, TerminalTone tone = TerminalTone::DEFAULT) -> void;
    auto tool(std::string_view name, const Json& arguments, bool full = false) -> void;
    auto thinking(std::string_view text) -> void;
    auto answer(std::string_view text) -> void;
    auto status(std::string_view text) -> void;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kidi::code
