#include "kidi/agent/terminal.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
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

#if defined(__unix__) || defined(__APPLE__)
#include <ftxui/component/component.hpp>
#include <ftxui/component/loop.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/string.hpp>
#include <unistd.h>
#endif

namespace kidi::agent {
namespace {

auto preview(std::string_view text, std::size_t lines = 6, std::size_t bytes = 1200) -> std::string {
    auto clean = terminal_text(text);
    std::size_t end = 0, count = 0;
    while (end < clean.size() && end < bytes && count < lines)
        if (clean[end++] == '\n') ++count;
    while (end < clean.size() && end && (static_cast<unsigned char>(clean[end]) & 0xc0) == 0x80) --end;
    if (end < clean.size()) {
        clean.resize(end);
        while (!clean.empty() && clean.back() == '\n') clean.pop_back();
        clean += "... (display truncated)";
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
            return "\033[2;37m";
        case TerminalTone::ASSISTANT:
            return "\033[1;37m";
        case TerminalTone::ERROR:
            return "\033[31m";
        case TerminalTone::CONTEXT:
            return "\033[36m";
        case TerminalTone::WARNING:
            return "\033[33m";
        case TerminalTone::DIM:
            return "\033[2;37m";
        case TerminalTone::SESSION:
            return "\033[2;37m";
        case TerminalTone::DEFAULT:
            return {};
    }
    return {};
}

#if defined(__unix__) || defined(__APPLE__)
auto ftxui_color(TerminalTone tone) -> ftxui::Color {
    switch (tone) {
        case TerminalTone::USER:
            return ftxui::Color::Green;
        case TerminalTone::TOOL:
            return ftxui::Color::Cyan;
        case TerminalTone::THINKING:
            return ftxui::Color::GrayDark;
        case TerminalTone::ASSISTANT:
            return ftxui::Color::White;
        case TerminalTone::ERROR:
            return ftxui::Color::Red;
        case TerminalTone::CONTEXT:
            return ftxui::Color::Cyan;
        case TerminalTone::WARNING:
            return ftxui::Color::Yellow;
        case TerminalTone::DIM:
            return ftxui::Color::GrayDark;
        case TerminalTone::SESSION:
            return ftxui::Color::GrayDark;
        case TerminalTone::DEFAULT:
            return ftxui::Color::White;
    }
    return ftxui::Color::White;
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

struct Terminal::Impl {
    explicit Impl(bool edit, bool color)
        : edit(edit),
          color(color)
#if defined(__unix__) || defined(__APPLE__)
          ,
          screen(ftxui::ScreenInteractive::Fullscreen())
#endif
    {
    }

    struct Entry {
        std::string text;
        TerminalTone tone;
    };
    struct Request {
        std::mutex mutex;
        std::condition_variable ready;
        bool completed = false;
        TerminalInput result{InputState::INTERRUPTED, {}};
        std::string prompt;
        bool remember = true;
    };

    bool edit;
    bool color;
    std::vector<Entry> entries;
    std::vector<std::string> history;
    std::string status;
    std::string activity;
    std::atomic<bool> activity_running{false};
    std::chrono::steady_clock::time_point activity_started;
    std::optional<std::size_t> active_tool;
    std::optional<TerminalTone> stream_tone;
    std::size_t reasoning_bytes = 0, reasoning_lines = 0;
    std::string input_value;
    std::string prompt;
    std::size_t history_index = 0;
    bool follow_tail = true;
    std::size_t scroll_end = 0;
    std::optional<std::string> paste;
    bool paste_truncated = false;
    std::shared_ptr<Request> request;
#if defined(__unix__) || defined(__APPLE__)
    ftxui::ScreenInteractive screen;
#endif
    ftxui::Component input_component;
    ftxui::Component root;
    std::function<void()> interrupt;
    bool worker_done = false;
    bool running = false;
    int worker_result = 1;
    std::exception_ptr worker_exception;

#if defined(__unix__) || defined(__APPLE__)
    auto viewport_height() const -> std::size_t {
        const auto extra_input = std::min<std::ptrdiff_t>(3, std::ranges::count(input_value, '\n'));
        const auto approval = request && !request->remember ? 1 : 0;
        return static_cast<std::size_t>(std::max<std::ptrdiff_t>(
            1, ftxui::Terminal::Size().dimy - 4 - extra_input - approval));
    }

    auto post(ftxui::Task task) -> void {
        screen.Post(std::move(task));
        screen.PostEvent(ftxui::Event::Custom);
    }
#endif

    auto append(std::string_view text, TerminalTone tone) -> void {
        std::size_t start = 0;
        do {
            const auto end = text.find('\n', start);
            entries.push_back(
                {terminal_text(text.substr(start, end == std::string_view::npos ? end : end - start)), tone});
            if (end == std::string_view::npos) break;
            start = end + 1;
        } while (start < text.size());
        if (entries.size() > 4000) {
            entries.erase(entries.begin(), entries.begin() + 1000);
            if (active_tool) {
                if (*active_tool < 1000) active_tool.reset();
                else *active_tool -= 1000;
            }
            scroll_end = scroll_end > 1000 ? scroll_end - 1000 : 0;
        }
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
            follow_tail = true;
            if (active->remember && !entries.empty()) append("", TerminalTone::DEFAULT);
            append((active->remember ? "> " : "? ") + text, TerminalTone::USER);
        }
        {
            const std::lock_guard lock(active->mutex);
            active->result = {state, std::move(text)};
            active->completed = true;
        }
        active->ready.notify_one();
        request.reset();
        prompt.clear();
        input_value.clear();
        history_index = history.size();
    }
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
    state.screen.TrackMouse(true);
    state.screen.ForceHandleCtrlC(false);

    ftxui::InputOption input_options;
    input_options.content = &state.input_value;
    input_options.placeholder = &state.prompt;
    input_options.multiline = true;
    input_options.transform = [&](ftxui::InputState input_state) {
        auto element = input_state.element;
        if (!state.color) return element;
        return element | ftxui::color(ftxui::Color::Green);
    };
    state.input_component = ftxui::Input(input_options);
    state.root = ftxui::Renderer(state.input_component, [&] {
        const auto dimensions = ftxui::Terminal::Size();
        const auto height = state.viewport_height();
        auto end = state.follow_tail ? state.entries.size() : std::min(state.scroll_end, state.entries.size());
        // Wrap long lines instead of clipping them; fill the viewport from the newest row upward.
        const auto width = static_cast<std::size_t>(std::max(8, dimensions.dimx - 2));
        ftxui::Elements transcript;
        for (auto index = end; index > 0 && transcript.size() < height; --index) {
            const auto& entry = state.entries[index - 1];
            std::vector<std::string> rows(1);
            std::size_t used = 0;
            for (auto& glyph : ftxui::Utf8ToGlyphs(entry.text)) {
                const auto cells = static_cast<std::size_t>(std::max(0, ftxui::string_width(glyph)));
                if (used + cells > width) {
                    rows.emplace_back();
                    used = 0;
                }
                rows.back() += glyph;
                used += cells;
            }
            for (auto row = rows.rbegin(); row != rows.rend() && transcript.size() < height; ++row) {
                auto line = ftxui::text(std::move(*row));
                if (state.color) line = line | ftxui::color(ftxui_color(entry.tone));
                transcript.push_back(std::move(line));
            }
        }
        std::ranges::reverse(transcript);
        if (transcript.empty()) transcript.push_back(ftxui::text(" "));
        auto log = ftxui::vbox(std::move(transcript)) | ftxui::vscroll_indicator | ftxui::frame | ftxui::flex;
        // Stable metrics on the left; the changing phase or live timer at the right end, never clipped.
        std::string metrics = state.status.empty() ? "Ready" : state.status, phase;
        if (const auto split = metrics.rfind(" | "); split != std::string::npos) {
            phase = metrics.substr(split + 3);
            metrics.resize(split);
        }
        if (!state.activity.empty()) {
            const auto seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - state.activity_started).count();
            std::ostringstream live;
            live << (state.request ? "Awaiting approval" : state.activity) << ' ' << std::fixed
                 << std::setprecision(1) << seconds << 's';
            phase = live.str();
        }
        auto status = ftxui::hbox({ftxui::text(metrics) | ftxui::xflex_shrink,
                                   ftxui::text(phase.empty() ? "" : " | " + phase), ftxui::filler()});
        if (state.color) status = status | ftxui::bgcolor(ftxui::Color::GrayDark) | ftxui::color(ftxui::Color::White);
        auto prompt = state.request ? state.input_component->Render() : ftxui::text(" ");
        prompt = prompt | ftxui::size(ftxui::HEIGHT, ftxui::LESS_THAN, 4);
        if (state.request && !state.request->remember) {
            auto approval = ftxui::text(" Approval required: y = allow once | a = allow tool for session | n = deny ");
            if (state.color) approval = approval | ftxui::color(ftxui::Color::Yellow);
            return ftxui::vbox({ftxui::text(" Kidi Agent | Approval") | ftxui::bold, log, ftxui::separator(), approval,
                                prompt, status | ftxui::size(ftxui::WIDTH, ftxui::EQUAL, dimensions.dimx)});
        }
        return ftxui::vbox({ftxui::text(" Kidi Agent") | ftxui::bold, log | ftxui::flex, ftxui::separator(), prompt,
                            status | ftxui::size(ftxui::WIDTH, ftxui::EQUAL, dimensions.dimx)});
    });
    state.root |= ftxui::CatchEvent([&](ftxui::Event event) {
        // Stock FTXUI reports bracketed-paste markers as special events and the
        // pasted bytes as ordinary key events; gather them into one insertion.
        static constexpr std::string_view PASTE_BEGIN = "\x1B[200~";
        static constexpr std::string_view PASTE_END = "\x1B[201~";
        static constexpr std::size_t MAX_PASTE_BYTES = 4 * 1024 * 1024;
        if (event.input() == PASTE_BEGIN) {
            state.paste.emplace();
            state.paste_truncated = false;
            return true;
        }
        if (state.paste) {
            if (event.input() != PASTE_END) {
                const auto bytes = event == ftxui::Event::Return ? std::string("\n") : event.input();
                if (state.paste->size() + bytes.size() <= MAX_PASTE_BYTES) state.paste->append(bytes);
                else state.paste_truncated = true;
                return true;
            }
            auto text = std::move(*state.paste);
            state.paste.reset();
            if (state.paste_truncated) text += "\n[paste truncated at 4 MiB]\n";
            if (state.request && !text.empty()) state.input_component->OnEvent(ftxui::Event::Character(std::move(text)));
            return true;
        }
        if (state.request && !state.request->remember && state.input_value.empty() && event.is_character() &&
            (event.input() == "y" || event.input() == "a" || event.input() == "n")) {
            state.complete(InputState::LINE, event.input());
            return true;
        }
        if (event == ftxui::Event::Return && state.request) {
            state.complete(InputState::LINE, std::exchange(state.input_value, {}));
            return true;
        }
        if (event == ftxui::Event::CtrlD && state.request && state.input_value.empty()) {
            state.complete(InputState::END);
            return true;
        }
        if (event == ftxui::Event::CtrlC) {
            if (state.request) {
                state.input_value.clear();
                state.complete(InputState::INTERRUPTED);
            } else if (state.interrupt) {
                state.append("Cancelling current operation...", TerminalTone::WARNING);
                state.interrupt();
            }
            return true;
        }
        if (event == ftxui::Event::ArrowUp && state.request && state.input_value.empty() && !state.history.empty()) {
            if (state.history_index) --state.history_index;
            state.input_value = state.history[state.history_index];
            return true;
        }
        if (event == ftxui::Event::ArrowDown && state.request && state.input_value.empty()) {
            if (state.history_index < state.history.size()) ++state.history_index;
            if (state.history_index < state.history.size()) state.input_value = state.history[state.history_index];
            return true;
        }
        if (event == ftxui::Event::PageUp) {
            state.follow_tail = false;
            const auto page = state.viewport_height();
            if (!state.scroll_end) state.scroll_end = state.entries.size();
            state.scroll_end = state.scroll_end > page ? state.scroll_end - page : page;
            return true;
        }
        if (event == ftxui::Event::PageDown) {
            const auto page = state.viewport_height();
            state.scroll_end = std::min(state.entries.size(), state.scroll_end + page);
            state.follow_tail = state.scroll_end >= state.entries.size();
            return true;
        }
        if (event == ftxui::Event::Tab && state.request && state.request->remember &&
            state.input_value.starts_with('/')) {
            static constexpr std::string_view COMMANDS[] = {
                "/help", "/exit", "/reset", "/permissions", "/permissions reset", "/context", "/compact"};
            const auto match =
                std::ranges::find_if(COMMANDS, [&](auto command) { return command.starts_with(state.input_value); });
            if (match != std::end(COMMANDS)) state.input_value = *match;
            return true;
        }
        if (!state.request && (event.is_character() || event == ftxui::Event::Return ||
                               event == ftxui::Event::Backspace || event == ftxui::Event::Delete))
            return true;
        return false;
    });

    // Install the screen before workers post: FTXUI drops tasks sent to an inactive screen.
    ftxui::Loop loop(&state.screen, state.root);
    state.running = true;
    struct BracketedPaste {
        BracketedPaste() { std::cout << "\033[?2004h" << std::flush; }
        ~BracketedPaste() { std::cout << "\033[?2004l" << std::flush; }
    } bracketed_paste;
    std::jthread background([&] {
        try {
            state.worker_result = worker();
        } catch (...) {
            state.worker_exception = std::current_exception();
        }
        state.post([&] {
            state.worker_done = true;
            state.screen.Exit();
        });
    });
    std::jthread ticker([&](std::stop_token stop) {
        std::mutex mutex;
        std::condition_variable_any wake;
        std::unique_lock lock(mutex);
        while (!stop.stop_requested()) {
            wake.wait_for(lock, stop, std::chrono::milliseconds(100), [] { return false; });
            if (!stop.stop_requested() && state.activity_running.load())
                state.screen.PostEvent(ftxui::Event::Custom);
        }
    });
    loop.Run();
    ticker.request_stop();
    ticker.join();
    background.join();
    state.running = false;
    if (state.worker_exception) std::rethrow_exception(state.worker_exception);
    return state.worker_result;
#else
    (void)interrupt;
    return worker();
#endif
}

auto Terminal::read(std::string_view prompt, bool remember) -> Result<TerminalInput> {
    if (!impl_->edit) {
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
#if defined(__unix__) || defined(__APPLE__)
    auto request = std::make_shared<Impl::Request>();
    request->prompt = terminal_text(prompt);
    request->remember = remember;
    impl_->post([state = impl_.get(), request] {
        state->request = request;
        state->prompt = request->prompt;
        state->input_value.clear();
        state->history_index = state->history.size();
        state->input_component->TakeFocus();
    });
    std::unique_lock lock(request->mutex);
    request->ready.wait(lock, [&] { return request->completed; });
    return std::move(request->result);
#else
    return TerminalInput{InputState::END, {}};
#endif
}

auto Terminal::label(std::string_view text, std::string_view color) const -> std::string {
    if (!impl_->color || color.empty()) return std::string(text);
    return std::string(color) + std::string(text) + "\033[0m";
}

auto Terminal::write(std::string_view text, TerminalTone tone) -> void {
    if (impl_->edit && impl_->running) {
#if defined(__unix__) || defined(__APPLE__)
        impl_->post([state = impl_.get(), text = std::string(text), tone] { state->append(text, tone); });
#endif
        return;
    }
    const auto clean = terminal_text(text);
    const auto color = ansi_color(tone);
    if (impl_->color && !color.empty()) std::cerr << color;
    std::cerr << clean;
    if (impl_->color && !color.empty()) std::cerr << "\033[0m";
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
    if (!full) write(preview(summary, 2, 300) + "\n", TerminalTone::TOOL);
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
    write("Assistant: " + terminal_text(text) + "\n", TerminalTone::ASSISTANT);
}

auto Terminal::status(std::string_view text) -> void {
    if (!impl_->edit || !impl_->running) return;
#if defined(__unix__) || defined(__APPLE__)
    impl_->post([state = impl_.get(), text = terminal_text(text)] { state->status = text; });
#endif
}

auto Terminal::activity(std::string_view text, bool tool) -> void {
    if (!impl_->edit || !impl_->running) return;
#if defined(__unix__) || defined(__APPLE__)
    impl_->post([state = impl_.get(), text = terminal_text(text), tool] {
        state->activity = text;
        state->activity_running.store(true);
        state->activity_started = std::chrono::steady_clock::now();
        state->active_tool.reset();
        if (tool)
            for (auto index = state->entries.size(); index > 0; --index)
                if (state->entries[index - 1].tone == TerminalTone::TOOL &&
                    state->entries[index - 1].text.starts_with("[tool] ")) {
                    state->active_tool = index - 1;
                    break;
                }
    });
#endif
}

auto Terminal::finish_activity(bool failed) -> void {
    if (!impl_->edit || !impl_->running) return;
#if defined(__unix__) || defined(__APPLE__)
    impl_->post([state = impl_.get(), failed] {
        if (state->active_tool) {
            std::ostringstream timing;
            timing << (failed ? " [failed, " : " [done, ") << std::fixed << std::setprecision(1)
                   << std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - state->activity_started).count()
                   << "s]";
            auto& text = state->entries[*state->active_tool].text;
            const auto arguments = text.find("  ");
            text.insert(arguments == std::string::npos ? text.size() : arguments, timing.str());
        }
        state->active_tool.reset();
        state->activity.clear();
        state->activity_running.store(false);
    });
#endif
}

auto Terminal::stream(std::string_view text, bool thinking) -> void {
    if (text.empty() || !impl_->edit || !impl_->running) return;
#if defined(__unix__) || defined(__APPLE__)
    impl_->post([state = impl_.get(), text = terminal_text(text), thinking] {
        const auto tone = thinking ? TerminalTone::THINKING : TerminalTone::ASSISTANT;
        if (thinking && (state->reasoning_bytes >= 300 || state->reasoning_lines >= 2)) return;
        if (state->stream_tone != tone) {
            if (state->stream_tone && !state->entries.empty() && state->entries.back().text.empty())
                state->entries.pop_back();
            state->append(thinking ? "Thinking: " : "Assistant: ", tone);
            state->stream_tone = tone;
        }
        auto visible = std::string_view(text);
        bool truncated = false;
        if (thinking) {
            auto length = std::min(visible.size(), 300 - state->reasoning_bytes);
            while (length < visible.size() && length &&
                   (static_cast<unsigned char>(visible[length]) & 0xc0) == 0x80) --length;
            std::size_t lines = state->reasoning_lines;
            for (std::size_t index = 0; index < length; ++index)
                if (visible[index] == '\n' && ++lines >= 2) {
                    length = index;
                    break;
                }
            truncated = length < visible.size();
            visible = visible.substr(0, length);
            state->reasoning_bytes += length;
            state->reasoning_lines = lines;
            if (truncated) state->reasoning_bytes = 300;
        }
        std::size_t start = 0;
        for (;;) {
            const auto end = visible.find('\n', start);
            state->entries.back().text += visible.substr(start, end == std::string::npos ? end : end - start);
            if (end == std::string::npos) break;
            state->append("", tone);
            start = end + 1;
        }
        if (truncated) state->entries.back().text += "... (thinking preview)";
    });
#endif
}

auto Terminal::end_stream() -> void {
    if (!impl_->edit || !impl_->running) return;
#if defined(__unix__) || defined(__APPLE__)
    impl_->post([state = impl_.get()] {
        if (state->stream_tone && !state->entries.empty() && state->entries.back().text.empty())
            state->entries.pop_back();
        state->stream_tone.reset();
        state->reasoning_bytes = state->reasoning_lines = 0;
    });
#endif
}

} // namespace kidi::agent
