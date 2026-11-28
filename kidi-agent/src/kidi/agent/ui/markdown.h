#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace kidi::agent::ui {

// Terminal cells taken by UTF-8 text; ANSI CSI sequences take none.
auto display_width(std::string_view text) -> std::size_t;
// Cuts text to at most `width` cells, keeping ANSI sequences, and marks a cut with "…".
auto fit_width(std::string_view text, std::size_t width) -> std::string;
// Splits text into rows of at most `width` cells, carrying ANSI styles across row breaks.
auto wrap_width(std::string_view text, std::size_t width) -> std::vector<std::string>;
// Like wrap_width, but breaks at spaces when it can.
auto wrap_words(std::string_view text, std::size_t width) -> std::vector<std::string>;

// Renders streamed Markdown for a terminal, one finished line at a time. Lines are never hard-wrapped, so copied text
// keeps its original line breaks. Tables are held until they end so their columns can be aligned. Without color,
// lines pass through unchanged.
class MarkdownStream {
public:
    explicit MarkdownStream(bool color) : color_(color) {}
    // Appends text and returns the lines it completed.
    auto feed(std::string_view text, std::size_t width) -> std::vector<std::string>;
    // Renders held table rows and the unfinished line without consuming them.
    auto preview(std::size_t width) const -> std::vector<std::string>;
    // Renders everything still held and resets the stream.
    auto finish(std::size_t width) -> std::vector<std::string>;

private:
    auto line(std::string_view text, std::size_t width, std::vector<std::string>& out) -> void;
    auto flush_table(std::size_t width, std::vector<std::string>& out) -> void;

    bool color_;
    std::string partial_;
    std::vector<std::string> table_;
    std::string fence_;
};

auto render_markdown(std::string_view text, bool color, std::size_t width) -> std::vector<std::string>;

} // namespace kidi::agent::ui
