#pragma once

#include <string>
#include <string_view>

namespace kidi::agent::tools {

struct HtmlPage {
    std::string title;
    std::string markdown;
};

// Converts an HTML document into compact Markdown for a model's context: the main content (<main>, else the largest
// <article>, else <body>) with headings, paragraphs, lists, tables, code blocks and links resolved against
// `base_url`; links within the same site are written as root-relative /paths. Scripts, styles, navigation, footers,
// forms, media and common boilerplate (cookie banners, share bars, sidebars, comments) are dropped; emphasis and
// images are omitted to save tokens.
auto html_to_markdown(std::string_view html, std::string_view base_url) -> HtmlPage;

// `href` resolved against `base_url` as an absolute http(s) URL without its fragment or utm_* tracking parameters;
// empty for other schemes (javascript:, mailto:, data:) and fragment-only links.
auto resolve_url(std::string_view base_url, std::string_view href) -> std::string;

// Valid UTF-8: Latin-1/Windows-1252 bodies are converted; invalid bytes in others become U+FFFD.
auto to_utf8(std::string_view bytes, std::string_view charset) -> std::string;

} // namespace kidi::agent::tools
