#include "kidi/agent/tools/html.h"

#include "kidi/agent/text.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace kidi::agent::tools {
namespace {

auto is_space(char character) -> bool {
    return character == ' ' || character == '\t' || character == '\n' || character == '\r' || character == '\f';
}

auto is_alpha(char character) -> bool {
    return (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z');
}

auto append_utf8(std::string& out, std::uint32_t code) -> void {
    if (code == 0 || code > 0x10FFFF || (code >= 0xD800 && code <= 0xDFFF)) code = 0xFFFD;
    if (code < 0x80) {
        out += static_cast<char>(code);
    } else if (code < 0x800) {
        out += static_cast<char>(0xC0 | (code >> 6));
        out += static_cast<char>(0x80 | (code & 0x3F));
    } else if (code < 0x10000) {
        out += static_cast<char>(0xE0 | (code >> 12));
        out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (code & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (code >> 18));
        out += static_cast<char>(0x80 | ((code >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (code & 0x3F));
    }
}

// The named entities common in prose; others stay as written.
constexpr std::array<std::pair<std::string_view, std::uint32_t>, 40> ENTITIES{{
    {"amp", '&'},       {"lt", '<'},          {"gt", '>'},          {"quot", '"'},       {"apos", '\''},
    {"nbsp", ' '},      {"ensp", ' '},        {"emsp", ' '},        {"thinsp", ' '},     {"shy", 0xAD},
    {"ndash", 0x2013},  {"mdash", 0x2014},    {"hellip", 0x2026},   {"lsquo", 0x2018},   {"rsquo", 0x2019},
    {"ldquo", 0x201C},  {"rdquo", 0x201D},    {"sbquo", 0x201A},    {"bdquo", 0x201E},   {"laquo", 0xAB},
    {"raquo", 0xBB},    {"middot", 0xB7},     {"bull", 0x2022},     {"copy", 0xA9},      {"reg", 0xAE},
    {"trade", 0x2122},  {"deg", 0xB0},        {"times", 0xD7},      {"divide", 0xF7},    {"plusmn", 0xB1},
    {"euro", 0x20AC},   {"pound", 0xA3},      {"yen", 0xA5},        {"cent", 0xA2},      {"sect", 0xA7},
    {"para", 0xB6},     {"rarr", 0x2192},     {"larr", 0x2190},     {"frac12", 0xBD},    {"micro", 0xB5},
}};

auto decode_entities(std::string_view text) -> std::string {
    std::string out;
    out.reserve(text.size());
    for (std::size_t index = 0; index < text.size();) {
        if (text[index] != '&') {
            out += text[index++];
            continue;
        }
        const auto end = text.find(';', index + 1);
        if (end == std::string_view::npos || end - index > 12) {
            out += text[index++];
            continue;
        }
        const auto name = text.substr(index + 1, end - index - 1);
        std::optional<std::uint32_t> code;
        if (name.size() > 1 && name[0] == '#') {
            std::uint32_t value = 0;
            const bool hex = name[1] == 'x' || name[1] == 'X';
            bool valid = name.size() > (hex ? 2U : 1U);
            for (const auto digit : name.substr(hex ? 2 : 1)) {
                std::uint32_t part;
                if (digit >= '0' && digit <= '9') part = static_cast<std::uint32_t>(digit - '0');
                else if (hex && digit >= 'a' && digit <= 'f') part = static_cast<std::uint32_t>(digit - 'a' + 10);
                else if (hex && digit >= 'A' && digit <= 'F') part = static_cast<std::uint32_t>(digit - 'A' + 10);
                else valid = false;
                if (!valid || value > 0x10FFFF) break;
                value = value * (hex ? 16 : 10) + part;
            }
            if (valid) code = value;
        } else {
            const auto entity = std::ranges::find(ENTITIES, name, &std::pair<std::string_view, std::uint32_t>::first);
            if (entity != ENTITIES.end()) code = entity->second;
        }
        if (!code) {
            out += text[index++];
            continue;
        }
        if (*code == 0xA0) *code = ' ';
        append_utf8(out, *code);
        index = end + 1;
    }
    return out;
}

struct Token {
    enum class Kind { TEXT, OPEN, CLOSE } kind;
    std::string name; // lowercase tag name
    std::string text; // decoded text, or the raw attribute source of an OPEN tag
    bool self_closing = false;
};

auto attribute(const Token& token, std::string_view wanted) -> std::optional<std::string> {
    const std::string_view source = token.text;
    for (std::size_t index = 0; index < source.size();) {
        while (index < source.size() && (is_space(source[index]) || source[index] == '/')) ++index;
        const auto name_start = index;
        while (index < source.size() && !is_space(source[index]) && source[index] != '=' && source[index] != '/')
            ++index;
        const auto name = lower(source.substr(name_start, index - name_start));
        while (index < source.size() && is_space(source[index])) ++index;
        std::string value;
        if (index < source.size() && source[index] == '=') {
            ++index;
            while (index < source.size() && is_space(source[index])) ++index;
            if (index < source.size() && (source[index] == '"' || source[index] == '\'')) {
                const auto quote = source[index++];
                const auto end = source.find(quote, index);
                value = source.substr(index, end == std::string_view::npos ? std::string_view::npos : end - index);
                index = end == std::string_view::npos ? source.size() : end + 1;
            } else {
                const auto start = index;
                while (index < source.size() && !is_space(source[index])) ++index;
                value = source.substr(start, index - start);
            }
        }
        if (name.empty()) {
            ++index;
            continue;
        }
        if (name == wanted) return decode_entities(value);
    }
    return std::nullopt;
}

// Elements whose content is raw text up to their end tag.
auto raw_text(std::string_view name) -> bool {
    return name == "script" || name == "style" || name == "textarea" || name == "title" || name == "noscript" ||
           name == "xmp" || name == "template";
}

auto tokenize(std::string_view html) -> std::vector<Token> {
    std::vector<Token> tokens;
    std::size_t index = 0;
    const auto text_until = [&](std::size_t end) {
        if (end > index) tokens.push_back({Token::Kind::TEXT, {}, decode_entities(html.substr(index, end - index))});
        index = end;
    };
    while (index < html.size()) {
        const auto open = html.find('<', index);
        if (open == std::string_view::npos) {
            text_until(html.size());
            break;
        }
        text_until(open);
        if (html.substr(index, 4) == "<!--") {
            const auto end = html.find("-->", index + 4);
            index = end == std::string_view::npos ? html.size() : end + 3;
            continue;
        }
        const auto next = index + 1 < html.size() ? html[index + 1] : '\0';
        if (next == '!' || next == '?') {
            const auto end = html.find('>', index);
            index = end == std::string_view::npos ? html.size() : end + 1;
            continue;
        }
        const bool closing = next == '/';
        const auto name_start = index + (closing ? 2 : 1);
        if (name_start >= html.size() || !is_alpha(html[name_start])) {
            tokens.push_back({Token::Kind::TEXT, {}, "<"});
            ++index;
            continue;
        }
        auto name_end = name_start;
        while (name_end < html.size() && !is_space(html[name_end]) && html[name_end] != '>' && html[name_end] != '/')
            ++name_end;
        auto name = lower(html.substr(name_start, name_end - name_start));
        // The tag ends at the first '>' outside a quoted attribute value.
        auto end = name_end;
        char quote = 0;
        for (; end < html.size(); ++end) {
            const auto character = html[end];
            if (quote) {
                if (character == quote) quote = 0;
            } else if (character == '"' || character == '\'') {
                quote = character;
            } else if (character == '>') {
                break;
            }
        }
        const auto source = html.substr(name_end, std::min(end, html.size()) - name_end);
        index = end < html.size() ? end + 1 : html.size();
        if (closing) {
            tokens.push_back({Token::Kind::CLOSE, std::move(name), {}});
            continue;
        }
        const bool self_closing = !source.empty() && source.back() == '/';
        tokens.push_back({Token::Kind::OPEN, name, std::string(source), self_closing});
        if (raw_text(name) && !self_closing) {
            // The content runs to the matching end tag; only <title> and <textarea> text is kept.
            std::size_t close = index;
            for (;;) {
                close = html.find("</", close);
                if (close == std::string_view::npos) break;
                if (lower(html.substr(close + 2, name.size())) == name) break;
                close += 2;
            }
            const auto content_end = close == std::string_view::npos ? html.size() : close;
            if (name == "title" || name == "textarea")
                tokens.push_back({Token::Kind::TEXT, {}, decode_entities(html.substr(index, content_end - index))});
            tokens.push_back({Token::Kind::CLOSE, name, {}});
            if (close == std::string_view::npos) {
                index = html.size();
            } else {
                const auto tag_end = html.find('>', close);
                index = tag_end == std::string_view::npos ? html.size() : tag_end + 1;
            }
        }
    }
    return tokens;
}

auto void_element(std::string_view name) -> bool {
    static constexpr std::array<std::string_view, 15> NAMES{"area", "base", "br",    "col",   "embed",
                                                            "hr",   "img",  "input", "link",  "meta",
                                                            "param", "source", "track", "wbr", "keygen"};
    return std::ranges::find(NAMES, name) != NAMES.end();
}

// Elements that never carry a page's readable content.
auto dropped_element(std::string_view name) -> bool {
    static constexpr std::array<std::string_view, 27> NAMES{
        "head",   "script", "style",  "noscript", "template", "svg",    "canvas", "iframe", "object",
        "embed",  "nav",    "footer", "aside",    "dialog",   "menu",   "select", "option", "button",
        "input",  "textarea", "label", "video",   "audio",    "picture", "map",   "img",    "math"};
    return std::ranges::find(NAMES, name) != NAMES.end();
}

// class/id words that mark page chrome rather than content.
auto boilerplate_word(std::string_view word) -> bool {
    static constexpr std::array<std::string_view, 36> WORDS{
        "nav",        "navbar",     "navigation", "menu",     "sidebar",   "footer",      "breadcrumb",
        "breadcrumbs", "cookie",    "cookies",    "consent",  "gdpr",      "newsletter",  "subscribe",
        "share",      "sharing",    "social",     "related",  "recommended", "comments",  "comment",
        "advert",     "advertisement", "ads",     "ad",       "sponsor",   "sponsored",   "promo",
        "popup",      "modal",      "skip",       "toc",      "editsection", "masthead",  "toolbar",
        "pagination"};
    return std::ranges::find(WORDS, word) != WORDS.end();
}

auto hidden_or_chrome(const Token& token, bool filter_boilerplate) -> bool {
    if (attribute(token, "hidden")) return true;
    if (auto aria = attribute(token, "aria-hidden"); aria && lower(*aria) == "true") return true;
    if (auto style = attribute(token, "style")) {
        auto compact = lower(*style);
        std::erase_if(compact, is_space);
        if (compact.find("display:none") != std::string::npos || compact.find("visibility:hidden") != std::string::npos)
            return true;
    }
    if (!filter_boilerplate) return false;
    // The content roots are kept even when a class mentions chrome ("has-sidebar").
    if (token.name == "html" || token.name == "body" || token.name == "main" || token.name == "article") return false;
    if (auto role = attribute(token, "role")) {
        const auto value = lower(*role);
        if (value == "navigation" || value == "banner" || value == "contentinfo" || value == "complementary" ||
            value == "search" || value == "dialog" || value == "alertdialog" || value == "menu" || value == "menubar")
            return true;
    }
    for (const auto* key : {"class", "id"}) {
        const auto value = attribute(token, key);
        if (!value) continue;
        const auto words = lower(*value);
        std::size_t start = 0;
        for (std::size_t index = 0; index <= words.size(); ++index) {
            const bool separator =
                index == words.size() || !((words[index] >= 'a' && words[index] <= 'z') ||
                                           (words[index] >= '0' && words[index] <= '9'));
            if (!separator) continue;
            if (index > start && boilerplate_word(std::string_view(words).substr(start, index - start))) return true;
            start = index + 1;
        }
    }
    return false;
}

// Index of the token after the element opened at `open` (its matching close, or the end).
auto element_end(const std::vector<Token>& tokens, std::size_t open, std::size_t limit) -> std::size_t {
    const auto& name = tokens[open].name;
    if (tokens[open].self_closing || void_element(name)) return open + 1;
    std::size_t depth = 0;
    for (auto index = open; index < limit; ++index) {
        if (tokens[index].name != name) continue;
        if (tokens[index].kind == Token::Kind::OPEN && !tokens[index].self_closing) ++depth;
        if (tokens[index].kind == Token::Kind::CLOSE && --depth == 0) return index + 1;
    }
    return limit;
}

auto text_length(const std::vector<Token>& tokens, std::size_t begin, std::size_t end) -> std::size_t {
    std::size_t length = 0;
    for (auto index = begin; index < end; ++index)
        if (tokens[index].kind == Token::Kind::TEXT) length += trim(tokens[index].text).size();
    return length;
}

// The token range holding the page's main content.
auto content_range(const std::vector<Token>& tokens) -> std::pair<std::size_t, std::size_t> {
    std::optional<std::pair<std::size_t, std::size_t>> main, article, body;
    std::size_t article_text = 0;
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (tokens[index].kind != Token::Kind::OPEN) continue;
        const auto& name = tokens[index].name;
        if (name == "main" && !main) {
            main = {index + 1, element_end(tokens, index, tokens.size()) - 1};
        } else if (name == "article") {
            const auto end = element_end(tokens, index, tokens.size());
            const auto length = text_length(tokens, index, end);
            if (length > article_text) article = {index + 1, end - 1}, article_text = length;
        } else if (name == "body" && !body) {
            body = {index + 1, tokens.size()};
        }
    }
    // A <main> or <article> holding little of the text is a fragment, not the page's content.
    const auto total = text_length(tokens, body ? body->first : 0, tokens.size());
    if (main && text_length(tokens, main->first, main->second) * 4 >= total) return *main;
    if (article && article_text * 4 >= total) return *article;
    return body.value_or(std::pair<std::size_t, std::size_t>{0, tokens.size()});
}

class Writer {
public:
    explicit Writer(std::string base_url) : base_url_(std::move(base_url)) {
        // Same-site links are written as /paths; the page's own URL gives their origin.
        const auto scheme = base_url_.find("://");
        if (scheme != std::string::npos) origin_ = base_url_.substr(0, base_url_.find('/', scheme + 3));
    }

    auto convert(const std::vector<Token>& tokens, std::size_t begin, std::size_t end, bool filter) -> std::string {
        for (auto index = begin; index < end;) {
            const auto& token = tokens[index];
            if (token.kind == Token::Kind::TEXT) {
                text(token.text);
                ++index;
                continue;
            }
            if (token.kind == Token::Kind::OPEN &&
                (dropped_element(token.name) || hidden_or_chrome(token, filter))) {
                index = element_end(tokens, index, end);
                continue;
            }
            if (token.kind == Token::Kind::OPEN) open(token);
            else close(token.name);
            if (token.kind == Token::Kind::OPEN && (token.self_closing || void_element(token.name)) &&
                !void_element(token.name))
                close(token.name);
            ++index;
        }
        return out_;
    }

private:
    struct List {
        bool ordered = false;
        int count = 0;
    };
    struct Link {
        std::size_t start;
        std::string href;
    };
    struct Table {
        int rows = 0;
        int cells = 0;
    };

    auto text(std::string_view value) -> void {
        if (pre_) {
            out_ += value;
            return;
        }
        for (const auto character : value) {
            if (is_space(character)) {
                pending_space_ = true;
                continue;
            }
            flush_space();
            out_ += character;
        }
    }

    auto flush_space() -> void {
        if (pending_space_ && !out_.empty() && out_.back() != '\n' && out_.back() != ' ' && out_.back() != '[' &&
            out_.back() != '(')
            out_ += ' ';
        pending_space_ = false;
    }

    auto raw(std::string_view value) -> void {
        flush_space();
        out_ += value;
    }

    // Table delimiters carry their own spacing.
    auto delimiter(std::string_view value) -> void {
        pending_space_ = false;
        while (!out_.empty() && out_.back() == ' ') out_.pop_back();
        out_ += value;
    }

    // Ends the current line; `lines` 2 leaves a blank line. Inside table cells, breaks become spaces.
    auto line(int lines) -> void {
        if (cell_ > 0 && !pre_) {
            pending_space_ = true;
            return;
        }
        while (!out_.empty() && out_.back() == ' ') out_.pop_back();
        pending_space_ = false;
        if (out_.empty()) return;
        int trailing = 0;
        for (auto it = out_.rbegin(); it != out_.rend() && *it == '\n' && trailing < 2; ++it) ++trailing;
        for (; trailing < lines; ++trailing) out_ += '\n';
    }

    auto open(const Token& token) -> void {
        const auto& name = token.name;
        if (name.size() == 2 && name[0] == 'h' && name[1] >= '1' && name[1] <= '6') {
            line(2);
            raw(std::string(static_cast<std::size_t>(name[1] - '0'), '#') + " ");
        } else if (name == "p" || name == "blockquote" || name == "figure" || name == "details" || name == "dl" ||
                   name == "address" || name == "fieldset") {
            line(2);
            if (name == "blockquote") raw("> ");
        } else if (name == "br") {
            if (pre_) out_ += '\n';
            else line(1);
        } else if (name == "hr") {
            line(2);
        } else if (name == "ul" || name == "ol") {
            line(lists_.empty() ? 2 : 1);
            lists_.push_back({name == "ol"});
        } else if (name == "li") {
            line(1);
            List orphan;
            auto& list = lists_.empty() ? orphan : lists_.back();
            raw(std::string(2 * (lists_.empty() ? 0 : lists_.size() - 1), ' ') +
                (list.ordered ? std::to_string(++list.count) + ". " : std::string("- ")));
        } else if (name == "pre") {
            line(2);
            std::string language;
            if (auto classes = attribute(token, "class")) language = language_of(*classes);
            raw("```" + language + "\n");
            ++pre_;
            pre_start_ = out_.size();
        } else if (name == "code") {
            if (!pre_) {
                raw("`");
                code_start_ = out_.size();
            } else if (auto classes = attribute(token, "class"); classes && out_.size() >= 4 &&
                                                                 out_.ends_with("```\n") && pre_start_ == out_.size()) {
                // <pre><code class="language-x">: the language belongs on the fence.
                out_.insert(out_.size() - 1, language_of(*classes));
                pre_start_ = out_.size();
            }
        } else if (name == "table") {
            if (tables_.empty()) line(2);
            tables_.push_back({});
        } else if (name == "tr") {
            if (tables_.size() == 1) {
                line(1);
                delimiter("| ");
                tables_.back().cells = 0;
            } else {
                pending_space_ = true;
            }
        } else if (name == "td" || name == "th") {
            if (tables_.size() == 1) {
                if (tables_.back().cells++ > 0) delimiter(" | ");
                ++cell_;
            } else {
                pending_space_ = true;
            }
        } else if (name == "a") {
            auto href = resolve_url(base_url_, attribute(token, "href").value_or(""));
            if (!origin_.empty() && href.size() > origin_.size() && href.starts_with(origin_) &&
                href[origin_.size()] == '/')
                href.erase(0, origin_.size());
            flush_space();
            links_.push_back({out_.size(), href.size() <= 200 ? std::move(href) : std::string{}});
        } else if (name == "div" || name == "section" || name == "article" || name == "main" || name == "header" ||
                   name == "figcaption" || name == "dt" || name == "dd" || name == "summary" || name == "center") {
            line(1);
        }
    }

    auto close(std::string_view name) -> void {
        if (name.size() == 2 && name[0] == 'h' && name[1] >= '1' && name[1] <= '6') {
            line(2);
        } else if (name == "p" || name == "blockquote" || name == "figure" || name == "details" || name == "dl" ||
                   name == "address" || name == "fieldset" || name == "hr") {
            line(2);
        } else if (name == "ul" || name == "ol") {
            if (!lists_.empty()) lists_.pop_back();
            line(lists_.empty() ? 2 : 1);
        } else if (name == "pre") {
            if (!pre_) return;
            --pre_;
            // Drop the conventional newline after <pre>, then close the fence on its own line.
            if (out_.size() > pre_start_ && out_[pre_start_] == '\n') out_.erase(pre_start_, 1);
            while (!out_.empty() && (out_.back() == '\n' || out_.back() == ' ')) out_.pop_back();
            out_ += "\n```";
            line(2);
        } else if (name == "code") {
            if (pre_) return;
            if (code_start_ && out_.size() == *code_start_) out_.pop_back();
            else raw("`");
            code_start_.reset();
        } else if (name == "td" || name == "th") {
            if (tables_.size() == 1 && cell_ > 0) --cell_;
        } else if (name == "tr") {
            if (tables_.size() != 1) return;
            auto& table = tables_.back();
            delimiter(" |");
            if (table.rows++ == 0) {
                line(1);
                std::string separator = "|";
                for (int cell = 0; cell < std::max(table.cells, 1); ++cell) separator += " --- |";
                delimiter(separator);
            }
        } else if (name == "table") {
            if (!tables_.empty()) tables_.pop_back();
            cell_ = 0;
            if (tables_.empty()) line(2);
        } else if (name == "a") {
            if (links_.empty()) return;
            auto link = std::move(links_.back());
            links_.pop_back();
            if (link.href.empty() || link.start > out_.size()) return;
            const auto label = std::string(trim(std::string_view(out_).substr(link.start)));
            if (label.empty() || label.find('\n') != std::string::npos || label.find('[') != std::string::npos) return;
            out_.resize(link.start);
            out_ += label == link.href ? label : "[" + label + "](" + link.href + ")";
        } else if (name == "div" || name == "section" || name == "article" || name == "main" || name == "header" ||
                   name == "figcaption" || name == "dt" || name == "dd" || name == "summary" || name == "center" ||
                   name == "li") {
            line(1);
        }
    }

    static auto language_of(std::string_view classes) -> std::string {
        for (const auto* prefix : {"language-", "lang-"}) {
            const auto at = classes.find(prefix);
            if (at == std::string_view::npos) continue;
            auto word = classes.substr(at + std::string_view(prefix).size());
            word = word.substr(0, word.find_first_of(" \t\n"));
            if (!word.empty() && word.size() <= 20) return std::string(word);
        }
        return {};
    }

    std::string base_url_;
    std::string origin_;
    std::string out_;
    bool pending_space_ = false;
    int pre_ = 0;
    std::size_t pre_start_ = 0;
    std::optional<std::size_t> code_start_;
    int cell_ = 0;
    std::vector<List> lists_;
    std::vector<Link> links_;
    std::vector<Table> tables_;
};

// Markers left without content ("- ", "#", "|  |"), repeated lines and runs of blank lines are removed.
auto tidy(std::string_view text) -> std::string {
    std::string out;
    std::string previous;
    bool blank = false, fence = false;
    std::size_t start = 0;
    while (start <= text.size()) {
        auto end = text.find('\n', start);
        if (end == std::string_view::npos) end = text.size();
        auto line = std::string(text.substr(start, end - start));
        start = end + 1;
        if (line.starts_with("```")) fence = !fence;
        if (!fence) {
            while (!line.empty() && is_space(line.back())) line.pop_back();
            const auto content = trim(line);
            // A line whose only content is Markdown structure ("- ", "##", "|  |", "3.") carries nothing.
            std::string rest;
            for (const auto character : content)
                if (character != '|') rest += character;
            auto view = trim(rest);
            while (!view.empty() && (view.front() == '-' || view.front() == '*' || view.front() == '#' ||
                                     view.front() == '>' || is_space(view.front())))
                view.remove_prefix(1);
            const auto digits = view.find_first_not_of("0123456789");
            if (digits != std::string_view::npos && digits > 0 && view[digits] == '.') view.remove_prefix(digits + 1);
            const bool separator = content.starts_with("| ---") || content.starts_with("|---");
            if (trim(view).empty() && !separator) line.clear();
            if (line.empty()) {
                blank = !out.empty();
                continue;
            }
            if (line == previous) continue;
        }
        if (blank) out += '\n';
        blank = false;
        out += line;
        out += '\n';
        previous = line;
    }
    while (!out.empty() && out.back() == '\n') out.pop_back();
    return out;
}

} // namespace

auto resolve_url(std::string_view base_url, std::string_view href) -> std::string {
    auto link = std::string(trim(href));
    std::erase_if(link, [](char character) { return character == '\n' || character == '\r' || character == '\t'; });
    if (link.empty() || link.front() == '#') return {};
    if (const auto hash = link.find('#'); hash != std::string::npos) link.resize(hash);
    const auto scheme_end = link.find(':');
    const auto first_delimiter = link.find_first_of("/?");
    std::string absolute;
    if (scheme_end != std::string::npos && (first_delimiter == std::string::npos || scheme_end < first_delimiter) &&
        scheme_end > 0 && is_alpha(link[0])) {
        const auto scheme = lower(link.substr(0, scheme_end));
        if (scheme != "http" && scheme != "https") return {};
        absolute = scheme + link.substr(scheme_end);
    } else {
        const auto base_scheme_end = base_url.find("://");
        if (base_scheme_end == std::string_view::npos) return {};
        const auto authority_end = base_url.find_first_of("/?#", base_scheme_end + 3);
        const auto origin = base_url.substr(0, authority_end);
        if (link.starts_with("//")) {
            absolute = std::string(base_url.substr(0, base_scheme_end + 1)) + link;
        } else if (link.front() == '/') {
            absolute = std::string(origin) + link;
        } else {
            auto path = authority_end == std::string_view::npos ? std::string_view("/")
                                                                  : base_url.substr(authority_end);
            path = path.substr(0, path.find_first_of("?#"));
            if (path.empty() || path.front() != '/') path = "/";
            if (link.front() == '?') {
                absolute = std::string(origin) + std::string(path) + link;
            } else {
                absolute = std::string(origin) + std::string(path.substr(0, path.rfind('/') + 1)) + link;
            }
        }
    }
    // Normalize "." and ".." path segments.
    const auto authority = absolute.find("://") + 3;
    const auto path_start = absolute.find('/', authority);
    if (path_start != std::string::npos) {
        const auto query_start = absolute.find('?', path_start);
        const auto path = absolute.substr(path_start, query_start == std::string::npos ? std::string::npos
                                                                                        : query_start - path_start);
        std::vector<std::string> segments;
        std::size_t begin = 1;
        while (begin <= path.size()) {
            auto end = path.find('/', begin);
            if (end == std::string::npos) end = path.size();
            const auto segment = path.substr(begin, end - begin);
            if (segment == "..") {
                if (!segments.empty()) segments.pop_back();
                if (end == path.size()) segments.emplace_back();
            } else if (segment != ".") {
                segments.push_back(segment);
            } else if (end == path.size()) {
                segments.emplace_back();
            }
            begin = end + 1;
        }
        std::string normalized;
        for (const auto& segment : segments) normalized += "/" + segment;
        if (normalized.empty()) normalized = "/";
        absolute = absolute.substr(0, path_start) + normalized +
                   (query_start == std::string::npos ? std::string{} : absolute.substr(query_start));
    }
    // Tracking parameters only cost tokens.
    if (const auto query = absolute.find('?'); query != std::string::npos) {
        std::string kept;
        std::size_t begin = query + 1;
        while (begin <= absolute.size()) {
            auto end = absolute.find('&', begin);
            if (end == std::string::npos) end = absolute.size();
            const auto parameter = std::string_view(absolute).substr(begin, end - begin);
            if (!parameter.empty() && !lower(parameter).starts_with("utm_"))
                kept += (kept.empty() ? "?" : "&") + std::string(parameter);
            begin = end + 1;
        }
        absolute = absolute.substr(0, query) + kept;
    }
    return absolute;
}

auto to_utf8(std::string_view bytes, std::string_view charset) -> std::string {
    const auto name = lower(trim(charset));
    std::string out;
    out.reserve(bytes.size());
    if (name == "iso-8859-1" || name == "latin1" || name == "latin-1" || name == "windows-1252" || name == "cp1252" ||
        name == "us-ascii" || name == "ascii") {
        static constexpr std::array<std::uint16_t, 32> CP1252{
            0x20AC, 0xFFFD, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021, 0x02C6, 0x2030, 0x0160,
            0x2039, 0x0152, 0xFFFD, 0x017D, 0xFFFD, 0xFFFD, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022,
            0x2013, 0x2014, 0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0xFFFD, 0x017E, 0x0178};
        for (const auto byte : bytes) {
            const auto value = static_cast<unsigned char>(byte);
            append_utf8(out, value >= 0x80 && value < 0xA0 ? CP1252[value - 0x80] : value);
        }
        return out;
    }
    for (std::size_t index = 0; index < bytes.size();) {
        const auto lead = static_cast<unsigned char>(bytes[index]);
        const std::size_t length = lead < 0x80             ? 1
                                   : (lead >> 5) == 0x6    ? 2
                                   : (lead >> 4) == 0xE    ? 3
                                   : (lead >> 3) == 0x1E   ? 4
                                                           : 0;
        bool valid = length > 0 && index + length <= bytes.size();
        for (std::size_t offset = 1; valid && offset < length; ++offset)
            valid = (static_cast<unsigned char>(bytes[index + offset]) & 0xC0) == 0x80;
        if (valid) {
            out.append(bytes.substr(index, length));
            index += length;
        } else {
            append_utf8(out, 0xFFFD);
            ++index;
        }
    }
    return out;
}

auto html_to_markdown(std::string_view html, std::string_view base_url) -> HtmlPage {
    const auto tokens = tokenize(html);
    HtmlPage page;
    for (std::size_t index = 0; index + 1 < tokens.size(); ++index) {
        if (tokens[index].kind == Token::Kind::OPEN && tokens[index].name == "title" &&
            tokens[index + 1].kind == Token::Kind::TEXT) {
            page.title = std::string(trim(tokens[index + 1].text));
            std::ranges::replace(page.title, '\n', ' ');
            break;
        }
    }
    const auto [begin, end] = content_range(tokens);
    page.markdown = tidy(Writer(std::string(base_url)).convert(tokens, begin, end, true));
    // Boilerplate words also appear on content containers ("has-sidebar"); when filtering removed most of the text,
    // the unfiltered page is the better answer.
    const auto unfiltered = tidy(Writer(std::string(base_url)).convert(tokens, begin, end, false));
    if (page.markdown.size() * 5 < unfiltered.size() && page.markdown.size() < 2000) page.markdown = unfiltered;
    return page;
}

} // namespace kidi::agent::tools
