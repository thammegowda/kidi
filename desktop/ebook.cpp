#include "ebook.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <map>
#include <optional>
#include <sstream>
#include <string_view>

namespace kidi::desktop {
namespace {

constexpr std::size_t MAX_EPUB_BYTES = 64U * 1024U * 1024U;
constexpr std::size_t MAX_EPUB_FILES = 4096;
constexpr std::size_t MAX_EPUB_FILE_BYTES = 4U * 1024U * 1024U;
constexpr std::size_t MAX_EPUB_EXPANDED_BYTES = 128U * 1024U * 1024U;
constexpr std::size_t MAX_CHAPTERS = 2048;

auto failure(std::string message) -> liteapp::Result<Ebook> {
    return std::unexpected(liteapp::Error{"invalid_ebook", std::move(message)});
}

auto text_failure(std::string message) -> liteapp::Result<std::string> {
    return std::unexpected(liteapp::Error{"invalid_ebook", std::move(message)});
}

auto local_name(std::string_view name) -> std::string_view {
    const auto colon = name.rfind(':');
    return colon == std::string_view::npos ? name : name.substr(colon + 1);
}

auto xml_name_character(char value) -> bool {
    const auto character = static_cast<unsigned char>(value);
    return std::isalnum(character) || value == '_' || value == '-' || value == '.' || value == ':';
}

struct XmlTag {
    std::string_view name;
    std::string_view attributes;
    std::size_t content = 0;
};

auto find_tag(std::string_view xml, std::string_view requested, std::size_t start = 0) -> std::optional<XmlTag> {
    for (auto cursor = start; cursor < xml.size();) {
        const auto opening = xml.find('<', cursor);
        if (opening == std::string_view::npos || opening + 1 >= xml.size()) return std::nullopt;
        if (xml.substr(opening, 4) == "<!--") {
            const auto end = xml.find("-->", opening + 4);
            if (end == std::string_view::npos) return std::nullopt;
            cursor = end + 3;
            continue;
        }
        if (xml[opening + 1] == '/' || xml[opening + 1] == '!' || xml[opening + 1] == '?') {
            const auto end = xml.find('>', opening + 2);
            if (end == std::string_view::npos) return std::nullopt;
            cursor = end + 1;
            continue;
        }
        auto name_end = opening + 1;
        while (name_end < xml.size() && xml_name_character(xml[name_end])) ++name_end;
        if (name_end == opening + 1) {
            cursor = opening + 1;
            continue;
        }
        char quote = 0;
        auto end = name_end;
        for (; end < xml.size(); ++end) {
            const auto value = xml[end];
            if (quote) {
                if (value == quote) quote = 0;
            } else if (value == '"' || value == '\'') {
                quote = value;
            } else if (value == '>') {
                break;
            }
        }
        if (end == xml.size()) return std::nullopt;
        const auto name = xml.substr(opening + 1, name_end - opening - 1);
        if (local_name(name) == requested)
            return XmlTag{.name = name, .attributes = xml.substr(name_end, end - name_end), .content = end + 1};
        cursor = end + 1;
    }
    return std::nullopt;
}

auto append_utf8(std::string& output, std::uint32_t codepoint) -> bool {
    if (codepoint > 0x10ffffU || (codepoint >= 0xd800U && codepoint <= 0xdfffU)) return false;
    if (codepoint <= 0x7fU) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ffU) {
        output.push_back(static_cast<char>(0xc0U | (codepoint >> 6)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else if (codepoint <= 0xffffU) {
        output.push_back(static_cast<char>(0xe0U | (codepoint >> 12)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6) & 0x3fU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else {
        output.push_back(static_cast<char>(0xf0U | (codepoint >> 18)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 12) & 0x3fU)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6) & 0x3fU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    }
    return true;
}

auto decode_xml(std::string_view source) -> std::string {
    std::string output;
    output.reserve(source.size());
    for (std::size_t cursor = 0; cursor < source.size();) {
        if (source[cursor] != '&') {
            output.push_back(source[cursor++]);
            continue;
        }
        const auto end = source.find(';', cursor + 1);
        if (end == std::string_view::npos || end - cursor > 16) {
            output.push_back(source[cursor++]);
            continue;
        }
        const auto entity = source.substr(cursor + 1, end - cursor - 1);
        std::optional<std::uint32_t> codepoint;
        if (entity == "amp") codepoint = '&';
        else if (entity == "lt") codepoint = '<';
        else if (entity == "gt") codepoint = '>';
        else if (entity == "quot") codepoint = '"';
        else if (entity == "apos") codepoint = '\'';
        else if (entity.starts_with("#x") || entity.starts_with("#X")) {
            std::uint32_t value = 0;
            const auto digits = entity.substr(2);
            const auto parsed = std::from_chars(digits.data(), digits.data() + digits.size(), value, 16);
            if (parsed.ec == std::errc{} && parsed.ptr == digits.data() + digits.size()) codepoint = value;
        } else if (entity.starts_with('#')) {
            std::uint32_t value = 0;
            const auto digits = entity.substr(1);
            const auto parsed = std::from_chars(digits.data(), digits.data() + digits.size(), value);
            if (parsed.ec == std::errc{} && parsed.ptr == digits.data() + digits.size()) codepoint = value;
        }
        if (!codepoint || !append_utf8(output, *codepoint)) {
            output.append(source.substr(cursor, end - cursor + 1));
        }
        cursor = end + 1;
    }
    return output;
}

auto attribute(const XmlTag& tag, std::string_view requested) -> std::optional<std::string> {
    auto source = tag.attributes;
    for (std::size_t cursor = 0; cursor < source.size();) {
        while (cursor < source.size() && std::isspace(static_cast<unsigned char>(source[cursor]))) ++cursor;
        if (cursor >= source.size() || source[cursor] == '/') break;
        const auto name_start = cursor;
        while (cursor < source.size() && xml_name_character(source[cursor])) ++cursor;
        if (cursor == name_start) {
            ++cursor;
            continue;
        }
        const auto name = source.substr(name_start, cursor - name_start);
        while (cursor < source.size() && std::isspace(static_cast<unsigned char>(source[cursor]))) ++cursor;
        if (cursor >= source.size() || source[cursor++] != '=') continue;
        while (cursor < source.size() && std::isspace(static_cast<unsigned char>(source[cursor]))) ++cursor;
        if (cursor >= source.size() || (source[cursor] != '"' && source[cursor] != '\'')) continue;
        const auto quote = source[cursor++];
        const auto value_start = cursor;
        const auto value_end = source.find(quote, cursor);
        if (value_end == std::string_view::npos) return std::nullopt;
        cursor = value_end + 1;
        if (local_name(name) == requested) return decode_xml(source.substr(value_start, value_end - value_start));
    }
    return std::nullopt;
}

auto normalize_space(std::string_view source) -> std::string {
    std::string result;
    result.reserve(source.size());
    bool spacing = true;
    for (const auto value : source) {
        if (std::isspace(static_cast<unsigned char>(value))) {
            spacing = !result.empty();
        } else {
            if (spacing && !result.empty()) result.push_back(' ');
            result.push_back(value);
            spacing = false;
        }
    }
    return result;
}

auto element_text(std::string_view xml, std::string_view requested) -> std::optional<std::string> {
    const auto tag = find_tag(xml, requested);
    if (!tag) return std::nullopt;
    const auto close = std::string{"</"} + std::string{tag->name};
    const auto end = xml.find(close, tag->content);
    if (end == std::string_view::npos) return std::nullopt;
    const auto source = xml.substr(tag->content, end - tag->content);
    std::string text;
    text.reserve(source.size());
    bool inside = false;
    for (const auto value : source) {
        if (value == '<') inside = true;
        else if (value == '>') {
            inside = false;
            text.push_back(' ');
        } else if (!inside) {
            text.push_back(value);
        }
    }
    auto decoded = normalize_space(decode_xml(text));
    if (decoded.empty()) return std::nullopt;
    return decoded;
}

auto file_text(const liteapp::vfs::FileSystem& files, std::string_view path) -> liteapp::Result<std::string> {
    auto file = files.read(path);
    if (!file) return std::unexpected(file.error());
    if (!*file) return text_failure("EPUB is missing " + std::string{path});
    const auto bytes = (**file).bytes();
    return std::string{reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

auto hex_value(char value) -> int {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

auto resolve_path(std::string_view base, std::string_view reference) -> std::optional<std::string> {
    const auto suffix = reference.find_first_of("?#");
    reference = reference.substr(0, suffix);
    std::string decoded;
    decoded.reserve(reference.size());
    for (std::size_t index = 0; index < reference.size(); ++index) {
        if (reference[index] == '%' && index + 2 < reference.size()) {
            const auto high = hex_value(reference[index + 1]);
            const auto low = hex_value(reference[index + 2]);
            if (high < 0 || low < 0) return std::nullopt;
            decoded.push_back(static_cast<char>((high << 4) | low));
            index += 2;
        } else {
            decoded.push_back(reference[index]);
        }
    }
    if (decoded.empty() || decoded.find('\\') != std::string::npos || decoded.find('\0') != std::string::npos)
        return std::nullopt;
    const auto path = (std::filesystem::path{base}.parent_path() / decoded).lexically_normal();
    const auto normalized = path.generic_string();
    if (path.is_absolute() || normalized == ".." || normalized.starts_with("../")) return std::nullopt;
    return normalized;
}

auto fingerprint(std::span<const std::byte> archive) -> std::string {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto byte : archive) {
        hash ^= static_cast<std::uint8_t>(byte);
        hash *= 1099511628211ULL;
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << hash;
    return output.str();
}

auto chapter_title(const liteapp::vfs::FileSystem& files, std::string_view path, std::size_t index) -> std::string {
    auto text = file_text(files, path);
    if (text) {
        for (const auto heading : {"h1", "h2", "h3", "title"})
            if (auto title = element_text(*text, heading); title) return *title;
    }
    return "Chapter " + std::to_string(index + 1);
}

} // namespace

auto parse_epub(std::span<const std::byte> archive) -> liteapp::Result<Ebook> {
    auto files = liteapp::vfs::FileSystem::from_zip(
        archive,
        {
            .max_archive_bytes = MAX_EPUB_BYTES,
            .max_files = MAX_EPUB_FILES,
            .max_file_bytes = MAX_EPUB_FILE_BYTES,
            .max_total_bytes = MAX_EPUB_EXPANDED_BYTES,
            .max_path_bytes = 2048,
        });
    if (!files) return failure(files.error().message);
    auto container = file_text(*files, "META-INF/container.xml");
    if (!container) return failure(container.error().message);
    const auto rootfile = find_tag(*container, "rootfile");
    const auto package_path = rootfile ? attribute(*rootfile, "full-path") : std::nullopt;
    if (!package_path) return failure("EPUB container does not name a package document");
    const auto normalized_package = resolve_path("container.xml", *package_path);
    if (!normalized_package) return failure("EPUB package path is invalid");
    auto package = file_text(*files, *normalized_package);
    if (!package) return failure(package.error().message);

    struct ManifestItem {
        std::string path;
        std::string media_type;
    };
    std::map<std::string, ManifestItem, std::less<>> manifest;
    std::size_t cursor = 0;
    while (const auto item = find_tag(*package, "item", cursor)) {
        cursor = item->content;
        const auto id = attribute(*item, "id");
        const auto href = attribute(*item, "href");
        const auto media_type = attribute(*item, "media-type");
        const auto path = href ? resolve_path(*normalized_package, *href) : std::nullopt;
        if (id && !id->empty() && path && media_type)
            manifest.emplace(*id, ManifestItem{.path = *path, .media_type = *media_type});
    }
    if (manifest.empty()) return failure("EPUB package manifest is empty");

    std::vector<EbookChapter> chapters;
    cursor = 0;
    while (const auto reference = find_tag(*package, "itemref", cursor)) {
        cursor = reference->content;
        const auto id = attribute(*reference, "idref");
        if (!id) continue;
        const auto found = manifest.find(*id);
        if (found == manifest.end()) return failure("EPUB spine references a missing manifest item");
        if (found->second.media_type != "application/xhtml+xml" && found->second.media_type != "text/html") continue;
        if (chapters.size() >= MAX_CHAPTERS) return failure("EPUB contains too many chapters");
        auto chapter = file_text(*files, found->second.path);
        if (!chapter) return failure(chapter.error().message);
        chapters.push_back({
            .id = *id,
            .title = chapter_title(*files, found->second.path, chapters.size()),
            .path = found->second.path,
        });
    }
    if (chapters.empty()) return failure("EPUB spine does not contain readable chapters");

    return Ebook{
        .id = fingerprint(archive),
        .title = element_text(*package, "title").value_or("Untitled book"),
        .author = element_text(*package, "creator").value_or("Unknown author"),
        .language = element_text(*package, "language").value_or("und"),
        .chapters = std::move(chapters),
        .files = std::move(*files),
    };
}

auto read_epub_chapter(const Ebook& book, std::size_t chapter) -> liteapp::Result<std::string> {
    if (chapter >= book.chapters.size()) return text_failure("EPUB chapter index is out of range");
    return file_text(book.files, book.chapters[chapter].path);
}

} // namespace kidi::desktop
