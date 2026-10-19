#include "ebook.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

void append_u16(std::vector<std::byte>& output, std::uint16_t value) {
    output.push_back(static_cast<std::byte>(value & 0xffU));
    output.push_back(static_cast<std::byte>((value >> 8) & 0xffU));
}

void append_u32(std::vector<std::byte>& output, std::uint32_t value) {
    for (std::size_t shift = 0; shift < 32; shift += 8)
        output.push_back(static_cast<std::byte>((value >> shift) & 0xffU));
}

auto crc32(std::string_view value) -> std::uint32_t {
    auto crc = 0xffffffffU;
    for (const auto character : value) {
        crc ^= static_cast<unsigned char>(character);
        for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ (0xedb88320U & (0U - (crc & 1U)));
    }
    return ~crc;
}

auto make_zip(const std::vector<std::pair<std::string, std::string>>& files) -> std::vector<std::byte> {
    struct Central {
        std::string name;
        std::string contents;
        std::uint32_t crc = 0;
        std::uint32_t offset = 0;
    };
    std::vector<std::byte> output;
    std::vector<Central> central;
    for (const auto& [name, contents] : files) {
        const auto offset = static_cast<std::uint32_t>(output.size());
        const auto crc = crc32(contents);
        append_u32(output, 0x04034b50U);
        append_u16(output, 20);
        append_u16(output, 0);
        append_u16(output, 0);
        append_u16(output, 0);
        append_u16(output, 0);
        append_u32(output, crc);
        append_u32(output, static_cast<std::uint32_t>(contents.size()));
        append_u32(output, static_cast<std::uint32_t>(contents.size()));
        append_u16(output, static_cast<std::uint16_t>(name.size()));
        append_u16(output, 0);
        for (const auto value : name) output.push_back(static_cast<std::byte>(value));
        for (const auto value : contents) output.push_back(static_cast<std::byte>(value));
        central.push_back({name, contents, crc, offset});
    }
    const auto central_offset = static_cast<std::uint32_t>(output.size());
    for (const auto& entry : central) {
        append_u32(output, 0x02014b50U);
        append_u16(output, 20);
        append_u16(output, 20);
        append_u16(output, 0);
        append_u16(output, 0);
        append_u16(output, 0);
        append_u16(output, 0);
        append_u32(output, entry.crc);
        append_u32(output, static_cast<std::uint32_t>(entry.contents.size()));
        append_u32(output, static_cast<std::uint32_t>(entry.contents.size()));
        append_u16(output, static_cast<std::uint16_t>(entry.name.size()));
        append_u16(output, 0);
        append_u16(output, 0);
        append_u16(output, 0);
        append_u16(output, 0);
        append_u32(output, 0);
        append_u32(output, entry.offset);
        for (const auto value : entry.name) output.push_back(static_cast<std::byte>(value));
    }
    const auto central_size = static_cast<std::uint32_t>(output.size()) - central_offset;
    append_u32(output, 0x06054b50U);
    append_u16(output, 0);
    append_u16(output, 0);
    append_u16(output, static_cast<std::uint16_t>(central.size()));
    append_u16(output, static_cast<std::uint16_t>(central.size()));
    append_u32(output, central_size);
    append_u32(output, central_offset);
    append_u16(output, 0);
    return output;
}

auto expect(bool condition, const char* message) -> bool {
    if (!condition) std::cerr << "FAILED: " << message << '\n';
    return condition;
}

} // namespace

auto main() -> int {
    const auto archive = make_zip({
        {"META-INF/container.xml",
         R"(<?xml version="1.0"?><container><rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>)"},
        {"OEBPS/content.opf",
         R"(<?xml version="1.0"?><package xmlns:dc="http://purl.org/dc/elements/1.1/">
            <metadata><dc:title>A &amp; B</dc:title><dc:creator>Test Author</dc:creator>
            <dc:language>en</dc:language></metadata>
            <manifest>
              <item id="first" href="chapters/one.xhtml" media-type="application/xhtml+xml"/>
              <item id="second" href="chapters/two.xhtml" media-type="application/xhtml+xml"/>
            </manifest>
            <spine><itemref idref="first"/><itemref idref="second"/></spine>
          </package>)"},
        {"OEBPS/chapters/one.xhtml", "<html><head><title>Fallback</title></head><body><h1>Opening</h1><p>Hello.</p></body></html>"},
        {"OEBPS/chapters/two.xhtml", "<html><head><title>Second</title></head><body><p>World.</p></body></html>"},
    });
    auto book = kidi::desktop::parse_epub(archive);
    if (!expect(static_cast<bool>(book), "valid EPUB parses")) return 1;
    if (!expect(book->title == "A & B", "metadata title decodes") ||
        !expect(book->author == "Test Author", "metadata author parses") ||
        !expect(book->language == "en", "metadata language parses") ||
        !expect(book->chapters.size() == 2, "spine chapters parse") ||
        !expect(book->chapters[0].title == "Opening", "chapter heading becomes title") ||
        !expect(book->chapters[1].title == "Second", "document title is fallback"))
        return 1;
    auto chapter = kidi::desktop::read_epub_chapter(*book, 1);
    if (!expect(chapter && chapter->find("World.") != std::string::npos, "chapter content reads")) return 1;
    if (!expect(!kidi::desktop::read_epub_chapter(*book, 2), "out-of-range chapter is rejected")) return 1;

    const auto unsafe = make_zip({
        {"META-INF/container.xml", R"(<container><rootfile full-path="../content.opf"/></container>)"},
    });
    if (!expect(!kidi::desktop::parse_epub(unsafe), "unsafe package path is rejected")) return 1;
    return 0;
}
