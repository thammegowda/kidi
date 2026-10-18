#pragma once

#include "liteapp/core/result.h"
#include "liteapp/vfs/filesystem.h"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace kidi::desktop {

struct EbookChapter {
    std::string id;
    std::string title;
    std::string path;
};

struct Ebook {
    std::string id;
    std::string title;
    std::string author;
    std::string language;
    std::vector<EbookChapter> chapters;
    liteapp::vfs::FileSystem files;
};

[[nodiscard]] auto parse_epub(std::span<const std::byte> archive) -> liteapp::Result<Ebook>;
[[nodiscard]] auto read_epub_chapter(const Ebook& book, std::size_t chapter) -> liteapp::Result<std::string>;

} // namespace kidi::desktop
