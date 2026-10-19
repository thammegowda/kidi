#include "liteapp/vfs/filesystem.h"

#include <kidi/desktop_assets.h>

#include <iostream>
#include <string_view>

namespace {
auto expect_asset(const liteapp::vfs::FileSystem& files, std::string_view path, std::string_view marker) -> bool {
    const auto loaded = files.read(path);
    if (!loaded || !*loaded) {
        std::cerr << "FAILED: missing embedded asset " << path << '\n';
        return false;
    }
    const auto bytes = (**loaded).bytes();
    const std::string_view text{reinterpret_cast<const char*>(bytes.data()), bytes.size()};
    if (!text.contains(marker)) {
        std::cerr << "FAILED: invalid embedded asset " << path << '\n';
        return false;
    }
    return true;
}
} // namespace

auto main() -> int {
    const auto files = liteapp::vfs::FileSystem::from_zip(kidi::desktop_assets::archive());
    if (!files) {
        std::cerr << "FAILED: " << files.error().message << '\n';
        return 1;
    }
    return expect_asset(*files, "index.html", "<title>Kidi</title>") &&
                   expect_asset(*files, "app.js", "new SpeechController") &&
                   expect_asset(*files, "libs/mermaid/mermaid.min.js", "mermaid")
               ? 0
               : 1;
}
