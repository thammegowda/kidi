#include "liteapp/vfs/filesystem.h"

#include <kidi/desktop_assets.h>

#include <string_view>

#include <gtest/gtest.h>

namespace {
auto expect_asset(const liteapp::vfs::FileSystem& files, std::string_view path, std::string_view marker) -> void {
    SCOPED_TRACE(path);
    const auto loaded = files.read(path);
    ASSERT_TRUE(loaded && *loaded) << "missing embedded asset " << path;
    const auto bytes = (**loaded).bytes();
    const std::string_view text{reinterpret_cast<const char*>(bytes.data()), bytes.size()};
    EXPECT_TRUE(text.contains(marker)) << "invalid embedded asset " << path;
}

TEST(AssetsTest, EmbedsDesktopAssets) {
    const auto files = liteapp::vfs::FileSystem::from_zip(kidi::desktop_assets::archive());
    ASSERT_TRUE(files) << files.error().message;
    expect_asset(*files, "index.html", "<title>Kidi</title>");
    expect_asset(*files, "app.js", "new SpeechController");
    expect_asset(*files, "libs/mermaid/mermaid.min.js", "mermaid");
}
} // namespace
