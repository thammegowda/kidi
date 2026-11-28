#include "kidi/core/files.h"

#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <span>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

namespace kidi::core {
namespace {

class FilesTest : public testing::Test {
protected:
    std::filesystem::path root_ =
        std::filesystem::temp_directory_path() / ("kidi-files-test-" + std::to_string(::getpid()));
    void SetUp() override { std::filesystem::create_directories(root_); }
    void TearDown() override { std::filesystem::remove_all(root_); }

    static auto contents(const std::filesystem::path& path) -> std::string {
        std::ifstream input(path, std::ios::binary);
        std::stringstream text;
        text << input.rdbuf();
        return text.str();
    }
};

TEST_F(FilesTest, LocksExcludeOtherHoldersAndConvert) {
    const auto path = root_ / "state.lock";
    EXPECT_FALSE(LockFile::open(path, false)) << "an existing file is required without create";
    auto first = LockFile::open(path);
    auto second = LockFile::open(path);
    ASSERT_TRUE(first && second);
    ASSERT_TRUE(first->lock(LockFile::Mode::SHARED));
    EXPECT_TRUE(*second->try_lock(LockFile::Mode::SHARED)) << "shared locks coexist";
    EXPECT_FALSE(*first->try_lock(LockFile::Mode::EXCLUSIVE)) << "another holder keeps it shared";
    ASSERT_TRUE(second->unlock());
    EXPECT_TRUE(*first->try_lock(LockFile::Mode::EXCLUSIVE)) << "the only holder converts its lock";
    EXPECT_FALSE(*second->try_lock(LockFile::Mode::SHARED));
    first = LockFile::open(path);
    EXPECT_TRUE(*second->try_lock(LockFile::Mode::EXCLUSIVE)) << "closing a LockFile releases its lock";
#if !defined(_WIN32)
    EXPECT_GE(second->native_handle(), 3) << "never a standard stream";
    std::filesystem::create_symlink(path, root_ / "link.lock");
    EXPECT_FALSE(LockFile::open(root_ / "link.lock")) << "symbolic links are refused";
#endif
}

TEST_F(FilesTest, WritesFilesAtomically) {
    const auto path = root_ / "data.json";
    ASSERT_TRUE(write_atomic(path, "first"));
    ASSERT_TRUE(write_atomic(path, "second"));
    EXPECT_EQ(contents(path), "second");
    const std::string head = "head-", tail = "tail";
    const std::array parts{std::as_bytes(std::span(head)), std::as_bytes(std::span(tail))};
    ASSERT_TRUE(write_atomic(path, parts));
    EXPECT_EQ(contents(path), "head-tail");
#if !defined(_WIN32)
    EXPECT_EQ(std::filesystem::status(path).permissions() & std::filesystem::perms::all,
              std::filesystem::perms::owner_read | std::filesystem::perms::owner_write);
#endif
    for (const auto& entry : std::filesystem::directory_iterator(root_)) {
        const auto name = entry.path().filename().string();
        EXPECT_TRUE(name == "data.json" || name == "data.json.lock") << "no temporary files remain: " << name;
    }
    EXPECT_FALSE(write_atomic(root_ / "missing" / "data.json", "x"));
}

// Read-modify-writes of one file from many writers (as from many processes) never lose an update.
TEST_F(FilesTest, UpdatesUnderTheLock) {
    const auto path = root_ / "count";
    const auto increment = [&] {
        return write_atomic(path, [](std::string current) -> std::optional<std::string> {
            return std::to_string((current.empty() ? 0 : std::stoi(current)) + 1);
        });
    };
    {
        std::vector<std::jthread> writers;
        for (int writer = 0; writer < 4; ++writer)
            writers.emplace_back([&] {
                for (int step = 0; step < 25; ++step) ASSERT_TRUE(increment());
            });
    }
    EXPECT_EQ(contents(path), "100");
    ASSERT_TRUE(write_atomic(path, [](std::string) { return std::optional<std::string>{}; }));
    EXPECT_EQ(contents(path), "100") << "nullopt leaves the file as it is";
}

} // namespace
} // namespace kidi::core
