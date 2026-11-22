#include "kidi/checkpoint/hub.h"

#include <fstream>
#include <stdexcept>

#include <gtest/gtest.h>

namespace {

namespace fs = std::filesystem;

TEST(HubTest, ResolvesLocalModels) {
    const auto root = fs::temp_directory_path() / "kidi-hub-test";
    std::error_code error;
    fs::remove_all(root, error);
    fs::create_directories(root / "owner/model");
    std::ofstream(root / "owner/model/config.json") << "{}\n";
    {
        const kidi::checkpoint::Hub hub{"https://huggingface.co", root};
        const auto model = hub.resolve("owner/model");
        const auto file = hub.get_file("owner/model", "config.json");
        ASSERT_TRUE(model.has_value()) << model.error().message;
        EXPECT_EQ(*model, root / "owner/model");
        ASSERT_TRUE(file.has_value()) << file.error().message;
        EXPECT_EQ(*file, *model / "config.json");
        EXPECT_EQ(hub.remote(), "https://huggingface.co");
        EXPECT_EQ(hub.local(), root);
        EXPECT_FALSE(hub.resolve("../model"));
        EXPECT_FALSE(hub.resolve("owner/.."));
        EXPECT_FALSE(hub.resolve("owner/model", "../main"));
        EXPECT_FALSE(hub.get_file("owner/model", "../secret"));
        EXPECT_FALSE(hub.download("owner/model@"));
    }
    fs::remove_all(root, error);
}

TEST(HubTest, ExpandsHome) {
    const auto home = kidi::checkpoint::Hub::expand_home("~");
    EXPECT_FALSE(home.empty());
    EXPECT_EQ(kidi::checkpoint::Hub::expand_home("~/models/gemma"), home / "models/gemma");
    EXPECT_EQ(kidi::checkpoint::Hub::expand_home("/abs/path"), "/abs/path");
    EXPECT_THROW((void)kidi::checkpoint::Hub::expand_home("~other/model"), std::invalid_argument);
}

} // namespace
