#include "kidi/checkpoint/hub.h"

#include <cstdlib>
#include <fstream>
#include <optional>
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
    fs::create_directories(root / "owner/model/tokenizer");
    std::ofstream(root / "owner/model/tokenizer/tokenizer.json") << "{}\n";
    {
        const kidi::checkpoint::Hub hub{"https://huggingface.co", root};
        const auto model = hub.resolve("owner/model");
        const auto file = hub.get_file("owner/model", "config.json");
        ASSERT_TRUE(model.has_value()) << model.error().message;
        EXPECT_EQ(*model, root / "owner/model");
        ASSERT_TRUE(file.has_value()) << file.error().message;
        EXPECT_EQ(*file, *model / "config.json");
        const auto reference_model = hub.resolve("@owner/model");
        const auto reference_file = hub.resolve("@owner/model/config.json");
        ASSERT_TRUE(reference_model.has_value()) << reference_model.error().message;
        ASSERT_TRUE(reference_file.has_value()) << reference_file.error().message;
        EXPECT_EQ(*reference_model, *model);
        EXPECT_EQ(*reference_file, *file);
        const auto tokenizer = hub.resolve("@owner/model/tokenizer/tokenizer.json");
        ASSERT_TRUE(tokenizer.has_value()) << tokenizer.error().message;
        EXPECT_EQ(*tokenizer, *model / "tokenizer/tokenizer.json");
        EXPECT_FALSE(hub.resolve("@owner/model/missing.json"));
        EXPECT_FALSE(hub.resolve("@owner/missing"));
        EXPECT_FALSE(hub.resolve("@owner/model/../../secret"));
        EXPECT_FALSE(hub.resolve("@owner/model/"));
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

TEST(HubTest, DefaultCacheOverride) {
    const auto* original = std::getenv("KIDI_HUB");
    struct RestoreEnvironment {
        std::optional<std::string> value;
        ~RestoreEnvironment() {
#if defined(_WIN32)
            _putenv_s("KIDI_HUB", value ? value->c_str() : "");
#else
            if (value)
                setenv("KIDI_HUB", value->c_str(), 1);
            else
                unsetenv("KIDI_HUB");
#endif
        }
    } restore{original ? std::optional<std::string>(original) : std::nullopt};
    const auto root = fs::temp_directory_path() / "kidi-default-hub-test";
    fs::create_directories(root / "owner/model");
    std::ofstream(root / "owner/model/tokenizer.json") << "{}\n";
#if defined(_WIN32)
    ASSERT_EQ(_putenv_s("KIDI_HUB", root.string().c_str()), 0);
#else
    ASSERT_EQ(setenv("KIDI_HUB", root.c_str(), 1), 0);
#endif
    const auto file = kidi::checkpoint::Hub{}.resolve("@owner/model/tokenizer.json");
    ASSERT_TRUE(file.has_value()) << file.error().message;
    EXPECT_EQ(*file, root / "owner/model/tokenizer.json");
    fs::remove_all(root);
#if defined(_WIN32)
    ASSERT_EQ(_putenv_s("KIDI_HUB", ""), 0);
#else
    ASSERT_EQ(unsetenv("KIDI_HUB"), 0);
#endif
    EXPECT_EQ(kidi::checkpoint::Hub{}.local(),
              kidi::checkpoint::Hub::expand_home("~/.cache/kidi/model-hub"));
}

TEST(HubTest, ExpandsHome) {
    const auto home = kidi::checkpoint::Hub::expand_home("~");
    EXPECT_FALSE(home.empty());
    EXPECT_EQ(kidi::checkpoint::Hub::expand_home("~/models/gemma"), home / "models/gemma");
    EXPECT_EQ(kidi::checkpoint::Hub::expand_home("/abs/path"), "/abs/path");
    EXPECT_THROW((void)kidi::checkpoint::Hub::expand_home("~other/model"), std::invalid_argument);
}

} // namespace
