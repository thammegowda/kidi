#include "kidi/model/kokoro.h"

#include <filesystem>

#include <gtest/gtest.h>

namespace {
TEST(KokoroTest, RejectsMalformedVoiceScales) {
    const auto path = std::filesystem::temp_directory_path() / "kidi-kokoro-invalid.safetensors";
    auto voice = kidi::tensor::Tensor::zeros({510, 256}, kidi::tensor::DType::I8);
    auto malformed_scales = kidi::tensor::Tensor::zeros({510, 1}, kidi::tensor::DType::F32);
    ASSERT_TRUE(voice && malformed_scales &&
                kidi::checkpoint::Weights::save(
                    path, {{"voices.af_heart.weight", *voice}, {"voices.af_heart.scale", *malformed_scales}}))
        << "cannot create malformed Kokoro fixture";
    auto weights = kidi::checkpoint::Weights::load(path);
    std::filesystem::remove(path);
    ASSERT_TRUE(weights.has_value()) << weights.error().message;
    YAML::Node config;
    config["type"] = "kokoro";
    config["sample_rate"] = 24000;
    config["weight_group_size"] = 8;
    config["voices"].push_back("af_heart");
    config["config"]["vocab"]["a"] = 1;
    auto model = kidi::model::KokoroImpl::load(config, std::move(*weights));
    ASSERT_FALSE(model.has_value()) << "malformed Kokoro voice scales were accepted";
    EXPECT_EQ(model.error().code, kidi::ErrorCode::INVALID_MANIFEST) << "malformed Kokoro voice scales were accepted";
}
} // namespace
