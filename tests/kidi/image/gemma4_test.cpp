#include "kidi/image/gemma4.h"
#include "kidi/model/gemma4_vision.h"
#include "kidi/ops/context.h"

#include <tahoma/vision/codec.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <gtest/gtest.h>

namespace {

auto max_difference(std::span<const float> left, std::span<const float> right) -> float {
    float maximum = 0;
    for (std::size_t index = 0; index < left.size(); ++index)
        maximum = std::max(maximum, std::abs(left[index] - right[index]));
    return maximum;
}

// Optional parity check: KIDI_TEST_GEMMA4_VISION_FIXTURE=<dir with config.yaml, model.safetensors,
// reference.safetensors>; additionally set KIDI_TEST_GEMMA4_VISION_QUANTIZED (any value) for a quantized fixture.
TEST(Gemma4ImageTest, VisionFixtureParity) {
    const char* fixture = std::getenv("KIDI_TEST_GEMMA4_VISION_FIXTURE");
    if (!fixture) GTEST_SKIP() << "KIDI_TEST_GEMMA4_VISION_FIXTURE is not set";
    const bool quantized = std::getenv("KIDI_TEST_GEMMA4_VISION_QUANTIZED") != nullptr;
    using kidi::ops::require;
    const std::filesystem::path directory(fixture);
    const auto config = YAML::LoadFile((directory / "config.yaml").string());
    const auto weights = require(kidi::checkpoint::Weights::load(directory / "model.safetensors"));
    const auto reference = require(kidi::checkpoint::Weights::load(directory / "reference.safetensors"));
    const kidi::ModuleScope scope(kidi::tensor::DType::F32, false, kidi::tensor::Device::cpu());
    auto model = kidi::model::Gemma4Vision(config, 20, quantized);
    require(model->set_checkpoint(weights));
    const auto patches = require(reference.tensor("patches"));
    const auto pixels = require(patches.data<float>());
    std::size_t observed_blocks = 0;
    std::size_t observed_stages = 0;
    const auto actual = require(model->forward(
        {6, 9, {pixels.begin(), pixels.end()}},
        [&](std::size_t index, const kidi::tensor::Tensor& block) {
            EXPECT_EQ(index, observed_blocks) << "invalid vision block observation";
            EXPECT_EQ(block.dimensions(), 3U) << "invalid vision block observation";
            EXPECT_EQ(block.size(0), 1) << "invalid vision block observation";
            EXPECT_EQ(block.size(1), 54) << "invalid vision block observation";
            EXPECT_EQ(static_cast<std::size_t>(block.size(2)), config["hidden_size"].as<std::size_t>())
                << "invalid vision block observation";
            ++observed_blocks;
        },
        [&](std::size_t layer, std::string_view stage, const kidi::tensor::Tensor& output) {
            EXPECT_EQ(layer, observed_blocks) << "invalid vision stage observation";
            EXPECT_FALSE(stage.empty()) << "invalid vision stage observation";
            EXPECT_TRUE(output.defined()) << "invalid vision stage observation";
            ++observed_stages;
        },
        config["num_hidden_layers"].as<std::size_t>()));
    EXPECT_EQ(observed_blocks, config["num_hidden_layers"].as<std::size_t>()) << "vision block observer missed layers";
    EXPECT_NE(observed_stages, 0U) << "vision stage observer missed operations";
    const auto expected = require(reference.tensor("features"));
    const auto actual_values = require(actual.data<float>()), expected_values = require(expected.data<float>());
    ASSERT_EQ(actual_values.size(), expected_values.size());
    const float maximum = max_difference(actual_values, expected_values);
    if (quantized) {
        model->set_precision(kidi::core::InferencePrecision::QAT_FP32);
        const auto reference_actual = require(model->forward({6, 9, {pixels.begin(), pixels.end()}}));
        const auto reference_values = require(reference_actual.data<float>());
        const float reference_maximum = max_difference(reference_values, expected_values);
        std::cout << "QAT-FP32 parity max error: " << reference_maximum << '\n';
        EXPECT_LT(reference_maximum, 2e-6F) << "FP32 packed-linear reference max error";
#if defined(__APPLE__)
        model->set_precision(kidi::core::InferencePrecision::LOWBIT_PARITY);
        const auto lowbit_actual = require(model->forward({6, 9, {pixels.begin(), pixels.end()}}));
        const auto lowbit_values = require(lowbit_actual.data<float>());
        const float lowbit_maximum = max_difference(lowbit_values, expected_values);
        std::cout << "low-bit parity max error: " << lowbit_maximum << '\n';
        EXPECT_LT(lowbit_maximum, 2e-6F) << "low-bit parity max error";
#endif
    }
#if defined(__APPLE__)
    {
        const kidi::ModuleScope gpu_scope(kidi::tensor::DType::F32, false, kidi::tensor::Device::apple_gpu());
        auto gpu_model = kidi::model::Gemma4Vision(config, 20, quantized);
        require(gpu_model->set_checkpoint(weights));
        const auto gpu = require(gpu_model->forward({6, 9, {pixels.begin(), pixels.end()}}));
        const auto gpu_values = require(gpu.data<float>());
        const float gpu_maximum = max_difference(gpu_values, actual_values);
        std::cout << "Metal vision max CPU difference: " << gpu_maximum << '\n';
        EXPECT_LT(gpu_maximum, 2e-6F) << "Metal vision max CPU difference";
    }
#endif
    std::cout << "vision feature max error: " << maximum << '\n';
    EXPECT_LT(maximum, 2e-4F) << "vision feature max error";
}

TEST(Gemma4ImageTest, WeightsKeepDtype) {
    using namespace kidi;
    using ops::require;
    const auto config = YAML::Load(R"(
hidden_size: 8
intermediate_size: 16
num_hidden_layers: 1
num_attention_heads: 2
num_key_value_heads: 2
head_dim: 4
patch_size: 16
pooling_kernel_size: 3
position_embedding_size: 32
rms_norm_eps: 0.000001
rope_parameters: {rope_theta: 10000}
)");
    StateDict state;
    {
        const ModuleScope scope(tensor::DType::F32, true, tensor::Device::cpu());
        const auto model = model::Gemma4Vision(config, 20, true);
        for (const auto& [name, value] : model->state_dict())
            state.emplace("model." + name,
                          require(tensor::Tensor::zeros({value.shape().begin(), value.shape().end()}, value.dtype())));
    }
    const auto file = std::filesystem::temp_directory_path() / "kidi-normalized-vision.safetensors";
    std::filesystem::remove(file);
    require(checkpoint::Weights::save(file, state));
    const auto weights = require(checkpoint::Weights::load(file));
    const ModuleScope scope(tensor::DType::F32, false, tensor::Device::cpu());
    const auto model = model::Gemma4Vision(config, 20, true);
    require(model->set_checkpoint(weights));
    for (const auto& [name, value] : model->state_dict())
        EXPECT_EQ(value.dtype(), state.at("model." + name).dtype()) << "normalized vision weight dtype changed: " << name;
    std::filesystem::remove(file);
}

auto solid_image() -> tahoma::vision::Image {
    auto source = tahoma::vision::make_image(48, 96);
    for (std::size_t index = 0; index < source.pixels.size(); ++index)
        source.pixels[index] = std::array<std::uint8_t, 3>{32, 128, 224}[index % 3];
    return source;
}

TEST(Gemma4ImageTest, PreparesPatches) {
    const auto source = solid_image();
    for (const auto format : {0, 1}) {
        SCOPED_TRACE(format ? "jpeg" : "png");
        const auto bytes =
            format ? tahoma::vision::encode_jpeg(source.view()) : tahoma::vision::encode_png(source.view());
        const auto image = kidi::image::prepare_gemma4(bytes, 70);
        ASSERT_TRUE(image.has_value()) << image.error().message;
        EXPECT_EQ(image->patch_columns % 3, 0);
        EXPECT_EQ(image->patch_rows % 3, 0);
        ASSERT_EQ(image->patches.size(), static_cast<std::size_t>(image->patch_columns * image->patch_rows * 768));
        EXPECT_LE(image->patch_rows * image->patch_columns / 9, 70);
        EXPECT_EQ(image->patch_rows, 33);
        EXPECT_EQ(image->patch_columns, 15);
        for (std::size_t index = 0; index < image->patches.size(); ++index)
            ASSERT_LE(std::abs(image->patches[index] * 255 - std::array<float, 3>{32, 128, 224}[index % 3]), 2)
                << "patch value " << index;
    }
}

TEST(Gemma4ImageTest, EnforcesCoreEdgeLimits) {
    for (const bool tall : {false, true}) {
        for (const int edge : {16384, 16385}) {
            const auto source = tahoma::vision::make_image(tall ? 1 : edge, tall ? edge : 1);
            for (const auto format : {0, 1, 2}) {
                SCOPED_TRACE(::testing::Message() << "tall=" << tall << " edge=" << edge << " format=" << format);
                const auto bytes = format == 0   ? tahoma::vision::encode_png(source.view())
                                   : format == 1 ? tahoma::vision::encode_jpeg(source.view())
                                                 : tahoma::vision::encode_ppm(source.view());
                const auto result = kidi::image::prepare_gemma4(bytes, 70);
                if (edge == 16384) {
                    EXPECT_TRUE(result.has_value()) << result.error().message;
                } else {
                    ASSERT_FALSE(result) << "oversized image was not rejected by core limits";
                    EXPECT_EQ(result.error().code, kidi::ErrorCode::INVALID_ARGUMENT)
                        << "oversized image was not rejected by core limits";
                }
            }
        }
    }
}

TEST(Gemma4ImageTest, ResizesToPixelBudget) {
    const auto source = solid_image();
    const auto original = tahoma::vision::encode_png(source.view());
    for (const auto limit : {3'000'000U, 1'000'000U, 500'000U, 161'280U}) {
        SCOPED_TRACE(::testing::Message() << "limit=" << limit);
        const auto resized = kidi::image::prepare_gemma4(original, 1120, limit);
        ASSERT_TRUE(resized.has_value()) << resized.error().message;
        EXPECT_LE(resized->patches.size() / 3, limit);
        const auto tokens = resized->patch_rows * resized->patch_columns / 9;
        const auto expected_budget = limit >= 2'580'480 ? 1120 : limit >= 645'120 ? 280 : limit >= 322'560 ? 140 : 70;
        EXPECT_LE(tokens, expected_budget);
        EXPECT_GE(tokens, expected_budget / 2);
    }
    EXPECT_FALSE(kidi::image::prepare_gemma4(original, 280, 0));
    EXPECT_FALSE(kidi::image::prepare_gemma4(original, 280, kidi::image::MIN_RESIZED_PIXELS - 1));
    EXPECT_FALSE(kidi::image::prepare_gemma4(original, 280, kidi::image::MAX_RESIZED_PIXELS + 1));
}

TEST(Gemma4ImageTest, RejectsInvalidImages) {
    const auto source = solid_image();
    auto oversized = tahoma::vision::encode_png(source.view());
    const std::array<std::uint32_t, 2> dimensions{6000, 4001};
    for (std::size_t axis = 0; axis < dimensions.size(); ++axis)
        for (std::size_t byte = 0; byte < 4; ++byte)
            oversized[16 + axis * 4 + byte] = static_cast<std::uint8_t>(dimensions[axis] >> (24 - byte * 8));
    const auto rejected = kidi::image::prepare_gemma4(oversized, 70);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, kidi::ErrorCode::INVALID_ARGUMENT);
    EXPECT_NE(rejected.error().message.find("pixel limits"), std::string::npos) << rejected.error().message;
    EXPECT_FALSE(kidi::image::prepare_gemma4({}));
    EXPECT_FALSE(kidi::image::prepare_gemma4(std::array<std::uint8_t, 3>{1, 2, 3}));
}

} // namespace
