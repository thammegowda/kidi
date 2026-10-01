#include "kidi/image/gemma4.h"
#include "kidi/model/gemma4_vision.h"
#include "kidi/ops/context.h"

#include <tahoma/vision/codec.h>

#include <cmath>
#include <iostream>

auto check_normalized_weights() -> void {
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
        if (value.dtype() != state.at("model." + name).dtype())
            throw std::runtime_error("normalized vision weight dtype changed");
    std::filesystem::remove(file);
}

auto main(int argc, char** argv) -> int {
    if (argc >= 2) {
        try {
            using kidi::ops::require;
            const std::filesystem::path directory(argv[1]);
            const auto config = YAML::LoadFile((directory / "config.yaml").string());
            const auto weights = require(kidi::checkpoint::Weights::load(directory / "model.safetensors"));
            const auto reference = require(kidi::checkpoint::Weights::load(directory / "reference.safetensors"));
            const kidi::ModuleScope scope(kidi::tensor::DType::F32, false, kidi::tensor::Device::cpu());
            auto model = kidi::model::Gemma4Vision(config, 20, argc == 3);
            require(model->set_checkpoint(weights));
            const auto patches = require(reference.tensor("patches"));
            const auto pixels = require(patches.data<float>());
            const auto actual = require(model->forward({6, 9, {pixels.begin(), pixels.end()}}));
            const auto expected = require(reference.tensor("features"));
            const auto actual_values = require(actual.data<float>()), expected_values = require(expected.data<float>());
            if (actual_values.size() != expected_values.size()) return 1;
            float maximum = 0;
            for (std::size_t index = 0; index < actual_values.size(); ++index)
                maximum = std::max(maximum, std::abs(actual_values[index] - expected_values[index]));
            std::cout << "vision feature max error: " << maximum << '\n';
            return maximum < 2e-4F ? 0 : 1;
        } catch (const kidi::ops::Failure& error) {
            std::cerr << error.error().message << '\n';
            return 1;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
    try {
        check_normalized_weights();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    auto source = tahoma::vision::make_image(48, 96);
    for (std::size_t index = 0; index < source.pixels.size(); ++index)
        source.pixels[index] = std::array<std::uint8_t, 3>{32, 128, 224}[index % 3];
    for (const auto format : {0, 1}) {
        const auto bytes =
            format ? tahoma::vision::encode_jpeg(source.view()) : tahoma::vision::encode_png(source.view());
        const auto image = kidi::image::prepare_gemma4(bytes, 70);
        if (!image || image->patch_columns % 3 || image->patch_rows % 3 ||
            image->patches.size() != static_cast<std::size_t>(image->patch_columns * image->patch_rows * 768) ||
            image->patch_rows * image->patch_columns / 9 > 70 || image->patch_rows != 33 || image->patch_columns != 15)
            return 1;
        for (std::size_t index = 0; index < image->patches.size(); ++index)
            if (std::abs(image->patches[index] * 255 - std::array<float, 3>{32, 128, 224}[index % 3]) > 2) return 1;
    }
    for (const bool tall : {false, true}) {
        for (const int edge : {16384, 16385}) {
            const auto source = tahoma::vision::make_image(tall ? 1 : edge, tall ? edge : 1);
            for (const auto format : {0, 1, 2}) {
                const auto bytes = format == 0   ? tahoma::vision::encode_png(source.view())
                                   : format == 1 ? tahoma::vision::encode_jpeg(source.view())
                                                 : tahoma::vision::encode_ppm(source.view());
                const auto result = kidi::image::prepare_gemma4(bytes, 70);
                if (edge == 16384) {
                    if (!result) return 1;
                } else if (result || result.error().code != kidi::ErrorCode::INVALID_ARGUMENT) {
                    std::cerr << "oversized image was not rejected by core limits\n";
                    return 1;
                }
            }
        }
    }
    const auto original = tahoma::vision::encode_png(source.view());
    for (const auto limit : {3'000'000U, 1'000'000U, 500'000U, 161'280U}) {
        const auto resized = kidi::image::prepare_gemma4(original, 1120, limit);
        if (!resized || resized->patches.size() / 3 > limit) return 1;
        const auto tokens = resized->patch_rows * resized->patch_columns / 9;
        const auto expected_budget = limit >= 2'580'480 ? 1120 : limit >= 645'120 ? 280 : limit >= 322'560 ? 140 : 70;
        if (tokens > expected_budget || tokens < expected_budget / 2) return 1;
    }
    if (kidi::image::prepare_gemma4(original, 280, 0) ||
        kidi::image::prepare_gemma4(original, 280, kidi::image::MIN_RESIZED_PIXELS - 1) ||
        kidi::image::prepare_gemma4(original, 280, kidi::image::MAX_RESIZED_PIXELS + 1))
        return 1;
    auto oversized = original;
    const std::array<std::uint32_t, 2> dimensions{6000, 4001};
    for (std::size_t axis = 0; axis < dimensions.size(); ++axis)
        for (std::size_t byte = 0; byte < 4; ++byte)
            oversized[16 + axis * 4 + byte] = static_cast<std::uint8_t>(dimensions[axis] >> (24 - byte * 8));
    const auto rejected = kidi::image::prepare_gemma4(oversized, 70);
    if (rejected || rejected.error().code != kidi::ErrorCode::INVALID_ARGUMENT ||
        rejected.error().message.find("pixel limits") == std::string::npos)
        return 1;
    if (kidi::image::prepare_gemma4({}) || kidi::image::prepare_gemma4(std::array<std::uint8_t, 3>{1, 2, 3})) return 1;
    return 0;
}