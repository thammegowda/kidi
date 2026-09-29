#include "kidi/image/gemma4.h"
#include "kidi/model/gemma4_vision.h"
#include "kidi/ops/context.h"

#include <tahoma/vision/codec.h>

#include <cmath>
#include <iostream>

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
    if (kidi::image::prepare_gemma4({}) || kidi::image::prepare_gemma4(std::array<std::uint8_t, 3>{1, 2, 3})) return 1;
    return 0;
}