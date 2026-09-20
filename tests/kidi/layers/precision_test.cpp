#include "kidi/layers/transformer.h"
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "kidi/model/precision.h"
#include "kidi/model/weights.h"

namespace {

auto write_bf16_weights(const std::filesystem::path& path) -> void {
    std::string header =
        R"({"linear.weight":{"dtype":"BF16","shape":[4,3],"data_offsets":[0,24]},"linear.bias":{"dtype":"F32","shape":[3],"data_offsets":[24,36]},"tgt_embed.0.lut.weight":{"dtype":"BF16","shape":[3,4],"data_offsets":[36,60]}})";
    while (header.size() % 8 != 0) header.push_back(' ');

    std::ofstream output(path, std::ios::binary);
    std::uint64_t header_size = header.size();
    if constexpr (std::endian::native == std::endian::big) header_size = std::byteswap(header_size);
    output.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));

    constexpr std::array<std::uint16_t, 12> WEIGHT = {
        0x3f00, 0xbf80, 0x3e00, 0xbe80, 0x3f00, 0x3e80, 0x3f80, 0x3e80, 0xbf00, 0x0000, 0x3f40, 0x3f80,
    };
    constexpr std::array BIAS = {0.1F, -0.2F, 0.3F};
    output.write(reinterpret_cast<const char*>(WEIGHT.data()), sizeof(WEIGHT));
    output.write(reinterpret_cast<const char*>(BIAS.data()), sizeof(BIAS));
    constexpr std::array<std::uint16_t, 12> EMBEDDING = {
        0x3f00, 0xbe80, 0x3f80, 0x0000, 0xbf80, 0x3f00, 0x3e80, 0x3f40, 0x3e00, 0x3e80, 0xbf00, 0x3f80,
    };
    output.write(reinterpret_cast<const char*>(EMBEDDING.data()), sizeof(EMBEDDING));
}

auto near(float left, float right) -> bool { return std::abs(left - right) < 1.0e-5F; }

} // namespace

auto main() -> int {
    using namespace kidi;
    try {
        const auto path = std::filesystem::temp_directory_path() / "kidi-bf16-precision-test.safetensors";
        write_bf16_weights(path);
        auto weights = ops::require(model::Weights::load(path));
        layers::Linear linear = std::make_shared<layers::LinearImpl>(weights, "linear", model::WeightEncoding::BF16);
        layers::Linear tied = std::make_shared<layers::LinearImpl>(weights, "tgt_embed.0.lut.weight", "linear.bias",
                                                                   model::WeightEncoding::BF16, true);
        const std::array inputs{1.F, 2.F, 3.F, 4.F, -1.F, 0.5F, 2.F, -0.5F};
        const std::array expected{3.1F, 3.55F, 3.425F, 1.475F, 1.175F, -1.2F};
        std::vector devices{tensor::Device::cpu()};
#if defined(__APPLE__)
        devices.push_back(tensor::Device::apple_gpu());
#endif
        for (auto device : devices) {
            ops::Context context(device);
            auto input = ops::require(tensor::Tensor::from_host({1, 2, 4}, std::span<const float>(inputs), device));
            auto output = linear->forward(context, input);
            auto tied_output = tied->forward(context, input);
            context.synchronize();
            auto values = ops::require(output.data<float>());
            auto tied_values = ops::require(tied_output.data<float>());
            for (std::size_t index = 0; index < expected.size(); ++index)
                if (!near(values[index], expected[index]) || !near(tied_values[index], expected[index])) return 1;
        }
        std::filesystem::remove(path);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
