#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "kidi/checkpoint/weights.h"
#include "kidi/ops/context.h"

namespace {

auto write_weights(const std::filesystem::path& path) -> void {
    std::string header = R"({"weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}})";
    while (header.size() % 8 != 0) {
        header.push_back(' ');
    }

    std::ofstream output(path, std::ios::binary);
    std::uint64_t header_size = header.size();
    if constexpr (std::endian::native == std::endian::big) {
        header_size = std::byteswap(header_size);
    }
    output.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    constexpr std::array VALUES = {1.0F, 2.0F, 3.0F, 4.0F};
    output.write(reinterpret_cast<const char*>(VALUES.data()), sizeof(VALUES));
}

} // namespace

auto main() -> int {
    const auto path = std::filesystem::temp_directory_path() / "kidi-ynnpack-weights.safetensors";
    write_weights(path);
    auto weights = kidi::checkpoint::Weights::load(path);
    auto weight =
        weights ? weights->tensor("weight") : kidi::Result<kidi::tensor::Tensor>{std::unexpected(weights.error())};
    if (!weight) {
        std::cerr << "failed to load mapped weight\n";
        return 1;
    }

    constexpr std::array INPUT_VALUES = {4.0F, 5.0F};
    auto input = kidi::ops::require(
        kidi::tensor::Tensor::from_host({1, 2}, std::span<const float>(INPUT_VALUES), kidi::tensor::Device::cpu()));
    kidi::ops::Context context(kidi::tensor::Device::cpu());
    auto output = context.linear(input, *weight, {}, true);
    context.synchronize();
    const auto out_values = kidi::ops::require(output.data<float>());
    std::filesystem::remove(path);

    if (out_values.size() != 2 || out_values[0] != 14.0F || out_values[1] != 32.0F) {
        std::cerr << "unexpected YNNPACK output: " << out_values[0] << ", " << out_values[1] << '\n';
        return 1;
    }
    return 0;
}