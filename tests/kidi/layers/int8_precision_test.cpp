#include "kidi/layers/transformer.h"
#include <array>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "kidi/model/weights.h"
#include "kidi/layers/position_encoding.h"
#if defined(__APPLE__)
#include "kidi/runtime/mps/quantized_linear.h"
#endif

namespace {

auto write_int8_weights(const std::filesystem::path& path) -> void {
    std::string header =
        R"({"linear.weight":{"dtype":"I8","shape":[4,3],"data_offsets":[0,12]},"linear.weight.scale":{"dtype":"F32","shape":[3,1],"data_offsets":[12,24]},"linear.bias":{"dtype":"F32","shape":[3],"data_offsets":[24,36]}})";
    while (header.size() % 8 != 0) header.push_back(' ');

    std::ofstream output(path, std::ios::binary);
    std::uint64_t header_size = header.size();
    if constexpr (std::endian::native == std::endian::big) header_size = std::byteswap(header_size);
    output.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));

    constexpr std::array<std::int8_t, 12> WEIGHT = {
        2, -4, 1, -1, 2, 2, 4, 1, -4, 0, 3, 8,
    };
    constexpr std::array SCALE = {0.25F, 0.25F, 0.125F};
    constexpr std::array BIAS = {0.1F, -0.2F, 0.3F};
    output.write(reinterpret_cast<const char*>(WEIGHT.data()), sizeof(WEIGHT));
    output.write(reinterpret_cast<const char*>(SCALE.data()), sizeof(SCALE));
    output.write(reinterpret_cast<const char*>(BIAS.data()), sizeof(BIAS));
}

auto write_int8_embeddings(const std::filesystem::path& path) -> void {
    std::string header =
        R"({"embedding":{"dtype":"I8","shape":[3,4],"data_offsets":[0,12]},"embedding.scale":{"dtype":"F32","shape":[3,1],"data_offsets":[12,24]}})";
    while (header.size() % 8 != 0) header.push_back(' ');

    std::ofstream output(path, std::ios::binary);
    std::uint64_t header_size = header.size();
    if constexpr (std::endian::native == std::endian::big) header_size = std::byteswap(header_size);
    output.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));

    constexpr std::array<std::int8_t, 12> WEIGHT = {0, 0, 0, 0, 2, 4, 6, 8, -2, -4, -6, -8};
    constexpr std::array SCALE = {1.0F, 0.5F, 0.5F};
    output.write(reinterpret_cast<const char*>(WEIGHT.data()), sizeof(WEIGHT));
    output.write(reinterpret_cast<const char*>(SCALE.data()), sizeof(SCALE));
}

} // namespace

auto main() -> int {
    using namespace kidi;
    using ops::require;
    using tensor::Tensor;
    try {
        const auto path = std::filesystem::temp_directory_path() / "kidi-int8-precision-test.safetensors";
        write_int8_weights(path);
        const std::array mappings{model::StateMappingSpec{{R"(linear\.weight\.scale)"}, "linear.scale"}};
        auto weights = require(model::Weights::load(path, mappings));
        const ModuleScope construction(tensor::DType::I8, false);
        layers::Linear linear(4, 3);
        ModuleMap<> modules;
        modules->insert("linear", linear);
        auto missing_scale = require(weights.state_dict());
        missing_scale.erase("linear.scale");
        if (modules->set_state(missing_scale)) return 1;
        require(modules->set_state(weights));
        const std::array input_values{-1.F, 0.F, 1.F, 2.F, -2.F, -1.F, 1.F, 2.F};
        const std::array expected{0.6F, 2.55F, 1.675F, 0.35F, 3.05F, 1.3F};
        std::vector devices{tensor::Device::cpu()};
#if defined(__APPLE__)
        devices.push_back(tensor::Device::apple_gpu());
#endif
        for (auto device : devices) {
            ops::Context context(device);
            auto input = require(Tensor::from_host({1, 2, 4}, std::span<const float>(input_values), device));
            auto output = linear->forward(context, input);
            context.synchronize();
            auto values = require(output.data<float>());
            for (std::size_t index = 0; index < expected.size(); ++index)
                if (std::abs(values[index] - expected[index]) > 0.04F) return 1;
        }
#if defined(__APPLE__)
        ops::Context cpu, gpu(tensor::Device::apple_gpu());
        const auto parameters = [](std::int64_t width, std::int64_t columns) {
            std::vector<std::int8_t> weights(width * columns);
            std::vector<float> scales(columns), bias(columns);
            for (std::size_t index = 0; index < weights.size(); ++index)
                weights[index] = static_cast<std::int8_t>(static_cast<int>((index * 37 + 13) % 255) - 127);
            for (std::int64_t index = 0; index < columns; ++index) {
                scales[index] = 0.001F * (1 + index % 4);
                bias[index] = 0.02F * (index - 2);
            }
            return std::array{require(Tensor::from_host({width, columns}, std::span<const std::int8_t>(weights))),
                              require(Tensor::from_host({columns, 1}, std::span<const float>(scales))),
                              require(Tensor::from_host({columns}, std::span<const float>(bias)))};
        };
        const auto first = parameters(7, 35), second = parameters(35, 5);
        for (std::int64_t rows : {4, 1, 2, 3, 33, 65, 4, 1, 33}) {
            std::vector<float> data(rows * 7);
            for (std::int64_t row = 0; row < rows; ++row)
                for (std::int64_t channel = 0; channel < 7; ++channel)
                    data[row * 7 + channel] = row == 3   ? 0.F
                                              : row == 1 ? 1.5F
                                              : row == 2 ? -0.75F
                                                         : (channel - 3) * 0.37F;
            const auto run = [&](ops::Context& context) {
                auto input = require(Tensor::from_host({rows, 7}, std::span<const float>(data), context.device()));
                auto scalar = [&](float value) {
                    return require(Tensor::from_host({}, std::span<const float>(&value, 1), context.device()));
                };
                auto projected =
                    context.quantized_linear(context.multiply(input, scalar(0.75F)), first[0], first[1], first[2]);
                auto output = context.add(context.quantized_linear(context.multiply(projected, scalar(0.5F)), second[0],
                                                                   second[1], second[2]),
                                          scalar(0.125F));
                context.synchronize();
                return std::array{projected, output};
            };
            auto reference = run(cpu), actual = run(gpu);
            for (std::size_t result = 0; result < 2; ++result) {
                auto expected = require(reference[result].data<float>()),
                     values = require(actual[result].data<float>());
                for (std::size_t index = 0; index < values.size(); ++index)
                    if (!std::isfinite(values[index]) || std::abs(values[index] - expected[index]) > 0.01F) return 1;
            }
        }
        constexpr std::int64_t WIDTH = 1025, COLUMNS = 35;
        std::vector<std::int8_t> weight_data(WIDTH * COLUMNS);
        for (std::int64_t channel = 0; channel < WIDTH; ++channel)
            for (std::int64_t column = 0; column < COLUMNS; ++column)
                weight_data[channel * COLUMNS + column] = column % 2 == 0 ? -128 : 127;
        std::vector<float> scales(COLUMNS, 1.F / 128), bias(COLUMNS, 0.25F);
        auto matrix = require(Tensor::from_host({WIDTH, COLUMNS}, std::span<const std::int8_t>(weight_data)));
        auto scale = require(Tensor::from_host({COLUMNS, 1}, std::span<const float>(scales)));
        auto offset = require(Tensor::from_host({COLUMNS}, std::span<const float>(bias)));
        for (std::int64_t rows : {3, 33, 2, 1}) {
            std::vector<float> data(rows * WIDTH);
            for (std::int64_t row = 0; row < rows; ++row)
                std::fill_n(data.begin() + row * WIDTH, WIDTH, row % 3 == 0 ? 0.F : row % 3 == 1 ? 255.F : -255.F);
            auto input = require(Tensor::from_host({rows, WIDTH}, std::span<const float>(data), gpu.device()));
            auto output = gpu.quantized_linear(input, matrix, scale, offset);
            gpu.synchronize();
            auto values = require(output.data<float>());
            for (std::int64_t row = 0; row < rows; ++row)
                for (std::int64_t column = 0; column < COLUMNS; ++column) {
                    const auto sum = static_cast<std::int32_t>(data[row * WIDTH]) *
                                     static_cast<std::int32_t>(weight_data[column]) * WIDTH;
                    if (values[row * COLUMNS + column] != static_cast<float>(sum) * scales[column] + bias[column])
                        return 1;
                }
        }
#endif
        std::filesystem::remove(path);
        const auto embedding_path = std::filesystem::temp_directory_path() / "kidi-int8-embedding-test.safetensors";
        write_int8_embeddings(embedding_path);
        auto embedding_weights = require(model::Weights::load(embedding_path));
        layers::Embedding embedding(3, 4, 2);
        require(embedding->set_state(StateDict{{"weight", require(embedding_weights.tensor("embedding"))},
                                               {"scale", require(embedding_weights.tensor("embedding.scale"))}}));
        const auto frequency = std::exp(-(std::log(10000.F) / 4.F) * 2.F);
        const std::array embedded_expected{2.F,
                                           5.F,
                                           6.F,
                                           9.F,
                                           -2.F + std::sin(1.F),
                                           -4.F + std::cos(1.F),
                                           -6.F + std::sin(frequency),
                                           -8.F + std::cos(frequency)};
        for (auto device : devices) {
            ops::Context context(device);
            auto output = embedding->forward(context, std::array<std::int32_t, 2>{1, 2}, 1);
            auto values = require(output.data<float>());
            for (std::size_t index = 0; index < values.size(); ++index)
                if (std::abs(values[index] - embedded_expected[index]) > 1e-5F) return 1;
        }
        std::filesystem::remove(embedding_path);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
