#include "kidi/layers/transformer.h"
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "kidi/model/weights.h"

namespace {

auto write_weights(const std::filesystem::path& path) -> void {
    std::string header =
        R"({"attn.qkv.weight":{"dtype":"F32","shape":[4,12],"data_offsets":[0,192]},"attn.qkv.bias":{"dtype":"F32","shape":[12],"data_offsets":[192,240]},"attn.out.weight":{"dtype":"F32","shape":[4,4],"data_offsets":[240,304]},"attn.out.bias":{"dtype":"F32","shape":[4],"data_offsets":[304,320]}})";
    while (header.size() % 8 != 0) header.push_back(' ');

    std::ofstream output(path, std::ios::binary);
    std::uint64_t header_size = header.size();
    if constexpr (std::endian::native == std::endian::big) header_size = std::byteswap(header_size);
    output.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));

    constexpr std::array QUERY_WEIGHT = {
        0.5F, -0.25F, 0.1F, 0.0F, 0.2F, 0.4F, -0.3F, 0.1F, -0.1F, 0.3F, 0.6F, -0.2F, 0.0F, -0.4F, 0.25F, 0.5F,
    };
    constexpr std::array QUERY_BIAS = {0.1F, -0.2F, 0.05F, 0.3F};
    constexpr std::array KEY_WEIGHT = {
        0.3F, 0.2F, -0.1F, 0.4F, -0.5F, 0.1F, 0.2F, 0.25F, 0.4F, -0.3F, 0.1F, -0.2F, 0.15F, 0.5F, -0.4F, 0.2F,
    };
    constexpr std::array KEY_BIAS = {-0.05F, 0.1F, 0.2F, -0.1F};
    constexpr std::array VALUE_WEIGHT = {
        0.6F, -0.1F, 0.2F, 0.0F, 0.1F, 0.3F, -0.5F, 0.4F, -0.2F, 0.25F, 0.5F, 0.1F, 0.4F, 0.0F, -0.3F, 0.2F,
    };
    constexpr std::array VALUE_BIAS = {0.05F, -0.15F, 0.2F, 0.0F};
    constexpr std::array OUTPUT_WEIGHT = {
        0.5F, 0.2F, -0.1F, 0.3F, -0.2F, 0.4F, 0.25F, 0.1F, 0.1F, -0.3F, 0.6F, 0.2F, 0.3F, 0.1F, -0.2F, 0.5F,
    };
    constexpr std::array OUTPUT_BIAS = {0.01F, 0.02F, -0.03F, 0.04F};
    const auto write_values = [&output](const auto& values) {
        output.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(values.size() * sizeof(values[0])));
    };
    const auto write_transposed = [&output](const auto& values, std::size_t rows, std::size_t columns) {
        for (std::size_t column = 0; column < columns; ++column) {
            for (std::size_t row = 0; row < rows; ++row) {
                const auto value = values[row * columns + column];
                output.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
        }
    };
    for (std::size_t input = 0; input < 4; ++input) {
        for (const auto* weight : {&QUERY_WEIGHT, &KEY_WEIGHT, &VALUE_WEIGHT}) {
            for (std::size_t channel = 0; channel < 4; ++channel) {
                const auto value = (*weight)[channel * 4 + input];
                output.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
        }
    }
    write_values(QUERY_BIAS);
    write_values(KEY_BIAS);
    write_values(VALUE_BIAS);
    write_transposed(OUTPUT_WEIGHT, 4, 4);
    write_values(OUTPUT_BIAS);
}

} // namespace

auto main() -> int {
    using namespace kidi;
    try {
        const auto path = std::filesystem::temp_directory_path() / "kidi-attention-test.safetensors";
        write_weights(path);
        const std::array mappings{model::StateMappingSpec{{R"(attn\.qkv\.(weight|bias))"}, "qkv.$1"},
                                  model::StateMappingSpec{{R"(attn\.out\.(weight|bias))"}, "attention.output.$1"}};
        auto weights = ops::require(model::Weights::load(path, mappings));
        const ModuleScope construction(tensor::DType::F32, false);
        layers::Linear qkv(4, 12);
        layers::Attention attention(layers::Shape{4, 8, 2, 1e-5F});
        ModuleMap<> modules;
        modules->insert("qkv", qkv);
        modules->insert("attention", attention);
        ops::require(modules->set_state(weights));
        const std::array input_values{1.F, 2.F, 3.F, 4.F, 0.5F, -1.F, 2.F, -0.5F, 3.F, 0.25F, -2.F, 1.F};
        const std::array mask_values{0.F, -1e9F, -1e9F, 0.F, 0.F, -1e9F, 0.F, 0.F, 0.F};
        const std::array expected{0.514999986F, 0.700000048F,  1.379999995F, 0.089999951F, 0.243209749F, 0.126859829F,
                                  1.196354151F, -0.066927783F, 0.657077312F, 0.518348098F, 0.961920202F, 0.348947883F};
        std::vector devices{tensor::Device::cpu()};
#if defined(__APPLE__)
        devices.push_back(tensor::Device::apple_gpu());
#endif
        for (auto device : devices) {
            ops::Context context(device);
            auto input =
                ops::require(tensor::Tensor::from_host({1, 3, 4}, std::span<const float>(input_values), device));
            auto mask =
                ops::require(tensor::Tensor::from_host({1, 1, 3, 3}, std::span<const float>(mask_values), device));
            auto projected = qkv->forward(context, input);
            auto query = context.slice(projected, 2, 0, 4);
            layers::KeyValue memory{context.slice(projected, 2, 4, 4), context.slice(projected, 2, 8, 4)};
            auto output = attention->forward(context, query, memory, mask);
            auto last = attention->forward(context, context.slice(query, 1, 2, 1), memory, {});
            context.synchronize();
            auto values = ops::require(output.data<float>()), final = ops::require(last.data<float>());
            for (std::size_t index = 0; index < expected.size(); ++index)
                if (std::abs(values[index] - expected[index]) > 3e-5F) return 1;
            for (std::size_t index = 0; index < 4; ++index)
                if (std::abs(final[index] - expected[index + 8]) > 3e-5F) return 1;

            auto batched_query = context.concat(std::array{query, query}, 0);
            auto batched_output =
                context.scaled_dot_product_attention(batched_query, memory.key, memory.value, 2, mask);
            auto single_output = context.scaled_dot_product_attention(query, memory.key, memory.value, 2, mask);
            context.synchronize();
            const auto batch_values = ops::require(batched_output.data<float>());
            const auto single_values = ops::require(single_output.data<float>());
            for (std::size_t index = 0; index < batch_values.size(); ++index)
                if (std::abs(batch_values[index] - single_values[index % single_values.size()]) > 3e-5F) return 1;
            bool rejected = false;
            try {
                context.scaled_dot_product_attention(query, memory.key, memory.value, 3, mask);
            } catch (const ops::Failure&) {
                rejected = true;
            }
            if (!rejected) return 1;
            auto mask_data = ops::require(mask.data<float>());
            std::fill(mask_data.begin(), mask_data.end(), 0.F);
            auto unmasked = context.scaled_dot_product_attention(query, memory.key, memory.value, 2);
            auto rebound = context.scaled_dot_product_attention(query, memory.key, memory.value, 2, mask);
            context.synchronize();
            const auto unmasked_values = ops::require(unmasked.data<float>()),
                       rebound_values = ops::require(rebound.data<float>());
            for (std::size_t index = 0; index < unmasked_values.size(); ++index)
                if (std::abs(unmasked_values[index] - rebound_values[index]) > 3e-5F) return 1;
        }
        std::filesystem::remove(path);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
