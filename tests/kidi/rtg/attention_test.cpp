#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <ynnpack.h>

#include "kidi/model/weights.h"
#include "kidi/rtg/transformer_builder.h"
#include "kidi/runtime/ynn.h"

namespace {

void write_weights(const std::filesystem::path& path) {
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

int main() {
    const auto path = std::filesystem::temp_directory_path() / "kidi-attention-test.safetensors";
    write_weights(path);
    auto weights = kidi::model::Weights::load(path);
    if (!weights) {
        std::cerr << weights.error().message << '\n';
        return 1;
    }

    auto graph = kidi::runtime::YnnGraph::create(7);
    if (!graph) {
        std::cerr << graph.error().message << '\n';
        return 1;
    }
    constexpr std::array<std::size_t, 3> INPUT_SHAPE = {1, 0, 4};
    constexpr std::array<std::size_t, 4> MASK_SHAPE = {1, 1, 0, 0};
    std::uint32_t input_id = 0;
    std::uint32_t mask_id = 1;
    std::uint32_t output_id = 2;
    std::uint32_t projected_output_id = 3;
    std::uint32_t last_query_id = 4;
    std::uint32_t decode_output_id = 5;
    std::uint32_t decode_one_output_id = 6;
    auto status = kidi::runtime::check_ynn_status(
        ynn_define_tensor(graph->get(), ynn_type_fp32, INPUT_SHAPE.size(), INPUT_SHAPE.data(), nullptr,
                          YNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id),
        "define attention input");
    if (status)
        status = kidi::runtime::check_ynn_status(
            ynn_define_tensor(graph->get(), ynn_type_fp32, MASK_SHAPE.size(), MASK_SHAPE.data(), nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_INPUT, &mask_id),
            "define attention mask");
    if (status)
        status =
            kidi::runtime::check_ynn_status(ynn_define_tensor(graph->get(), ynn_type_fp32, INPUT_SHAPE.size(), nullptr,
                                                              nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id),
                                            "define attention output");
    if (status)
        status = kidi::runtime::check_ynn_status(
            ynn_define_tensor(graph->get(), ynn_type_fp32, INPUT_SHAPE.size(), nullptr, nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &projected_output_id),
            "define projected attention output");
    constexpr std::array<std::size_t, 3> LAST_QUERY_SHAPE = {1, 1, 4};
    if (status)
        status = kidi::runtime::check_ynn_status(
            ynn_define_tensor(graph->get(), ynn_type_fp32, LAST_QUERY_SHAPE.size(), LAST_QUERY_SHAPE.data(), nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_INPUT, &last_query_id),
            "define last attention query");
    if (status)
        status = kidi::runtime::check_ynn_status(
            ynn_define_tensor(graph->get(), ynn_type_fp32, LAST_QUERY_SHAPE.size(), nullptr, nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &decode_output_id),
            "define decode attention output");
    if (status)
        status = kidi::runtime::check_ynn_status(
            ynn_define_tensor(graph->get(), ynn_type_fp32, LAST_QUERY_SHAPE.size(), nullptr, nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &decode_one_output_id),
            "define decode-one attention output");

    kidi::rtg::TransformerBuilder builder(graph->get(), *weights, 4, 8, 2, 1.0e-5F);
    auto result_id = status ? builder.self_attention(input_id, mask_id, "attn", output_id)
                            : kidi::Result<std::uint32_t>{std::unexpected(std::move(status.error()))};
    if (!result_id) {
        std::cerr << result_id.error().message << '\n';
        return 1;
    }
    auto projected_ids = builder.linear_split(input_id, "attn.qkv", 4, 4, 3);
    auto projected_result_id =
        projected_ids ? builder.attention_from_projections((*projected_ids)[0], (*projected_ids)[1],
                                                           (*projected_ids)[2], mask_id, "attn", projected_output_id)
                      : kidi::Result<std::uint32_t>{std::unexpected(std::move(projected_ids.error()))};
    if (!projected_result_id) {
        std::cerr << projected_result_id.error().message << '\n';
        return 1;
    }
    auto last_projection_ids = builder.linear_split(last_query_id, "attn.qkv", 4, 4, 3);
    auto decode_result_id =
        last_projection_ids
            ? builder.attention_from_projections((*last_projection_ids)[0], (*projected_ids)[1], (*projected_ids)[2],
                                                 YNN_INVALID_VALUE_ID, "attn", decode_output_id)
            : kidi::Result<std::uint32_t>{std::unexpected(std::move(last_projection_ids.error()))};
    if (!decode_result_id) {
        std::cerr << decode_result_id.error().message << '\n';
        return 1;
    }
    auto decode_one_result_id = builder.attention_decode_one_from_projections(
        (*last_projection_ids)[0], (*projected_ids)[1], (*projected_ids)[2], YNN_INVALID_VALUE_ID, "attn",
        decode_one_output_id);
    if (!decode_one_result_id) {
        std::cerr << decode_one_result_id.error().message << '\n';
        return 1;
    }

    auto executable = std::move(*graph).compile();
    if (!executable) {
        std::cerr << executable.error().message << '\n';
        return 1;
    }
    constexpr std::array<std::size_t, 3> RUN_INPUT_SHAPE = {1, 3, 4};
    constexpr std::array<std::size_t, 4> RUN_MASK_SHAPE = {1, 1, 3, 3};
    std::array input = {
        1.0F, 2.0F, 3.0F, 4.0F, 0.5F, -1.0F, 2.0F, -0.5F, 3.0F, 0.25F, -2.0F, 1.0F,
    };
    constexpr std::array mask = {
        0.0F, -1.0e9F, -1.0e9F, 0.0F, 0.0F, -1.0e9F, 0.0F, 0.0F, 0.0F,
    };
    std::array<float, 12> output{};
    std::array<float, 12> projected_output{};
    std::array<float, 4> decode_output{};
    std::array<float, 4> decode_one_output{};
    status = executable->set_shape(input_id, RUN_INPUT_SHAPE);
    if (status) status = executable->set_shape(mask_id, RUN_MASK_SHAPE);
    if (status) status = executable->reshape();
    if (status) status = executable->bind(input_id, input.data());
    if (status) status = executable->bind(last_query_id, input.data() + 8);
    if (status) status = executable->bind(mask_id, const_cast<float*>(mask.data()));
    if (status) status = executable->bind(output_id, output.data());
    if (status) status = executable->bind(projected_output_id, projected_output.data());
    if (status) status = executable->bind(decode_output_id, decode_output.data());
    if (status) status = executable->bind(decode_one_output_id, decode_one_output.data());
    if (status) status = executable->invoke();
    if (!status) {
        std::cerr << status.error().message << '\n';
        return 1;
    }

    constexpr std::array EXPECTED = {
        0.514999986F, 0.700000048F,  1.379999995F, 0.089999951F, 0.243209749F, 0.126859829F,
        1.196354151F, -0.066927783F, 0.657077312F, 0.518348098F, 0.961920202F, 0.348947883F,
    };
    for (std::size_t index = 0; index < output.size(); ++index) {
        if (std::abs(output[index] - EXPECTED[index]) > 3.0e-5F ||
            std::abs(projected_output[index] - output[index]) > 3.0e-5F) {
            std::cerr << "attention mismatch at " << index << ": " << output[index] << " vs " << EXPECTED[index]
                      << '\n';
            return 1;
        }
    }
    for (std::size_t index = 0; index < decode_output.size(); ++index) {
        if (std::abs(decode_output[index] - output[index + 8]) > 3.0e-5F ||
            std::abs(decode_one_output[index] - decode_output[index]) > 3.0e-5F) {
            std::cerr << "decode attention mismatch at " << index << ": " << decode_output[index] << " vs "
                      << output[index + 8] << '\n';
            return 1;
        }
    }

    std::filesystem::remove(path);
    return 0;
}