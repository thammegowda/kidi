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
        R"({"attn.linears.0.weight":{"dtype":"F32","shape":[4,4],"data_offsets":[0,64]},"attn.linears.0.bias":{"dtype":"F32","shape":[4],"data_offsets":[64,80]},"attn.linears.1.weight":{"dtype":"F32","shape":[4,4],"data_offsets":[80,144]},"attn.linears.1.bias":{"dtype":"F32","shape":[4],"data_offsets":[144,160]},"attn.linears.2.weight":{"dtype":"F32","shape":[4,4],"data_offsets":[160,224]},"attn.linears.2.bias":{"dtype":"F32","shape":[4],"data_offsets":[224,240]},"attn.linears.3.weight":{"dtype":"F32","shape":[4,4],"data_offsets":[240,304]},"attn.linears.3.bias":{"dtype":"F32","shape":[4],"data_offsets":[304,320]}})";
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
    write_values(QUERY_WEIGHT);
    write_values(QUERY_BIAS);
    write_values(KEY_WEIGHT);
    write_values(KEY_BIAS);
    write_values(VALUE_WEIGHT);
    write_values(VALUE_BIAS);
    write_values(OUTPUT_WEIGHT);
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

    auto graph = kidi::runtime::YnnGraph::create(3);
    if (!graph) {
        std::cerr << graph.error().message << '\n';
        return 1;
    }
    constexpr std::array<std::size_t, 3> INPUT_SHAPE = {1, 0, 4};
    constexpr std::array<std::size_t, 4> MASK_SHAPE = {1, 1, 0, 0};
    std::uint32_t input_id = 0;
    std::uint32_t mask_id = 1;
    std::uint32_t output_id = 2;
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

    kidi::rtg::TransformerBuilder builder(graph->get(), *weights, 4, 8, 2, 1.0e-5F);
    auto result_id = status
                         ? builder.attention(input_id, input_id, input_id, mask_id, "attn", output_id)
                         : std::expected<std::uint32_t, kidi::core::Error>{std::unexpected(std::move(status.error()))};
    if (!result_id) {
        std::cerr << result_id.error().message << '\n';
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
    status = executable->set_shape(input_id, RUN_INPUT_SHAPE);
    if (status) status = executable->set_shape(mask_id, RUN_MASK_SHAPE);
    if (status) status = executable->reshape();
    if (status) status = executable->bind(input_id, input.data());
    if (status) status = executable->bind(mask_id, const_cast<float*>(mask.data()));
    if (status) status = executable->bind(output_id, output.data());
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
        if (std::abs(output[index] - EXPECTED[index]) > 3.0e-5F) {
            std::cerr << "attention mismatch at " << index << ": " << output[index] << " vs " << EXPECTED[index]
                      << '\n';
            return 1;
        }
    }

    std::filesystem::remove(path);
    return 0;
}