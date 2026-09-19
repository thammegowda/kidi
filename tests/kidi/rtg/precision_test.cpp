#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <ynnpack.h>

#include "kidi/model/precision.h"
#include "kidi/model/weights.h"
#include "kidi/rtg/transformer_builder.h"
#include "kidi/runtime/ynn.h"

namespace {

void write_bf16_weights(const std::filesystem::path& path) {
    std::string header =
        R"({"linear.weight":{"dtype":"BF16","shape":[4,3],"data_offsets":[0,24]},"linear.bias":{"dtype":"F32","shape":[3],"data_offsets":[24,36]}})";
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
}

bool near(float left, float right) { return std::abs(left - right) < 1.0e-5F; }

} // namespace

int main() {
    const auto path = std::filesystem::temp_directory_path() / "kidi-bf16-precision-test.safetensors";
    write_bf16_weights(path);
    auto weights = kidi::model::Weights::load(path);
    if (!weights) {
        std::cerr << weights.error().message << '\n';
        return 1;
    }

    auto graph = kidi::runtime::YnnGraph::create(2, YNN_FLAG_NO_EXCESS_PRECISION);
    if (!graph) {
        std::cerr << graph.error().message << '\n';
        return 1;
    }
    constexpr std::array<std::size_t, 3> INPUT_SHAPE = {1, 0, 4};
    std::uint32_t input_id = 0;
    std::uint32_t output_id = 1;
    auto status = kidi::runtime::check_ynn_status(
        ynn_define_tensor(graph->get(), ynn_type_fp32, INPUT_SHAPE.size(), INPUT_SHAPE.data(), nullptr,
                          YNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id),
        "define BF16 test input");
    if (status)
        status = kidi::runtime::check_ynn_status(ynn_define_tensor(graph->get(), ynn_type_fp32, 3, nullptr, nullptr,
                                                                   YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id),
                                                 "define BF16 test output");

    kidi::rtg::TransformerBuilder builder(graph->get(), *weights, 4, 8, 2, 1.0e-5F, kidi::model::WeightEncoding::BF16,
                                          kidi::model::LinearWeightLayout::INPUT_OUTPUT);
    auto result_id = status ? builder.linear(input_id, "linear", 4, 3, output_id)
                            : kidi::Result<std::uint32_t>{std::unexpected(std::move(status.error()))};
    if (!result_id) {
        std::cerr << result_id.error().message << '\n';
        return 1;
    }
    auto executable = std::move(*graph).compile();
    if (!executable) {
        std::cerr << executable.error().message << '\n';
        return 1;
    }

    constexpr std::array<std::size_t, 3> RUN_SHAPE = {1, 2, 4};
    std::array input = {1.0F, 2.0F, 3.0F, 4.0F, -1.0F, 0.5F, 2.0F, -0.5F};
    std::array<float, 6> output{};
    status = executable->set_shape(input_id, RUN_SHAPE);
    if (status) status = executable->reshape();
    if (status) status = executable->bind(input_id, input.data());
    if (status) status = executable->bind(output_id, output.data());
    if (status) status = executable->invoke();
    if (!status) {
        std::cerr << status.error().message << '\n';
        return 1;
    }

    constexpr std::array EXPECTED = {3.1F, 3.55F, 3.425F, 1.475F, 1.175F, -1.2F};
    for (std::size_t index = 0; index < output.size(); ++index) {
        if (!near(output[index], EXPECTED[index])) {
            std::cerr << "BF16 linear mismatch at " << index << ": " << output[index] << " vs " << EXPECTED[index]
                      << '\n';
            return 1;
        }
    }

    std::filesystem::remove(path);
    return 0;
}