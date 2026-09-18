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
        R"({"norm.weight":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},"norm.bias":{"dtype":"F32","shape":[4],"data_offsets":[16,32]},"ff.w_1.weight":{"dtype":"F32","shape":[3,4],"data_offsets":[32,80]},"ff.w_1.bias":{"dtype":"F32","shape":[3],"data_offsets":[80,92]},"ff.w_2.weight":{"dtype":"F32","shape":[4,3],"data_offsets":[92,140]},"ff.w_2.bias":{"dtype":"F32","shape":[4],"data_offsets":[140,156]}})";
    while (header.size() % 8 != 0) header.push_back(' ');

    std::ofstream output(path, std::ios::binary);
    std::uint64_t header_size = header.size();
    if constexpr (std::endian::native == std::endian::big) header_size = std::byteswap(header_size);
    output.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));

    constexpr std::array NORMALIZATION_WEIGHT = {1.5F, 0.5F, 2.0F, -1.0F};
    constexpr std::array NORMALIZATION_BIAS = {0.1F, -0.2F, 0.3F, 0.4F};
    constexpr std::array FIRST_WEIGHT = {
        0.25F, -0.5F, 1.0F, 0.75F, -1.0F, 0.5F, 0.125F, 0.25F, 0.6F, -0.2F, -0.4F, 0.9F,
    };
    constexpr std::array FIRST_BIAS = {0.2F, -0.1F, 0.3F};
    constexpr std::array SECOND_WEIGHT = {
        0.3F, -0.7F, 0.2F, 1.0F, 0.1F, -0.5F, -0.4F, 0.8F, 0.6F, 0.25F, 0.5F, -1.0F,
    };
    constexpr std::array SECOND_BIAS = {-0.2F, 0.4F, 0.05F, -0.3F};
    const auto write_values = [&output](const auto& values) {
        output.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(values.size() * sizeof(values[0])));
    };
    write_values(NORMALIZATION_WEIGHT);
    write_values(NORMALIZATION_BIAS);
    write_values(FIRST_WEIGHT);
    write_values(FIRST_BIAS);
    write_values(SECOND_WEIGHT);
    write_values(SECOND_BIAS);
}

bool near(float left, float right) { return std::abs(left - right) < 2.0e-5F; }

} // namespace

int main() {
    const auto path = std::filesystem::temp_directory_path() / "kidi-transformer-builder-test.safetensors";
    write_weights(path);
    auto weights = kidi::model::Weights::load(path);
    if (!weights) {
        std::cerr << weights.error().message << '\n';
        return 1;
    }

    auto graph = kidi::runtime::YnnGraph::create(5);
    if (!graph) {
        std::cerr << graph.error().message << '\n';
        return 1;
    }
    constexpr std::array<std::size_t, 3> INPUT_SHAPE = {1, 0, 4};
    std::uint32_t input_id = 0;
    std::uint32_t normalized_output_id = 1;
    std::uint32_t linear_output_id = 2;
    std::uint32_t gelu_output_id = 3;
    std::uint32_t output_id = 4;
    auto status = kidi::runtime::check_ynn_status(
        ynn_define_tensor(graph->get(), ynn_type_fp32, INPUT_SHAPE.size(), INPUT_SHAPE.data(), nullptr,
                          YNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id),
        "define input");
    if (status)
        status = kidi::runtime::check_ynn_status(
            ynn_define_tensor(graph->get(), ynn_type_fp32, INPUT_SHAPE.size(), nullptr, nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &normalized_output_id),
            "define normalized output");
    if (status)
        status = kidi::runtime::check_ynn_status(
            ynn_define_tensor(graph->get(), ynn_type_fp32, INPUT_SHAPE.size(), nullptr, nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &linear_output_id),
            "define linear output");
    if (status)
        status =
            kidi::runtime::check_ynn_status(ynn_define_tensor(graph->get(), ynn_type_fp32, INPUT_SHAPE.size(), nullptr,
                                                              nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &gelu_output_id),
                                            "define GELU output");
    if (status)
        status =
            kidi::runtime::check_ynn_status(ynn_define_tensor(graph->get(), ynn_type_fp32, INPUT_SHAPE.size(), nullptr,
                                                              nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id),
                                            "define feed-forward output");

    kidi::rtg::TransformerBuilder builder(graph->get(), *weights, 4, 3, 2, 1.0e-5F);
    auto normalized_id = status ? builder.layer_norm(input_id, "norm", normalized_output_id)
                                : kidi::Result<std::uint32_t>{std::unexpected(std::move(status.error()))};
    auto linear_id = normalized_id ? builder.linear(*normalized_id, "ff.w_1", 4, 3, linear_output_id)
                                   : kidi::Result<std::uint32_t>{std::unexpected(std::move(normalized_id.error()))};
    auto gelu_id = linear_id ? builder.gelu(*linear_id, gelu_output_id)
                             : kidi::Result<std::uint32_t>{std::unexpected(std::move(linear_id.error()))};
    auto result_id = gelu_id ? builder.linear(*gelu_id, "ff.w_2", 3, 4, output_id)
                             : kidi::Result<std::uint32_t>{std::unexpected(std::move(gelu_id.error()))};
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
    std::array input = {-1.0F, 0.5F, 2.0F, 3.5F, 4.0F, -2.0F, 1.0F, 0.25F};
    std::array<float, 8> normalized_output{};
    std::array<float, 6> linear_output{};
    std::array<float, 6> gelu_output{};
    std::array<float, 8> output{};
    status = executable->set_shape(input_id, RUN_SHAPE);
    if (status) status = executable->reshape();
    if (status) status = executable->bind(input_id, input.data());
    if (status) status = executable->bind(normalized_output_id, normalized_output.data());
    if (status) status = executable->bind(linear_output_id, linear_output.data());
    if (status) status = executable->bind(gelu_output_id, gelu_output.data());
    if (status) status = executable->bind(output_id, output.data());
    if (status) status = executable->invoke();
    if (!status) {
        std::cerr << status.error().message << '\n';
        return 1;
    }

    constexpr std::array EXPECTED_NORMALIZED = {
        -1.9124613F, -0.4236068F, 1.1944273F, -0.9416408F, 2.3279426F, -0.8552770F, 0.4747319F, 0.6621111F,
    };
    constexpr std::array EXPECTED_LINEAR = {
        0.4218847F, 1.5145510F, -2.0880032F, 2.1809392F, -2.6307118F, 2.2738280F,
    };
    constexpr std::array EXPECTED_GELU = {
        0.279898256F, 1.416187644F, -0.038417075F, 2.149120331F, -0.011207701F, 2.247702360F,
    };
    constexpr std::array EXPECTED_OUTPUT = {
        -1.115045309F, 0.840725541F, 1.047940612F, 0.516485453F,
        0.902121961F,  1.424148321F, 0.530007124F, -2.016026020F,
    };
    bool matches = true;
    for (std::size_t index = 0; index < normalized_output.size(); ++index) {
        if (!near(normalized_output[index], EXPECTED_NORMALIZED[index])) {
            std::cerr << "layer norm mismatch at " << index << ": " << normalized_output[index] << " vs "
                      << EXPECTED_NORMALIZED[index] << '\n';
            matches = false;
        }
    }
    for (std::size_t index = 0; index < linear_output.size(); ++index) {
        if (!near(linear_output[index], EXPECTED_LINEAR[index])) {
            std::cerr << "linear mismatch at " << index << ": " << linear_output[index] << " vs "
                      << EXPECTED_LINEAR[index] << '\n';
            matches = false;
        }
        if (!near(gelu_output[index], EXPECTED_GELU[index])) {
            std::cerr << "GELU mismatch at " << index << ": " << gelu_output[index] << " vs " << EXPECTED_GELU[index]
                      << '\n';
            matches = false;
        }
    }
    for (std::size_t index = 0; index < output.size(); ++index) {
        if (!near(output[index], EXPECTED_OUTPUT[index])) {
            std::cerr << "transformer primitive mismatch at " << index << ": " << output[index] << " vs "
                      << EXPECTED_OUTPUT[index] << '\n';
            matches = false;
        }
    }
    if (!matches) return 1;

    std::filesystem::remove(path);
    return 0;
}