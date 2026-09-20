#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <ynnpack.h>

#include "kidi/model/weights.h"

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

auto check(ynn_status status, std::string_view operation) -> bool {
    if (status == ynn_status_success) {
        return true;
    }
    std::cerr << operation << " failed with YNNPACK status " << status << '\n';
    return false;
}

} // namespace

auto main() -> int {
    const auto path = std::filesystem::temp_directory_path() / "kidi-ynnpack-weights.safetensors";
    write_weights(path);
    auto weights = kidi::model::Weights::load(path);
    auto weight =
        weights ? weights->tensor("weight") : kidi::Result<kidi::tensor::Tensor>{std::unexpected(weights.error())};
    if (!weight) {
        std::cerr << "failed to load mapped weight\n";
        return 1;
    }
    const auto& weight_tensor = *weight;
    auto weight_bytes = weight_tensor.host_bytes();
    if (!weight_bytes) {
        std::cerr << weight_bytes.error().message << '\n';
        return 1;
    }

    ynn_subgraph_t subgraph = nullptr;
    if (!check(ynn_create_subgraph(2, 0, &subgraph), "create subgraph")) {
        return 1;
    }

    constexpr std::array<std::size_t, 2> INPUT_SHAPE = {1, 2};
    constexpr std::array<std::size_t, 2> WEIGHT_SHAPE = {2, 2};
    std::uint32_t input_id = YNN_INVALID_VALUE_ID;
    std::uint32_t output_id = YNN_INVALID_VALUE_ID;
    std::uint32_t weight_id = YNN_INVALID_VALUE_ID;
    std::uint32_t transposed_weight_id = YNN_INVALID_VALUE_ID;
    constexpr std::array<std::int32_t, 2> TRANSPOSE_AXES = {1, 0};
    bool ok = check(ynn_define_tensor(subgraph, ynn_type_fp32, INPUT_SHAPE.size(), INPUT_SHAPE.data(), nullptr,
                                      YNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id),
                    "define input") &&
              check(ynn_define_tensor(subgraph, ynn_type_fp32, INPUT_SHAPE.size(), INPUT_SHAPE.data(), nullptr,
                                      YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id),
                    "define output") &&
              check(ynn_define_tensor(subgraph, ynn_type_fp32, WEIGHT_SHAPE.size(), WEIGHT_SHAPE.data(),
                                      weight_bytes->data(), 0, &weight_id),
                    "define mapped weight") &&
              check(ynn_define_static_transpose(subgraph, TRANSPOSE_AXES.size(), TRANSPOSE_AXES.data(), weight_id,
                                                &transposed_weight_id, 0),
                    "transpose PyTorch weight") &&
              check(ynn_define_dot(subgraph, 1, input_id, transposed_weight_id, YNN_INVALID_VALUE_ID, &output_id, 0),
                    "define dot") &&
              check(ynn_optimize_subgraph(subgraph, nullptr, 0), "optimize graph");
    if (!ok) {
        ynn_delete_subgraph(subgraph);
        return 1;
    }

    ynn_runtime_t runtime = nullptr;
    if (!check(ynn_create_runtime(subgraph, nullptr, YNN_RUNTIME_FLAG_NO_SCHEDULE, &runtime), "create runtime")) {
        ynn_delete_subgraph(subgraph);
        return 1;
    }
    std::array input = {4.0F, 5.0F};
    std::array output = {0.0F, 0.0F};
    ok = check(ynn_reshape_runtime(runtime), "reshape runtime") &&
         check(ynn_set_external_value_data(runtime, input_id, input.data()), "bind input") &&
         check(ynn_set_external_value_data(runtime, output_id, output.data()), "bind output") &&
         check(ynn_invoke_runtime(runtime), "invoke runtime");

    ynn_delete_runtime(runtime);
    ynn_delete_subgraph(subgraph);
    std::filesystem::remove(path);

    if (!ok || output != std::array{14.0F, 32.0F}) {
        std::cerr << "unexpected YNNPACK output: " << output[0] << ", " << output[1] << '\n';
        return 1;
    }
    return 0;
}