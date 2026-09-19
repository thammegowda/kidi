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
#include "kidi/rtg/embedding.h"
#include "kidi/rtg/transformer_builder.h"
#include "kidi/runtime/ynn.h"

namespace {

void write_int8_weights(const std::filesystem::path& path) {
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

void write_int8_embeddings(const std::filesystem::path& path) {
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

int main() {
    const auto path = std::filesystem::temp_directory_path() / "kidi-int8-precision-test.safetensors";
    write_int8_weights(path);
    auto weights = kidi::model::Weights::load(path);
    if (!weights) {
        std::cerr << weights.error().message << '\n';
        return 1;
    }

    auto graph = kidi::runtime::YnnGraph::create(2);
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
        "define INT8 test input");
    if (status)
        status = kidi::runtime::check_ynn_status(ynn_define_tensor(graph->get(), ynn_type_fp32, 3, nullptr, nullptr,
                                                                   YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id),
                                                 "define INT8 test output");

    kidi::rtg::TransformerBuilder builder(graph->get(), *weights, 4, 8, 2, 1.0e-5F,
                                          kidi::model::WeightEncoding::INT8_PER_CHANNEL);
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
    std::array input = {-1.0F, 0.0F, 1.0F, 2.0F, -2.0F, -1.0F, 1.0F, 2.0F};
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

    constexpr std::array EXPECTED = {0.6F, 2.55F, 1.675F, 0.35F, 3.05F, 1.3F};
    for (std::size_t index = 0; index < output.size(); ++index) {
        if (std::abs(output[index] - EXPECTED[index]) > 0.04F) {
            std::cerr << "INT8 linear mismatch at " << index << ": " << output[index] << " vs " << EXPECTED[index]
                      << '\n';
            return 1;
        }
    }

    std::filesystem::remove(path);

    const auto embedding_path = std::filesystem::temp_directory_path() / "kidi-int8-embedding-test.safetensors";
    write_int8_embeddings(embedding_path);
    auto embedding_weights = kidi::model::Weights::load(embedding_path);
    auto embedding = embedding_weights
                         ? kidi::rtg::EmbeddingGraph::create(*embedding_weights, "embedding", 3, 4,
                                                             kidi::model::WeightEncoding::INT8_PER_CHANNEL)
                         : kidi::Result<kidi::rtg::EmbeddingGraph>{std::unexpected(embedding_weights.error())};
    if (!embedding) {
        std::cerr << embedding.error().message << '\n';
        return 1;
    }
    constexpr std::array<std::int32_t, 2> TOKEN_IDS = {1, 2};
    auto embedded = embedding->run(TOKEN_IDS);
    if (!embedded || embedded->size() != 8) {
        std::cerr << (embedded ? "INT8 embedding output has wrong size" : embedded.error().message) << '\n';
        return 1;
    }
    const auto frequency = std::exp(-(std::log(10000.0F) / 4.0F) * 2.0F);
    const std::array expected_embedding = {
        2.0F,
        5.0F,
        6.0F,
        9.0F,
        -2.0F + std::sin(1.0F),
        -4.0F + std::cos(1.0F),
        -6.0F + std::sin(frequency),
        -8.0F + std::cos(frequency),
    };
    for (std::size_t index = 0; index < embedded->size(); ++index) {
        if (std::abs((*embedded)[index] - expected_embedding[index]) > 1.0e-5F) {
            std::cerr << "INT8 embedding mismatch at " << index << '\n';
            return 1;
        }
    }
    std::filesystem::remove(embedding_path);
    return 0;
}