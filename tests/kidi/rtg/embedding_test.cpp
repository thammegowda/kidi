#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "kidi/model/weights.h"
#include "kidi/rtg/embedding.h"

namespace {

void write_weights(const std::filesystem::path& path) {
    std::string header = R"({"embedding":{"dtype":"F32","shape":[3,4],"data_offsets":[0,48]}})";
    while (header.size() % 8 != 0) header.push_back(' ');
    std::ofstream output(path, std::ios::binary);
    std::uint64_t header_size = header.size();
    if constexpr (std::endian::native == std::endian::big) header_size = std::byteswap(header_size);
    output.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    constexpr std::array VALUES = {
        0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, -1.0F, -2.0F, -3.0F, -4.0F,
    };
    output.write(reinterpret_cast<const char*>(VALUES.data()), sizeof(VALUES));
}

bool near(float left, float right) { return std::abs(left - right) < 1.0e-5F; }

} // namespace

int main() {
    const auto path = std::filesystem::temp_directory_path() / "kidi-embedding-test.safetensors";
    write_weights(path);
    auto weights = kidi::model::Weights::load(path);
    auto graph = weights ? kidi::rtg::EmbeddingGraph::create(*weights, "embedding", 3, 4)
                         : kidi::Result<kidi::rtg::EmbeddingGraph>{std::unexpected(weights.error())};
    if (!graph) {
        std::cerr << graph.error().message << '\n';
        return 1;
    }

    constexpr std::array<std::int32_t, 2> IDS = {1, 2};
    auto output = graph->run(IDS);
    if (!output || output->size() != 8) {
        std::cerr << "embedding graph failed: " << (output ? "wrong output size" : output.error().message) << '\n';
        return 1;
    }
    const auto frequency = std::exp(-(std::log(10000.0F) / 4.0F) * 2.0F);
    const std::array expected = {
        2.0F,
        5.0F,
        6.0F,
        9.0F,
        -2.0F + std::sin(1.0F),
        -4.0F + std::cos(1.0F),
        -6.0F + std::sin(frequency),
        -8.0F + std::cos(frequency),
    };
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (!near((*output)[index], expected[index])) {
            std::cerr << "embedding mismatch at " << index << ": " << (*output)[index] << " vs " << expected[index]
                      << '\n';
            return 1;
        }
    }

    constexpr std::array<std::int32_t, 1> OFFSET_ID = {2};
    auto offset_output = graph->run(OFFSET_ID, 1, 1);
    if (!offset_output || offset_output->size() != 4 ||
        !std::equal(offset_output->begin(), offset_output->end(), output->begin() + 4,
                    [](float left, float right) { return near(left, right); })) {
        std::cerr << "offset embedding mismatch\n";
        return 1;
    }

    constexpr std::array<std::int32_t, 4> BATCH_IDS = {1, 2, 2, 1};
    auto batch_output = graph->run(BATCH_IDS, 2);
    if (!batch_output || batch_output->size() != 16) {
        std::cerr << "batched embedding graph failed\n";
        return 1;
    }
    const std::array expected_second_batch = {
        -2.0F,
        -3.0F,
        -6.0F,
        -7.0F,
        2.0F + std::sin(1.0F),
        4.0F + std::cos(1.0F),
        6.0F + std::sin(frequency),
        8.0F + std::cos(frequency),
    };
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (!near((*batch_output)[index], expected[index]) ||
            !near((*batch_output)[index + expected.size()], expected_second_batch[index])) {
            std::cerr << "batched embedding mismatch at " << index << '\n';
            return 1;
        }
    }
    std::filesystem::remove(path);
    return 0;
}