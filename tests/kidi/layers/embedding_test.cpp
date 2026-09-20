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

auto near(float left, float right) -> bool { return std::abs(left - right) < 1.0e-5F; }

} // namespace

auto main() -> int {
    using namespace kidi;
    try {
        const auto path = std::filesystem::temp_directory_path() / "kidi-embedding-test.safetensors";
        write_weights(path);
        auto weights = ops::require(model::Weights::load(path));
        layers::Embedding embedding =
            std::make_shared<layers::EmbeddingImpl>(weights, "embedding", model::WeightEncoding::F32, 8);
        std::vector devices{tensor::Device::cpu()};
#if defined(__APPLE__)
        devices.push_back(tensor::Device::apple_gpu());
#endif
        const auto frequency = std::exp(-(std::log(10000.F) / 4.F) * 2.F);
        const std::array expected{2.F,
                                  5.F,
                                  6.F,
                                  9.F,
                                  -2.F + std::sin(1.F),
                                  -4.F + std::cos(1.F),
                                  -6.F + std::sin(frequency),
                                  -8.F + std::cos(frequency)};
        for (auto device : devices) {
            ops::Context context(device);
            auto output = embedding->forward(context, std::array<std::int32_t, 4>{1, 2, 2, 1}, 2);
            auto offset = embedding->forward(context, std::array<std::int32_t, 1>{2}, 1, 1);
            context.synchronize();
            auto values = ops::require(output.data<float>());
            auto selected = ops::require(offset.data<float>());
            for (std::size_t index = 0; index < expected.size(); ++index)
                if (!near(values[index], expected[index])) return 1;
            for (std::size_t index = 0; index < 4; ++index)
                if (!near(selected[index], expected[index + 4])) return 1;
            if (values[8] != -2.F || values[11] != -7.F) return 1;
            bool rejected = false;
            try {
                embedding->forward(context, std::array<std::int32_t, 1>{3}, 1);
            } catch (const ops::Failure&) {
                rejected = true;
            }
            if (!rejected) return 1;
        }
        std::filesystem::remove(path);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
