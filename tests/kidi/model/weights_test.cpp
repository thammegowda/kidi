#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "kidi/model/weights.h"

namespace {

void write_fixture(const std::filesystem::path& path) {
    std::string header = R"({"bf16":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},)"
                         R"("weight":{"dtype":"F32","shape":[2],"data_offsets":[4,12]},)"
                         R"("int8":{"dtype":"I8","shape":[2],"data_offsets":[12,14]},)"
                         R"("e4m3":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[14,16]},)"
                         R"("e5m2":{"dtype":"F8_E5M2","shape":[2],"data_offsets":[16,18]}})";
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
    constexpr std::array<std::uint16_t, 2> BF16 = {0x3FA0, 0xC020};
    constexpr std::array VALUES = {1.25F, -2.5F};
    constexpr std::array<std::int8_t, 2> INT8 = {12, -7};
    constexpr std::array<std::uint8_t, 2> E4M3 = {0x3A, 0xC2};
    constexpr std::array<std::uint8_t, 2> E5M2 = {0x3D, 0xC1};
    output.write(reinterpret_cast<const char*>(BF16.data()), sizeof(BF16));
    output.write(reinterpret_cast<const char*>(VALUES.data()), sizeof(VALUES));
    output.write(reinterpret_cast<const char*>(INT8.data()), sizeof(INT8));
    output.write(reinterpret_cast<const char*>(E4M3.data()), sizeof(E4M3));
    output.write(reinterpret_cast<const char*>(E5M2.data()), sizeof(E5M2));
}

} // namespace

int main() {
    const auto path = std::filesystem::temp_directory_path() / "kidi-weights-test.safetensors";
    write_fixture(path);

    auto weights = kidi::model::Weights::load(path);
    if (!weights || weights->size() != 5 || !weights->contains("weight")) {
        std::cerr << "failed to map Safetensors fixture\n";
        return 1;
    }
    auto tensor = weights->tensor("weight");
    const auto* values = tensor ? static_cast<const float*>(tensor->data()) : nullptr;
    if (!tensor || tensor->data_type != kidi::model::DataType::F32 || tensor->element_size() != sizeof(float) ||
        tensor->shape.size() != 1 || tensor->shape[0] != 2 || tensor->element_count() != 2 || values[0] != 1.25F ||
        values[1] != -2.5F) {
        std::cerr << "mapped tensor view is incorrect\n";
        return 1;
    }
    const auto bf16 = weights->tensor("bf16");
    const auto int8 = weights->tensor("int8");
    const auto e4m3 = weights->tensor("e4m3");
    const auto e5m2 = weights->tensor("e5m2");
    if (!bf16 || bf16->data_type != kidi::model::DataType::BF16 || bf16->element_size() != 2 || !int8 ||
        int8->data_type != kidi::model::DataType::I8 || !e4m3 || e4m3->data_type != kidi::model::DataType::E4M3 ||
        !e5m2 || e5m2->data_type != kidi::model::DataType::E5M2) {
        std::cerr << "Safetensors dtype mapping is incorrect\n";
        return 1;
    }

    std::filesystem::remove(path);
    return 0;
}