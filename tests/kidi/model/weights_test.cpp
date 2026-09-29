#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "kidi/model/ggml.h"
#include "kidi/model/weights.h"

namespace {

auto write_import_fixture(const std::filesystem::path& path, std::uint32_t type, bool legacy = false)
    -> std::pair<std::array<float, 32>, std::streamoff> {
    std::ofstream output(path, std::ios::binary);
    const auto write = [&](auto value) { output.write(reinterpret_cast<const char*>(&value), sizeof(value)); };
    std::streamoff type_offset;
    if (legacy) {
        write(kidi::model::GGML_MAGIC);
        for (const std::int32_t value : {51865, 1500, 384, 6, 4, 448, 384, 6, 4, 80, 2007}) write(value);
        write(std::uint32_t{80});
        write(std::uint32_t{201});
        for (int index = 0; index < 80 * 201; ++index) write(0.F);
        write(std::uint32_t{1});
        write(std::uint32_t{1});
        output.put('a');
        const std::string name = "encoder.blocks.0.attn.query.weight";
        write(std::uint32_t{2});
        write(static_cast<std::uint32_t>(name.size()));
        type_offset = output.tellp();
        write(type);
        write(std::int32_t{32});
        write(std::int32_t{1});
        output.write(name.data(), name.size());
    } else {
        const auto string = [&](std::string_view value) {
            write(static_cast<std::uint64_t>(value.size()));
            output.write(value.data(), value.size());
        };
        write(kidi::model::GGUF_MAGIC);
        write(std::uint32_t{3});
        write(std::uint64_t{1});
        write(std::uint64_t{2});
        string("general.alignment");
        write(std::uint32_t{4});
        write(std::uint32_t{64});
        string("test.array");
        write(std::uint32_t{9});
        write(std::uint32_t{8});
        write(std::uint64_t{2});
        string("metadata");
        string("");
        string("weight");
        write(std::uint32_t{2});
        write(std::uint64_t{32});
        write(std::uint64_t{1});
        type_offset = output.tellp();
        write(type);
        write(std::uint64_t{0});
        while (static_cast<std::uint64_t>(output.tellp()) % 64) output.put('\0');
    }
    std::array<float, 32> expected;
    if (type == 0) {
        for (std::size_t index = 0; index < 32; ++index) write(expected[index] = static_cast<float>(index) - 16.F);
    } else if (type == 1 || type == 30) {
        const std::array<std::uint16_t, 4> bits{static_cast<std::uint16_t>(type == 1 ? 0x3c00 : 0x3f80), 0xc000, 1,
                                                0x8000};
        const std::array<float, 4> values{1.F, -2.F, std::ldexp(1.F, type == 1 ? -24 : -133), -0.F};
        for (std::size_t index = 0; index < 32; ++index) {
            write(bits[index % 4]);
            expected[index] = values[index % 4];
        }
    } else {
        write(std::uint16_t{0x3800});
        const float minimum = type == 3 || type == 7 ? 1.F : 0.F;
        if (minimum) write(std::uint16_t{0x3c00});
        if (type == 6 || type == 7) write(std::uint32_t{0xaaaa5555});
        if (type == 8) {
            for (int index = 0; index < 32; ++index) {
                write(static_cast<std::int8_t>(index - 16));
                expected[index] = (index - 16) * 0.5F;
            }
        } else {
            for (int index = 0; index < 16; ++index) write(static_cast<std::uint8_t>(index | ((15 - index) << 4)));
            for (std::size_t index = 0; index < 32; ++index) {
                int value = index < 16 ? static_cast<int>(index) : 31 - static_cast<int>(index);
                if (type == 6 || type == 7) value += ((0xaaaa5555U >> index) & 1U) * 16;
                if (type == 2) value -= 8;
                if (type == 6) value -= 16;
                expected[index] = value * 0.5F + minimum;
            }
        }
    }
    return {expected, type_offset};
}

auto write_fixture(const std::filesystem::path& path) -> void {
    std::string header = R"({"bf16":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},)"
                         R"("weight":{"dtype":"F32","shape":[2],"data_offsets":[4,12]},)"
                         R"("weight2":{"dtype":"F32","shape":[2],"data_offsets":[12,20]},)"
                         R"("int8":{"dtype":"I8","shape":[2],"data_offsets":[20,22]},)"
                         R"("e4m3":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[22,24]},)"
                         R"("e5m2":{"dtype":"F8_E5M2","shape":[2],"data_offsets":[24,26]}})";
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
    constexpr std::array VALUES2 = {3.0F, 4.0F};
    constexpr std::array<std::int8_t, 2> INT8 = {12, -7};
    constexpr std::array<std::uint8_t, 2> E4M3 = {0x3A, 0xC2};
    constexpr std::array<std::uint8_t, 2> E5M2 = {0x3D, 0xC1};
    output.write(reinterpret_cast<const char*>(BF16.data()), sizeof(BF16));
    output.write(reinterpret_cast<const char*>(VALUES.data()), sizeof(VALUES));
    output.write(reinterpret_cast<const char*>(VALUES2.data()), sizeof(VALUES2));
    output.write(reinterpret_cast<const char*>(INT8.data()), sizeof(INT8));
    output.write(reinterpret_cast<const char*>(E4M3.data()), sizeof(E4M3));
    output.write(reinterpret_cast<const char*>(E5M2.data()), sizeof(E5M2));
}

} // namespace

auto main(int argc, char** argv) -> int {
    if (argc == 3) {
        const auto weights = kidi::model::Weights::load(argv[1]);
        if (!weights) {
            std::cerr << weights.error().message << '\n';
            return 1;
        }
        const auto value = weights->tensor(argv[2]);
        if (!value) {
            std::cerr << value.error().message << '\n';
            return 1;
        }
        std::cout << "tensors=" << weights->size() << " values=" << value->numel() << '\n';
        for (const auto sample : value->data<float>()->first(std::min<std::size_t>(32, value->numel())))
            std::cout << sample << ' ';
        std::cout << '\n';
        return 0;
    }
    const auto gguf_path = std::filesystem::temp_directory_path() / "kidi-weights-test.gguf";
    for (const bool legacy : {false, true}) {
        for (const std::uint32_t type : {0, 1, 2, 3, 6, 7, 8, 30}) {
            if (legacy && type == 30) continue;
            const auto [expected, type_offset] = write_import_fixture(gguf_path, type, legacy);
            const std::string name = legacy ? "model.encoder.layers.0.self_attn.q_proj.weight" : "weight";
            auto imported = kidi::model::Weights::load(gguf_path);
            if (!imported || imported->names() != std::vector<std::string>{name}) return 1;
            const auto value = imported->tensor(name);
            const auto data =
                value ? value->data<float>() : kidi::Result<std::span<const float>>{std::unexpected(value.error())};
            if (!data || value->size(0) != 1 || value->size(1) != 32 || !std::ranges::equal(*data, expected)) return 1;
            const std::array concat = {kidi::model::StateMappingSpec{{"^" + name + "$", name}, "fused", 0}};
            auto mapped = kidi::model::Weights::load(gguf_path, concat);
            const auto fused =
                mapped ? mapped->tensor("fused") : kidi::Result<kidi::tensor::Tensor>{std::unexpected(mapped.error())};
            if (!fused || mapped->size() != 1 || mapped->contains(name) || fused->size(0) != 2 ||
                !std::ranges::equal(fused->data<float>()->first(32), expected) ||
                !std::ranges::equal(fused->data<float>()->subspan(32), expected))
                return 1;
            {
                std::fstream stream(gguf_path, std::ios::binary | std::ios::in | std::ios::out);
                stream.seekp(type_offset);
                const std::uint32_t unsupported = 999;
                stream.write(reinterpret_cast<const char*>(&unsupported), sizeof(unsupported));
            }
            if (kidi::model::Weights::load(gguf_path)) return 1;
        }
    }
    const auto overwrite = [&](std::streamoff offset, auto value) {
        std::fstream stream(gguf_path, std::ios::binary | std::ios::in | std::ios::out);
        stream.seekp(offset);
        stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
    };
    for (const std::uint64_t dimension : {std::uint64_t{0}, std::uint64_t{31}, UINT64_MAX}) {
        const auto [expected, type_offset] = write_import_fixture(gguf_path, 8);
        overwrite(type_offset - 16, dimension);
        if (kidi::model::Weights::load(gguf_path)) return 1;
    }
    {
        const auto [expected, type_offset] = write_import_fixture(gguf_path, 8);
        overwrite(type_offset + 4, std::uint64_t{1} << 63);
        if (kidi::model::Weights::load(gguf_path)) return 1;
    }
    {
        const auto [expected, type_offset] = write_import_fixture(gguf_path, 8);
        overwrite((type_offset + 12 + 63) / 64 * 64, std::uint16_t{0x7c00});
        auto imported = kidi::model::Weights::load(gguf_path);
        if (!imported || imported->tensor("weight")) return 1;
    }
    {
        const auto [expected, type_offset] = write_import_fixture(gguf_path, 8, true);
        const auto offset = type_offset - 8;
        std::vector<char> duplicate(std::filesystem::file_size(gguf_path) - offset);
        std::ifstream input(gguf_path, std::ios::binary);
        input.seekg(offset);
        input.read(duplicate.data(), duplicate.size());
        input.close();
        std::ofstream output(gguf_path, std::ios::binary | std::ios::app);
        output.write(duplicate.data(), duplicate.size());
        output.close();
        if (kidi::model::Weights::load(gguf_path)) return 1;
    }
    write_import_fixture(gguf_path, 8);
    std::filesystem::resize_file(gguf_path, std::filesystem::file_size(gguf_path) - 1);
    if (kidi::model::Weights::load(gguf_path)) return 1;
    std::filesystem::remove(gguf_path);
    const auto path = std::filesystem::temp_directory_path() / "kidi-weights-test.safetensors";
    write_fixture(path);

    auto weights = kidi::model::Weights::load(path);
    if (!weights || weights->size() != 6 || !weights->contains("weight")) {
        std::cerr << "failed to map Safetensors fixture\n";
        return 1;
    }
    auto tensor = weights->tensor("weight");
    auto values = tensor ? std::as_const(*tensor).data<float>()
                         : kidi::Result<std::span<const float>>{std::unexpected(tensor.error())};
    if (!tensor || !values || tensor->dtype() != kidi::model::DataType::F32 ||
        kidi::tensor::element_size(tensor->dtype()) != sizeof(float) || tensor->shape().size() != 1 ||
        tensor->shape()[0] != 2 || tensor->numel() != 2 || (*values)[0] != 1.25F || (*values)[1] != -2.5F ||
        tensor->data<float>().has_value()) {
        std::cerr << "mapped tensor view is incorrect\n";
        return 1;
    }
    const auto bf16 = weights->tensor("bf16");
    const auto int8 = weights->tensor("int8");
    const auto e4m3 = weights->tensor("e4m3");
    const auto e5m2 = weights->tensor("e5m2");
    if (!bf16 || bf16->dtype() != kidi::model::DataType::BF16 || kidi::tensor::element_size(bf16->dtype()) != 2 ||
        !int8 || int8->dtype() != kidi::model::DataType::I8 || !e4m3 || e4m3->dtype() != kidi::model::DataType::E4M3 ||
        !e5m2 || e5m2->dtype() != kidi::model::DataType::E5M2) {
        std::cerr << "Safetensors dtype mapping is incorrect\n";
        return 1;
    }

    const std::array mappings = {kidi::model::StateMappingSpec{
        .sources = {R"(^weight$)", "weight2"},
        .destination = "fused",
        .concat_axis = 0,
    }};
    auto mapped = kidi::model::Weights::load(path, mappings);
    auto fused = mapped ? mapped->tensor("fused") : kidi::Result<kidi::tensor::Tensor>{std::unexpected(mapped.error())};
    auto fused_values = fused ? std::as_const(*fused).data<float>()
                              : kidi::Result<std::span<const float>>{std::unexpected(fused.error())};
    if (!mapped || mapped->size() != 5 || mapped->contains("weight") || mapped->contains("weight2") || !fused ||
        !fused_values || fused->shape().size() != 1 || fused->shape()[0] != 4 || (*fused_values)[0] != 1.25F ||
        (*fused_values)[1] != -2.5F || (*fused_values)[2] != 3.0F || (*fused_values)[3] != 4.0F) {
        std::cerr << "state mapping fusion is incorrect\n";
        return 1;
    }

    const auto saved_path = std::filesystem::temp_directory_path() / "kidi-weights-roundtrip.safetensors";
    const auto state = weights->state_dict();
    if (!state || !kidi::model::Weights::save(saved_path, *state) || kidi::model::Weights::save(saved_path, *state)) {
        std::cerr << "checkpoint serialization or no-overwrite contract failed\n";
        return 1;
    }
    const auto reloaded = kidi::model::Weights::load(saved_path);
    if (!reloaded || reloaded->size() != state->size()) return 1;
    for (const auto& [name, expected] : *state) {
        const auto actual = reloaded->tensor(name);
        if (!actual || actual->dtype() != expected.dtype() || !std::ranges::equal(actual->shape(), expected.shape()))
            return 1;
        const auto expected_bytes = expected.host_bytes(), actual_bytes = actual->host_bytes();
        if (!expected_bytes || !actual_bytes || !std::ranges::equal(*expected_bytes, *actual_bytes)) return 1;
    }
    std::filesystem::remove(saved_path);
    std::filesystem::remove(path);
    return 0;
}