#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "kidi/checkpoint/ggml/reader.h"
#include "kidi/checkpoint/weights.h"

namespace {

auto write_import_fixture(const std::filesystem::path& path, std::uint32_t type, bool legacy = false)
    -> std::pair<std::array<float, 32>, std::streamoff> {
    std::ofstream output(path, std::ios::binary);
    const auto write = [&](auto value) { output.write(reinterpret_cast<const char*>(&value), sizeof(value)); };
    std::streamoff type_offset;
    if (legacy) {
        write(kidi::checkpoint::ggml::GGML_MAGIC);
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
        write(kidi::checkpoint::ggml::GGUF_MAGIC);
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
                         R"("e5m2":{"dtype":"F8_E5M2","shape":[2],"data_offsets":[24,26]},)"
                         R"("f16":{"dtype":"F16","shape":[4],"data_offsets":[26,34]}})";
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
    const std::array<std::uint16_t, 4> f16{0x3c00, 0xc000, 1, 0x8000};
    output.write(reinterpret_cast<const char*>(f16.data()), sizeof(f16));
}

auto temp_path(std::string_view stem, std::string_view extension) -> std::filesystem::path {
    return std::filesystem::temp_directory_path() /
           (std::string(stem) + "-" + ::testing::UnitTest::GetInstance()->current_test_info()->name() +
            std::string(extension));
}

auto overwrite(const std::filesystem::path& path, std::streamoff offset, auto value) -> void {
    std::fstream stream(path, std::ios::binary | std::ios::in | std::ios::out);
    stream.seekp(offset);
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

// Diagnostic: set KIDI_TEST_WEIGHTS_FILE (checkpoint path) and KIDI_TEST_WEIGHTS_TENSOR (tensor name) to dump a tensor.
TEST(WeightsTest, DumpsTensorFromEnvironment) {
    const char* file = std::getenv("KIDI_TEST_WEIGHTS_FILE");
    const char* name = std::getenv("KIDI_TEST_WEIGHTS_TENSOR");
    if (!file || !name) GTEST_SKIP() << "KIDI_TEST_WEIGHTS_FILE and KIDI_TEST_WEIGHTS_TENSOR are not set";
    const auto weights = kidi::checkpoint::Weights::load(file);
    ASSERT_TRUE(weights.has_value()) << weights.error().message;
    const auto value = weights->tensor(name);
    ASSERT_TRUE(value.has_value()) << value.error().message;
    std::cout << "tensors=" << weights->size() << " values=" << value->numel() << '\n';
    for (const auto sample : value->data<float>()->first(std::min<std::size_t>(32, value->numel())))
        std::cout << sample << ' ';
    std::cout << '\n';
}

TEST(WeightsTest, ImportsGgmlAndGguf) {
    const auto gguf_path = temp_path("kidi-weights-test", ".gguf");
    for (const bool legacy : {false, true}) {
        for (const std::uint32_t type : {0, 1, 2, 3, 6, 7, 8, 30}) {
            if (legacy && type == 30) continue;
            SCOPED_TRACE(::testing::Message() << "legacy=" << legacy << " type=" << type);
            const auto [expected, type_offset] = write_import_fixture(gguf_path, type, legacy);
            const std::string name = legacy ? "model.encoder.layers.0.self_attn.q_proj.weight" : "weight";
            auto imported = kidi::checkpoint::Weights::load(gguf_path);
            ASSERT_TRUE(imported.has_value()) << imported.error().message;
            EXPECT_EQ(imported->names(), std::vector<std::string>{name});
            const auto value = imported->tensor(name);
            ASSERT_TRUE(value.has_value()) << value.error().message;
            const auto data = value->data<float>();
            ASSERT_TRUE(data.has_value()) << data.error().message;
            EXPECT_EQ(value->size(0), 1);
            EXPECT_EQ(value->size(1), 32);
            EXPECT_TRUE(std::ranges::equal(*data, expected));
            const std::array concat = {kidi::checkpoint::StateMappingSpec{{"^" + name + "$", name}, "fused", 0}};
            auto mapped = kidi::checkpoint::Weights::load(gguf_path, concat);
            ASSERT_TRUE(mapped.has_value()) << mapped.error().message;
            const auto fused = mapped->tensor("fused");
            ASSERT_TRUE(fused.has_value()) << fused.error().message;
            EXPECT_EQ(mapped->size(), 1U);
            EXPECT_FALSE(mapped->contains(name));
            ASSERT_EQ(fused->size(0), 2);
            EXPECT_TRUE(std::ranges::equal(fused->data<float>()->first(32), expected));
            EXPECT_TRUE(std::ranges::equal(fused->data<float>()->subspan(32), expected));
            overwrite(gguf_path, type_offset, std::uint32_t{999});
            EXPECT_FALSE(kidi::checkpoint::Weights::load(gguf_path)) << "unsupported tensor type was accepted";
        }
    }
    std::filesystem::remove(gguf_path);
}

TEST(WeightsTest, RejectsCorruptGguf) {
    const auto gguf_path = temp_path("kidi-weights-test", ".gguf");
    for (const std::uint64_t dimension : {std::uint64_t{0}, std::uint64_t{31}, UINT64_MAX}) {
        const auto [expected, type_offset] = write_import_fixture(gguf_path, 8);
        overwrite(gguf_path, type_offset - 16, dimension);
        EXPECT_FALSE(kidi::checkpoint::Weights::load(gguf_path)) << "dimension=" << dimension;
    }
    {
        const auto [expected, type_offset] = write_import_fixture(gguf_path, 8);
        overwrite(gguf_path, type_offset + 4, std::uint64_t{1} << 63);
        EXPECT_FALSE(kidi::checkpoint::Weights::load(gguf_path));
    }
    {
        const auto [expected, type_offset] = write_import_fixture(gguf_path, 8);
        overwrite(gguf_path, (type_offset + 12 + 63) / 64 * 64, std::uint16_t{0x7c00});
        auto imported = kidi::checkpoint::Weights::load(gguf_path);
        ASSERT_TRUE(imported.has_value()) << imported.error().message;
        EXPECT_FALSE(imported->tensor("weight"));
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
        EXPECT_FALSE(kidi::checkpoint::Weights::load(gguf_path)) << "duplicate tensor was accepted";
    }
    write_import_fixture(gguf_path, 8);
    std::filesystem::resize_file(gguf_path, std::filesystem::file_size(gguf_path) - 1);
    EXPECT_FALSE(kidi::checkpoint::Weights::load(gguf_path)) << "truncated file was accepted";
    std::filesystem::remove(gguf_path);
}

class SafetensorsTest : public ::testing::Test {
protected:
    void SetUp() override {
        path = temp_path("kidi-weights-test", ".safetensors");
        write_fixture(path);
        auto loaded = kidi::checkpoint::Weights::load(path);
        ASSERT_TRUE(loaded.has_value()) << "failed to map Safetensors fixture: " << loaded.error().message;
        weights.emplace(std::move(*loaded));
        ASSERT_EQ(weights->size(), 7U) << "failed to map Safetensors fixture";
        ASSERT_TRUE(weights->contains("weight")) << "failed to map Safetensors fixture";
        auto loaded_tensor = weights->tensor("weight");
        ASSERT_TRUE(loaded_tensor.has_value()) << "mapped tensor view is incorrect: " << loaded_tensor.error().message;
        tensor.emplace(std::move(*loaded_tensor));
    }

    void TearDown() override {
        tensor.reset();
        weights.reset();
        std::error_code error;
        std::filesystem::remove(path, error);
    }

    const std::array<kidi::checkpoint::StateMappingSpec, 1> mappings = {kidi::checkpoint::StateMappingSpec{
        .sources = {R"(^weight$)", "weight2"},
        .destination = "fused",
        .concat_axis = 0,
    }};
    std::filesystem::path path;
    std::optional<kidi::checkpoint::Weights> weights;
    std::optional<kidi::tensor::Tensor> tensor;
};

TEST_F(SafetensorsTest, MapsTensorViews) {
    auto values = std::as_const(*tensor).data<float>();
    ASSERT_TRUE(values.has_value()) << "mapped tensor view is incorrect";
    EXPECT_EQ(tensor->dtype(), kidi::checkpoint::DataType::F32) << "mapped tensor view is incorrect";
    EXPECT_EQ(kidi::tensor::element_size(tensor->dtype()), sizeof(float)) << "mapped tensor view is incorrect";
    ASSERT_EQ(tensor->shape().size(), 1U) << "mapped tensor view is incorrect";
    EXPECT_EQ(tensor->shape()[0], 2) << "mapped tensor view is incorrect";
    EXPECT_EQ(tensor->numel(), 2U) << "mapped tensor view is incorrect";
    EXPECT_EQ((*values)[0], 1.25F) << "mapped tensor view is incorrect";
    EXPECT_EQ((*values)[1], -2.5F) << "mapped tensor view is incorrect";
    EXPECT_FALSE(tensor->data<float>().has_value()) << "mapped tensor view is incorrect";
}

TEST_F(SafetensorsTest, MapsDataTypes) {
    const auto bf16 = weights->tensor("bf16");
    const auto int8 = weights->tensor("int8");
    const auto e4m3 = weights->tensor("e4m3");
    const auto e5m2 = weights->tensor("e5m2");
    ASSERT_TRUE(bf16.has_value()) << "Safetensors dtype mapping is incorrect";
    EXPECT_EQ(bf16->dtype(), kidi::checkpoint::DataType::BF16) << "Safetensors dtype mapping is incorrect";
    EXPECT_EQ(kidi::tensor::element_size(bf16->dtype()), 2U) << "Safetensors dtype mapping is incorrect";
    ASSERT_TRUE(int8.has_value()) << "Safetensors dtype mapping is incorrect";
    EXPECT_EQ(int8->dtype(), kidi::checkpoint::DataType::I8) << "Safetensors dtype mapping is incorrect";
    ASSERT_TRUE(e4m3.has_value()) << "Safetensors dtype mapping is incorrect";
    EXPECT_EQ(e4m3->dtype(), kidi::checkpoint::DataType::E4M3) << "Safetensors dtype mapping is incorrect";
    ASSERT_TRUE(e5m2.has_value()) << "Safetensors dtype mapping is incorrect";
    EXPECT_EQ(e5m2->dtype(), kidi::checkpoint::DataType::E5M2) << "Safetensors dtype mapping is incorrect";
}

TEST_F(SafetensorsTest, ExpandsToFloat32) {
    const auto expanded = weights->float32_tensor("f16");
    const auto expanded_bf16 = weights->float32_tensor("bf16");
    const auto unchanged = weights->float32_tensor("weight");
    ASSERT_TRUE(expanded.has_value()) << "checkpoint FP32 expansion or zero-copy contract failed";
    ASSERT_TRUE(expanded_bf16.has_value()) << "checkpoint FP32 expansion or zero-copy contract failed";
    ASSERT_TRUE(unchanged.has_value()) << "checkpoint FP32 expansion or zero-copy contract failed";
    EXPECT_EQ(unchanged->host_bytes()->data(), std::as_const(*tensor).host_bytes()->data())
        << "checkpoint FP32 expansion or zero-copy contract failed";
    EXPECT_FALSE(weights->float32_tensor("int8")) << "checkpoint FP32 expansion or zero-copy contract failed";
    EXPECT_FALSE(weights->float32_tensor("missing")) << "checkpoint FP32 expansion or zero-copy contract failed";
    const auto half_values = expanded->data<float>(), bf16_values = expanded_bf16->data<float>();
    ASSERT_TRUE(half_values.has_value()) << "FP16/BF16 expansion produced incorrect values";
    EXPECT_EQ((*half_values)[0], 1.F) << "FP16/BF16 expansion produced incorrect values";
    EXPECT_EQ((*half_values)[1], -2.F) << "FP16/BF16 expansion produced incorrect values";
    EXPECT_EQ((*half_values)[2], std::ldexp(1.F, -24)) << "FP16/BF16 expansion produced incorrect values";
    EXPECT_TRUE(std::signbit((*half_values)[3])) << "FP16/BF16 expansion produced incorrect values";
    ASSERT_TRUE(bf16_values.has_value()) << "FP16/BF16 expansion produced incorrect values";
    EXPECT_EQ((*bf16_values)[0], 1.25F) << "FP16/BF16 expansion produced incorrect values";
    EXPECT_EQ((*bf16_values)[1], -2.5F) << "FP16/BF16 expansion produced incorrect values";
}

TEST_F(SafetensorsTest, FusesStateMappings) {
    auto mapped = kidi::checkpoint::Weights::load(path, mappings);
    ASSERT_TRUE(mapped.has_value()) << "state mapping fusion is incorrect: " << mapped.error().message;
    auto fused = mapped->tensor("fused");
    ASSERT_TRUE(fused.has_value()) << "state mapping fusion is incorrect: " << fused.error().message;
    auto fused_values = std::as_const(*fused).data<float>();
    EXPECT_EQ(mapped->size(), 6U) << "state mapping fusion is incorrect";
    EXPECT_FALSE(mapped->contains("weight")) << "state mapping fusion is incorrect";
    EXPECT_FALSE(mapped->contains("weight2")) << "state mapping fusion is incorrect";
    ASSERT_TRUE(fused_values.has_value()) << "state mapping fusion is incorrect";
    ASSERT_EQ(fused->shape().size(), 1U) << "state mapping fusion is incorrect";
    ASSERT_EQ(fused->shape()[0], 4) << "state mapping fusion is incorrect";
    EXPECT_EQ((*fused_values)[0], 1.25F) << "state mapping fusion is incorrect";
    EXPECT_EQ((*fused_values)[1], -2.5F) << "state mapping fusion is incorrect";
    EXPECT_EQ((*fused_values)[2], 3.0F) << "state mapping fusion is incorrect";
    EXPECT_EQ((*fused_values)[3], 4.0F) << "state mapping fusion is incorrect";
    const auto zero_copy_mapped = weights->mapped(mappings);
    ASSERT_TRUE(zero_copy_mapped.has_value()) << "adjacent state mapping fusion was not zero-copy";
    auto zero_copy_fused = zero_copy_mapped->tensor("fused");
    ASSERT_TRUE(zero_copy_fused.has_value()) << "adjacent state mapping fusion was not zero-copy";
    EXPECT_FALSE(zero_copy_fused->data<float>().has_value()) << "adjacent state mapping fusion was not zero-copy";
    EXPECT_EQ(std::as_const(*zero_copy_fused).host_bytes()->data(), std::as_const(*tensor).host_bytes()->data())
        << "adjacent state mapping fusion was not zero-copy";
}

TEST_F(SafetensorsTest, AppliesMappingRules) {
    using kidi::checkpoint::StateMappingOp;
    using kidi::checkpoint::StateMappingSpec;
    const std::array rules = {
        StateMappingSpec{.sources = {R"(^e[45]m.*$)"}, .op = StateMappingOp::IGNORE},
        StateMappingSpec{
            .sources = {R"(^weight$)", "weight"}, .destination = "same_scale", .op = StateMappingOp::IDENTICAL},
        StateMappingSpec{.sources = {"0:weight2", R"(^weight2$)"}, .destination = "zero_prefixed"},
    };
    const auto view = weights->mapped(rules);
    ASSERT_TRUE(view.has_value()) << "mapped IGNORE/IDENTICAL/zero-reference rules failed: " << view.error().message;
    const auto zero_prefixed = view->tensor("zero_prefixed");
    ASSERT_TRUE(zero_prefixed.has_value()) << "mapped IGNORE/IDENTICAL/zero-reference rules failed";
    const auto zp_values = std::as_const(*zero_prefixed).data<float>();
    EXPECT_FALSE(view->contains("e4m3")) << "mapped IGNORE/IDENTICAL/zero-reference rules failed";
    EXPECT_FALSE(view->contains("e5m2")) << "mapped IGNORE/IDENTICAL/zero-reference rules failed";
    EXPECT_TRUE(view->contains("same_scale")) << "mapped IGNORE/IDENTICAL/zero-reference rules failed";
    ASSERT_TRUE(zp_values.has_value()) << "mapped IGNORE/IDENTICAL/zero-reference rules failed";
    ASSERT_EQ(zp_values->size(), 4U) << "mapped IGNORE/IDENTICAL/zero-reference rules failed";
    EXPECT_EQ((*zp_values)[0], 0.F) << "mapped IGNORE/IDENTICAL/zero-reference rules failed";
    EXPECT_EQ((*zp_values)[1], 0.F) << "mapped IGNORE/IDENTICAL/zero-reference rules failed";
    EXPECT_EQ((*zp_values)[2], 3.0F) << "mapped IGNORE/IDENTICAL/zero-reference rules failed";
    EXPECT_EQ((*zp_values)[3], 4.0F) << "mapped IGNORE/IDENTICAL/zero-reference rules failed";
    const std::array mismatch = {StateMappingSpec{
        .sources = {R"(^weight$)", "weight2"}, .destination = "bad_scale", .op = StateMappingOp::IDENTICAL}};
    EXPECT_FALSE(weights->mapped(mismatch)) << "mismatched IDENTICAL parameters were accepted";
    const auto prefixed =
        kidi::checkpoint::prefix_state_mappings("llm.", "llm.", std::span<const StateMappingSpec>(mappings));
    ASSERT_EQ(prefixed.size(), 1U) << "prefix_state_mappings rewrote patterns incorrectly";
    EXPECT_EQ(prefixed[0].sources[0], R"(^llm\.weight$)") << "prefix_state_mappings rewrote patterns incorrectly";
    EXPECT_EQ(prefixed[0].sources[1], "llm.weight2") << "prefix_state_mappings rewrote patterns incorrectly";
    EXPECT_EQ(prefixed[0].destination, "llm.fused") << "prefix_state_mappings rewrote patterns incorrectly";
}

TEST_F(SafetensorsTest, SavesAndReloads) {
    const auto saved_path = temp_path("kidi-weights-roundtrip", ".safetensors");
    std::filesystem::remove(saved_path);
    const auto state = weights->state_dict();
    ASSERT_TRUE(state.has_value()) << "checkpoint serialization or no-overwrite contract failed";
    ASSERT_TRUE(kidi::checkpoint::Weights::save(saved_path, *state))
        << "checkpoint serialization or no-overwrite contract failed";
    EXPECT_FALSE(kidi::checkpoint::Weights::save(saved_path, *state))
        << "checkpoint serialization or no-overwrite contract failed";
    const auto reloaded = kidi::checkpoint::Weights::load(saved_path);
    ASSERT_TRUE(reloaded.has_value()) << reloaded.error().message;
    ASSERT_EQ(reloaded->size(), state->size());
    for (const auto& [name, expected] : *state) {
        SCOPED_TRACE(name);
        const auto actual = reloaded->tensor(name);
        ASSERT_TRUE(actual.has_value()) << actual.error().message;
        EXPECT_EQ(actual->dtype(), expected.dtype());
        EXPECT_TRUE(std::ranges::equal(actual->shape(), expected.shape()));
        const auto expected_bytes = expected.host_bytes(), actual_bytes = actual->host_bytes();
        ASSERT_TRUE(expected_bytes.has_value());
        ASSERT_TRUE(actual_bytes.has_value());
        EXPECT_TRUE(std::ranges::equal(*expected_bytes, *actual_bytes));
    }
    std::filesystem::remove(saved_path);
}

} // namespace
