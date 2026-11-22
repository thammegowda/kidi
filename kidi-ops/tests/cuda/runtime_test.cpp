#include "cuda/runtime.h"
#include "cuda/launch.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
#if defined(KIDI_CUDA_USE_CUTLASS_INT8)
constexpr bool USE_INT8_GEMM = true;
#else
constexpr bool USE_INT8_GEMM = false;
#endif

void check_bfloat_bias(int device) {
    using namespace kidi::cuda;
    using Buffer = std::unique_ptr<void, decltype(&release)>;
    const std::uint16_t value = 0x3f81;
    const float offset = 0.00390625F;
    Buffer input(allocate(sizeof(value), device), release), weight(allocate(sizeof(value), device), release),
        bias(allocate(sizeof(offset), device), release), output(allocate(sizeof(value), device), release);
    copy_to_device(input.get(), &value, sizeof(value), device);
    copy_to_device(weight.get(), &value, sizeof(value), device);
    copy_to_device(bias.get(), &offset, sizeof(offset), device);
    const std::array<Tensor, 3> inputs{Tensor{input.get(), Type::BF16, 2, {1, 1}, device},
                                       Tensor{weight.get(), Type::BF16, 2, {1, 1}, device},
                                       Tensor{bias.get(), Type::F32, 1, {1}, device}};
    Descriptor descriptor;
    descriptor.operation = Op::LINEAR;
    descriptor.input_count = 3;
    std::copy(inputs.begin(), inputs.end(), descriptor.inputs.begin());
    descriptor.output.type = Type::BF16;
    Program program(descriptor, device);
    program.run(inputs.data(), 3, output.get());
    std::uint16_t actual = 0;
    copy_to_host(&actual, output.get(), sizeof(actual), device);
    if (actual != 0x3f83) throw std::runtime_error("CUDA BF16 linear rounded before applying its bias");
}

void check_integer_layout(int device) {
    using namespace kidi::cuda;
    using Buffer = std::unique_ptr<void, decltype(&release)>;
    const std::array<std::int32_t, 6> values{16777217, -16777219, 2147483647, -2147483647, 123456789, -123456789};
    const std::array<std::int32_t, 2> rows{1, 0};
    Buffer input(allocate(sizeof(values), device), release), indices(allocate(sizeof(rows), device), release),
        output(allocate(2 * sizeof(values), device), release);
    copy_to_device(input.get(), values.data(), sizeof(values), device);
    copy_to_device(indices.get(), rows.data(), sizeof(rows), device);
    const Tensor source{input.get(), Type::I32, 2, {2, 3}, device};
    for (const auto operation : {Op::TRANSPOSE, Op::GATHER, Op::CONCAT}) {
        std::array<Tensor, 2> inputs{
            source, operation == Op::GATHER ? Tensor{indices.get(), Type::I32, 1, {2}, device} : source};
        Descriptor descriptor;
        descriptor.operation = operation;
        descriptor.input_count = operation == Op::TRANSPOSE ? 1 : 2;
        std::copy(inputs.begin(), inputs.end(), descriptor.inputs.begin());
        descriptor.attribute_count = operation == Op::TRANSPOSE ? 2 : 1;
        descriptor.attributes =
            operation == Op::TRANSPOSE ? std::array<std::int64_t, 32>{1, 0} : std::array<std::int64_t, 32>{0};
        Program program(descriptor, device);
        program.run(inputs.data(), descriptor.input_count, output.get());
        std::vector<std::int32_t> actual(program.output().elements());
        copy_to_host(actual.data(), output.get(), actual.size() * sizeof(std::int32_t), device);
        for (std::size_t index = 0; index < actual.size(); ++index) {
            const auto original = operation == Op::TRANSPOSE ? (index % 2) * 3 + index / 2
                                  : operation == Op::GATHER  ? (1 - index / 3) * 3 + index % 3
                                                             : index % values.size();
            if (actual[index] != values[original])
                throw std::runtime_error("CUDA integer layout lost storage precision");
        }
    }
}

void check_gelu(int device) {
    using namespace kidi::cuda;
    using Buffer = std::unique_ptr<void, decltype(&release)>;
    const std::array<float, 8> values{-5.F, -3.F, -1.F, -0.5F, 0.5F, 1.F, 3.F, 5.F};
    Buffer input(allocate(sizeof(values), device), release), output(allocate(sizeof(values), device), release);
    copy_to_device(input.get(), values.data(), sizeof(values), device);
    const Tensor operand{input.get(), Type::F32, 1, {8}, device};
    for (const bool approximate : {false, true}) {
        Descriptor descriptor;
        descriptor.operation = Op::GELU;
        descriptor.input_count = 1;
        descriptor.inputs[0] = operand;
        descriptor.attribute_count = 1;
        descriptor.attributes[0] = approximate;
        Program program(descriptor, device);
        program.run(&operand, 1, output.get());
        std::array<float, 8> actual{};
        copy_to_host(actual.data(), output.get(), sizeof(actual), device);
        for (std::size_t index = 0; index < values.size(); ++index) {
            const auto value = values[index];
            const float probability = approximate
                                          ? std::tanh(0.7978845608028654F * (value + 0.044715F * value * value * value))
                                          : std::erf(value * 0.7071067811865475F);
            const float expected = 0.5F * value * (1.F + probability);
            if (std::abs(actual[index] - expected) > 2e-6F)
                throw std::runtime_error("CUDA GELU ignored the requested approximation mode");
        }
    }
}

void check_scalar_and_empty(int device) {
    using namespace kidi::cuda;
    using Buffer = std::unique_ptr<void, decltype(&release)>;
    const float scalar = 2.F;
    const std::array<float, 3> values{3.F, 4.F, 5.F};
    Buffer scalar_buffer(allocate(sizeof(scalar), device), release),
        vector_buffer(allocate(sizeof(values), device), release), output(allocate(6 * sizeof(float), device), release);
    copy_to_device(scalar_buffer.get(), &scalar, sizeof(scalar), device);
    copy_to_device(vector_buffer.get(), values.data(), sizeof(values), device);
    for (const bool broadcast : {false, true}) {
        std::array<Tensor, 2> inputs{Tensor{scalar_buffer.get(), Type::F32, 0, {}, device},
                                     broadcast ? Tensor{vector_buffer.get(), Type::F32, 1, {3}, device}
                                               : Tensor{scalar_buffer.get(), Type::F32, 0, {}, device}};
        Descriptor descriptor;
        descriptor.operation = Op::ADD;
        descriptor.input_count = 2;
        std::copy(inputs.begin(), inputs.end(), descriptor.inputs.begin());
        Program program(descriptor, device);
        program.run(inputs.data(), 2, output.get());
        std::array<float, 3> actual{};
        copy_to_host(actual.data(), output.get(), program.output().elements() * sizeof(float), device);
        if (broadcast ? actual != std::array<float, 3>{5.F, 6.F, 7.F} : actual[0] != 4.F)
            throw std::runtime_error("CUDA scalar preparation or broadcasting failed");
    }
    const std::array<Tensor, 2> inputs{Tensor{nullptr, Type::F32, 2, {2, 0}, device},
                                       Tensor{nullptr, Type::F32, 2, {0, 3}, device}};
    Descriptor descriptor;
    descriptor.operation = Op::MATMUL;
    descriptor.input_count = 2;
    std::copy(inputs.begin(), inputs.end(), descriptor.inputs.begin());
    Program program(descriptor, device);
    program.run(inputs.data(), 2, output.get());
    std::array<float, 6> actual{};
    copy_to_host(actual.data(), output.get(), sizeof(actual), device);
    if (actual != std::array<float, 6>{}) throw std::runtime_error("CUDA zero-width matrix produced nonzero output");
}

void check_greedy_selection(int device) {
    using namespace kidi::cuda;
    using Buffer = std::unique_ptr<void, decltype(&release)>;
    for (const std::int64_t width : {17, 4096, 4097}) {
        std::vector<float> scores(4 * width, -INFINITY);
        scores[1] = scores[width - 1] = 3.F;
        scores[2 * width] = 1.F;
        scores[3 * width - 1] = NAN;
        scores[3 * width] = 1.F;
        scores[4 * width - 1] = INFINITY;
        Buffer input(allocate(scores.size() * sizeof(float), device), release),
            output(allocate(4 * sizeof(std::int32_t), device), release);
        copy_to_device(input.get(), scores.data(), scores.size() * sizeof(float), device);
        const Tensor operand{input.get(), Type::F32, 2, {4, width}, device};
        Descriptor descriptor;
        descriptor.operation = Op::GREEDY_TOKEN;
        descriptor.input_count = 1;
        descriptor.inputs[0] = operand;
        Program program(descriptor, device);
        program.run(&operand, 1, output.get());
        std::array<std::int32_t, 4> actual{};
        copy_to_host(actual.data(), output.get(), sizeof(actual), device);
        if (actual != std::array<std::int32_t, 4>{1, -1, -1, -1})
            throw std::runtime_error("CUDA greedy selection accepted invalid scores or changed tie-breaking");
    }
}

void check_slice(int device) {
    using namespace kidi::cuda;
    using Buffer = std::unique_ptr<void, decltype(&release)>;
    std::array<std::int32_t, 2 * 3 * 16> values{};
    for (std::size_t index = 0; index < values.size(); ++index)
        values[index] = 16777217 + static_cast<std::int32_t>(index * 2);
    Buffer input(allocate(sizeof(values), device), release), output(allocate(sizeof(values), device), release);
    copy_to_device(input.get(), values.data(), sizeof(values), device);
    const Tensor source{input.get(), Type::I32, 3, {2, 3, 16}, device};
    for (const int axis : {0, 1, -1}) {
        const auto normalized = axis < 0 ? axis + source.rank : axis;
        const auto length = source.shape[normalized] - 1;
        Descriptor descriptor;
        descriptor.operation = Op::SLICE;
        descriptor.input_count = 1;
        descriptor.inputs[0] = source;
        descriptor.attribute_count = 3;
        descriptor.attributes = {axis, 1, length};
        Program program(descriptor, device);
        program.run(&source, 1, output.get());
        std::vector<std::int32_t> actual(program.output().elements());
        copy_to_host(actual.data(), output.get(), actual.size() * sizeof(actual[0]), device);
        std::size_t index = 0;
        for (int batch = 0; batch < 2; ++batch)
            for (int row = 0; row < 3; ++row)
                for (int column = 0; column < 16; ++column) {
                    const std::array coordinates{batch, row, column};
                    if (coordinates[normalized] < 1) continue;
                    if (actual[index++] != values[(batch * 3 + row) * 16 + column])
                        throw std::runtime_error("CUDA pitched slice did not preserve exact storage");
                }
    }
}

template <typename Element = std::uint16_t, typename CacheElement = Element>
void check_attention(int device, std::int64_t keys = 19, bool long_decode_only = false) {
    using namespace kidi::cuda;
    using Buffer = std::unique_ptr<void, decltype(&release)>;
    constexpr std::int64_t batches = 2, heads = 4, offset = 3;
    constexpr bool bfloat = std::is_same_v<Element, std::uint16_t>;
    constexpr bool bfloat_cache = std::is_same_v<CacheElement, std::uint16_t>;
    constexpr Type type = bfloat ? Type::BF16 : Type::F32;
    constexpr Type cache_type = bfloat_cache ? Type::BF16 : Type::F32;
    const auto capacity = keys + 5;
    for (const std::int64_t queries : {1, 17})
        for (const std::int64_t width : {32, 64, 128, 256, 512})
            for (const std::int64_t key_heads : {1, 2, 4})
                for (const bool shared : {false, true}) {
                    if (long_decode_only &&
                        (queries != 1 || (width != 256 && width != 512) || key_heads != 2 || shared))
                        continue;
                    const auto key_batches = shared ? 1 : batches;
                    std::vector<Element> query(batches * queries * heads * width);
                    std::vector<CacheElement> key(key_batches * capacity * key_heads * width), value(key.size());
                    const auto fill = [](auto& values, int multiplier) {
                        for (std::size_t index = 0; index < values.size(); ++index) {
                            const float number =
                                static_cast<float>(static_cast<int>(index * multiplier % 31) - 15) * 0.015625F;
                            if constexpr (std::is_same_v<typename std::decay_t<decltype(values)>::value_type,
                                                         std::uint16_t>) {
                                std::uint32_t bits = 0;
                                std::memcpy(&bits, &number, sizeof(bits));
                                values[index] = static_cast<std::uint16_t>(bits >> 16);
                            } else {
                                values[index] = number;
                            }
                        }
                    };
                    fill(query, 7);
                    if constexpr (!bfloat && bfloat_cache)
                        for (auto& number : query) number += 0.00017F;
                    fill(key, 11);
                    fill(value, 13);
                    std::vector<float> mask(queries * keys);
                    for (std::int64_t row = 0; row < queries; ++row)
                        for (std::int64_t column = 0; column < keys; ++column)
                            mask[row * keys + column] = column > row + 1 ? -INFINITY : (column % 3 ? 0.F : 0.125F);
                    Buffer query_buffer(allocate(query.size() * sizeof(Element), device), release),
                        key_buffer(allocate(key.size() * sizeof(CacheElement), device), release),
                        value_buffer(allocate(value.size() * sizeof(CacheElement), device), release),
                        mask_buffer(allocate(mask.size() * sizeof(float), device), release),
                        output(allocate(query.size() * sizeof(Element), device), release);
                    copy_to_device(query_buffer.get(), query.data(), query.size() * sizeof(Element), device);
                    copy_to_device(key_buffer.get(), key.data(), key.size() * sizeof(CacheElement), device);
                    copy_to_device(value_buffer.get(), value.data(), value.size() * sizeof(CacheElement), device);
                    copy_to_device(mask_buffer.get(), mask.data(), mask.size() * sizeof(float), device);
                    std::array<Tensor, 4> inputs{
                        Tensor{query_buffer.get(), type, 3, {batches, queries, heads * width}, device},
                        Tensor{key_buffer.get(), cache_type, 3, {key_batches, capacity, key_heads * width}, device},
                        Tensor{value_buffer.get(), cache_type, 3, {key_batches, capacity, key_heads * width}, device},
                        Tensor{mask_buffer.get(), Type::F32, 4, {1, 1, queries, keys}, device}};
                    for (const float scale : {0.F, 1.F})
                        for (const bool masked : {false, true}) {
                            Descriptor descriptor;
                            descriptor.operation = Op::ATTENTION;
                            descriptor.input_count = masked ? 4 : 3;
                            std::copy(inputs.begin(), inputs.end(), descriptor.inputs.begin());
                            descriptor.attribute_count = 4;
                            descriptor.attributes = {heads, key_heads, offset, keys};
                            descriptor.epsilon = scale;
                            descriptor.reference_attention = true;
                            Program reference(descriptor, device);
                            reference.run(inputs.data(), descriptor.input_count, output.get());
                            std::vector<Element> expected(query.size()), actual(query.size());
                            copy_to_host(expected.data(), output.get(), expected.size() * sizeof(Element), device);
                            descriptor.reference_attention = false;
                            const auto previous_bytes = allocated_bytes(device);
                            Program native(descriptor, device);
                            if (queries > 1 && (bfloat_cache || USE_INT8_GEMM) &&
                                allocated_bytes(device) != previous_bytes)
                                throw std::runtime_error("CUDA fused attention allocated a score buffer");
                            native.run(inputs.data(), descriptor.input_count, output.get());
                            copy_to_host(actual.data(), output.get(), actual.size() * sizeof(Element), device);
                            for (std::size_t index = 0; index < actual.size(); ++index) {
                                const auto decode = [](Element rounded) -> float {
                                    if constexpr (bfloat) {
                                        const std::uint32_t bits = static_cast<std::uint32_t>(rounded) << 16;
                                        float number = 0;
                                        std::memcpy(&number, &bits, sizeof(number));
                                        return number;
                                    } else {
                                        return rounded;
                                    }
                                };
                                const auto target = decode(expected[index]), result = decode(actual[index]);
                                const auto probability_rounding = queries > 1 ? (15.F * 0.015625F) / 256.F : 0.F;
                                const auto tolerance =
                                    bfloat || (bfloat_cache && queries > 1)
                                        ? probability_rounding + 0.0001F + 0.008F * std::abs(target)
                                        : 0.000002F;
                                if (!std::isfinite(result) || std::abs(result - target) > tolerance) {
                                    std::cerr << "attention mismatch queries=" << queries << " width=" << width
                                              << " key_heads=" << key_heads << " shared=" << shared
                                              << " masked=" << masked << " scale=" << scale << " index=" << index
                                              << " expected=" << target << " actual=" << result << '\n';
                                    throw std::runtime_error("CUDA attention differs from scalar reference");
                                }
                            }
                        }
                }
}

template <typename CacheElement>
void check_nonbroadcast_causal_attention(int device, bool per_key) {
    using namespace kidi::cuda;
    using Buffer = std::unique_ptr<void, decltype(&release)>;
    constexpr int heads = 2, queries = 2, keys = 1024, width = 256;
    constexpr bool bfloat_cache = std::is_same_v<CacheElement, std::uint16_t>;
    constexpr Type cache_type = bfloat_cache ? Type::BF16 : Type::F32;
    std::vector<float> query(queries * heads * width, 0.F);
    std::vector<CacheElement> key(keys * width, 0), value(keys * width, 0);
    const CacheElement one = bfloat_cache ? static_cast<CacheElement>(0x3f80) : static_cast<CacheElement>(1.F);
    std::fill(value.begin() + 512 * width, value.end(), one);
    const int mask_keys = per_key ? keys : 1;
    const int mask_heads = per_key ? 1 : heads;
    std::vector<std::int32_t> positions(mask_heads * queries * mask_keys);
    for (int head = 0; head < mask_heads; ++head)
        for (int row = 0; row < queries; ++row)
            for (int column = 0; column < mask_keys; ++column)
                positions[(head * queries + row) * mask_keys + column] =
                    per_key ? (column < 512 ? 100 : 900) + row : (head == 0 ? 100 : 900) + row;
    Buffer query_buffer(allocate(query.size() * sizeof(float), device), release),
        key_buffer(allocate(key.size() * sizeof(CacheElement), device), release),
        value_buffer(allocate(value.size() * sizeof(CacheElement), device), release),
        positions_buffer(allocate(positions.size() * sizeof(std::int32_t), device), release),
        output(allocate(query.size() * sizeof(float), device), release);
    copy_to_device(query_buffer.get(), query.data(), query.size() * sizeof(float), device);
    copy_to_device(key_buffer.get(), key.data(), key.size() * sizeof(CacheElement), device);
    copy_to_device(value_buffer.get(), value.data(), value.size() * sizeof(CacheElement), device);
    copy_to_device(positions_buffer.get(), positions.data(), positions.size() * sizeof(std::int32_t), device);
    const std::array<Tensor, 4> inputs{
        Tensor{query_buffer.get(), Type::F32, 3, {1, queries, heads * width}, device},
        Tensor{key_buffer.get(), cache_type, 3, {1, keys, width}, device},
        Tensor{value_buffer.get(), cache_type, 3, {1, keys, width}, device},
        Tensor{positions_buffer.get(), Type::I32, 4, {1, mask_heads, queries, mask_keys}, device}};
    Descriptor descriptor;
    descriptor.operation = Op::ATTENTION;
    descriptor.input_count = 4;
    std::copy(inputs.begin(), inputs.end(), descriptor.inputs.begin());
    descriptor.attribute_count = 7;
    descriptor.attributes = {heads, 1, 0, keys, 0, 1, 0};
    Program program(descriptor, device);
    program.run(inputs.data(), 4, output.get());
    std::vector<float> actual(query.size());
    copy_to_host(actual.data(), output.get(), actual.size() * sizeof(float), device);
    for (int row = 0; row < queries; ++row)
        for (int head = 0; head < heads; ++head) {
            int allowed = 0, ones = 0;
            for (int column = 0; column < keys; ++column)
                if (column <= positions[((per_key ? 0 : head) * queries + row) * mask_keys +
                                        (per_key ? column : 0)]) {
                    ++allowed;
                    if (column >= 512) ++ones;
                }
            const float expected = static_cast<float>(ones) / allowed;
            for (int col = 0; col < width; ++col) {
                const auto result = actual[(row * heads + head) * width + col];
                if (!std::isfinite(result) || std::abs(result - expected) > (bfloat_cache ? 0.002F : 1e-4F))
                    throw std::runtime_error("CUDA nonbroadcast causal attention differs from reference");
            }
        }
}

// Prefill-sized causal and sliding-window attention with position masks, including a later chunk whose queries start
// partway into the cache, against the scalar reference.
void check_structured_prefill_attention(int device) {
    using namespace kidi::cuda;
    using Buffer = std::unique_ptr<void, decltype(&release)>;
    constexpr std::int64_t keys = 300;
    struct Case {
        std::int64_t first, window, offset = 0, batches = 1, heads = 8, key_heads = 2;
        bool bfloat_io = false, masked = true;
    };
    const Case cases[]{{0, 0}, {0, 128}, {190, 0}, {190, 128},
                       {0, 0, 64, 2, 6, 2, true, false},
                       {190, 128, 64, 2, 6, 2, true},
                       {-65, 128, 64, 2}};
    const auto bfloat = [](float number) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &number, sizeof(bits));
        return static_cast<std::uint16_t>(bits >> 16);
    };
    for (const std::int64_t width : {256, 512})
        for (const auto& test : cases) {
                const auto [first, window, offset, batches, heads, key_heads, bfloat_io, masked] = test;
                const auto queries = keys - first;
                std::vector<float> query(batches * queries * heads * width);
                std::vector<std::uint16_t> key(batches * keys * key_heads * width), value(key.size()),
                    bfloat_query(query.size());
                for (std::size_t index = 0; index < query.size(); ++index)
                    query[index] = static_cast<float>(static_cast<int>(index * 7 % 61) - 30) * 0.03125F;
                for (std::size_t index = 0; index < query.size(); ++index) bfloat_query[index] = bfloat(query[index]);
                for (std::size_t index = 0; index < key.size(); ++index) {
                    key[index] = bfloat(static_cast<float>(static_cast<int>(index * 11 % 37) - 18) * 0.0625F);
                    value[index] = bfloat(static_cast<float>(static_cast<int>(index * 13 % 31) - 15) * 0.0625F);
                }
                std::vector<std::int32_t> positions(batches * queries);
                for (std::int64_t row = 0; row < batches * queries; ++row)
                    positions[row] = static_cast<std::int32_t>(first + row % queries);
                const auto query_bytes = query.size() * (bfloat_io ? sizeof(std::uint16_t) : sizeof(float));
                Buffer query_buffer(allocate(query.size() * sizeof(float), device), release),
                    key_buffer(allocate(key.size() * sizeof(std::uint16_t), device), release),
                    value_buffer(allocate(value.size() * sizeof(std::uint16_t), device), release),
                    positions_buffer(allocate(positions.size() * sizeof(std::int32_t), device), release),
                    output(allocate(query.size() * sizeof(float), device), release);
                copy_to_device(query_buffer.get(), bfloat_io ? static_cast<const void*>(bfloat_query.data())
                                                             : static_cast<const void*>(query.data()),
                               query_bytes, device);
                copy_to_device(key_buffer.get(), key.data(), key.size() * sizeof(std::uint16_t), device);
                copy_to_device(value_buffer.get(), value.data(), value.size() * sizeof(std::uint16_t), device);
                copy_to_device(positions_buffer.get(), positions.data(), positions.size() * sizeof(std::int32_t),
                               device);
                const std::array<Tensor, 4> inputs{
                    Tensor{query_buffer.get(), bfloat_io ? Type::BF16 : Type::F32, 3,
                           {batches, queries, heads * width}, device},
                    Tensor{key_buffer.get(), Type::BF16, 3, {batches, keys, key_heads * width}, device},
                    Tensor{value_buffer.get(), Type::BF16, 3, {batches, keys, key_heads * width}, device},
                    Tensor{positions_buffer.get(), Type::I32, 4, {batches, 1, queries, 1}, device}};
                Descriptor descriptor;
                descriptor.operation = Op::ATTENTION;
                descriptor.input_count = masked ? 4 : 3;
                std::copy(inputs.begin(), inputs.end(), descriptor.inputs.begin());
                descriptor.attribute_count = 7;
                descriptor.attributes = {heads, key_heads, offset, keys - offset, 0, 1, window};
                descriptor.epsilon = 0.0625F;
                std::vector<float> expected(query.size()), actual(query.size());
                const auto read_output = [&](std::vector<float>& values) {
                    if (!bfloat_io) {
                        copy_to_host(values.data(), output.get(), values.size() * sizeof(float), device);
                        return;
                    }
                    std::vector<std::uint16_t> rounded(values.size());
                    copy_to_host(rounded.data(), output.get(), rounded.size() * sizeof(std::uint16_t), device);
                    for (std::size_t index = 0; index < values.size(); ++index) {
                        const std::uint32_t bits = static_cast<std::uint32_t>(rounded[index]) << 16;
                        std::memcpy(&values[index], &bits, sizeof(float));
                    }
                };
                descriptor.reference_attention = true;
                Program(descriptor, device).run(inputs.data(), descriptor.input_count, output.get());
                read_output(expected);
                descriptor.reference_attention = false;
                Program(descriptor, device).run(inputs.data(), descriptor.input_count, output.get());
                read_output(actual);
                for (std::size_t index = 0; index < actual.size(); ++index) {
                    // The scalar reference's all-masked softmax is undefined; fused attention returns zero.
                    if (masked && positions[index / (heads * width)] < offset) expected[index] = 0.F;
                    if (!std::isfinite(actual[index]) ||
                        std::abs(actual[index] - expected[index]) > 0.004F + 0.008F * std::abs(expected[index])) {
                        std::cerr << "structured attention mismatch width=" << width << " first=" << first
                                  << " window=" << window << " index=" << index << " expected=" << expected[index]
                                  << " actual=" << actual[index] << '\n';
                        throw std::runtime_error("CUDA structured prefill attention differs from reference");
                    }
                }
            }
}

void check_bfloat_matrix(int device, int rows, int width, int columns = 33, bool allow_split_workspace = false) {
    using namespace kidi::cuda;
    using Buffer = std::unique_ptr<void, decltype(&release)>;
    std::vector<std::uint16_t> values(rows * width), weights(columns * width);
    for (std::size_t index = 0; index < values.size(); ++index) values[index] = index % 2 ? 0x3f80 : 0xbf80;
    for (std::size_t index = 0; index < weights.size(); ++index) weights[index] = index % 3 ? 0x3f00 : 0x3f80;
    Buffer input(allocate(values.size() * sizeof(std::uint16_t), device), release),
        weight(allocate(weights.size() * sizeof(std::uint16_t), device), release),
        output(allocate(rows * columns * sizeof(std::uint16_t), device), release);
    copy_to_device(input.get(), values.data(), values.size() * sizeof(std::uint16_t), device);
    copy_to_device(weight.get(), weights.data(), weights.size() * sizeof(std::uint16_t), device);
    for (const bool transpose : {false, true}) {
        std::array<Tensor, 2> inputs{
            Tensor{input.get(), Type::BF16, 2, {rows, width}, device},
            Tensor{weight.get(), Type::BF16, 2, {transpose ? columns : width, transpose ? width : columns}, device}};
        Descriptor descriptor;
        descriptor.operation = Op::LINEAR;
        descriptor.input_count = 2;
        descriptor.inputs[0] = inputs[0];
        descriptor.inputs[1] = inputs[1];
        descriptor.output.type = Type::BF16;
        descriptor.attribute_count = 1;
        descriptor.attributes[0] = transpose;
        descriptor.reference_projection = true;
        Program reference(descriptor, device);
        reference.run(inputs.data(), 2, output.get());
        std::vector<std::uint16_t> expected(rows * columns), actual(rows * columns);
        copy_to_host(expected.data(), output.get(), expected.size() * sizeof(std::uint16_t), device);
        descriptor.reference_projection = false;
        const auto previous_bytes = allocated_bytes(device);
        Program native(descriptor, device);
        const auto prepared_bytes = allocated_bytes(device);
        const auto workspace_limit = transpose && allow_split_workspace ? 8 * 1024 * 1024 : 0;
        if (prepared_bytes - previous_bytes > static_cast<std::size_t>(workspace_limit))
            throw std::runtime_error("CUDA native BF16 GEMM exceeded the workspace budget");
        if (native.output().type != Type::BF16) throw std::runtime_error("CUDA BF16 output dtype changed");
        native.run(inputs.data(), 2, output.get());
        copy_to_host(actual.data(), output.get(), actual.size() * sizeof(std::uint16_t), device);
        if (actual != expected) throw std::runtime_error("CUDA native BF16 GEMM differs from reference");
        if (allocated_bytes(device) != prepared_bytes)
            throw std::runtime_error("CUDA native BF16 GEMM allocated during launch");
    }
}

void check_float_matrix(int device, int rows, int width = 128) {
    using namespace kidi::cuda;
    using Buffer = std::unique_ptr<void, decltype(&release)>;
    constexpr int columns = 72;
    std::vector<float> values(rows * width), weights(columns * width), actual(rows * columns);
    for (std::size_t index = 0; index < values.size(); ++index)
        values[index] = static_cast<float>(index / width % 3 + 1) * (index % 2 ? 1.F : -1.F);
    for (std::size_t index = 0; index < weights.size(); ++index)
        weights[index] = static_cast<float>(index / width % 3 + 1) * (index % 2 ? 1.00048828125F : 1.F);
    Buffer input(allocate(values.size() * sizeof(float), device), release),
        weight(allocate(weights.size() * sizeof(float), device), release),
        output(allocate(actual.size() * sizeof(float), device), release);
    copy_to_device(input.get(), values.data(), values.size() * sizeof(float), device);
    copy_to_device(weight.get(), weights.data(), weights.size() * sizeof(float), device);
    const std::array<Tensor, 2> inputs{Tensor{input.get(), Type::F32, 2, {rows, width}, device},
                                       Tensor{weight.get(), Type::F32, 2, {columns, width}, device}};
    Descriptor descriptor;
    descriptor.operation = Op::LINEAR;
    descriptor.input_count = 2;
    std::copy(inputs.begin(), inputs.end(), descriptor.inputs.begin());
    descriptor.attribute_count = 1;
    descriptor.attributes[0] = 1;
    Program program(descriptor, device);
    const auto prepared_bytes = allocated_bytes(device);
    program.run(inputs.data(), 2, output.get());
    copy_to_host(actual.data(), output.get(), actual.size() * sizeof(float), device);
    if (allocated_bytes(device) != prepared_bytes)
        throw std::runtime_error("CUDA native FP32 GEMM allocated during launch");
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < columns; ++column) {
            const auto expected = static_cast<float>((row % 3 + 1) * (column % 3 + 1) * (width / 2)) / 2048.F;
            if (actual[row * columns + column] != expected)
                throw std::runtime_error("CUDA FP32 projection lost precision");
        }
}

template <typename Element>
void check_wide_rms(int device) {
    using namespace kidi::cuda;
    using Buffer = std::unique_ptr<void, decltype(&release)>;
    constexpr bool bfloat = std::is_same_v<Element, std::uint16_t>;
    constexpr Type type = bfloat ? Type::BF16 : Type::F32;
    constexpr int rows = 2, width = 2560;
    const auto encode = [](float value) -> Element {
        if constexpr (bfloat)
            return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16);
        else
            return value;
    };
    std::vector<Element> values(rows * width, encode(1.F)), residual(rows * width, encode(2.F)), actual(values.size());
    std::fill(values.begin() + width, values.end(), encode(-2.F));
    const std::vector<float> scales(width, 1.00390625F);
    const float multiplier = 0.5F;
    Buffer input(allocate(values.size() * sizeof(Element), device), release),
        scale(allocate(scales.size() * sizeof(float), device), release),
        skip(allocate(residual.size() * sizeof(Element), device), release),
        factor(allocate(sizeof(multiplier), device), release),
        output(allocate(actual.size() * sizeof(Element), device), release);
    copy_to_device(input.get(), values.data(), values.size() * sizeof(Element), device);
    copy_to_device(scale.get(), scales.data(), scales.size() * sizeof(float), device);
    copy_to_device(skip.get(), residual.data(), residual.size() * sizeof(Element), device);
    copy_to_device(factor.get(), &multiplier, sizeof(multiplier), device);
    const std::array<Tensor, 4> inputs{
        Tensor{input.get(), type, 2, {rows, width}, device}, Tensor{scale.get(), Type::F32, 1, {width}, device},
        Tensor{skip.get(), type, 2, {rows, width}, device}, Tensor{factor.get(), Type::F32, 1, {1}, device}};
    Descriptor descriptor;
    descriptor.operation = Op::RMS_NORM_RESIDUAL;
    descriptor.input_count = 4;
    descriptor.epsilon = 0;
    std::copy(inputs.begin(), inputs.end(), descriptor.inputs.begin());
    Program program(descriptor, device);
    program.run(inputs.data(), 4, output.get());
    copy_to_host(actual.data(), output.get(), actual.size() * sizeof(Element), device);
    const auto normalized = bfloat ? 1.F : scales.front();
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const auto expected = (2.F + (index < width ? normalized : -normalized)) * multiplier;
        if (actual[index] != encode(expected)) throw std::runtime_error("CUDA wide RMS residual rounding mismatch");
    }
}

void check_w8a16_projection(int device) {
    using namespace kidi::cuda;
    using Buffer = std::unique_ptr<void, decltype(&release)>;
    const auto encode = [](float value) -> std::uint16_t {
        return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16);
    };
    const auto decode = [](std::uint16_t rounded) -> float {
        return std::bit_cast<float>(static_cast<std::uint32_t>(rounded) << 16);
    };
    for (const int rows : {1, 3, 19, 65, 129}) {
        for (const int columns : {72, 4112}) {
            if (rows == 65 && columns > 72) continue;
            for (const int width : {64, 1600}) {
                for (const int group_size : {64, width}) {
                    if (width % group_size != 0) continue;
                    const int groups = width / group_size;
                    std::vector<std::uint16_t> values(static_cast<std::size_t>(rows) * width);
                    std::vector<std::int8_t> weights(static_cast<std::size_t>(columns) * width);
                    std::vector<float> scales(static_cast<std::size_t>(columns) * groups);
                    for (std::size_t index = 0; index < values.size(); ++index)
                        values[index] = encode(static_cast<float>(static_cast<int>(index * 13 % 29) - 14) * 0.0625F);
                    for (std::size_t index = 0; index < weights.size(); ++index)
                        weights[index] = static_cast<std::int8_t>(static_cast<int>(index * 17 % 251) - 125);
                    for (std::size_t index = 0; index < scales.size(); ++index)
                        scales[index] = static_cast<float>(index % 7 + 1) * 0.00390625F;
                    Buffer input(allocate(values.size() * sizeof(std::uint16_t), device), release),
                        weight(allocate(weights.size(), device), release),
                        scale(allocate(scales.size() * sizeof(float), device), release),
                        output(allocate(static_cast<std::size_t>(rows) * columns * sizeof(std::uint16_t), device),
                               release);
                    copy_to_device(input.get(), values.data(), values.size() * sizeof(std::uint16_t), device);
                    copy_to_device(weight.get(), weights.data(), weights.size(), device);
                    copy_to_device(scale.get(), scales.data(), scales.size() * sizeof(float), device);
                    const std::array<Tensor, 3> inputs{Tensor{input.get(), Type::BF16, 2, {rows, width}, device},
                                                       Tensor{weight.get(), Type::U8, 2, {columns, width}, device},
                                                       Tensor{scale.get(), Type::F32, 2, {columns, groups}, device}};
                    Descriptor descriptor;
                    descriptor.operation = Op::PACKED_LINEAR;
                    descriptor.input_count = 3;
                    std::copy(inputs.begin(), inputs.end(), descriptor.inputs.begin());
                    descriptor.output.type = Type::BF16;
                    descriptor.attribute_count = 3;
                    descriptor.attributes = {8, group_size, 0};
                    descriptor.reference_projection = true;
                    Program reference(descriptor, device);
                    reference.run(inputs.data(), 3, output.get());
                    std::vector<std::uint16_t> expected(static_cast<std::size_t>(rows) * columns),
                        actual(expected.size());
                    copy_to_host(expected.data(), output.get(), expected.size() * sizeof(std::uint16_t), device);
                    descriptor.reference_projection = false;
                    Program optimized(descriptor, device);
                    if (optimized.output().type != Type::BF16)
                        throw std::runtime_error("CUDA W8A16 output dtype changed");
                    const auto prepared_bytes = allocated_bytes(device);
                    optimized.run(inputs.data(), 3, output.get());
                    if (allocated_bytes(device) != prepared_bytes)
                        throw std::runtime_error("CUDA W8A16 allocated during launch");
                    copy_to_host(actual.data(), output.get(), actual.size() * sizeof(std::uint16_t), device);
                    for (std::size_t index = 0; index < actual.size(); ++index) {
                        const float lhs = decode(actual[index]), rhs = decode(expected[index]);
                        if (!std::isfinite(lhs) || std::abs(lhs - rhs) > 0.008F * (1.F + std::abs(rhs)))
                            throw std::runtime_error("CUDA W8A16 projection differs from reference");
                    }
                }
            }
        }
    }
}

void check_w8a8_projection(int device) {
    using namespace kidi::cuda;
    using Buffer = std::unique_ptr<void, decltype(&release)>;
    for (const int rows : {1, 3, 17, 65, 129, 1025}) {
        for (const int columns : {128, 4096}) {
            if ((rows == 65 || rows == 1025) && columns > 128) continue;
            constexpr int width = 256;
            std::vector<float> values(static_cast<std::size_t>(rows) * width);
            std::vector<std::int8_t> weights(static_cast<std::size_t>(columns) * width);
            std::vector<float> scales(columns), biases(columns);
            for (std::size_t index = 0; index < values.size(); ++index)
                values[index] = static_cast<float>(static_cast<int>(index * 13 % 29) - 14) * 0.0625F;
            for (std::size_t index = 0; index < weights.size(); ++index)
                weights[index] = static_cast<std::int8_t>(static_cast<int>(index * 17 % 251) - 125);
            for (int index = 0; index < columns; ++index) {
                scales[index] = static_cast<float>(index % 7 + 1) * 0.00390625F;
                biases[index] = static_cast<float>(index % 5 - 2) * 0.015625F;
            }
            Buffer input(allocate(values.size() * sizeof(float), device), release),
                weight(allocate(weights.size(), device), release),
                scale(allocate(scales.size() * sizeof(float), device), release),
                bias(allocate(biases.size() * sizeof(float), device), release),
                output(allocate(static_cast<std::size_t>(rows) * columns * sizeof(float), device), release);
            copy_to_device(input.get(), values.data(), values.size() * sizeof(float), device);
            copy_to_device(weight.get(), weights.data(), weights.size(), device);
            copy_to_device(scale.get(), scales.data(), scales.size() * sizeof(float), device);
            copy_to_device(bias.get(), biases.data(), biases.size() * sizeof(float), device);
            for (const bool calibrated : {false, true}) {
                for (const bool with_bias : {false, true}) {
                    if (calibrated && with_bias) continue;
                    const std::array<Tensor, 4> inputs{
                        Tensor{input.get(), Type::F32, 2, {rows, width}, device},
                        Tensor{weight.get(), with_bias ? Type::I8 : Type::U8, 2, {columns, width}, device},
                        Tensor{scale.get(), Type::F32, 2, {columns, 1}, device},
                        Tensor{bias.get(), Type::F32, 1, {columns}, device}};
                    Descriptor descriptor;
                    descriptor.operation = with_bias ? Op::QUANTIZED_LINEAR : Op::PACKED_LINEAR;
                    descriptor.input_count = with_bias ? 4 : 3;
                    std::copy(inputs.begin(), inputs.end(), descriptor.inputs.begin());
                    descriptor.output.type = Type::F32;
                    descriptor.attribute_count = with_bias ? 1 : 3;
                    if (with_bias) {
                        descriptor.attributes[0] = 1;
                    } else {
                        descriptor.attributes = {8, width, 0};
                        if (calibrated) {
                            descriptor.epsilon = 0.0625F;
                            const float out_scale = 0.125F;
                            std::int32_t encoded = 0;
                            std::memcpy(&encoded, &out_scale, sizeof(encoded));
                            descriptor.attributes[2] = encoded;
                        }
                    }
                    descriptor.reference_projection = true;
                    Program reference(descriptor, device);
                    reference.run(inputs.data(), descriptor.input_count, output.get());
                    std::vector<float> expected(static_cast<std::size_t>(rows) * columns), actual(expected.size());
                    copy_to_host(expected.data(), output.get(), expected.size() * sizeof(float), device);
                    descriptor.reference_projection = false;
                    Program optimized(descriptor, device);
                    optimized.run(inputs.data(), descriptor.input_count, output.get());
                    const auto prepared_bytes = allocated_bytes(device);
                    optimized.run(inputs.data(), descriptor.input_count, output.get());
                    if (allocated_bytes(device) != prepared_bytes)
                        throw std::runtime_error("CUDA W8A8 allocated during launch");
                    copy_to_host(actual.data(), output.get(), actual.size() * sizeof(float), device);
                    for (std::size_t index = 0; index < actual.size(); ++index) {
                        if (!std::isfinite(actual[index]) ||
                            std::abs(actual[index] - expected[index]) > 1e-4F * (1.F + std::abs(expected[index])))
                            throw std::runtime_error("CUDA W8A8 projection differs from reference");
                    }
                }
            }
        }
    }
}

void check_sparse_scatter(int device) {
    using namespace kidi::cuda;
    using Buffer = std::unique_ptr<void, decltype(&release)>;
    std::array<std::int32_t, 16> initial{}, actual{};
    const std::array<std::int32_t, 12> values{16777217,  16777219,  16777221,  16777223,   -16777217,  -16777219,
                                              -16777221, -16777223, 123456789, -123456789, 2147483647, -2147483647};
    const std::array<std::int32_t, 3> rows{3, 1, 3};
    Buffer cache(allocate(sizeof(initial), device), release), updates(allocate(sizeof(values), device), release),
        indices(allocate(sizeof(rows), device), release);
    copy_to_device(cache.get(), initial.data(), sizeof(initial), device);
    copy_to_device(updates.get(), values.data(), sizeof(values), device);
    copy_to_device(indices.get(), rows.data(), sizeof(rows), device);
    std::array<Tensor, 3> inputs{Tensor{cache.get(), Type::I32, 3, {1, 4, 4}, device},
                                 Tensor{updates.get(), Type::I32, 3, {1, 3, 4}, device},
                                 Tensor{indices.get(), Type::I32, 1, {3}, device}};
    Descriptor descriptor;
    descriptor.operation = Op::SCATTER;
    descriptor.input_count = 3;
    std::copy(inputs.begin(), inputs.end(), descriptor.inputs.begin());
    Program program(descriptor, device);
    program.run(inputs.data(), 3, cache.get());
    copy_to_host(actual.data(), cache.get(), sizeof(actual), device);
    for (std::size_t row = 0; row < 4; ++row)
        for (std::size_t channel = 0; channel < 4; ++channel) {
            const auto expected = row == 1 ? values[4 + channel] : row == 3 ? values[8 + channel] : 0;
            if (actual[row * 4 + channel] != expected) throw std::runtime_error("CUDA sparse scatter mismatch");
        }
    inputs[1].data = cache.get();
    bool rejected = false;
    try {
        program.run(inputs.data(), 3, cache.get());
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    if (!rejected) throw std::runtime_error("CUDA sparse scatter accepted overlapping inputs");
    inputs[1].data = updates.get();
    Buffer copied(allocate(sizeof(initial), device), release);
    program.run(inputs.data(), 3, copied.get());
    std::array<std::int32_t, 16> fallback{};
    copy_to_host(fallback.data(), copied.get(), sizeof(fallback), device);
    if (fallback != actual) throw std::runtime_error("CUDA fallback scatter lost integer precision");
}

void check_mixed_matrix(int device) {
    using namespace kidi::cuda;
    using Buffer = std::unique_ptr<void, decltype(&release)>;
    constexpr int rows = 4, width = 8, columns = 6;
    std::array<float, 2 * rows * width> values{};
    for (std::size_t index = 0; index < values.size(); ++index)
        values[index] = static_cast<float>(static_cast<int>(index % 13) - 6) * 0.03125F;
    Buffer input(allocate(sizeof(values), device), release);
    std::array<std::uint16_t, 2 * rows * width> rounded{};
    for (std::size_t index = 0; index < rounded.size(); ++index) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &values[index], sizeof(bits));
        rounded[index] = static_cast<std::uint16_t>(bits >> 16);
    }
    Buffer bfloat_input(allocate(sizeof(rounded), device), release);
    copy_to_device(bfloat_input.get(), rounded.data(), sizeof(rounded), device);
    Buffer weight(allocate(width * columns * sizeof(std::uint16_t), device), release);
    Buffer output(allocate(2 * rows * columns * sizeof(float), device), release);
    copy_to_device(input.get(), values.data(), sizeof(values), device);
    for (const bool bfloat : {false, true})
        for (const auto type : {Type::BF16, Type::F16})
            for (const bool transpose : {false, true}) {
                std::array<std::uint16_t, width * columns> weights{};
                for (std::size_t index = 0; index < weights.size(); ++index)
                    weights[index] =
                        static_cast<std::uint16_t>((type == Type::BF16 ? 0x3f80 : 0x3c00) | (index % 3 ? 0 : 0x8000));
                copy_to_device(weight.get(), weights.data(), sizeof(weights), device);
                std::array<Tensor, 2> inputs{
                    Tensor{bfloat ? bfloat_input.get() : input.get(),
                           bfloat ? Type::BF16 : Type::F32,
                           3,
                           {2, rows, width},
                           device},
                    Tensor{weight.get(), type, 2, {transpose ? columns : width, transpose ? width : columns}, device}};
                Descriptor descriptor;
                descriptor.operation = Op::MATMUL;
                descriptor.input_count = 2;
                descriptor.inputs[0] = inputs[0];
                descriptor.inputs[1] = inputs[1];
                descriptor.attribute_count = 1;
                descriptor.attributes[0] = transpose;
                descriptor.reference_projection = true;
                Program reference(descriptor, device);
                const auto verify = [&](Program& program) {
                    std::array<float, 2 * rows * columns> expected{}, actual{};
                    reference.run(inputs.data(), 2, output.get());
                    copy_to_host(expected.data(), output.get(), sizeof(expected), device);
                    program.run(inputs.data(), 2, output.get());
                    copy_to_host(actual.data(), output.get(), sizeof(actual), device);
                    if (actual != expected) throw std::runtime_error("CUDA mixed matrix differs from reference");
                };
                descriptor.reference_projection = false;
                descriptor.projection_cache = make_projection_cache();
                descriptor.immutable_weights = true;
                {
                    Program immutable(descriptor, device);
                    verify(immutable);
                    const auto bytes = allocated_bytes(device);
                    Program shared(descriptor, device);
                    verify(shared);
                    if (allocated_bytes(device) != bytes)
                        throw std::runtime_error("CUDA mixed matrix cache duplicated immutable weights");
                }
                descriptor.immutable_weights = false;
                Program dynamic(descriptor, device);
                verify(dynamic);
                weights.fill(0x4000);
                copy_to_device(weight.get(), weights.data(), sizeof(weights), device);
                verify(dynamic);
            }
}

void check_projection_rebinding(int device) {
    using namespace kidi::cuda;
    using Buffer = std::unique_ptr<void, decltype(&release)>;
    constexpr int rows = 9, width = 16, columns = 8;
    std::array<float, rows * 2 * width> values{};
    std::array<float, columns> scales{}, bias{};
    values.fill(1);
    scales.fill(0.25F);
    for (const int bits : {4, 8}) {
        std::vector<std::uint8_t> weights(columns * width / (8 / bits), bits == 4 ? 0x22 : 2);
        const std::vector<std::uint8_t> replacement(weights.size(), bits == 4 ? 0x44 : 4);
        Buffer input(allocate(sizeof(values), device), release), weight(allocate(weights.size(), device), release),
            changed(allocate(weights.size(), device), release), scale(allocate(sizeof(scales), device), release),
            offsets(allocate(sizeof(bias), device), release),
            output(allocate(rows * columns * sizeof(float), device), release);
        copy_to_device(input.get(), values.data(), sizeof(values), device);
        copy_to_device(weight.get(), weights.data(), weights.size(), device);
        copy_to_device(changed.get(), replacement.data(), replacement.size(), device);
        copy_to_device(scale.get(), scales.data(), sizeof(scales), device);
        copy_to_device(offsets.get(), bias.data(), sizeof(bias), device);
        std::array<Tensor, 4> inputs{
            {{input.get(), Type::F32, 2, {rows, width}, device},
             {weight.get(), bits == 4 ? Type::U8 : Type::I8, 2, {columns, width / (8 / bits)}, device},
             {scale.get(), Type::F32, 2, {columns, 1}, device},
             {offsets.get(), Type::F32, 1, {columns}, device}}};
        Descriptor descriptor;
        descriptor.operation = bits == 4 ? Op::PACKED_LINEAR : Op::QUANTIZED_LINEAR;
        for (int index = 0; index < 4; ++index) descriptor.inputs[index] = inputs[index];
        descriptor.input_count = bits == 4 ? 3 : 4;
        descriptor.attribute_count = bits == 4 ? 3 : 1;
        descriptor.attributes[0] = bits == 4 ? bits : 1;
        descriptor.attributes[1] = width;
        descriptor.epsilon = bits == 4 ? 0.125F : 0;
        descriptor.projection_cache = make_projection_cache();
        const auto verify = [&](Program& program, float expected) {
            program.run(inputs.data(), descriptor.input_count, output.get());
            std::array<float, rows * columns> actual{};
            copy_to_host(actual.data(), output.get(), sizeof(actual), device);
            for (const auto value : actual)
                if (std::abs(value - expected) > 1e-5F)
                    throw std::runtime_error("CUDA projection weight rebinding failed");
        };
        Program dynamic(descriptor, device);
        verify(dynamic, 8.F);
        std::fill(weights.begin(), weights.end(), bits == 4 ? 0xDD : static_cast<std::uint8_t>(-3));
        copy_to_device(weight.get(), weights.data(), weights.size(), device);
        verify(dynamic, -12.F);
        descriptor.immutable_weights = true;
        Program immutable(descriptor, device);
        verify(immutable, -12.F);
        inputs[1].data = changed.get();
        verify(immutable, 16.F);
        verify(immutable, 16.F);
        const auto previous_bytes = allocated_bytes(device);
        {
            auto larger = inputs;
            larger[0].shape[0] = rows * 2;
            descriptor.inputs[0] = larger[0];
            Program shared(descriptor, device);
            Buffer larger_output(allocate(rows * 2 * columns * sizeof(float), device), release);
            shared.run(larger.data(), descriptor.input_count, larger_output.get());
            std::array<float, rows * 2 * columns> actual{};
            copy_to_host(actual.data(), larger_output.get(), sizeof(actual), device);
            for (const auto value : actual)
                if (std::abs(value - 16.F) > 1e-5F) throw std::runtime_error("CUDA shared projection cache mismatch");
            const auto workspace_bytes = rows * 2 * width + rows * 2 * sizeof(float) + rows * 2 * sizeof(int) +
                                         (USE_INT8_GEMM ? 2 : 1) * sizeof(actual);
            if (allocated_bytes(device) - previous_bytes != workspace_bytes)
                throw std::runtime_error("CUDA projection cache duplicated immutable weights");
        }

        Descriptor activation;
        activation.operation = Op::GELU_GATE;
        activation.input_count = 1;
        activation.inputs[0] = immutable.output();
        activation.inputs[0].data = output.get();
        Program geglu(activation, device);
        std::array<float, rows * columns / 2> expected{}, actual{};
        Buffer activated(allocate(sizeof(actual), device), release);
        const auto check = [](cudaError_t status) {
            if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        };
        const DeviceScope device_scope(device);
        cudaStream_t stream{};
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(&cudaStreamDestroy)> stream_owner(
            stream, cudaStreamDestroy);
        const StreamScope scope(stream);
        const auto run = [&](Program& projection) {
            projection.run(inputs.data(), descriptor.input_count, output.get());
            geglu.run(activation.inputs.data(), 1, activated.get());
        };
        descriptor.inputs[0] = inputs[0];
        descriptor.reference_projection = true;
        Program reference(descriptor, device);
        run(immutable);
        check(cudaStreamSynchronize(stream));
        const auto warm_bytes = allocated_bytes(device);
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        run(immutable);
        cudaGraph_t graph{};
        check(cudaStreamEndCapture(stream, &graph));
        const std::unique_ptr<std::remove_pointer_t<cudaGraph_t>, decltype(&cudaGraphDestroy)> graph_owner(
            graph, cudaGraphDestroy);
        cudaGraphExec_t executable{};
        check(cudaGraphInstantiate(&executable, graph, 0));
        const std::unique_ptr<std::remove_pointer_t<cudaGraphExec_t>, decltype(&cudaGraphExecDestroy)> executable_owner(
            executable, cudaGraphExecDestroy);
        if (allocated_bytes(device) != warm_bytes)
            throw std::runtime_error("CUDA graph allocated projection workspace");
        auto replay_values = values;
        for (const float shift : {0.F, 1.F}) {
            for (std::size_t index = 0; index < replay_values.size(); ++index)
                replay_values[index] = shift + (static_cast<int>(index * 13 % 19) - 6) * 0.125F;
            copy_to_device(input.get(), replay_values.data(), sizeof(replay_values), device);
            run(reference);
            copy_to_host(expected.data(), activated.get(), sizeof(expected), device);
            check(cudaMemsetAsync(output.get(), 0, rows * columns * sizeof(float), stream));
            check(cudaMemsetAsync(activated.get(), 0, sizeof(actual), stream));
            check(cudaGraphLaunch(executable, stream));
            copy_to_host(actual.data(), activated.get(), sizeof(actual), device);
            for (std::size_t index = 0; index < actual.size(); ++index)
                if (!std::isfinite(actual[index]) || std::abs(actual[index] - expected[index]) > 1e-5F)
                    throw std::runtime_error("CUDA projection and GEGLU graph replay mismatch");
        }
    }
}

void check_descriptor_validation(int device) {
    using namespace kidi::cuda;
    using Buffer = std::unique_ptr<void, decltype(&release)>;
    Buffer buffer(allocate(256 * sizeof(float), device), release);
    const auto rejected = [&](const Descriptor& descriptor) {
        try {
            Program program(descriptor, device);
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };
    Descriptor descriptor;
    descriptor.input_count = 1;
    descriptor.inputs[0] = Tensor{buffer.get(), Type::F32, 3, {1, 2, 8}, device};
    for (const auto operation :
         {Op::ADD, Op::MULTIPLY, Op::MATMUL, Op::LINEAR, Op::GATHER, Op::RMS_NORM, Op::LAYER_NORM, Op::ROTARY,
          Op::PACKED_LINEAR, Op::QUANTIZED_LINEAR, Op::ATTENTION, Op::EMBEDDING_SUM, Op::DIFFUSION_UPDATE}) {
        descriptor.operation = operation;
        if (!rejected(descriptor)) throw std::runtime_error("CUDA accepted an operation with too few inputs");
    }

    Descriptor attention;
    attention.operation = Op::ATTENTION;
    attention.input_count = 3;
    for (int index = 0; index < 3; ++index)
        attention.inputs[index] = Tensor{buffer.get(), Type::F32, 3, {1, 4, 8}, device};
    attention.attribute_count = 1;
    for (const std::int64_t heads : {0, 3, 16}) {
        attention.attributes[0] = heads;
        if (!rejected(attention)) throw std::runtime_error("CUDA accepted invalid attention heads");
    }
    attention.attribute_count = 4;
    attention.attributes = {2, 2, 3, 2};
    if (!rejected(attention)) throw std::runtime_error("CUDA accepted attention keys beyond the cache");
    attention.attributes = {2, 2, 2, 2};
    if (rejected(attention)) throw std::runtime_error("CUDA rejected valid attention");
}

void check_relocated_scatter(int device) {
    using namespace kidi::cuda;
    using Buffer = std::unique_ptr<void, decltype(&release)>;
    const std::array<std::int32_t, 8> values{1, 2, 3, 4, 5, 6, 7, 8};
    const std::array<std::int32_t, 2> rows{3, 1}, stale{0, 0};
    std::array<std::int32_t, 16> initial{}, original{}, moved{};
    Buffer cache(allocate(sizeof(initial), device), release), target(allocate(sizeof(initial), device), release),
        updates(allocate(sizeof(values), device), release), indices(allocate(sizeof(rows), device), release),
        stale_indices(allocate(sizeof(stale), device), release), table(allocate(2 * sizeof(void*), device), release);
    copy_to_device(cache.get(), initial.data(), sizeof(initial), device);
    copy_to_device(target.get(), initial.data(), sizeof(initial), device);
    copy_to_device(updates.get(), values.data(), sizeof(values), device);
    copy_to_device(indices.get(), rows.data(), sizeof(rows), device);
    copy_to_device(stale_indices.get(), stale.data(), sizeof(stale), device);
    const std::array<void*, 2> current{target.get(), indices.get()};
    copy_to_device(table.get(), current.data(), sizeof(current), device);
    const auto* slots = static_cast<void* const*>(table.get());
    const std::array<TensorBinding, 2> bindings{TensorBinding{cache.get(), sizeof(initial), slots},
                                                TensorBinding{stale_indices.get(), sizeof(stale), slots + 1}};
    const std::array<Tensor, 3> inputs{Tensor{cache.get(), Type::I32, 3, {1, 4, 4}, device},
                                       Tensor{updates.get(), Type::I32, 3, {1, 2, 4}, device},
                                       Tensor{stale_indices.get(), Type::I32, 1, {2}, device}};
    Descriptor descriptor;
    descriptor.operation = Op::SCATTER;
    descriptor.input_count = 3;
    std::copy(inputs.begin(), inputs.end(), descriptor.inputs.begin());
    Program program(descriptor, device);
    {
        const BindingScope scope(bindings.data(), bindings.size());
        program.run(inputs.data(), 3, cache.get());
    }
    copy_to_host(original.data(), cache.get(), sizeof(initial), device);
    copy_to_host(moved.data(), target.get(), sizeof(initial), device);
    if (original != initial) throw std::runtime_error("CUDA scatter wrote to the address a cache was relocated from");
    for (std::size_t row = 0; row < 4; ++row)
        for (std::size_t channel = 0; channel < 4; ++channel) {
            const auto expected = row == 3 ? values[channel] : row == 1 ? values[4 + channel] : 0;
            if (moved[row * 4 + channel] != expected)
                throw std::runtime_error("CUDA scatter ignored its relocated operands");
        }
}

void check_relocated_attention(int device) {
    using namespace kidi::cuda;
    using Buffer = std::unique_ptr<void, decltype(&release)>;
    constexpr std::int64_t heads = 2, key_heads = 1, keys = 19, offset = 3, capacity = keys + 5;
    const auto bfloat = [](float number) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &number, sizeof(bits));
        return static_cast<std::uint16_t>(bits >> 16);
    };
    for (const std::int64_t width : {64, 256}) {
        std::vector<std::uint16_t> query(heads * width), cache(capacity * key_heads * width),
            zeros(cache.size(), bfloat(0.F));
        for (std::size_t index = 0; index < query.size(); ++index)
            query[index] = bfloat(static_cast<float>(static_cast<int>(index * 7 % 31) - 15) * 0.015625F);
        for (std::size_t index = 0; index < cache.size(); ++index)
            cache[index] = bfloat(static_cast<float>(static_cast<int>(index * 11 % 31) - 15) * 0.015625F);
        const auto bytes = cache.size() * sizeof(std::uint16_t);
        Buffer queries(allocate(query.size() * sizeof(std::uint16_t), device), release),
            key(allocate(bytes, device), release), value(allocate(bytes, device), release),
            old_key(allocate(bytes, device), release), old_value(allocate(bytes, device), release),
            output(allocate(query.size() * sizeof(std::uint16_t), device), release),
            table(allocate(2 * sizeof(void*), device), release);
        copy_to_device(queries.get(), query.data(), query.size() * sizeof(std::uint16_t), device);
        copy_to_device(key.get(), cache.data(), bytes, device);
        copy_to_device(value.get(), cache.data(), bytes, device);
        copy_to_device(old_key.get(), zeros.data(), bytes, device);
        copy_to_device(old_value.get(), zeros.data(), bytes, device);
        const std::array<void*, 2> current{key.get(), value.get()};
        copy_to_device(table.get(), current.data(), sizeof(current), device);
        const auto* slots = static_cast<void* const*>(table.get());
        const std::array<TensorBinding, 2> bindings{TensorBinding{old_key.get(), bytes, slots},
                                                    TensorBinding{old_value.get(), bytes, slots + 1}};
        const auto operands = [&](std::int64_t rows, void* keys_data, void* values_data) {
            return std::array<Tensor, 3>{Tensor{queries.get(), Type::BF16, 3, {1, rows, heads * width}, device},
                                         Tensor{keys_data, Type::BF16, 3, {1, capacity, key_heads * width}, device},
                                         Tensor{values_data, Type::BF16, 3, {1, capacity, key_heads * width}, device}};
        };
        Descriptor descriptor;
        descriptor.operation = Op::ATTENTION;
        descriptor.input_count = 3;
        const auto fixed = operands(1, key.get(), value.get());
        std::copy(fixed.begin(), fixed.end(), descriptor.inputs.begin());
        descriptor.attribute_count = 4;
        descriptor.attributes = {heads, key_heads, offset, keys};
        Program program(descriptor, device);
        const auto result = [&](const std::array<Tensor, 3>& inputs) {
            std::vector<std::uint16_t> values(query.size());
            program.run(inputs.data(), 3, output.get());
            copy_to_host(values.data(), output.get(), values.size() * sizeof(std::uint16_t), device);
            return values;
        };
        const auto expected = result(fixed);
        const auto moved = operands(1, old_key.get(), old_value.get());
        if (result(moved) == expected) throw std::runtime_error("CUDA relocation test cannot tell stale caches apart");
        {
            const BindingScope scope(bindings.data(), bindings.size());
            if (result(moved) != expected) throw std::runtime_error("CUDA decode attention ignored a relocated cache");
        }
        auto prefill = descriptor;
        const auto prefill_inputs = operands(2, old_key.get(), old_value.get());
        std::copy(prefill_inputs.begin(), prefill_inputs.end(), prefill.inputs.begin());
        Program batched(prefill, device);
        const BindingScope scope(bindings.data(), bindings.size());
        bool rejected = false;
        try {
            batched.run(prefill_inputs.data(), 3, output.get());
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        if (!rejected) throw std::runtime_error("CUDA accepted a relocated cache in a multi-query attention");
    }
}

void check_relocation_limits(int device) {
    using namespace kidi::cuda;
    using Buffer = std::unique_ptr<void, decltype(&release)>;
    const std::array<float, 4> numbers{1, 2, 3, 4};
    Buffer left(allocate(sizeof(numbers), device), release), right(allocate(sizeof(numbers), device), release),
        output(allocate(sizeof(numbers), device), release), table(allocate(sizeof(void*), device), release);
    copy_to_device(left.get(), numbers.data(), sizeof(numbers), device);
    copy_to_device(right.get(), numbers.data(), sizeof(numbers), device);
    const void* current = right.get();
    copy_to_device(table.get(), &current, sizeof(current), device);
    const TensorBinding binding{left.get(), sizeof(numbers), static_cast<void* const*>(table.get())};
    const std::array<Tensor, 2> inputs{Tensor{left.get(), Type::F32, 1, {4}, device},
                                       Tensor{right.get(), Type::F32, 1, {4}, device}};
    Descriptor descriptor;
    descriptor.operation = Op::ADD;
    descriptor.input_count = 2;
    std::copy(inputs.begin(), inputs.end(), descriptor.inputs.begin());
    Program program(descriptor, device);
    const BindingScope scope(&binding, 1);
    bool rejected = false;
    try {
        program.run(inputs.data(), 2, output.get());
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    if (!rejected) throw std::runtime_error("CUDA accepted a relocated operand on a path that ignores relocation");
}
} // namespace

int main() {
    using namespace kidi::cuda;
    try {
        if (!available()) throw std::runtime_error("CUDA device unavailable");
        if (available(-1) || available(device_count())) throw std::runtime_error("CUDA invalid device accepted");
        const std::array<float, 6> left{1, 2, 3, 4, 5, 6};
        const std::array<float, 3> right{10, 20, 30};
        Tensor inputs[2]{{allocate(sizeof(left)), Type::F32, 2, {2, 3}}, {allocate(sizeof(right)), Type::F32, 1, {3}}};
        copy_to_device(inputs[0].data, left.data(), sizeof(left));
        copy_to_device(inputs[1].data, right.data(), sizeof(right));
        Descriptor descriptor;
        descriptor.operation = Op::ADD;
        descriptor.input_count = 2;
        descriptor.inputs[0] = inputs[0];
        descriptor.inputs[1] = inputs[1];
        Program program(descriptor);
        void* output = allocate(sizeof(left));
        program.run(inputs, 2, output);
        std::array<float, 6> result{};
        copy_to_host(result.data(), output, sizeof(result));
        if (result != std::array<float, 6>{11, 22, 33, 14, 25, 36}) throw std::runtime_error("CUDA broadcast mismatch");
        release(output);
        release(inputs[0].data);
        release(inputs[1].data);
        if (allocated_bytes()) throw std::runtime_error("CUDA allocation leak");
        const int previous = current_device();
        for (int device = 0; device < device_count(); ++device) {
            check_bfloat_bias(device);
            check_integer_layout(device);
            check_gelu(device);
            check_scalar_and_empty(device);
            check_greedy_selection(device);
            check_slice(device);
            check_attention(device);
            check_attention(device, 257);
            check_attention<float>(device);
            check_attention<float, std::uint16_t>(device);
            check_attention<float, std::uint16_t>(device, 257);
            check_attention<float, std::uint16_t>(device, 513);
            for (const std::int64_t keys : {2047, 2048, 2049})
                check_attention<float, std::uint16_t>(device, keys, true);
            for (bool per_key : {false, true}) {
                if (USE_INT8_GEMM) check_nonbroadcast_causal_attention<float>(device, per_key);
                check_nonbroadcast_causal_attention<std::uint16_t>(device, per_key);
            }
            if (USE_INT8_GEMM) check_structured_prefill_attention(device);
            {
                // Gemma 4 E4B GQA prefill check: 8 query heads, 2 KV heads, sliding + causal (-1e9F and -INFINITY).
                constexpr std::int64_t batches = 1, queries = 129, heads = 8, key_heads = 2, keys = 640, offset = 0;
                for (const std::int64_t width : {256, 512}) {
                    for (const float masked_value : {-1e9F, -INFINITY}) {
                        std::vector<std::uint16_t> query(batches * queries * heads * width),
                            key(batches * keys * key_heads * width), value(key.size());
                        for (std::size_t index = 0; index < query.size(); ++index) {
                            const float number = static_cast<float>(static_cast<int>(index * 7 % 31) - 15) * 0.015625F;
                            std::uint32_t bits = 0;
                            std::memcpy(&bits, &number, sizeof(bits));
                            query[index] = static_cast<std::uint16_t>(bits >> 16);
                        }
                        for (std::size_t index = 0; index < key.size(); ++index) {
                            const float n1 = static_cast<float>(static_cast<int>(index * 11 % 31) - 15) * 0.015625F;
                            const float n2 = static_cast<float>(static_cast<int>(index * 13 % 31) - 15) * 0.015625F;
                            std::uint32_t b1 = 0, b2 = 0;
                            std::memcpy(&b1, &n1, sizeof(b1));
                            std::memcpy(&b2, &n2, sizeof(b2));
                            key[index] = static_cast<std::uint16_t>(b1 >> 16);
                            value[index] = static_cast<std::uint16_t>(b2 >> 16);
                        }
                        std::vector<float> mask(queries * keys);
                        for (std::int64_t row = 0; row < queries; ++row)
                            for (std::int64_t column = 0; column < keys; ++column) {
                                const std::int64_t q_abs = (keys - queries) + row;
                                const bool allowed = column <= q_abs && (width == 512 || q_abs - column < 256);
                                mask[row * keys + column] = allowed ? (column % 5 ? 0.F : 0.125F) : masked_value;
                            }
                        std::unique_ptr<void, decltype(&release)> q_buf(allocate(query.size() * 2, device), release),
                            k_buf(allocate(key.size() * 2, device), release),
                            v_buf(allocate(value.size() * 2, device), release),
                            m_buf(allocate(mask.size() * 4, device), release),
                            o_buf(allocate(query.size() * 2, device), release);
                        copy_to_device(q_buf.get(), query.data(), query.size() * 2, device);
                        copy_to_device(k_buf.get(), key.data(), key.size() * 2, device);
                        copy_to_device(v_buf.get(), value.data(), value.size() * 2, device);
                        copy_to_device(m_buf.get(), mask.data(), mask.size() * 4, device);
                        std::array<Tensor, 4> inputs{
                            Tensor{q_buf.get(), Type::BF16, 3, {batches, queries, heads * width}, device},
                            Tensor{k_buf.get(), Type::BF16, 3, {batches, keys, key_heads * width}, device},
                            Tensor{v_buf.get(), Type::BF16, 3, {batches, keys, key_heads * width}, device},
                            Tensor{m_buf.get(), Type::F32, 4, {1, 1, queries, keys}, device}};
                        Descriptor d;
                        d.operation = Op::ATTENTION;
                        d.input_count = 4;
                        std::copy(inputs.begin(), inputs.end(), d.inputs.begin());
                        d.attribute_count = 4;
                        d.attributes = {heads, key_heads, offset, keys};
                        d.epsilon = 1.F;
                        d.reference_attention = true;
                        Program ref(d, device);
                        ref.run(inputs.data(), 4, o_buf.get());
                        std::vector<std::uint16_t> expected(query.size()), actual(query.size());
                        copy_to_host(expected.data(), o_buf.get(), expected.size() * 2, device);
                        d.reference_attention = false;
                        const auto prev_bytes = allocated_bytes(device);
                        Program nat(d, device);
                        if (allocated_bytes(device) != prev_bytes)
                            throw std::runtime_error("CUDA FlashAttention GQA allocated a score buffer");
                        nat.run(inputs.data(), 4, o_buf.get());
                        copy_to_host(actual.data(), o_buf.get(), actual.size() * 2, device);
                        for (std::size_t index = 0; index < actual.size(); ++index) {
                            const auto dec = [](std::uint16_t u) {
                                return std::bit_cast<float>(static_cast<std::uint32_t>(u) << 16);
                            };
                            const float t = dec(expected[index]), r = dec(actual[index]);
                            const float tol = (15.F * 0.015625F) / 256.F + 0.0001F + 0.008F * std::abs(t);
                            if (!std::isfinite(r) || std::abs(r - t) > tol)
                                throw std::runtime_error("CUDA FlashAttention GQA mismatch");
                        }
                    }
                }
            }
            check_sparse_scatter(device);
            for (const int rows : {1, 4})
                for (const int width : {8, 256, 1536, 6144, 12288}) check_bfloat_matrix(device, rows, width);
            check_bfloat_matrix(device, 1, 256, 4112);
            check_bfloat_matrix(device, 9, 256, 72);
            check_bfloat_matrix(device, 9, 1560, 72, true);
            check_bfloat_matrix(device, 65, 40, 4096, true);
            check_bfloat_matrix(device, 129, 1600, 72, true);
            check_bfloat_matrix(device, 128, 2560, 8192, true);
            check_bfloat_matrix(device, 128, 256, 2560, true);
            check_w8a16_projection(device);
            check_w8a8_projection(device);
            check_float_matrix(device, 1);
            check_float_matrix(device, 37, 132);
            check_float_matrix(device, 34, 1564);
            check_wide_rms<std::uint16_t>(device);
            check_wide_rms<float>(device);
            check_mixed_matrix(device);
            check_projection_rebinding(device);
            check_descriptor_validation(device);
            check_relocated_scatter(device);
            check_relocated_attention(device);
            check_relocation_limits(device);
            auto selected_inputs = std::array<Tensor, 2>{inputs[0], inputs[1]};
            selected_inputs[0].device = selected_inputs[1].device = device;
            selected_inputs[0].data = allocate(sizeof(left), device);
            selected_inputs[1].data = allocate(sizeof(right), device);
            copy_to_device(selected_inputs[0].data, left.data(), sizeof(left), device);
            copy_to_device(selected_inputs[1].data, right.data(), sizeof(right), device);
            Program selected_program(descriptor, device);
            void* selected_output = allocate(sizeof(left), device);
            selected_program.run(selected_inputs.data(), 2, selected_output);
            copy_to_host(result.data(), selected_output, sizeof(result), device);
            if (result != std::array<float, 6>{11, 22, 33, 14, 25, 36})
                throw std::runtime_error("CUDA selected-device mismatch");
            if (current_device() != previous) throw std::runtime_error("CUDA current device was not restored");
            selected_inputs[0].device = device_count();
            bool rejected = false;
            try {
                selected_program.run(selected_inputs.data(), 2, selected_output);
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            if (!rejected) throw std::runtime_error("CUDA mixed-device operands accepted");
            release(selected_output);
            release(selected_inputs[0].data);
            release(selected_inputs[1].data);
        }
        if (allocated_bytes()) throw std::runtime_error("CUDA selected-device allocation leak");
        std::cout << "CUDA broadcast and allocation tests PASS on " << device_name() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}