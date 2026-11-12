#pragma once

#include "kidi/core/error.h"
#include "kidi/tensor/dtype.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace kidi::search {

class Failure {
public:
    explicit Failure(Error error) : error_(std::move(error)) {}
    auto error() const -> const Error& { return error_; }

private:
    Error error_;
};

template <typename T>
auto require(Result<T> result) -> T {
    if (!result) throw Failure(std::move(result.error()));
    if constexpr (!std::is_void_v<T>) return std::move(*result);
}

inline auto invalid(bool condition, const char* message) -> void {
    if (condition) throw Failure({ErrorCode::INVALID_MANIFEST, message});
}

class Reader {
public:
    explicit Reader(std::span<const std::byte> bytes) : bytes_(bytes) {}
    auto take(std::size_t count) -> std::span<const std::byte> {
        invalid(count > bytes_.size() - position_, "truncated HNSW index");
        const auto result = bytes_.subspan(position_, count);
        position_ += count;
        return result;
    }
    auto integer(std::size_t count) -> std::uint64_t {
        std::uint64_t value = 0;
        const auto bytes = take(count);
        for (std::size_t index = 0; index < count; ++index)
            value |= std::to_integer<std::uint64_t>(bytes[index]) << (8 * index);
        return value;
    }
    auto position() const -> std::size_t { return position_; }
    auto remaining() const -> std::size_t { return bytes_.size() - position_; }

private:
    std::span<const std::byte> bytes_;
    std::size_t position_ = 0;
};

inline auto append(std::vector<std::byte>& bytes, std::uint64_t value, std::size_t count) -> void {
    for (std::size_t index = 0; index < count; ++index)
        bytes.push_back(static_cast<std::byte>((value >> (8 * index)) & 255U));
}

inline auto write(std::span<std::byte> bytes, std::size_t offset, std::uint64_t value, std::size_t count) -> void {
    for (std::size_t index = 0; index < count; ++index)
        bytes[offset + index] = static_cast<std::byte>((value >> (8 * index)) & 255U);
}

struct ByDistance {
    auto operator()(const std::pair<float, std::uint32_t>& left, const std::pair<float, std::uint32_t>& right) const
        -> bool {
        return left.first < right.first;
    }
};

inline auto push(std::vector<std::pair<float, std::uint32_t>>& heap, float distance, std::uint32_t node) -> void {
    heap.emplace_back(distance, node);
    std::push_heap(heap.begin(), heap.end(), ByDistance{});
}

inline auto pop(std::vector<std::pair<float, std::uint32_t>>& heap) -> std::pair<float, std::uint32_t> {
    std::pop_heap(heap.begin(), heap.end(), ByDistance{});
    const auto result = heap.back();
    heap.pop_back();
    return result;
}

using Dot = float (*)(const float*, const std::byte*, std::size_t);
using Dot4 = std::array<float, 4> (*)(const float*, const std::byte*, const std::byte*, const std::byte*,
                                      const std::byte*, std::size_t);
using RowDot = float (*)(const std::byte*, const std::byte*, std::size_t);

auto hnsw_crc32(std::span<const std::byte> bytes) -> std::uint32_t;
auto read_file(const std::filesystem::path& path) -> Result<std::vector<std::byte>>;
auto save_atomic_file(const std::filesystem::path& path, std::span<const std::byte> bytes) -> Result<void>;
auto dot_kernels(tensor::DType dtype) -> std::tuple<Dot, Dot4, RowDot>;
auto weight_element(const std::byte* row, std::size_t column, bool bfloat) noexcept -> float;
auto prefetch_slice(const std::byte* slice_ptr, std::size_t slice_bytes) noexcept -> void;
auto quantized_slice_bytes(std::size_t width, std::int32_t bits, std::int32_t group_size, bool scale_f16, bool zero_f16,
                           bool has_zero) noexcept -> std::size_t;
auto pack_quantized_slice(std::byte* dst, const std::byte* w_row, const std::byte* s_row, const std::byte* z_row,
                          std::size_t width, std::int32_t bits, std::int32_t group_size, bool scale_f16, bool zero_f16,
                          bool has_zero) noexcept -> void;
auto prepare_quantized_query(std::span<const float> query, std::size_t width, std::int32_t bits,
                             std::int32_t group_size, float input_scale, std::vector<std::int8_t>& q_i8,
                             std::vector<float>& q_scale, std::vector<float>& q_bsum) -> void;
auto quantized_slice_dot(const std::byte* slice_ptr, std::size_t width, std::int32_t bits, std::int32_t group_size,
                         bool scale_f16, bool zero_f16, bool has_zero, const std::vector<std::int8_t>& q_i8,
                         const std::vector<float>& q_scale, const std::vector<float>& q_bsum) -> float;
auto quantized_slices4_dot(const std::byte* s0, const std::byte* s1, const std::byte* s2, const std::byte* s3,
                           std::size_t width, std::int32_t bits, std::int32_t group_size, bool scale_f16, bool zero_f16,
                           bool has_zero, const std::vector<std::int8_t>& q_i8, const std::vector<float>& q_scale,
                           const std::vector<float>& q_bsum) -> std::array<float, 4>;

} // namespace kidi::search
