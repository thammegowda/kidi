#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace kidi::tensor {

enum class DType {
    BOOL,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F16,
    BF16,
    F32,
    F64,
    E4M3,
    E5M2,
};

constexpr auto element_size(DType dtype) noexcept -> std::size_t {
    switch (dtype) {
        case DType::BOOL:
        case DType::U8:
        case DType::I8:
        case DType::E4M3:
        case DType::E5M2:
            return 1;
        case DType::U16:
        case DType::I16:
        case DType::F16:
        case DType::BF16:
            return 2;
        case DType::U32:
        case DType::I32:
        case DType::F32:
            return 4;
        case DType::U64:
        case DType::I64:
        case DType::F64:
            return 8;
    }
}

constexpr auto to_string(DType dtype) noexcept -> std::string_view {
    switch (dtype) {
        case DType::BOOL:
            return "bool";
        case DType::U8:
            return "u8";
        case DType::I8:
            return "i8";
        case DType::U16:
            return "u16";
        case DType::I16:
            return "i16";
        case DType::U32:
            return "u32";
        case DType::I32:
            return "i32";
        case DType::U64:
            return "u64";
        case DType::I64:
            return "i64";
        case DType::F16:
            return "f16";
        case DType::BF16:
            return "bf16";
        case DType::F32:
            return "f32";
        case DType::F64:
            return "f64";
        case DType::E4M3:
            return "e4m3";
        case DType::E5M2:
            return "e5m2";
    }
}

template <typename Value>
struct DTypeOf;

template <>
struct DTypeOf<float> {
    static constexpr DType value = DType::F32;
};

template <>
struct DTypeOf<double> {
    static constexpr DType value = DType::F64;
};

template <>
struct DTypeOf<std::int8_t> {
    static constexpr DType value = DType::I8;
};

template <>
struct DTypeOf<std::uint8_t> {
    static constexpr DType value = DType::U8;
};

template <>
struct DTypeOf<std::int16_t> {
    static constexpr DType value = DType::I16;
};

template <>
struct DTypeOf<std::uint16_t> {
    static constexpr DType value = DType::U16;
};

template <>
struct DTypeOf<std::int32_t> {
    static constexpr DType value = DType::I32;
};

template <>
struct DTypeOf<std::uint32_t> {
    static constexpr DType value = DType::U32;
};

template <>
struct DTypeOf<std::int64_t> {
    static constexpr DType value = DType::I64;
};

template <>
struct DTypeOf<std::uint64_t> {
    static constexpr DType value = DType::U64;
};

template <typename Value>
inline constexpr DType dtype_of = DTypeOf<Value>::value;

} // namespace kidi::tensor