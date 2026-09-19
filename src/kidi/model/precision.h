#pragma once

#include <string>
#include <string_view>

namespace kidi::model {

enum class DataType {
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

enum class WeightEncoding {
    F32,
    BF16,
    INT8_PER_CHANNEL,
};

enum class LinearWeightLayout {
    OUTPUT_INPUT,
    INPUT_OUTPUT,
};

struct WeightSpec {
    std::string format;
    WeightEncoding encoding;
    LinearWeightLayout linear_layout;
};

[[nodiscard]] constexpr DataType matrix_data_type(WeightEncoding encoding) noexcept {
    switch (encoding) {
        case WeightEncoding::F32:
            return DataType::F32;
        case WeightEncoding::BF16:
            return DataType::BF16;
        case WeightEncoding::INT8_PER_CHANNEL:
            return DataType::I8;
    }
}

[[nodiscard]] constexpr std::string_view to_string(WeightEncoding encoding) noexcept {
    switch (encoding) {
        case WeightEncoding::F32:
            return "F32";
        case WeightEncoding::BF16:
            return "BF16";
        case WeightEncoding::INT8_PER_CHANNEL:
            return "INT8_PER_CHANNEL";
    }
}

[[nodiscard]] constexpr std::string_view to_string(LinearWeightLayout layout) noexcept {
    switch (layout) {
        case LinearWeightLayout::OUTPUT_INPUT:
            return "OUTPUT_INPUT";
        case LinearWeightLayout::INPUT_OUTPUT:
            return "INPUT_OUTPUT";
    }
}

[[nodiscard]] inline std::string quantization_scale_name(std::string_view weight_name) {
    return std::string(weight_name) + ".scale";
}

} // namespace kidi::model