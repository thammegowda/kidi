#pragma once

#include <string>
#include <string_view>

#include "kidi/tensor/dtype.h"

namespace kidi::model {

using DataType = tensor::DType;

enum class WeightEncoding {
    F32,
    BF16,
    INT8_PER_CHANNEL,
};

struct WeightSpec {
    std::string format;
    WeightEncoding encoding;
};

constexpr auto matrix_data_type(WeightEncoding encoding) noexcept -> DataType {
    switch (encoding) {
        case WeightEncoding::F32:
            return DataType::F32;
        case WeightEncoding::BF16:
            return DataType::BF16;
        case WeightEncoding::INT8_PER_CHANNEL:
            return DataType::I8;
    }
}

constexpr auto to_string(WeightEncoding encoding) noexcept -> std::string_view {
    switch (encoding) {
        case WeightEncoding::F32:
            return "F32";
        case WeightEncoding::BF16:
            return "BF16";
        case WeightEncoding::INT8_PER_CHANNEL:
            return "INT8_PER_CHANNEL";
    }
}

inline auto quantization_scale_name(std::string_view weight_name) -> std::string {
    return std::string(weight_name) + ".scale";
}

} // namespace kidi::model