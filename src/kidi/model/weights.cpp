#include "kidi/model/weights.h"

#include <optional>
#include <numeric>
#include <string>
#include <utility>

#include "kidi/model/safetensors/mapped.h"

namespace kidi::model {
namespace {

std::optional<DataType> parse_data_type(std::string_view value) {
    if (value == "BOOL") return DataType::BOOL;
    if (value == "U8") return DataType::U8;
    if (value == "I8") return DataType::I8;
    if (value == "U16") return DataType::U16;
    if (value == "I16") return DataType::I16;
    if (value == "U32") return DataType::U32;
    if (value == "I32") return DataType::I32;
    if (value == "U64") return DataType::U64;
    if (value == "I64") return DataType::I64;
    if (value == "F16") return DataType::F16;
    if (value == "BF16") return DataType::BF16;
    if (value == "F32") return DataType::F32;
    if (value == "F64") return DataType::F64;
    if (value == "F8_E4M3") return DataType::E4M3;
    if (value == "F8_E5M2") return DataType::E5M2;
    return std::nullopt;
}

std::size_t data_type_size(DataType data_type) {
    switch (data_type) {
        case DataType::BOOL:
        case DataType::U8:
        case DataType::I8:
        case DataType::E4M3:
        case DataType::E5M2:
            return 1;
        case DataType::U16:
        case DataType::I16:
        case DataType::F16:
        case DataType::BF16:
            return 2;
        case DataType::U32:
        case DataType::I32:
        case DataType::F32:
            return 4;
        case DataType::U64:
        case DataType::I64:
        case DataType::F64:
            return 8;
    }
}

} // namespace

struct Weights::Impl {
    explicit Impl(const std::filesystem::path& path) : checkpoint(path.string()) {}

    safetensors::MappedCheckpoint checkpoint;
};

std::size_t TensorView::element_count() const noexcept {
    return std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<>{});
}

std::size_t TensorView::element_size() const noexcept { return data_type_size(data_type); }

const void* TensorView::data() const noexcept { return bytes.data(); }

Weights::Weights(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Weights::Weights(Weights&&) noexcept = default;
Weights& Weights::operator=(Weights&&) noexcept = default;
Weights::~Weights() = default;

std::expected<Weights, core::Error> Weights::load(const std::filesystem::path& path) {
    try {
        return Weights(std::make_unique<Impl>(path));
    } catch (const safetensors::SafetensorsException& error) {
        return std::unexpected(core::Error{
            core::ErrorCode::INVALID_ARGUMENT,
            "cannot load Safetensors weights " + path.string() + ": " + error.what(),
        });
    }
}

bool Weights::contains(std::string_view name) const { return impl_->checkpoint.contains(std::string(name)); }

std::size_t Weights::size() const noexcept { return impl_->checkpoint.tensors().size(); }

std::expected<TensorView, core::Error> Weights::tensor(std::string_view name) const {
    if (!contains(name)) {
        return std::unexpected(
            core::Error{core::ErrorCode::INVALID_ARGUMENT, "weight tensor not found: " + std::string(name)});
    }
    const auto& tensor = impl_->checkpoint.at(std::string(name));
    const auto data_type = parse_data_type(tensor.dtype);
    if (!data_type) {
        return std::unexpected(core::Error{
            core::ErrorCode::UNSUPPORTED,
            "weight tensor " + std::string(name) + " has unsupported dtype " + tensor.dtype,
        });
    }
    const auto alignment = data_type_size(*data_type);
    if (!tensor.is_aligned(alignment)) {
        return std::unexpected(core::Error{
            core::ErrorCode::INVALID_ARGUMENT,
            "weight tensor " + std::string(name) + " is not aligned for zero-copy access",
        });
    }
    return TensorView{
        .data_type = *data_type,
        .shape = tensor.shape,
        .bytes = tensor.bytes(),
    };
}

} // namespace kidi::model