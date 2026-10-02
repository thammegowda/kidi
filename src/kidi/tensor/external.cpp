#include "kidi/tensor/external.h"

#include <cstring>

namespace kidi::tensor {
namespace {
class ExternalStorage final : public Storage {
public:
    ExternalStorage(std::size_t bytes, std::shared_ptr<const RowSource> source)
        : bytes_(bytes), source_(std::move(source)) {}
    auto device() const noexcept -> Device override { return Device::cpu(); }
    auto size_bytes() const noexcept -> std::size_t override { return bytes_; }
    auto source() const noexcept -> const RowSource& { return *source_; }

private:
    std::size_t bytes_;
    std::shared_ptr<const RowSource> source_;
};

auto row_bytes(const Tensor& table) -> std::size_t { return table.size(0) ? table.nbytes() / table.size(0) : 0; }
} // namespace

auto external_tensor(std::vector<std::int64_t> shape, DType dtype, std::shared_ptr<const RowSource> source)
    -> Result<Tensor> {
    if (shape.empty() || !source || std::ranges::any_of(shape, [](std::int64_t extent) { return extent <= 0; }))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "external tensors need a source and a positive shape"});
    std::size_t bytes = element_size(dtype);
    for (const auto extent : shape) bytes *= static_cast<std::size_t>(extent);
    auto backend = BackendRegistry::instance().backend(Device::cpu());
    if (!backend) return std::unexpected(std::move(backend.error()));
    detail::Dimensions strides(shape.size());
    std::int64_t stride = 1;
    for (std::size_t axis = shape.size(); axis-- > 0;) {
        strides[axis] = stride;
        stride *= shape[axis];
    }
    return Tensor(std::move(*backend), std::make_shared<ExternalStorage>(bytes, std::move(source)), dtype,
                  std::move(shape), std::move(strides), 0);
}

auto is_external(const Tensor& tensor) noexcept -> bool {
    return dynamic_cast<const ExternalStorage*>(tensor.storage_.get()) != nullptr;
}

auto gather_rows(const Tensor& table, std::span<const std::int32_t> rows, std::span<std::byte> destination)
    -> Result<void> {
    if (!table.defined() || !table.dimensions() || !table.is_contiguous())
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "row gathers need a contiguous table"});
    const auto width = row_bytes(table);
    if (destination.size() != rows.size() * width)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "row gather destination has the wrong size"});
    for (const auto row : rows)
        if (row < 0 || static_cast<std::size_t>(row) >= table.size(0))
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "gathered row outside table"});
    if (const auto* external = dynamic_cast<const ExternalStorage*>(table.storage_.get()))
        return external->source().gather(rows, destination);
    const auto bytes = table.host_bytes();
    if (!bytes) return std::unexpected(bytes.error());
    for (std::size_t index = 0; index < rows.size(); ++index)
        std::memcpy(destination.data() + index * width, bytes->data() + static_cast<std::size_t>(rows[index]) * width,
                    width);
    return {};
}

} // namespace kidi::tensor
