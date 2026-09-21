#pragma once

#include <functional>
#include <memory>
#include <string>
#include "kidi/core/error.h"

namespace kidi::tensor {
class Tensor;
}

namespace kidi::runtime::mps {
class CommandBatch {
public:
    CommandBatch(CommandBatch&&) noexcept;
    auto operator=(CommandBatch&&) noexcept -> CommandBatch&;
    ~CommandBatch();
    static auto create() -> Result<CommandBatch>;
    auto restart() -> Result<void>;
    auto finish() -> Result<void>;
    auto native_handle() const noexcept -> void*;
    auto track_completion() -> std::function<void(std::string)>;
    auto copy_(const tensor::Tensor& source, tensor::Tensor& destination) -> Result<void>;
    auto copy_slice_(const tensor::Tensor& source, tensor::Tensor& destination, std::size_t outer,
                     std::size_t source_bytes, std::size_t destination_bytes, std::size_t offset_bytes) -> Result<void>;
    auto scatter_(tensor::Tensor& destination, const tensor::Tensor& updates, const tensor::Tensor& indices)
        -> Result<void>;

private:
    struct Impl;
    explicit CommandBatch(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
} // namespace kidi::runtime::mps