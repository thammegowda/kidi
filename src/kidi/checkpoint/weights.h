#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "kidi/core/error.h"
#include "kidi/core/module.h"
#include "kidi/tensor/tensor.h"

namespace kidi::checkpoint {

using DataType = tensor::DType;

enum class StateMappingOp {
    CONCAT,
    IDENTICAL,
    IGNORE,
};

struct StateMappingSpec {
    std::vector<std::string> sources;
    std::string destination;
    std::int64_t concat_axis = 0;
    StateMappingOp op = StateMappingOp::CONCAT;
};

auto prefix_state_mappings(std::string_view source_prefix, std::string_view destination_prefix,
                           std::span<const StateMappingSpec> mappings) -> std::vector<StateMappingSpec>;

class Weights {
public:
    Weights(Weights&&) noexcept;
    auto operator=(Weights&&) noexcept -> Weights&;
    ~Weights();

    Weights(const Weights&) = delete;
    auto operator=(const Weights&) -> Weights& = delete;

    static auto load(const std::filesystem::path& path, std::span<const StateMappingSpec> mappings = {})
        -> Result<Weights>;
    static auto save(const std::filesystem::path& path, const StateDict& state) -> Result<void>;

    /// Returns a view with declarative model state mappings (renaming, projection fusion, scalar checks, ignores)
    /// applied without copying untouched tensors.
    auto mapped(std::span<const StateMappingSpec> mappings) const -> Result<Weights>;

    /// Adds a tensor supplied by the embedder rather than the file, such as a table kept outside linear memory.
    auto add(std::string name, tensor::Tensor value) -> Result<void>;
    auto contains(std::string_view name) const -> bool;
    auto size() const noexcept -> std::size_t;
    auto names() const -> std::vector<std::string>;
    auto tensor(std::string_view name) const -> Result<tensor::Tensor>;
    /// Expands FP16/BF16 checkpoint parameters to FP32; FP32 parameters remain zero-copy.
    auto float32_tensor(std::string_view name) const -> Result<tensor::Tensor>;
    auto state_dict() const -> Result<StateDict>;

private:
    struct Impl;
    explicit Weights(std::shared_ptr<Impl> impl);

    std::shared_ptr<Impl> impl_;
};

} // namespace kidi::checkpoint