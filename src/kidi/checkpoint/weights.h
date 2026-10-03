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

struct StateMappingSpec {
    std::vector<std::string> sources;
    std::string destination;
    std::int64_t concat_axis = 0;
};

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

    /// Adds a tensor supplied by the embedder rather than the file, such as a table kept outside linear memory.
    auto add(std::string name, tensor::Tensor value) -> Result<void>;
    auto contains(std::string_view name) const -> bool;
    auto size() const noexcept -> std::size_t;
    auto names() const -> std::vector<std::string>;
    auto tensor(std::string_view name) const -> Result<tensor::Tensor>;
    auto state_dict() const -> Result<StateDict>;

private:
    struct Impl;
    explicit Weights(std::shared_ptr<Impl> impl);

    std::shared_ptr<Impl> impl_;
};

} // namespace kidi::checkpoint