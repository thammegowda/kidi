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
#include "kidi/model/precision.h"
#include "kidi/tensor/tensor.h"

namespace kidi::model {

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

    auto contains(std::string_view name) const -> bool;
    auto size() const noexcept -> std::size_t;
    auto tensor(std::string_view name) const -> Result<tensor::Tensor>;

private:
    struct Impl;
    explicit Weights(std::shared_ptr<Impl> impl);

    std::shared_ptr<Impl> impl_;
};

} // namespace kidi::model