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

namespace kidi::model {

struct TensorView {
    DataType data_type;
    std::span<const std::int64_t> shape;
    std::span<const std::byte> bytes;

    [[nodiscard]] std::size_t element_count() const noexcept;
    [[nodiscard]] std::size_t element_size() const noexcept;
    [[nodiscard]] const void* data() const noexcept;
};

struct StateMappingSpec {
    std::vector<std::string> sources;
    std::string destination;
    std::int64_t concat_axis = 0;
};

class Weights {
public:
    Weights(Weights&&) noexcept;
    Weights& operator=(Weights&&) noexcept;
    ~Weights();

    Weights(const Weights&) = delete;
    Weights& operator=(const Weights&) = delete;

    [[nodiscard]] static Result<Weights> load(const std::filesystem::path& path,
                                              std::span<const StateMappingSpec> mappings = {});

    [[nodiscard]] bool contains(std::string_view name) const;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] Result<TensorView> tensor(std::string_view name) const;

private:
    struct Impl;
    explicit Weights(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

} // namespace kidi::model