#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string_view>

#include "kidi/core/error.h"

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

struct TensorView {
    DataType data_type;
    std::span<const std::int64_t> shape;
    std::span<const std::byte> bytes;

    [[nodiscard]] std::size_t element_count() const noexcept;
    [[nodiscard]] std::size_t element_size() const noexcept;
    [[nodiscard]] const void* data() const noexcept;
};

class Weights {
public:
    Weights(Weights&&) noexcept;
    Weights& operator=(Weights&&) noexcept;
    ~Weights();

    Weights(const Weights&) = delete;
    Weights& operator=(const Weights&) = delete;

    [[nodiscard]] static Result<Weights> load(const std::filesystem::path& path);

    [[nodiscard]] bool contains(std::string_view name) const;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] Result<TensorView> tensor(std::string_view name) const;

private:
    struct Impl;
    explicit Weights(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

} // namespace kidi::model