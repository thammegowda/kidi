#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "kidi/core/error.h"

namespace kidi::text {

class Tokenizer {
public:
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;
    ~Tokenizer();

    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    [[nodiscard]] static std::expected<Tokenizer, core::Error> load(const std::filesystem::path& path);

    [[nodiscard]] std::expected<std::vector<std::int32_t>, core::Error> encode(std::string_view text) const;
    [[nodiscard]] std::expected<std::string, core::Error> decode(std::span<const std::int32_t> ids) const;
    [[nodiscard]] std::size_t vocabulary_size() const noexcept;
    [[nodiscard]] std::optional<std::int32_t> token_id(std::string_view token) const;

private:
    struct Impl;
    explicit Tokenizer(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

} // namespace kidi::text