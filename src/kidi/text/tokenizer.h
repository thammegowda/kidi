#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "kidi/core/error.h"

namespace kidi::text {

struct ChatMessage {
    std::string role, content;
};

class Tokenizer {
public:
    Tokenizer(Tokenizer&&) noexcept;
    auto operator=(Tokenizer&&) noexcept -> Tokenizer&;
    ~Tokenizer();

    Tokenizer(const Tokenizer&) = delete;
    auto operator=(const Tokenizer&) -> Tokenizer& = delete;

    static auto load(const std::filesystem::path& path) -> Result<Tokenizer>;

    auto encode(std::string_view text) const -> Result<std::vector<std::int32_t>>;
    auto decode(std::span<const std::int32_t> ids) const -> Result<std::string>;
    auto decode_delta(std::span<const std::int32_t> ids, std::string& emitted, bool final = false) const
        -> Result<std::string>;
    auto format_chat(std::span<const ChatMessage> messages) const -> Result<std::string>;
    auto vocabulary_size() const noexcept -> std::size_t;
    auto token_id(std::string_view token) const -> std::optional<std::int32_t>;

private:
    struct Impl;
    explicit Tokenizer(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

} // namespace kidi::text