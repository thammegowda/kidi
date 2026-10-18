#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "kidi/core/error.h"

namespace kidi::text {

using TemplateArguments = std::map<std::string, std::string, std::less<>>;
using TemplateFiles = std::map<std::string, std::filesystem::path, std::less<>>;

struct ChatMessage {
    std::string role, content;
    std::vector<std::filesystem::path> images;
};

class Tokenizer {
public:
    Tokenizer(Tokenizer&&) noexcept;
    auto operator=(Tokenizer&&) noexcept -> Tokenizer&;
    ~Tokenizer();

    Tokenizer(const Tokenizer&) = delete;
    auto operator=(const Tokenizer&) -> Tokenizer& = delete;

    static auto load(const std::filesystem::path& path, const TemplateFiles& templates = {}) -> Result<Tokenizer>;

    auto encode(std::string_view text) const -> Result<std::vector<std::int32_t>>;
    auto decode(std::span<const std::int32_t> ids) const -> Result<std::string>;
    auto decode_delta(std::span<const std::int32_t> ids, std::string& emitted, bool final = false) const
        -> Result<std::string>;
    auto format_chat(std::span<const ChatMessage> messages) const -> Result<std::string>;
    auto format_template(std::string_view name, const TemplateArguments& arguments) const -> Result<std::string>;
    auto encode_template(std::string_view name, const TemplateArguments& arguments) const
        -> Result<std::vector<std::int32_t>>;
    auto vocabulary_size() const noexcept -> std::size_t;
    auto token_id(std::string_view token) const -> std::optional<std::int32_t>;

private:
    struct Impl;
    explicit Tokenizer(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

} // namespace kidi::text