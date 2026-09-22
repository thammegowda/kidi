#include "kidi/text/tokenizer.h"

#include <array>
#include <fstream>
#include <iterator>
#include <utility>

#include <tokenizers/tokenizer.h>
#include <tokenizers/tokenizer_config.h>
#include <zlib.h>

namespace kidi::text {
namespace {

auto read_plain(const std::filesystem::path& path) -> Result<std::string> {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return std::unexpected(Error{ErrorCode::IO, "cannot open tokenizer: " + path.string()});
    }
    std::string contents(std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{});
    if (input.bad()) {
        return std::unexpected(Error{ErrorCode::IO, "cannot read tokenizer: " + path.string()});
    }
    return contents;
}

auto read_gzip(const std::filesystem::path& path) -> Result<std::string> {
    gzFile input = gzopen(path.c_str(), "rb");
    if (input == nullptr) {
        return std::unexpected(Error{ErrorCode::IO, "cannot open gzip tokenizer: " + path.string()});
    }

    std::string contents;
    std::array<char, 64 * 1024> buffer{};
    int count = 0;
    while ((count = gzread(input, buffer.data(), static_cast<unsigned int>(buffer.size()))) > 0) {
        contents.append(buffer.data(), static_cast<std::size_t>(count));
    }
    if (count < 0) {
        int error_number = Z_OK;
        const char* detail = gzerror(input, &error_number);
        const std::string message = detail == nullptr ? "unknown gzip error" : detail;
        gzclose(input);
        return std::unexpected(Error{
            ErrorCode::IO,
            "cannot decompress tokenizer " + path.string() + ": " + message,
        });
    }
    if (gzclose(input) != Z_OK) {
        return std::unexpected(Error{ErrorCode::IO, "cannot close gzip tokenizer: " + path.string()});
    }
    return contents;
}

auto has_gzip_suffix(const std::filesystem::path& path) -> bool { return path.extension() == ".gz"; }

} // namespace

struct Tokenizer::Impl {
    explicit Impl(tokenizers::Tokenizer value) : value(std::move(value)) {}

    tokenizers::Tokenizer value;
};

Tokenizer::Tokenizer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
auto Tokenizer::operator=(Tokenizer&&) noexcept -> Tokenizer& = default;
Tokenizer::~Tokenizer() = default;

auto Tokenizer::load(const std::filesystem::path& path) -> Result<Tokenizer> {
    if (!std::filesystem::is_regular_file(path)) {
        return std::unexpected(Error{ErrorCode::IO, "tokenizer does not exist: " + path.string()});
    }
    if (path.extension() != ".json" && !has_gzip_suffix(path)) {
        return std::unexpected(Error{
            ErrorCode::UNSUPPORTED,
            "tokenizer must use a .json or .json.gz extension: " + path.string(),
        });
    }

    auto contents = has_gzip_suffix(path) ? read_gzip(path) : read_plain(path);
    if (!contents) {
        return std::unexpected(std::move(contents.error()));
    }
    auto tokenizer = tokenizers::Tokenizer::from_string(*contents);
    if (!tokenizer) {
        return std::unexpected(Error{
            ErrorCode::INVALID_ARGUMENT,
            "cannot parse tokenizer " + path.string() + ": " + tokenizer.error().message(),
        });
    }
    tokenizers::TokenizerConfig config;
    const auto config_path = path.parent_path() / "tokenizer_config.json";
    if (std::filesystem::is_regular_file(config_path)) {
        auto source = read_plain(config_path);
        if (!source) return std::unexpected(std::move(source.error()));
        if (source->empty())
            return std::unexpected(Error{ErrorCode::IO, "empty tokenizer config read: " + config_path.string()});
        auto loaded = tokenizers::TokenizerConfig::from_json(*source);
        if (!loaded) return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, loaded.error().message()});
        config = std::move(*loaded);
    }
    const auto template_path = path.parent_path() / "chat_template.jinja";
    if (std::filesystem::is_regular_file(template_path)) {
        if (!std::filesystem::is_regular_file(config_path))
            return std::unexpected(
                Error{ErrorCode::INVALID_ARGUMENT, "chat_template.jinja requires tokenizer_config.json"});
        auto source = read_plain(template_path);
        if (!source) return std::unexpected(std::move(source.error()));
        config.default_chat_template = std::move(*source);
    }
    tokenizer->with_config(std::move(config));
    return Tokenizer(std::make_unique<Impl>(std::move(*tokenizer)));
}

auto Tokenizer::format_chat(std::span<const ChatMessage> messages) const -> Result<std::string> {
    if (messages.empty() || messages.back().role != "user")
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "messages must end with a user turn"});
    std::vector<tokenizers::ChatMessage> conversation;
    conversation.reserve(messages.size());
    for (std::size_t index = 0; index < messages.size(); ++index) {
        const auto& message = messages[index];
        const bool instruction = message.role == "system" || message.role == "developer";
        if ((instruction && index != 0) || (!instruction && message.role != "user" && message.role != "assistant"))
            return std::unexpected(
                Error{ErrorCode::INVALID_ARGUMENT, "supported roles: initial system/developer, user, assistant"});
        conversation.push_back({message.role, message.content});
    }
    if (!impl_->value.has_chat_template())
        return std::unexpected(
            Error{ErrorCode::INVALID_ARGUMENT, "chat requires tokenizer_config.json and a checkpoint chat template"});
    auto result = impl_->value.apply_chat_template(conversation, true, "default", {{"enable_thinking", false}});
    if (!result)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "chat template: " + result.error().message()});
    return std::move(*result);
}

auto Tokenizer::encode(std::string_view text) const -> Result<std::vector<std::int32_t>> {
    auto encoding = impl_->value.encode(text, false);
    if (!encoding) {
        return std::unexpected(Error{ErrorCode::RUNTIME, "tokenizer encode failed: " + encoding.error().message()});
    }
    return encoding->get_ids();
}

auto Tokenizer::decode(std::span<const std::int32_t> ids) const -> Result<std::string> {
    const std::vector<std::int32_t> owned_ids(ids.begin(), ids.end());
    auto text = impl_->value.decode(owned_ids, false);
    if (!text) {
        return std::unexpected(Error{ErrorCode::RUNTIME, "tokenizer decode failed: " + text.error().message()});
    }
    return std::move(*text);
}

auto Tokenizer::decode_delta(std::span<const std::int32_t> ids, std::string& emitted, bool final) const
    -> Result<std::string> {
    auto decoded = decode(ids);
    if (!decoded) return std::unexpected(std::move(decoded.error()));
    if (!decoded->starts_with(emitted))
        return std::unexpected(Error{ErrorCode::UNSUPPORTED, "tokenizer revised already streamed text"});
    auto end = decoded->size();
    if (!final) {
        const auto replacement = decoded->find("\xEF\xBF\xBD", emitted.size());
        if (replacement != std::string::npos) end = replacement;
        if (end) {
            auto lead = end - 1;
            while (lead && (static_cast<unsigned char>((*decoded)[lead]) & 0xC0) == 0x80) --lead;
            const auto byte = static_cast<unsigned char>((*decoded)[lead]);
            const std::size_t width = byte < 0x80 ? 1 : byte < 0xE0 ? 2 : byte < 0xF0 ? 3 : 4;
            if (end - lead < width) end = lead;
        }
    }
    auto delta = decoded->substr(emitted.size(), end - emitted.size());
    emitted.append(delta);
    return delta;
}

auto Tokenizer::vocabulary_size() const noexcept -> std::size_t { return impl_->value.get_vocab_size(); }

auto Tokenizer::token_id(std::string_view token) const -> std::optional<std::int32_t> {
    return impl_->value.token_to_id(token);
}

} // namespace kidi::text