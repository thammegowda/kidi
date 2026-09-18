#include "kidi/text/tokenizer.h"

#include <array>
#include <fstream>
#include <iterator>
#include <utility>

#include <tokenizers/tokenizer.h>
#include <zlib.h>

namespace kidi::text {
namespace {

Result<std::string> read_plain(const std::filesystem::path& path) {
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

Result<std::string> read_gzip(const std::filesystem::path& path) {
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

bool has_gzip_suffix(const std::filesystem::path& path) { return path.extension() == ".gz"; }

} // namespace

struct Tokenizer::Impl {
    explicit Impl(tokenizers::Tokenizer value) : value(std::move(value)) {}

    tokenizers::Tokenizer value;
};

Tokenizer::Tokenizer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;
Tokenizer::~Tokenizer() = default;

Result<Tokenizer> Tokenizer::load(const std::filesystem::path& path) {
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
    return Tokenizer(std::make_unique<Impl>(std::move(*tokenizer)));
}

Result<std::vector<std::int32_t>> Tokenizer::encode(std::string_view text) const {
    auto encoding = impl_->value.encode(text, false);
    if (!encoding) {
        return std::unexpected(Error{ErrorCode::RUNTIME, "tokenizer encode failed: " + encoding.error().message()});
    }
    return encoding->get_ids();
}

Result<std::string> Tokenizer::decode(std::span<const std::int32_t> ids) const {
    const std::vector<std::int32_t> owned_ids(ids.begin(), ids.end());
    auto text = impl_->value.decode(owned_ids, false);
    if (!text) {
        return std::unexpected(Error{ErrorCode::RUNTIME, "tokenizer decode failed: " + text.error().message()});
    }
    return std::move(*text);
}

std::size_t Tokenizer::vocabulary_size() const noexcept { return impl_->value.get_vocab_size(); }

std::optional<std::int32_t> Tokenizer::token_id(std::string_view token) const {
    return impl_->value.token_to_id(token);
}

} // namespace kidi::text