#include "kidi/model/package.h"

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace kidi::model {
namespace {

constexpr std::string_view MANIFEST_FILENAME = "model.yaml";

auto validate_tokenizer(const text::Tokenizer& tokenizer, std::int32_t expected_size,
                        const model::SpecialTokenIds& special_tokens, std::string_view label) -> std::optional<Error> {
    if (tokenizer.vocabulary_size() != static_cast<std::size_t>(expected_size)) {
        return Error{
            ErrorCode::INVALID_MANIFEST,
            std::string(label) + " tokenizer vocabulary size is " + std::to_string(tokenizer.vocabulary_size()) +
                "; expected " + std::to_string(expected_size),
        };
    }

    const std::array expected_tokens = {
        std::pair<std::string_view, std::int32_t>{"<pad>", special_tokens.pad},
        std::pair<std::string_view, std::int32_t>{"<unk>", special_tokens.unknown},
        std::pair<std::string_view, std::int32_t>{"<s>", special_tokens.begin},
        std::pair<std::string_view, std::int32_t>{"</s>", special_tokens.end},
    };
    for (const auto& [token, expected_id] : expected_tokens) {
        const auto actual_id = tokenizer.token_id(token);
        if (!actual_id || *actual_id != expected_id) {
            return Error{
                ErrorCode::INVALID_MANIFEST,
                std::string(label) + " tokenizer maps " + std::string(token) + " to " +
                    (actual_id ? std::to_string(*actual_id) : "no ID") + "; expected " + std::to_string(expected_id),
            };
        }
    }
    return std::nullopt;
}

} // namespace

Package::Package(model::ModelManifest manifest, model::Weights weights, text::Tokenizer source_tokenizer,
                 text::Tokenizer target_tokenizer)
    : manifest_(std::move(manifest)),
      weights_(std::move(weights)),
      source_tokenizer_(std::move(source_tokenizer)),
      target_tokenizer_(std::move(target_tokenizer)) {}

auto Package::load(const std::filesystem::path& model_directory) -> Result<Package> {
    if (!std::filesystem::is_directory(model_directory)) {
        return std::unexpected(Error{ErrorCode::IO, "not a model directory: " + model_directory.string()});
    }
    auto manifest = model::ModelManifest::load(model_directory / MANIFEST_FILENAME);
    if (!manifest) {
        return std::unexpected(std::move(manifest.error()));
    }
    auto weights = model::Weights::load(manifest->weights_file);
    if (!weights) {
        return std::unexpected(std::move(weights.error()));
    }
    auto source = text::Tokenizer::load(manifest->tokenizers.source);
    if (!source) {
        return std::unexpected(std::move(source.error()));
    }
    auto target = text::Tokenizer::load(manifest->tokenizers.target);
    if (!target) {
        return std::unexpected(std::move(target.error()));
    }

    if (auto error = validate_tokenizer(*source, manifest->architecture.source_vocabulary_size,
                                        manifest->special_tokens, "source")) {
        return std::unexpected(std::move(*error));
    }
    if (auto error = validate_tokenizer(*target, manifest->architecture.target_vocabulary_size,
                                        manifest->special_tokens, "target")) {
        return std::unexpected(std::move(*error));
    }
    return Package(std::move(*manifest), std::move(*weights), std::move(*source), std::move(*target));
}

auto Package::manifest() const noexcept -> const model::ModelManifest& { return manifest_; }

auto Package::weights() const noexcept -> const model::Weights& { return weights_; }

auto Package::source_tokenizer() const noexcept -> const text::Tokenizer& { return source_tokenizer_; }

auto Package::target_tokenizer() const noexcept -> const text::Tokenizer& { return target_tokenizer_; }

} // namespace kidi::model