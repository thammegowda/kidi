#pragma once

#include <filesystem>

#include "kidi/core/error.h"
#include "kidi/model/manifest.h"
#include "kidi/model/weights.h"
#include "kidi/text/tokenizer.h"

namespace kidi::rtg {

class Package {
public:
    Package(Package&&) noexcept = default;
    Package& operator=(Package&&) noexcept = default;

    Package(const Package&) = delete;
    Package& operator=(const Package&) = delete;

    [[nodiscard]] static Result<Package> load(const std::filesystem::path& manifest_path);

    [[nodiscard]] const model::ModelManifest& manifest() const noexcept;
    [[nodiscard]] const model::Weights& weights() const noexcept;
    [[nodiscard]] const text::Tokenizer& source_tokenizer() const noexcept;
    [[nodiscard]] const text::Tokenizer& target_tokenizer() const noexcept;

private:
    Package(model::ModelManifest manifest, model::Weights weights, text::Tokenizer source_tokenizer,
            text::Tokenizer target_tokenizer);

    model::ModelManifest manifest_;
    model::Weights weights_;
    text::Tokenizer source_tokenizer_;
    text::Tokenizer target_tokenizer_;
};

} // namespace kidi::rtg