#pragma once

#include <filesystem>

#include "kidi/core/error.h"
#include "kidi/model/config.h"
#include "kidi/model/weights.h"
#include "kidi/text/tokenizer.h"

namespace kidi::model {

class Package {
public:
    Package(Package&&) noexcept = default;
    auto operator=(Package&&) noexcept -> Package& = default;

    Package(const Package&) = delete;
    auto operator=(const Package&) -> Package& = delete;

    static auto load(const std::filesystem::path& model_directory) -> Result<Package>;

    auto config() const noexcept -> const YAML::Node&;
    auto weights() const noexcept -> const model::Weights&;
    auto source_tokenizer() const noexcept -> const text::Tokenizer&;
    auto target_tokenizer() const noexcept -> const text::Tokenizer&;

private:
    Package(YAML::Node config, model::Weights weights, text::Tokenizer source_tokenizer,
            text::Tokenizer target_tokenizer);

    YAML::Node config_;
    model::Weights weights_;
    text::Tokenizer source_tokenizer_;
    text::Tokenizer target_tokenizer_;
};

} // namespace kidi::model