#include "kidi/model/package.h"
#include "kidi/model/transformer.h"

#include <array>
#include <set>
#include <string>
#include <string_view>
#include <utility>

namespace kidi::model {
namespace {

constexpr std::string_view MANIFEST_FILENAME = "model.yaml";

auto tokenizer_tokens(const text::Tokenizer& tokenizer, std::int32_t expected_size, std::string_view label)
    -> Result<YAML::Node> {
    if (tokenizer.vocabulary_size() != static_cast<std::size_t>(expected_size)) {
        return std::unexpected(Error{
            ErrorCode::INVALID_MANIFEST,
            std::string(label) + " tokenizer vocabulary size is " + std::to_string(tokenizer.vocabulary_size()) +
                "; expected " + std::to_string(expected_size),
        });
    }

    const std::array tokens = {
        std::pair{"pad", "<pad>"},
        std::pair{"unknown", "<unk>"},
        std::pair{"begin", "<s>"},
        std::pair{"end", "</s>"},
    };
    YAML::Node result(YAML::NodeType::Map);
    std::set<std::int32_t> ids;
    for (const auto& [name, token] : tokens) {
        const auto id = tokenizer.token_id(token);
        if (!id || *id < 0 || *id >= expected_size || !ids.insert(*id).second)
            return std::unexpected(
                Error{ErrorCode::INVALID_MANIFEST,
                      std::string(label) + " tokenizer has a missing, invalid or duplicate special token: " + token});
        result[name] = *id;
    }
    return result;
}

} // namespace

Package::Package(YAML::Node config, model::Weights weights, text::Tokenizer source_tokenizer,
                 text::Tokenizer target_tokenizer)
    : config_(std::move(config)),
      weights_(std::move(weights)),
      source_tokenizer_(std::move(source_tokenizer)),
      target_tokenizer_(std::move(target_tokenizer)) {}

auto Package::load(const std::filesystem::path& model_directory) -> Result<Package> {
    try {
        if (!std::filesystem::is_directory(model_directory)) {
            return std::unexpected(Error{ErrorCode::IO, "not a model directory: " + model_directory.string()});
        }
        auto config = load_config(model_directory / MANIFEST_FILENAME);
        if (!config) {
            return std::unexpected(std::move(config.error()));
        }
        auto weights =
            model::Weights::load((*config)["weights_file"].as<std::string>(), TransformerImpl::state_mapping_specs());
        if (!weights) {
            return std::unexpected(std::move(weights.error()));
        }
        auto source = text::Tokenizer::load((*config)["tokenizers"]["source"].as<std::string>());
        if (!source) {
            return std::unexpected(std::move(source.error()));
        }
        auto target = text::Tokenizer::load((*config)["tokenizers"]["target"].as<std::string>());
        if (!target) {
            return std::unexpected(std::move(target.error()));
        }

        const auto model = (*config)["model"];
        auto source_tokens = tokenizer_tokens(*source, model["source_vocabulary_size"].as<std::int32_t>(), "source");
        if (!source_tokens) return std::unexpected(std::move(source_tokens.error()));
        auto target_tokens = tokenizer_tokens(*target, model["target_vocabulary_size"].as<std::int32_t>(), "target");
        if (!target_tokens) return std::unexpected(std::move(target_tokens.error()));
        (*config)["model"]["source_pad_id"] = (*source_tokens)["pad"].as<std::int32_t>();
        auto decode = (*config)["decode"];
        decode["source_end_id"] = (*source_tokens)["end"].as<std::int32_t>();
        decode["begin_id"] = (*target_tokens)["begin"].as<std::int32_t>();
        decode["end_id"] = (*target_tokens)["end"].as<std::int32_t>();
        decode["pad_id"] = (*target_tokens)["pad"].as<std::int32_t>();
        decode["vocabulary_size"] = target->vocabulary_size();
        return Package(std::move(*config), std::move(*weights), std::move(*source), std::move(*target));
    } catch (const YAML::Exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, error.what()});
    }
}

auto Package::config() const noexcept -> const YAML::Node& { return config_; }

auto Package::weights() const noexcept -> const model::Weights& { return weights_; }

auto Package::source_tokenizer() const noexcept -> const text::Tokenizer& { return source_tokenizer_; }

auto Package::target_tokenizer() const noexcept -> const text::Tokenizer& { return target_tokenizer_; }

} // namespace kidi::model