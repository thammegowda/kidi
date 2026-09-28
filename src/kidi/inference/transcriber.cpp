#include "kidi/inference/transcriber.h"

#include <algorithm>
#include <chrono>
#include <limits>

#include "kidi/model/config.h"

namespace kidi::inference {
using ops::require;
namespace {
using Clock = std::chrono::steady_clock;
auto elapsed(Clock::time_point start) -> std::uint64_t {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}
auto token_list(const YAML::Node& node) -> std::vector<std::int32_t> {
    std::vector<std::int32_t> result;
    for (const auto& value : node) result.push_back(value.as<std::int32_t>());
    return result;
}
} // namespace

Transcriber::Transcriber(YAML::Node config, YAML::Node generation, audio::WhisperFeatureExtractor extractor,
                         text::Tokenizer tokenizer, model::Whisper model, std::map<std::string, std::int32_t> languages,
                         std::vector<std::int32_t> suppress, std::vector<std::int32_t> begin_suppress)
    : config_(std::move(config)),
      generation_(std::move(generation)),
      extractor_(std::move(extractor)),
      tokenizer_(std::move(tokenizer)),
      model_(std::move(model)),
      languages_(std::move(languages)),
      suppress_(std::move(suppress)),
      begin_suppress_(std::move(begin_suppress)) {}

auto Transcriber::load(const std::filesystem::path& directory, tensor::Device device) -> Result<Transcriber> {
    try {
        if (device != tensor::Device::cpu())
            throw ops::Failure({ErrorCode::UNSUPPORTED, "Whisper currently requires the CPU backend"});
        auto config = require(model::load_whisper_config(directory));
        require(model::WhisperImpl::validate_config(config["model"]));
        auto tokenizer = require(text::Tokenizer::load(config["tokenizer_file"].as<std::string>()));
        if (tokenizer.vocabulary_size() != config["model"]["vocab_size"].as<std::size_t>())
            throw ops::Failure({ErrorCode::INVALID_MANIFEST, "Whisper tokenizer vocabulary mismatch"});
        auto generation = YAML::LoadFile(config["generation_config_file"].as<std::string>());
        if (!generation.IsMap() || !generation["lang_to_id"].IsMap() || generation["lang_to_id"].size() == 0 ||
            !generation["task_to_id"].IsMap())
            throw ops::Failure({ErrorCode::INVALID_MANIFEST, "invalid Whisper generation config"});
        const auto vocabulary = config["model"]["vocab_size"].as<std::int32_t>();
        const auto valid_token = [&](std::int32_t token) { return token >= 0 && token < vocabulary; };
        std::map<std::string, std::int32_t> languages;
        for (const auto& item : generation["lang_to_id"]) {
            auto name = item.first.as<std::string>();
            if (!name.starts_with("<|") || !name.ends_with("|>") || name.size() <= 4)
                throw ops::Failure({ErrorCode::INVALID_MANIFEST, "invalid Whisper language token"});
            const auto token = item.second.as<std::int32_t>();
            if (!valid_token(token))
                throw ops::Failure({ErrorCode::INVALID_MANIFEST, "Whisper language token is outside vocabulary"});
            languages.emplace(name.substr(2, name.size() - 4), token);
        }
        for (const auto& item : generation["task_to_id"])
            if (!valid_token(item.second.as<std::int32_t>()))
                throw ops::Failure({ErrorCode::INVALID_MANIFEST, "Whisper task token is outside vocabulary"});
        for (const auto* name : {"decoder_start_token_id", "eos_token_id", "no_timestamps_token_id"})
            if (!valid_token(generation[name].as<std::int32_t>()))
                throw ops::Failure({ErrorCode::INVALID_MANIFEST, "Whisper special token is outside vocabulary"});
        auto weights = require(model::Weights::load(config["model_file"].as<std::string>()));
        const auto parameter = require(weights.tensor("model.encoder.layers.0.fc1.weight"));
        const ModuleScope construction(parameter.dtype(), false, device);
        auto whisper = require(model::WhisperImpl::create(config["model"]));
        require(whisper->set_checkpoint(weights));
        auto extractor =
            require(audio::WhisperFeatureExtractor::load(config["preprocessor_config_file"].as<std::string>()));
        auto suppress = token_list(generation["suppress_tokens"]);
        auto begin_suppress = token_list(generation["begin_suppress_tokens"]);
        if (!std::ranges::all_of(suppress, valid_token) || !std::ranges::all_of(begin_suppress, valid_token))
            throw ops::Failure({ErrorCode::INVALID_MANIFEST, "Whisper suppression token is outside vocabulary"});
        return Transcriber(std::move(config), std::move(generation), std::move(extractor), std::move(tokenizer),
                           std::move(whisper), std::move(languages), std::move(suppress), std::move(begin_suppress));
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const YAML::Exception& error) {
        return std::unexpected(
            Error{ErrorCode::INVALID_MANIFEST, "invalid Whisper metadata: " + std::string(error.what())});
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, error.what()});
    }
}

auto Transcriber::transcribe(std::span<const float> waveform, std::uint32_t sample_rate,
                             const TranscriptionOptions& options) -> Result<Transcription> {
    try {
        const auto task = generation_["task_to_id"][options.task];
        if (!task) throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Whisper task must be transcribe or translate"});
        const auto maximum_position = config_["model"]["max_target_positions"].as<std::size_t>();
        if (!options.maximum_tokens || options.maximum_tokens + 4 > maximum_position)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Whisper output token limit is outside model capacity"});
        Transcription result;
        const auto feature_start = Clock::now();
        auto features = require(extractor_.extract(waveform, sample_rate));
        result.stats.feature_ns = elapsed(feature_start);
        const auto preparation = model_->preparation_ns();
        const auto encode_start = Clock::now();
        auto source = require(model_->encode(features));
        result.stats.encode_ns = elapsed(encode_start);
        auto state = require(model_->create_state(options.maximum_tokens + 4));
        const auto decode_start = Clock::now();
        const auto start_token = generation_["decoder_start_token_id"].as<std::int32_t>();
        auto logits = require(model_->forward(source, std::span(&start_token, 1), state));
        auto scores = require(logits.data<float>());
        std::int32_t language_token = -1;
        if (options.language == "auto") {
            auto best = -std::numeric_limits<float>::infinity();
            for (const auto& [language, token] : languages_)
                if (scores[token] > best) {
                    best = scores[token];
                    result.language = language;
                    language_token = token;
                }
        } else {
            const auto found = languages_.find(options.language);
            if (found == languages_.end())
                throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "unsupported Whisper language: " + options.language});
            result.language = found->first;
            language_token = found->second;
        }
        for (const auto token :
             {language_token, task.as<std::int32_t>(), generation_["no_timestamps_token_id"].as<std::int32_t>()})
            logits = require(model_->forward(source, std::span(&token, 1), state));

        const auto vocabulary = config_["model"]["vocab_size"].as<std::size_t>();
        const auto end = generation_["eos_token_id"].as<std::int32_t>();
        const auto no_timestamps = generation_["no_timestamps_token_id"].as<std::int32_t>();
        std::string emitted;
        for (std::size_t step = 0; step < options.maximum_tokens; ++step) {
            scores = require(logits.data<float>());
            const auto suppress = [&](std::int32_t token) {
                if (token >= 0 && static_cast<std::size_t>(token) < scores.size())
                    scores[token] = -std::numeric_limits<float>::infinity();
            };
            for (const auto token : suppress_) suppress(token);
            if (!step)
                for (const auto token : begin_suppress_) suppress(token);
            suppress(no_timestamps);
            for (std::size_t token = static_cast<std::size_t>(no_timestamps + 1); token < vocabulary; ++token)
                scores[token] = -std::numeric_limits<float>::infinity();
            const auto selected = static_cast<std::int32_t>(std::ranges::max_element(scores) - scores.begin());
            if (selected == end) break;
            result.token_ids.push_back(selected);
            if (options.on_partial) {
                const auto delta = require(tokenizer_.decode_delta(result.token_ids, emitted));
                if (!delta.empty()) options.on_partial(emitted, result.language);
            }
            logits = require(model_->forward(source, std::span(&selected, 1), state));
        }
        result.stats.decode_ns = elapsed(decode_start);
        result.stats.preparation_ns = model_->preparation_ns() - preparation;
        result.text = require(tokenizer_.decode(result.token_ids));
        if (options.on_partial && result.text != emitted) options.on_partial(result.text, result.language);
        return result;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const YAML::Exception& error) {
        return std::unexpected(
            Error{ErrorCode::INVALID_MANIFEST, "invalid Whisper generation config: " + std::string(error.what())});
    }
}

} // namespace kidi::inference