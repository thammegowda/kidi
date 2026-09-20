#include "kidi/inference/generator.h"
#include "kidi/model/config.h"

#include <chrono>

namespace kidi::inference {
using ops::require;
namespace {
using Clock = std::chrono::steady_clock;
auto elapsed(Clock::time_point start) -> std::uint64_t {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}
} // namespace

Generator::Generator(YAML::Node config, text::Tokenizer tokenizer, model::Gemma4 model,
                     std::array<std::int32_t, 3> special)
    : config_(std::move(config)), tokenizer_(std::move(tokenizer)), model_(std::move(model)), special_(special) {}
auto Generator::load(const std::filesystem::path& directory, tensor::Device device, std::int32_t weight_bits,
                     std::int32_t group_size, bool packed_prefill) -> Result<Generator> {
    try {
        auto config = require(model::load_config(directory / "model.yaml"));
        require(model::Gemma4Impl::validate_config(config["model"]));
        auto tokenizer = require(text::Tokenizer::load(config["tokenizer_file"].as<std::string>()));
        if (tokenizer.vocabulary_size() != config["model"]["vocab_size"].as<std::size_t>())
            throw ops::Failure({ErrorCode::INVALID_MANIFEST, "Gemma tokenizer vocabulary mismatch"});
        std::array<std::int32_t, 3> special;
        const std::array names{"<eos>", "<turn|>", "<pad>"};
        for (std::size_t index = 0; index < names.size(); ++index) {
            const auto token = tokenizer.token_id(names[index]);
            if (!token)
                throw ops::Failure({ErrorCode::INVALID_MANIFEST, "Gemma tokenizer lacks required special token"});
            special[index] = *token;
        }
        if (!tokenizer.token_id("<bos>") || !tokenizer.token_id("<|turn>"))
            throw ops::Failure({ErrorCode::INVALID_MANIFEST, "Gemma tokenizer lacks chat delimiters"});
        auto weights = require(model::Weights::load(config["weights_file"].as<std::string>()));
        const auto parameter = require(weights.tensor(config["model"]["quantization_config"]
                                                          ? "model.language_model.norm.weight"
                                                          : "model.language_model.embed_tokens.weight"));
        const ModuleScope construction(parameter.dtype(), false, device);
        auto model = require(model::Gemma4Impl::create(config["model"]));
        require(model->set_checkpoint(weights, weight_bits, group_size, packed_prefill));
        return Generator(std::move(config), std::move(tokenizer), std::move(model), special);
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, error.what()});
    }
}
auto Generator::generate(std::string_view prompt, GenerationOptions options) -> Result<TextGeneration> {
    try {
        const auto started = Clock::now();
        const auto preparation = model_->preparation_ns();
        TextGeneration result;
        const auto serialized = options.raw_prompt
                                    ? std::string(prompt)
                                    : "<bos><|turn>user\n" + std::string(prompt) + "<turn|>\n<|turn>model\n";
        const auto tokens = require(tokenizer_.encode(serialized));
        result.stats.tokenize_ns = elapsed(started);
        result.stats.prompt_tokens = tokens.size();
        const auto maximum = options.maximum_new_tokens ? options.maximum_new_tokens
                                                        : config_["decode"]["maximum_new_tokens"].as<std::size_t>();
        const auto capacity =
            options.context_size ? options.context_size : config_["decode"]["context_size"].as<std::size_t>();
        if (!maximum || !options.prefill_chunk_size || tokens.empty() || tokens.size() >= capacity ||
            maximum > capacity - tokens.size())
            throw ops::Failure(
                {ErrorCode::INVALID_ARGUMENT, "prompt and generation must fit the context; counts must be positive"});
        auto state = require(model_->create_state(capacity));
        state.crop_local_attention = !options.full_attention_cache;
        const std::array extra_stops{special_[1]};
        const SearchOptions search{.vocabulary_size = tokenizer_.vocabulary_size(),
                                   .end_id = special_[0],
                                   .pad_id = special_[2],
                                   .maximum_steps = maximum,
                                   .compute_score = false,
                                   .stop_ids = extra_stops,
                                   .stop_on_eos = !options.ignore_eos};
        tensor::Tensor storage;
        result.generation =
            require(Decoder::generate(tokens, search, [&](const DecodeRequest& request) -> Result<TokenScores> {
                if (request.generated_steps == 1) result.stats.time_to_first_token_ns = elapsed(started);
                const auto step_start = Clock::now();
                if (!request.generated_steps) {
                    std::size_t offset = 0;
                    while (tokens.size() - offset > options.prefill_chunk_size) {
                        auto status =
                            model_->prefill(std::span(tokens).subspan(offset, options.prefill_chunk_size), state);
                        if (!status) return std::unexpected(std::move(status.error()));
                        offset += options.prefill_chunk_size;
                    }
                    auto logits = model_->forward(std::span(tokens).subspan(offset), state);
                    if (!logits) return std::unexpected(std::move(logits.error()));
                    storage = std::move(*logits);
                    result.stats.prefill_ns += elapsed(step_start);
                } else {
                    auto logits = model_->forward(request.prefixes.last(1), state);
                    if (!logits) return std::unexpected(std::move(logits.error()));
                    storage = std::move(*logits);
                    result.stats.decode_ns += elapsed(step_start);
                    ++result.stats.decode_tokens;
                }
                auto values = storage.data<float>();
                if (!values) return std::unexpected(std::move(values.error()));
                return TokenScores{*values};
            }));
        if (!result.stats.time_to_first_token_ns) result.stats.time_to_first_token_ns = elapsed(started);
        result.stats.preparation_ns = model_->preparation_ns() - preparation;
        result.text = require(tokenizer_.decode(result.generation.token_ids));
        return result;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, error.what()});
    }
}
} // namespace kidi::inference