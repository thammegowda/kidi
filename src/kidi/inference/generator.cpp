#include "kidi/inference/generator.h"
#include "kidi/runtime/operator.h"
#include "kidi/tensor/backend.h"
#include "kidi/checkpoint/config.h"

#include <chrono>
#include <optional>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <fstream>

namespace kidi::inference {
using ops::require;
namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t QNN_PREFILL_CAPTURE_TOKENS = 2048;

auto elapsed(Clock::time_point start) -> std::uint64_t {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}
auto should_capture_prefill(std::string_view accelerator, std::size_t prompt_tokens) -> bool {
    return accelerator != "qnn-htp" || prompt_tokens >= QNN_PREFILL_CAPTURE_TOKENS;
}
} // namespace

Generator::Generator(YAML::Node config, text::Tokenizer tokenizer, model::Gemma4 model,
                     std::array<std::int32_t, 3> special)
    : config_(std::move(config)), tokenizer_(std::move(tokenizer)), model_(std::move(model)), special_(special) {}
auto Generator::load(const std::filesystem::path& directory, tensor::Device device, std::int32_t weight_bits,
                     std::int32_t group_size, bool packed_prefill) -> Result<Generator> {
    try {
        auto config = require(checkpoint::load_config(directory / "model.yaml"));
        require(model::Gemma4Impl::validate_config(config["model"]));
        auto tokenizer = require(text::Tokenizer::load(config["tokenizer_file"].as<std::string>()));
        if (tokenizer.vocabulary_size() != config["model"]["vocab_size"].as<std::size_t>())
            throw ops::Failure({ErrorCode::INVALID_MANIFEST, "Gemma 4 tokenizer vocabulary mismatch"});
        std::array<std::int32_t, 3> special;
        const std::array names{"<eos>", "<turn|>", "<pad>"};
        for (std::size_t index = 0; index < names.size(); ++index) {
            const auto token = tokenizer.token_id(names[index]);
            if (!token)
                throw ops::Failure({ErrorCode::INVALID_MANIFEST, "Gemma 4 tokenizer lacks required special token"});
            special[index] = *token;
        }
        if (!tokenizer.token_id("<bos>") || !tokenizer.token_id("<|turn>"))
            throw ops::Failure({ErrorCode::INVALID_MANIFEST, "Gemma 4 tokenizer lacks chat delimiters"});
        auto weights = require(checkpoint::Weights::load(config["weights_file"].as<std::string>()));
        if (std::filesystem::is_regular_file(directory / "config.json") &&
            weights.contains("model.vision_tower.patch_embedder.input_proj.weight") &&
            weights.contains("model.embed_vision.embedding_projection.weight")) {
            const auto upstream = YAML::LoadFile((directory / "config.json").string());
            if (upstream["vision_config"] && !config["model"]["use_bidirectional_attention"].IsScalar()) {
                config["vision"] = upstream["vision_config"];
                config["image_token_id"] = upstream["image_token_id"];
            }
        }
        const auto parameter = require(weights.tensor(config["model"]["quantization_config"]
                                                          ? "model.language_model.norm.weight"
                                                          : "model.language_model.embed_tokens.weight"));
        std::optional<ops::StepCompilerScope> accelerator;
        if (device.kind == tensor::DeviceKind::Q_NPU) {
            accelerator.emplace(require(runtime::npu_step_compiler()));
            device = tensor::Device::cpu();
        }
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
auto select_device(std::string_view accelerator, bool speech) -> Result<tensor::Device> {
    const auto gpu = [] {
#if defined(__APPLE__)
        return tensor::Device::apple_gpu();
#else
        return tensor::Device::vulkan();
#endif
    }();
    const auto available = [](tensor::Device device) {
        for (const auto& backend : tensor::BackendRegistry::instance().backends(device.index))
            if (backend.device_kind == device.kind) return backend.execution_available;
        return false;
    };
    if (accelerator == "cpu") return tensor::Device::cpu();
    if (accelerator == "gpu") {
        if (!available(gpu))
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, "no GPU backend is available on this device"});
        return gpu;
    }
    if (accelerator == "npu") {
        auto compiler = runtime::npu_step_compiler();
        if (!compiler) return std::unexpected(std::move(compiler.error()));
        return tensor::Device::qualcomm_npu();
    }
    if (accelerator != "auto")
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "accelerator must be auto, cpu, gpu, or npu"});
    if (speech) return tensor::Device::cpu();
    if (runtime::npu_step_compiler()) return tensor::Device::qualcomm_npu();
    // Vulkan is an explicit, experimental choice until it measures faster than the CPU; Metal is Apple's default.
    if (gpu.kind != tensor::DeviceKind::VULKAN && available(gpu)) return gpu;
    return tensor::Device::cpu();
}
auto Generator::execution() const -> std::string {
    const auto accelerator = model_->accelerator();
    auto result = std::string(tensor::to_string(model_->device().kind));
    if (!accelerator.empty()) result += "+" + std::string(accelerator);
    return result;
}
auto Generator::generate(std::string_view prompt, GenerationOptions options) -> Result<TextGeneration> {
    try {
        options.prefill_chunk_size = model_->prefill_chunk_size(options.prefill_chunk_size);
        if (pending_requests() || serving_failed_)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT,
                                "drain serving requests before serial generation; reload after a serving failure"});
        serving_prefix_.reset();
        const auto started = Clock::now();
        const auto preparation = model_->preparation_ns();
        TextGeneration result;
        result.stats.last_token_prefill = model_->last_token_prefill();
        result.stats.shared_prefill_tail = model_->shared_prefill_tail();
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
        if (prefix_ && (!options.prefix_cache_bytes || prefix_->bytes > options.prefix_cache_bytes ||
                        prefix_->chunk_size != options.prefill_chunk_size ||
                        prefix_->crop_local_attention != !options.full_attention_cache))
            prefix_.reset();
        if (prefix_) {
            const auto limit = std::min(prefix_->tokens.size(), tokens.size() - 1);
            std::size_t common = 0;
            while (common < limit && prefix_->tokens[common] == tokens[common]) ++common;
            result.stats.reused_prompt_tokens = (common / options.prefill_chunk_size) * options.prefill_chunk_size;
        }
        auto state = result.stats.reused_prompt_tokens
                         ? require(model_->fork_state(prefix_->state, result.stats.reused_prompt_tokens, capacity))
                         : require(model_->create_state(capacity));
        state.crop_local_attention = !options.full_attention_cache;
        state.capture_prefill = should_capture_prefill(model_->accelerator(), tokens.size());
        std::size_t bytes_per_token = 0;
        for (const auto& layer : state.layers)
            bytes_per_token += (layer.key.nbytes() + layer.value.nbytes()) / capacity;
        const auto cached_length = std::min((tokens.size() - 1) / options.prefill_chunk_size,
                                            options.prefix_cache_bytes / bytes_per_token / options.prefill_chunk_size) *
                                   options.prefill_chunk_size;
        const std::array extra_stops{special_[1]};
        const SearchOptions search{.vocabulary_size = tokenizer_.vocabulary_size(),
                                   .end_id = special_[0],
                                   .pad_id = special_[2],
                                   .maximum_steps = maximum,
                                   .compute_score = false,
                                   .stop_ids = extra_stops,
                                   .stop_on_eos = !options.ignore_eos};
        tensor::Tensor storage;
        std::optional<std::int32_t> selected;
        result.stats.device_selection = model_->device() != tensor::Device::cpu();
        const auto project = [&](std::span<const std::int32_t> input) -> Result<void> {
            if (result.stats.device_selection) {
                auto token = model_->forward_token(input, state);
                if (!token) return std::unexpected(std::move(token.error()));
                selected = *token;
            } else {
                auto logits = model_->forward(input, state);
                if (!logits) return std::unexpected(std::move(logits.error()));
                storage = std::move(*logits);
            }
            return {};
        };
        result.generation =
            require(Decoder::generate(tokens, search, [&](const DecodeRequest& request) -> Result<TokenScores> {
                if (request.generated_steps == 1) result.stats.time_to_first_token_ns = elapsed(started);
                const auto step_start = Clock::now();
                if (!request.generated_steps) {
                    std::size_t offset = result.stats.reused_prompt_tokens;
                    while (tokens.size() - offset > options.prefill_chunk_size) {
                        auto status =
                            model_->prefill(std::span(tokens).subspan(offset, options.prefill_chunk_size), state);
                        if (!status) return std::unexpected(std::move(status.error()));
                        offset += options.prefill_chunk_size;
                        if (offset == cached_length && offset > result.stats.reused_prompt_tokens) {
                            prefix_.reset();
                            auto snapshot = require(model_->fork_state(state, cached_length, cached_length));
                            prefix_ = PrefixEntry{{tokens.begin(), tokens.begin() + cached_length},
                                                  std::move(snapshot),
                                                  cached_length * bytes_per_token,
                                                  options.prefill_chunk_size,
                                                  state.crop_local_attention};
                        }
                    }
                    auto status = project(std::span(tokens).subspan(offset));
                    if (!status) return std::unexpected(std::move(status.error()));
                    result.stats.prefill_ns += elapsed(step_start);
                } else {
                    auto status = project(request.prefixes.last(1));
                    if (!status) return std::unexpected(std::move(status.error()));
                    result.stats.decode_ns += elapsed(step_start);
                    ++result.stats.decode_tokens;
                }
                if (selected) return TokenScores{{}, ScoreKind::LOGITS, selected};
                auto values = storage.data<float>();
                if (!values) return std::unexpected(std::move(values.error()));
                return TokenScores{*values};
            }));
        if (!result.stats.time_to_first_token_ns) result.stats.time_to_first_token_ns = elapsed(started);
        result.stats.preparation_ns = model_->preparation_ns() - preparation;
        result.stats.prefix_cache_bytes = prefix_ ? prefix_->bytes : 0;
        result.stats.prefix_reserved_bytes = result.stats.prefix_cache_bytes;
        result.text = require(tokenizer_.decode(result.generation.token_ids));
        result.stats.generation_ns = elapsed(started);
        return result;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, error.what()});
    }
}
auto Generator::generate_batch(std::span<const std::string> prompts, GenerationOptions options)
    -> Result<GenerationBatch> {
    try {
        options.prefill_chunk_size = model_->prefill_chunk_size(options.prefill_chunk_size);
        if (pending_requests() || serving_failed_)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT,
                                "drain serving requests before batch generation; reload after a serving failure"});
        serving_prefix_.reset();
        if (prompts.empty() || prompts.size() > 16 || options.prefix_cache_bytes || !options.prefill_chunk_size)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT,
                                "batch generation supports 1-16 prompts, positive chunks and no prefix-cache policy"});
        const auto started = Clock::now();
        const auto preparation_before = model_->preparation_ns();
        const auto maximum = options.maximum_new_tokens ? options.maximum_new_tokens
                                                        : config_["decode"]["maximum_new_tokens"].as<std::size_t>();
        const auto capacity =
            options.context_size ? options.context_size : config_["decode"]["context_size"].as<std::size_t>();
        const std::array extra_stops{special_[1]};
        const SearchOptions search{.vocabulary_size = tokenizer_.vocabulary_size(),
                                   .end_id = special_[0],
                                   .pad_id = special_[2],
                                   .maximum_steps = maximum,
                                   .compute_score = false,
                                   .stop_ids = extra_stops,
                                   .stop_on_eos = !options.ignore_eos};
        GenerationBatch result;
        result.rows.resize(prompts.size());
        std::vector<model::Gemma4State> states;
        states.reserve(prompts.size());
        std::vector<std::int32_t> chosen(prompts.size()), seeds(prompts.size());
        for (std::size_t row = 0; row < prompts.size(); ++row) {
            const auto tokenizing = Clock::now();
            const auto text =
                options.raw_prompt ? prompts[row] : "<bos><|turn>user\n" + prompts[row] + "<turn|>\n<|turn>model\n";
            const auto tokens = require(tokenizer_.encode(text));
            auto& stats = result.rows[row].stats;
            stats.tokenize_ns = elapsed(tokenizing);
            stats.prompt_tokens = tokens.size();
            stats.device_selection = model_->device() == tensor::Device::apple_gpu();
            stats.last_token_prefill = model_->last_token_prefill();
            stats.shared_prefill_tail = model_->shared_prefill_tail();
            if (!maximum || tokens.empty() || tokens.size() >= capacity || maximum > capacity - tokens.size())
                throw ops::Failure(
                    {ErrorCode::INVALID_ARGUMENT, "every batch prompt and generation must fit the context"});
            seeds[row] = tokens.back();
            states.push_back(require(model_->create_state(capacity)));
            states.back().crop_local_attention = !options.full_attention_cache;
            states.back().capture_prefill = should_capture_prefill(model_->accelerator(), tokens.size());
            const auto preparation = model_->preparation_ns();
            const auto prefill_start = Clock::now();
            std::size_t offset = 0;
            while (tokens.size() - offset > options.prefill_chunk_size) {
                require(model_->prefill(std::span(tokens).subspan(offset, options.prefill_chunk_size), states.back()));
                offset += options.prefill_chunk_size;
            }
            chosen[row] = require(model_->forward_token(std::span(tokens).subspan(offset), states.back()));
            stats.prefill_ns = elapsed(prefill_start);
            stats.preparation_ns = model_->preparation_ns() - preparation;
            stats.time_to_first_token_ns = elapsed(started);
            result.prefill_ns += stats.prefill_ns;
        }
        std::vector<SearchOptions> policies(prompts.size(), search);
        std::vector<std::int32_t> inputs;
        std::vector<model::Gemma4State*> active_states;
        inputs.reserve(prompts.size());
        active_states.reserve(prompts.size());
        auto generated =
            require(Decoder::generate_batch(seeds, policies, [&](const GreedyRequest& request) -> Result<TokenScores> {
                if (request.generated_steps) {
                    inputs.clear();
                    active_states.clear();
                    for (auto row : request.active_rows) {
                        inputs.push_back(request.tokens[row]);
                        active_states.push_back(&states[row]);
                    }
                    const auto decode_start = Clock::now();
                    auto output = model_->forward_batch_tokens(inputs, active_states);
                    if (!output) return std::unexpected(std::move(output.error()));
                    result.decode_ns += elapsed(decode_start);
                    for (std::size_t index = 0; index < request.active_rows.size(); ++index)
                        chosen[request.active_rows[index]] = (*output)[index];
                }
                return TokenScores{{}, ScoreKind::LOGITS, {}, chosen};
            }));
        for (std::size_t row = 0; row < prompts.size(); ++row) {
            result.rows[row].generation = std::move(generated[row]);
            result.rows[row].text = require(tokenizer_.decode(result.rows[row].generation.token_ids));
            result.rows[row].stats.decode_tokens = result.rows[row].generation.decoder_steps - 1;
        }
        result.generation_ns = elapsed(started);
        result.preparation_ns = model_->preparation_ns() - preparation_before;
        return result;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, error.what()});
    }
}
auto Generator::configure_serving(ServingOptions options) -> Result<void> {
    try {
        options.prefill_tokens_per_step = model_->prefill_chunk_size(options.prefill_tokens_per_step);
        if (pending_requests() || serving_failed_ || !options.maximum_active || options.maximum_active > 16 ||
            options.maximum_requests < options.maximum_active || !options.cache_token_budget ||
            !options.prefill_tokens_per_step)
            throw ops::Failure(
                {ErrorCode::INVALID_ARGUMENT, "invalid serving limits, outstanding requests or failed engine"});
        prefix_.reset();
        serving_prefix_.reset();
        running_.reserve(options.maximum_active);
        serving_ = options;
        return {};
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, error.what()});
    }
}
auto Generator::enqueue_chat(std::span<const text::ChatMessage> messages, GenerationOptions options)
    -> Result<std::uint64_t> {
    try {
        std::vector<text::ChatMessage> expanded(messages.begin(), messages.end());
        std::vector<CachedImage> current;
        std::string image_key;
        for (auto& message : expanded) {
            std::string placeholders;
            for (const auto& path : message.images) {
                if (!vision_supported())
                    throw ops::Failure({ErrorCode::UNSUPPORTED, "this checkpoint has no supported vision tower"});
                if (message.role != "user" || current.size() >= 8)
                    throw ops::Failure(
                        {ErrorCode::INVALID_ARGUMENT, "images require user messages, at most eight per context"});
                const auto size = std::filesystem::file_size(path);
                if (!size || size > 32 * 1024 * 1024 || image_key.size() + size > 32 * 1024 * 1024)
                    throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "images exceed the 32 MiB request limit"});
                std::ifstream stream(path, std::ios::binary);
                std::string encoded(size, '\0');
                if (!stream.read(encoded.data(), encoded.size()) || stream.peek() != std::char_traits<char>::eof())
                    throw ops::Failure({ErrorCode::IO, "unable to read complete image"});
                auto cached = std::ranges::find(images_, encoded, &CachedImage::encoded);
                tensor::Tensor embeddings;
                if (cached != images_.end()) {
                    embeddings = cached->embeddings;
                } else {
                    if (!vision_) {
                        const ModuleScope construction(tensor::DType::F32, false, tensor::Device::cpu());
                        auto vision = model::Gemma4Vision(config_["vision"], config_["model"]["hidden_size"].as<int>(),
                                                          native_qat());
                        auto weights = require(checkpoint::Weights::load(config_["weights_file"].as<std::string>()));
                        require(vision->set_checkpoint(weights));
                        vision_ = std::move(vision);
                    }
                    const auto pixels = require(image::prepare_gemma4(
                        std::span(reinterpret_cast<const std::uint8_t*>(encoded.data()), encoded.size())));
                    embeddings = require(require(vision_->forward(pixels)).to(model_->device()));
                }
                placeholders += "<|image>";
                for (std::size_t index = 0; index < embeddings.size(1); ++index) placeholders += "<|image|>";
                placeholders += "<image|>";
                image_key += std::to_string(encoded.size()) + ":" + encoded;
                current.push_back({std::move(encoded), std::move(embeddings)});
            }
            message.images.clear();
            message.content = placeholders + message.content;
        }
        const auto prompt = require(tokenizer_.format_chat(expanded));
        std::vector<model::Gemma4ImageTokens> positions;
        if (!current.empty()) {
            const auto ids = require(tokenizer_.encode(prompt));
            const auto image_token = config_["image_token_id"].as<std::int32_t>();
            std::size_t image_index = 0;
            for (std::size_t position = 0; position < ids.size();) {
                if (ids[position] != image_token) {
                    ++position;
                    continue;
                }
                if (image_index >= current.size())
                    throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "extra image placeholders"});
                const auto& embeddings = current[image_index++].embeddings;
                const auto count = embeddings.size(1);
                if (count > ids.size() - position ||
                    !std::ranges::all_of(std::span(ids).subspan(position, count),
                                         [&](auto token) { return token == image_token; }))
                    throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "image token count mismatch"});
                positions.push_back({position, embeddings});
                position += count;
            }
            if (image_index != current.size())
                throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "missing image placeholders"});
        }
        options.raw_prompt = true;
        const auto id = require(enqueue(prompt, options));
        waiting_.back().images = std::move(positions);
        waiting_.back().image_key = std::move(image_key);
        images_ = std::move(current);
        return id;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, error.what()});
    }
}
auto Generator::enqueue(std::string_view prompt, GenerationOptions options) -> Result<std::uint64_t> {
    try {
        options.prefill_chunk_size = model_->prefill_chunk_size(options.prefill_chunk_size);
        if (!serving_ || serving_failed_ || pending_requests() >= serving_->maximum_requests ||
            next_request_id_ == std::numeric_limits<std::uint64_t>::max())
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "serving queue is unavailable or full"});
        const auto started = Clock::now();
        const auto maximum = options.maximum_new_tokens ? options.maximum_new_tokens
                                                        : config_["decode"]["maximum_new_tokens"].as<std::size_t>();
        const auto capacity =
            options.context_size ? options.context_size : config_["decode"]["context_size"].as<std::size_t>();
        if (!maximum || !options.prefill_chunk_size || !capacity || capacity > serving_->cache_token_budget ||
            capacity > config_["model"]["max_position_embeddings"].as<std::size_t>())
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "request exceeds serving context budget"});
        const auto serialized = options.raw_prompt
                                    ? std::string(prompt)
                                    : "<bos><|turn>user\n" + std::string(prompt) + "<turn|>\n<|turn>model\n";
        auto tokens = require(tokenizer_.encode(serialized));
        if (tokens.empty() || tokens.size() >= capacity || maximum > capacity - tokens.size())
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "prompt and generation must fit the reserved context"});
        if (options.prefix_cache_bytes) tokens.reserve(capacity);
        const std::array extra_stops{special_[1]};
        auto search = require(GreedyState::create({.vocabulary_size = tokenizer_.vocabulary_size(),
                                                   .end_id = special_[0],
                                                   .pad_id = special_[2],
                                                   .maximum_steps = maximum,
                                                   .compute_score = false,
                                                   .stop_ids = extra_stops,
                                                   .stop_on_eos = !options.ignore_eos}));
        GenerationStats stats;
        stats.tokenize_ns = elapsed(started);
        stats.prompt_tokens = tokens.size();
        stats.device_selection = model_->device() != tensor::Device::cpu();
        stats.last_token_prefill = model_->last_token_prefill();
        stats.shared_prefill_tail = model_->shared_prefill_tail();
        waiting_.push_back(
            {next_request_id_, std::move(tokens), capacity, options, std::move(search), {}, stats, started, {}});
        return next_request_id_++;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, error.what()});
    }
}
auto Generator::retain_serving_prefix(QueuedGeneration& request) -> void {
    serving_prefix_.reset();
    if (!request.options.prefix_cache_bytes || !request.state || !request.state->position) return;
    std::size_t bytes = 0;
    for (const auto& layer : request.state->layers) bytes += layer.key.nbytes() + layer.value.nbytes();
    if (bytes > request.options.prefix_cache_bytes) return;
    const auto position = request.state->position;
    const auto& generated = request.search.result().token_ids;
    if (position > request.prompt.size() + generated.size()) return;
    auto tokens = std::move(request.prompt);
    if (position > tokens.size()) {
        const auto count = position - tokens.size();
        tokens.insert(tokens.end(), generated.begin(), generated.begin() + count);
    } else {
        tokens.resize(position);
    }
    request.stats.prefix_cache_bytes = bytes / request.capacity * position;
    request.stats.prefix_reserved_bytes = bytes;
    const auto chunk = std::min(request.options.prefill_chunk_size, serving_->prefill_tokens_per_step);
    serving_prefix_.emplace(PrefixEntry{std::move(tokens), std::move(*request.state), bytes, chunk,
                                        !request.options.full_attention_cache, std::move(request.image_key)});
    request.state.reset();
}

auto Generator::cancel(std::uint64_t request_id) -> Result<void> {
    const auto waiting = std::ranges::find(waiting_, request_id, &QueuedGeneration::id);
    if (waiting != waiting_.end()) {
        waiting_.erase(waiting);
        return {};
    }
    const auto running = std::ranges::find(running_, request_id, &QueuedGeneration::id);
    if (running != running_.end()) {
        retain_serving_prefix(*running);
        reserved_cache_tokens_ -= running->capacity;
        running_.erase(running);
        return {};
    }
    return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "unknown serving request"});
}
auto Generator::step() -> Result<GenerationStep> {
    if (!serving_ || serving_failed_)
        return std::unexpected(
            Error{ErrorCode::INVALID_ARGUMENT, "serving is unconfigured or failed; reload after failure"});
    try {
        GenerationStep result;
        result.events.reserve(serving_->maximum_active + 1);
        const auto preparation = model_->preparation_ns();
        const auto admit = [&] {
            while (!waiting_.empty() && running_.size() < serving_->maximum_active &&
                   waiting_.front().capacity <= serving_->cache_token_budget - reserved_cache_tokens_) {
                auto& request = waiting_.front();
                const auto chunk = std::min(request.options.prefill_chunk_size, serving_->prefill_tokens_per_step);
                if (serving_prefix_ && request.options.prefix_cache_bytes >= serving_prefix_->bytes &&
                    request.image_key == serving_prefix_->image_key &&
                    request.capacity == serving_prefix_->state.capacity && chunk == serving_prefix_->chunk_size &&
                    serving_prefix_->crop_local_attention == !request.options.full_attention_cache) {
                    const auto limit = std::min(serving_prefix_->tokens.size(), request.prompt.size() - 1);
                    std::size_t common = 0;
                    while (common < limit && serving_prefix_->tokens[common] == request.prompt[common]) ++common;
                    request.stats.reused_prompt_tokens = common;
                    request.state.emplace(std::move(serving_prefix_->state));
                    request.state->position = common;
                }
                serving_prefix_.reset();
                if (!request.state) request.state = require(model_->create_state(request.capacity));
                request.state->images = request.images;
                request.state->crop_local_attention = !request.options.full_attention_cache;
                request.state->capture_prefill =
                    should_capture_prefill(model_->accelerator(), request.prompt.size());
                running_.push_back(std::move(request));
                reserved_cache_tokens_ += running_.back().capacity;
                waiting_.pop_front();
            }
        };
        const auto retire = [&] {
            std::erase_if(running_, [&](const auto& request) {
                if (!request.search.finished()) return false;
                reserved_cache_tokens_ -= request.capacity;
                return true;
            });
        };
        const auto accept = [&](QueuedGeneration& request, std::int32_t token) {
            const auto previous = request.search.result().token_ids.size();
            require(request.search.accept(TokenScores{{}, ScoreKind::LOGITS, token}));
            if (request.search.result().decoder_steps == 1)
                request.stats.time_to_first_token_ns = elapsed(request.enqueued);
            GenerationEvent event{request.id, {}, {}};
            if (request.search.result().token_ids.size() > previous) event.token = token;
            if (request.options.stream_text)
                event.text = require(tokenizer_.decode_delta(request.search.result().token_ids, request.streamed_text,
                                                             request.search.finished()));
            if (request.search.finished()) {
                retain_serving_prefix(request);
                TextGeneration completed;
                completed.generation = std::move(request.search).result();
                completed.text = require(tokenizer_.decode(completed.generation.token_ids));
                completed.stats = request.stats;
                completed.stats.decode_tokens = completed.generation.decoder_steps - 1;
                completed.stats.generation_ns = elapsed(request.enqueued);
                event.completed = std::move(completed);
            }
            result.events.push_back(std::move(event));
        };
        admit();
        std::array<std::int32_t, 16> tokens{};
        std::array<model::Gemma4State*, 16> states{};
        std::array<std::size_t, 16> rows{};
        std::size_t count = 0;
        for (std::size_t row = 0; row < running_.size(); ++row) {
            auto& request = running_[row];
            if (!request.search.result().decoder_steps) continue;
            tokens[count] = request.search.token();
            states[count] = &*request.state;
            rows[count++] = row;
        }
        if (count) {
            const auto started = Clock::now();
            const auto prepared = model_->preparation_ns();
            std::vector<std::int32_t> selected;
            if (count == 1)
                selected.push_back(require(model_->forward_token(std::span(tokens).first(1), *states[0])));
            else
                selected = require(
                    model_->forward_batch_tokens(std::span(tokens).first(count), std::span(states).first(count)));
            result.decode_ns = elapsed(started);
            if (serving_->maximum_active == 1) {
                running_[rows[0]].stats.decode_ns += result.decode_ns;
                running_[rows[0]].stats.preparation_ns += model_->preparation_ns() - prepared;
            }
            for (std::size_t index = 0; index < count; ++index) accept(running_[rows[index]], selected[index]);
        }
        retire();
        admit();
        auto prefill = running_.end();
        for (auto candidate = running_.begin(); candidate != running_.end(); ++candidate) {
            if (candidate->search.result().decoder_steps) continue;
            if (prefill == running_.end()) prefill = candidate;
            if (candidate->id > prefill_after_) {
                prefill = candidate;
                break;
            }
        }
        if (prefill != running_.end()) {
            auto& request = *prefill;
            prefill_after_ = request.id;
            const auto length = std::min({request.prompt.size() - request.state->position,
                                          request.options.prefill_chunk_size, serving_->prefill_tokens_per_step});
            const auto input = std::span(request.prompt).subspan(request.state->position, length);
            const bool final_chunk = request.state->position + length == request.prompt.size();
            const auto started = Clock::now();
            const auto prepared = model_->preparation_ns();
            std::optional<std::int32_t> selected;
            if (final_chunk) {
                selected = require(model_->forward_token(input, *request.state));
            } else {
                require(model_->prefill(input, *request.state));
            }
            result.prefill_ns = elapsed(started);
            request.stats.prefill_ns += result.prefill_ns;
            request.stats.preparation_ns += model_->preparation_ns() - prepared;
            if (selected) accept(request, *selected);
        }
        retire();
        result.active_requests = running_.size();
        result.waiting_requests = waiting_.size();
        result.reserved_cache_tokens = reserved_cache_tokens_;
        result.preparation_ns = model_->preparation_ns() - preparation;
        return result;
    } catch (const ops::Failure& error) {
        serving_failed_ = true;
        serving_prefix_.reset();
        waiting_.clear();
        running_.clear();
        reserved_cache_tokens_ = 0;
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        serving_failed_ = true;
        serving_prefix_.reset();
        waiting_.clear();
        running_.clear();
        reserved_cache_tokens_ = 0;
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}
} // namespace kidi::inference