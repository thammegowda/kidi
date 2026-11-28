#pragma once
// Model-independent text generation, batching and request-state management over registered causal models.

#include "kidi/checkpoint/config.h"
#include "kidi/core/precision.h"
#include "kidi/image/gemma4.h"
#include "kidi/inference/decoder.h"
#include "kidi/model/model.h"
#include "kidi/text/tokenizer.h"
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <random>

namespace kidi::inference {
struct GenerationOptions {
    std::size_t maximum_new_tokens = 0, context_size = 0, prefill_chunk_size = 128;
    std::size_t prefix_cache_bytes = 0, maximum_prompt_tokens = 0;
    bool raw_prompt = false, ignore_eos = false, full_attention_cache = false;
    bool stream_text = false;
    // temperature 0 selects greedily on the fastest path; > 0 samples on the host from the last-row logits.
    float temperature = 0, top_p = 1;
    std::size_t top_k = 0;
    std::optional<std::uint64_t> seed = std::nullopt;
    std::int32_t image_tokens = 280;
    std::size_t image_max_pixels = image::MAX_RESIZED_PIXELS;
    std::function<void(std::string_view)> on_image_progress;
};
struct GenerationStats {
    bool device_selection = false;
    bool last_token_prefill = false;
    bool shared_prefill_tail = false;
    bool speculative = false;
    std::uint64_t tokenize_ns = 0, prefill_ns = 0, decode_ns = 0, preparation_ns = 0, time_to_first_token_ns = 0;
    std::uint64_t generation_ns = 0;
    std::size_t prompt_tokens = 0, decode_tokens = 0;
    std::size_t reused_prompt_tokens = 0, prefix_cache_bytes = 0, prefix_reserved_bytes = 0;
    std::size_t decode_submitted_tokens = 0, decode_queue_peak = 0;
    std::uint64_t draft_ns = 0, verify_ns = 0;
    std::size_t draft_tokens = 0, accepted_draft_tokens = 0, verification_steps = 0;
};
struct TextGeneration {
    std::string text;
    Generation generation;
    GenerationStats stats;
};
struct GenerationBatch {
    std::vector<TextGeneration> rows;
    std::uint64_t prefill_ns = 0, decode_ns = 0, generation_ns = 0, preparation_ns = 0;
};
struct ServingOptions {
    std::size_t maximum_active = 4, maximum_requests = 64, cache_token_budget = 16384, prefill_tokens_per_step = 128;
    bool compact_cache = false;
    /// `step()` submits work and reports the tokens it selects on the following call, so a device may finish
    /// asynchronously in between without the caller blocking (the browser awaits WebGPU between calls).
    bool deferred_tokens = false;
    std::size_t decode_ahead_tokens = 0;
};
struct GenerationEvent {
    std::uint64_t request_id;
    std::optional<std::int32_t> token;
    std::optional<TextGeneration> completed;
    std::string text;
    bool replace = false;
};
struct GenerationStep {
    std::vector<GenerationEvent> events;
    std::uint64_t prefill_ns = 0, decode_ns = 0, preparation_ns = 0;
    std::size_t active_requests = 0, waiting_requests = 0, reserved_cache_tokens = 0;
};
/// Resolves an accelerator preference ("auto", "cpu", "gpu", or "npu") to a device. "auto" picks the Qualcomm NPU,
/// then a GPU, then the CPU, by availability; an explicit choice fails when that accelerator is unavailable.
/// `speech` selects the policy for Whisper, whose "auto" stays on the CPU INT8 path.
auto select_device(std::string_view accelerator, bool speech = false) -> Result<tensor::Device>;

/// Live bytes held by generation state, on whichever device owns it.
struct GeneratorMemory {
    std::size_t cache_bytes = 0; // KV caches of running requests, retained prefixes and reusable CUDA states
    std::size_t image_bytes = 0; // reusable image embeddings
};

/// Creates a tensor that config.json lists under `external_tensors`: checkpoint data the embedder keeps outside
/// linear memory, such as browser-held lookup tables.
using ExternalTensorSource = std::function<Result<tensor::Tensor>(std::string_view name, tensor::DType dtype,
                                                                  std::span<const std::int64_t> shape)>;

struct ModelLoadOptions {
    tensor::Device device = tensor::Device::cpu();
    /// Also selects load-time packing: Q8 and Q4 quantize plain BF16 checkpoints when they load.
    core::InferencePrecision precision = core::InferencePrecision::CHECKPOINT;
    core::KVCachePrecision kv_cache_precision = core::KVCachePrecision::AUTO;
    bool packed_prefill = false;
    bool dense_lm_head = false;
    std::filesystem::path assistant_model{};
    bool prompt_lookup = false;
    std::size_t draft_tokens = 3;
};

class Generator {
public:
    Generator(Generator&&) noexcept;
    auto operator=(Generator&&) noexcept -> Generator&;
    ~Generator();

    static auto load(const std::filesystem::path& directory, ModelLoadOptions options,
                     const ExternalTensorSource& external = {}) -> Result<Generator>;
    /// Loads a registered causal model on `device`; the Qualcomm NPU device keeps the model on the CPU and
    /// compiles captured decoding steps for the NPU. Nonzero `weight_bits` (4 or 8) pack plain BF16 weights at load
    /// with an explicit `group_size` for developer tools; applications select the Q4 or Q8 precision instead.
    static auto load(const std::filesystem::path& directory, tensor::Device device, std::int32_t weight_bits = 0,
                     std::int32_t group_size = 128, bool packed_prefill = false,
                     const ExternalTensorSource& external = {}) -> Result<Generator>;
    /// Where text generation runs, e.g. "cpu", "vulkan", or "cpu+qnn-htp".
    auto execution() const -> std::string;
    auto generate(std::string_view prompt, GenerationOptions options = {}) -> Result<TextGeneration>;
    auto generate_batch(std::span<const std::string> prompts, GenerationOptions options = {})
        -> Result<GenerationBatch>;
    auto configure_serving(ServingOptions options) -> Result<void>;
    auto enqueue(std::string_view prompt, GenerationOptions options = {}) -> Result<std::uint64_t>;
    auto enqueue_chat(std::span<const text::ChatMessage> messages, GenerationOptions options = {})
        -> Result<std::uint64_t>;
    auto chat_prompt_tokens(std::span<const text::ChatMessage> messages) const -> Result<std::size_t>;
    /// Structured chat in OpenAI form; user messages may hold image_url parts with base64 data URLs.
    auto enqueue_chat(const nlohmann::ordered_json& messages, GenerationOptions options = {},
                      const nlohmann::ordered_json& template_args = nlohmann::ordered_json::object())
        -> Result<std::uint64_t>;
    auto chat_prompt_tokens(const nlohmann::ordered_json& messages,
                            const nlohmann::ordered_json& template_args = nlohmann::ordered_json::object()) const
        -> Result<std::size_t>;
    auto context_capacity(GenerationOptions options = {}) const -> Result<std::size_t>;
    /// Largest context, up to the model limit, whose KV cache fits in `memory_bytes`; a multiple of 1024 tokens.
    auto fit_context_size(std::size_t memory_bytes) -> Result<std::size_t>;
    auto step() -> Result<GenerationStep>;
    auto cancel(std::uint64_t request_id) -> Result<void>;
    auto pending_requests() const -> std::size_t { return waiting_.size() + running_.size(); }
    auto precision() const noexcept -> core::InferencePrecision { return options_.precision; }
    auto kv_cache_precision() const noexcept -> core::KVCachePrecision { return options_.kv_cache_precision; }
    auto tokenizer() const noexcept -> const text::Tokenizer& { return tokenizer_; }
    /// Runs teacher-forced prefill over `tokens` and returns host FP32 logits shaped [tokens.size(), vocab_size].
    auto prefill_logits(std::span<const std::int32_t> tokens) -> Result<tensor::Tensor>;
    auto native_qat() const -> bool { return checkpoint::native_qat(config_["model"]); }
    auto speculative() const noexcept -> bool { return static_cast<bool>(drafter_); }
    auto vision_supported() const -> bool {
        return config_["model"]["type"].as<std::string>() == "gemma4_text" && static_cast<bool>(config_["vision"]) &&
               !config_["vision"]["use_clipped_linears"].as<bool>(false);
    }
    auto memory() const -> GeneratorMemory;
    /// Prepares reusable request shapes and weights before the first request. Serving recycles compatible
    /// text-only CUDA states after all pending work completes.
    auto warm_up(std::size_t prefill_chunk_size = 128, std::size_t context_size = 0) -> Result<void>;
    /// Releases request-shaped execution graphs and workspaces while retaining model weights.
    /// All queued and running requests must be drained first.
    auto release_workspaces() -> Result<void>;
    auto trim_workspaces() -> Result<void>;

private:
    struct GeneratorOptions {
        std::vector<std::int32_t> stop_ids;
        std::int32_t pad_id = 0;
        std::int32_t warmup_token_id = 0;
        core::InferencePrecision precision = core::InferencePrecision::CHECKPOINT;
        core::KVCachePrecision kv_cache_precision = core::KVCachePrecision::AUTO;
    };
    Generator(YAML::Node config, text::Tokenizer tokenizer, std::shared_ptr<model::CausalLanguageModel> model,
              GeneratorOptions options);
    static auto load_with_packing(const std::filesystem::path& directory, ModelLoadOptions options,
                                  std::int32_t weight_bits, std::int32_t group_size,
                                  const ExternalTensorSource& external) -> Result<Generator>;
    struct DraftBackend;
    YAML::Node config_;
    text::Tokenizer tokenizer_;
    std::shared_ptr<model::CausalLanguageModel> model_;
    std::unique_ptr<DraftBackend> drafter_;
    std::size_t draft_tokens_ = 0;
    std::shared_ptr<model::VisionModel> vision_;
    struct CachedImage {
        std::string encoded;
        tensor::Tensor embeddings;
        std::int32_t tokens;
        std::size_t max_pixels;
    };
    std::vector<CachedImage> images_;
    struct ImageBatch {
        std::vector<CachedImage> images;
        std::string key;
    };
    auto check_image_request(const GenerationOptions& options) const -> void;
    // Encodes one image (or reuses its cached embeddings) and returns the prompt placeholders that stand for it.
    auto image_placeholders(std::string encoded, const GenerationOptions& options, ImageBatch& batch) -> std::string;
    auto enqueue_with_images(const std::string& prompt, GenerationOptions options, ImageBatch batch) -> std::uint64_t;
    GeneratorOptions options_;
    bool prefill_warmed_ = false;
    struct PrefixEntry {
        std::vector<std::int32_t> tokens;
        std::unique_ptr<model::LanguageModelState> state;
        std::size_t bytes = 0, chunk_size = 0;
        bool crop_local_attention = true;
        std::string image_key;
    };
    std::optional<PrefixEntry> prefix_;
    std::optional<PrefixEntry> serving_prefix_;
    struct DecodeReadback {
        std::shared_ptr<tensor::Readback> transfer;
        std::size_t position;
    };
    struct DeviceDecode {
        tensor::Tensor token;
        std::deque<DecodeReadback> pending;
        std::size_t submitted = 0, logical_position = 0;
        std::chrono::steady_clock::time_point started;
        std::chrono::steady_clock::time_point next_poll;
    };
    struct QueuedGeneration {
        std::uint64_t id;
        std::vector<std::int32_t> prompt;
        std::size_t capacity;
        GenerationOptions options;
        GreedyState search;
        std::unique_ptr<model::LanguageModelState> state;
        GenerationStats stats;
        std::chrono::steady_clock::time_point enqueued;
        std::string streamed_text;
        std::vector<model::EmbeddingSpan> images;
        std::string image_key;
        std::optional<DeviceDecode> decode;
        bool completion_emitted = false;
        tensor::Tensor draft_hidden;
        std::mt19937_64 random{};
    };
    auto retain_serving_prefix(QueuedGeneration& request) -> void;
    /// Wraps a plain prompt as one user turn using the checkpoint chat template.
    auto user_prompt(std::string_view prompt) const -> std::string;
    auto acquire_state(std::size_t capacity) -> std::unique_ptr<model::LanguageModelState>;
    auto recycle_state(std::unique_ptr<model::LanguageModelState>& state, bool has_images = false) -> void;
    auto reap_retired() -> void;
    /// Throws on devices whose selected tokens reach the host only on a later call (see `deferred_tokens`).
    auto require_immediate_tokens() const -> void;
    struct PendingToken {
        std::uint64_t id;
        tensor::Tensor selected;
        std::size_t index;
        std::chrono::steady_clock::time_point started;
        bool prefill;
    };
    std::vector<PendingToken> pending_tokens_;
    std::optional<ServingOptions> serving_;
    std::deque<QueuedGeneration> waiting_;
    std::vector<QueuedGeneration> running_;
    std::vector<QueuedGeneration> retired_;
    std::vector<std::unique_ptr<model::LanguageModelState>> reusable_states_;
    std::size_t reserved_cache_tokens_ = 0;
    std::uint64_t next_request_id_ = 1, prefill_after_ = 0;
    bool serving_failed_ = false;
};
} // namespace kidi::inference