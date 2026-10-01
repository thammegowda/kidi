#pragma once

#include "kidi/inference/decoder.h"
#include "kidi/model/gemma4.h"
#include "kidi/model/gemma4_vision.h"
#include "kidi/text/tokenizer.h"
#include <chrono>
#include <deque>
#include <functional>

namespace kidi::inference {
struct GenerationOptions {
    std::size_t maximum_new_tokens = 0, context_size = 0, prefill_chunk_size = 128;
    std::size_t prefix_cache_bytes = 0;
    bool raw_prompt = false, ignore_eos = false, full_attention_cache = false;
    bool stream_text = false;
    std::int32_t image_tokens = 280;
    std::size_t image_max_pixels = image::MAX_RESIZED_PIXELS;
    std::function<void(std::string_view)> on_image_progress;
};
struct GenerationStats {
    bool device_selection = false;
    bool last_token_prefill = false;
    bool shared_prefill_tail = false;
    std::uint64_t tokenize_ns = 0, prefill_ns = 0, decode_ns = 0, preparation_ns = 0, time_to_first_token_ns = 0;
    std::uint64_t generation_ns = 0;
    std::size_t prompt_tokens = 0, decode_tokens = 0;
    std::size_t reused_prompt_tokens = 0, prefix_cache_bytes = 0, prefix_reserved_bytes = 0;
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
};
struct GenerationEvent {
    std::uint64_t request_id;
    std::optional<std::int32_t> token;
    std::optional<TextGeneration> completed;
    std::string text;
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

class Generator {
public:
    /// Loads Gemma 4 on `device`; the Qualcomm NPU device keeps the model on the CPU and compiles captured
    /// decoding steps for the NPU.
    static auto load(const std::filesystem::path& directory, tensor::Device device, std::int32_t weight_bits = 0,
                     std::int32_t group_size = 128, bool packed_prefill = false) -> Result<Generator>;
    /// Where text generation runs, e.g. "cpu", "vulkan", or "cpu+qnn-htp".
    auto execution() const -> std::string;
    auto generate(std::string_view prompt, GenerationOptions options = {}) -> Result<TextGeneration>;
    auto generate_batch(std::span<const std::string> prompts,
                        GenerationOptions options = {}) -> Result<GenerationBatch>;
    auto configure_serving(ServingOptions options) -> Result<void>;
    auto enqueue(std::string_view prompt, GenerationOptions options = {}) -> Result<std::uint64_t>;
    auto enqueue_chat(std::span<const text::ChatMessage> messages,
                      GenerationOptions options = {}) -> Result<std::uint64_t>;
    auto step() -> Result<GenerationStep>;
    auto cancel(std::uint64_t request_id) -> Result<void>;
    auto pending_requests() const -> std::size_t { return waiting_.size() + running_.size(); }
    auto native_qat() const -> bool { return static_cast<bool>(config_["model"]["quantization_config"]); }
    auto vision_supported() const -> bool { return static_cast<bool>(config_["vision"]); }

private:
    Generator(YAML::Node config, text::Tokenizer tokenizer, model::Gemma4 model, std::array<std::int32_t, 3> special);
    YAML::Node config_;
    text::Tokenizer tokenizer_;
    model::Gemma4 model_;
    model::Gemma4Vision vision_{nullptr};
    struct CachedImage {
        std::string encoded;
        tensor::Tensor embeddings;
        std::int32_t tokens;
        std::size_t max_pixels;
    };
    std::vector<CachedImage> images_;
    std::array<std::int32_t, 3> special_;
    struct PrefixEntry {
        std::vector<std::int32_t> tokens;
        model::Gemma4State state;
        std::size_t bytes = 0, chunk_size = 0;
        bool crop_local_attention = true;
        std::string image_key;
    };
    std::optional<PrefixEntry> prefix_;
    std::optional<PrefixEntry> serving_prefix_;
    struct QueuedGeneration {
        std::uint64_t id;
        std::vector<std::int32_t> prompt;
        std::size_t capacity;
        GenerationOptions options;
        GreedyState search;
        std::optional<model::Gemma4State> state;
        GenerationStats stats;
        std::chrono::steady_clock::time_point enqueued;
        std::string streamed_text;
        std::vector<model::Gemma4ImageTokens> images;
        std::string image_key;
    };
    auto retain_serving_prefix(QueuedGeneration& request) -> void;
    std::optional<ServingOptions> serving_;
    std::deque<QueuedGeneration> waiting_;
    std::vector<QueuedGeneration> running_;
    std::size_t reserved_cache_tokens_ = 0;
    std::uint64_t next_request_id_ = 1, prefill_after_ = 0;
    bool serving_failed_ = false;
};
} // namespace kidi::inference