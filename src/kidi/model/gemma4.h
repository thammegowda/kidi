#pragma once

#include "kidi/core/precision.h"
#include "kidi/layers/gemma4.h"
#include "kidi/checkpoint/weights.h"
#include "kidi/checkpoint/config.h"
#include <yaml-cpp/yaml.h>

namespace kidi::model {
struct Gemma4ImageTokens {
    std::size_t position;
    tensor::Tensor embeddings;
};
/// Per-token inputs of a captured decode step, written in place before each step.
struct Gemma4StepInputs {
    tensor::Tensor token, index;                         // I32 [1]
    std::array<tensor::Tensor, 2> masks;                 // per attention type: [1, 1, 1, keys]
    std::array<std::array<tensor::Tensor, 2>, 2> angles; // per attention type: cosine, sine
};
struct Gemma4State {
    std::vector<layers::KeyValue> layers;
    std::vector<Gemma4ImageTokens> images;
    std::size_t position = 0, capacity = 0;
    bool crop_local_attention = true;
    bool capture_prefill = true;
    bool prefilling = true;
    bool profile_started = false;
    Gemma4StepInputs step, prefill_step;
};

struct Gemma4Output {
    tensor::Tensor hidden, logits;
};
struct Gemma4Selection {
    tensor::Tensor hidden, tokens;
};

KIDI_MODULE(Gemma4);
class Gemma4Impl : public Module {
public:
    explicit Gemma4Impl(const YAML::Node& config);
    ~Gemma4Impl();
    static auto validate_config(const YAML::Node& config) -> Result<void>;
    static auto checkpoint_config() -> checkpoint::ConfigAdapter;
    static auto create(const YAML::Node& config) -> Result<Gemma4>;
    auto set_checkpoint(const checkpoint::Weights& weights, std::int32_t weight_bits = 0, std::int32_t group_size = 128,
                        bool packed_prefill = false) -> Result<void>;
    auto configure_cache(const std::filesystem::path& directory, const std::filesystem::path& source) -> void;
    auto set_precision(core::InferencePrecision precision) noexcept -> void;
    auto set_kv_cache_precision(core::KVCachePrecision precision) noexcept -> void;
    auto configure_lm_head(const std::filesystem::path& index) -> Result<void>;
    auto create_state(std::size_t capacity) -> Result<Gemma4State>;
    auto fork_state(const Gemma4State& source, std::size_t prefix_length, std::size_t capacity) -> Result<Gemma4State>;
    /// Discards an uncommitted suffix of the dense KV cache without copying its accepted prefix.
    auto rewind(Gemma4State& state, std::size_t position) -> Result<void>;
    auto prefill(std::span<const std::int32_t> tokens, Gemma4State& state) -> Result<void>;
    auto forward(std::span<const std::int32_t> tokens, Gemma4State& state, bool all_logits = false)
        -> Result<tensor::Tensor>;
    /// Returns normalized final-layer activations for MTP, paired with the corresponding target logits.
    auto forward_features(std::span<const std::int32_t> tokens, Gemma4State& state, bool all_logits = true)
        -> Result<Gemma4Output>;
    /// Like `forward_features`, but uses the ordinary greedy head. Captured outputs are reused by matching calls.
    auto select_features(std::span<const std::int32_t> tokens, Gemma4State& state, bool all_tokens = true)
        -> Result<Gemma4Selection>;
    auto forward_token(std::span<const std::int32_t> tokens, Gemma4State& state) -> Result<std::int32_t>;
    /// Decodes an I32 `[1]` token tensor and returns the greedy next token as an I32 `[1]` tensor on the model
    /// device, so it can feed the next call without a host read. The result is overwritten by the next call;
    /// -1 marks invalid scores.
    auto forward_token(const tensor::Tensor& token, Gemma4State& state) -> Result<tensor::Tensor>;
    auto forward_batch(std::span<const std::int32_t> tokens, std::span<Gemma4State*> states) -> Result<tensor::Tensor>;
    auto forward_batch_tokens(std::span<const std::int32_t> tokens, std::span<Gemma4State*> states)
        -> Result<std::vector<std::int32_t>>;
    /// Like `forward_token`, but returns the selected token as an I32 `[1]` host tensor without reading it. Devices
    /// that finish asynchronously fill it once submitted work completes; -1 marks invalid scores. The next call may
    /// reuse it.
    auto select_token(std::span<const std::int32_t> tokens, Gemma4State& state) -> Result<tensor::Tensor>;
    /// Batched `select_token`: an I32 `[states]` host tensor.
    auto select_batch_tokens(std::span<const std::int32_t> tokens, std::span<Gemma4State*> states)
        -> Result<tensor::Tensor>;
    auto preparation_ns() const -> std::uint64_t;
    auto release_workspaces() -> void;
    auto trim_workspaces() -> void;
    /// Accelerator compiling this model's captured steps, or empty.
    auto accelerator() const -> std::string_view;
    auto prefill_chunk_size(std::size_t requested) const -> std::size_t;
    auto last_token_prefill() const -> bool;
    auto shared_prefill_tail() const -> bool;
    auto supports_device_tokens() const -> bool;

private:
    friend class Gemma4AssistantImpl;
    struct State;
    struct Attention;
    auto assistant_embedding(ops::Context& context, std::int32_t token) const -> tensor::Tensor;
    auto assistant_cache(Gemma4State& state, int type) const -> layers::KeyValue&;
    auto assistant_positions(int type, std::size_t position) -> std::array<tensor::Tensor, 2>;
    auto embed(std::span<const std::int32_t> tokens, std::span<const Gemma4ImageTokens> images = {},
               std::size_t position = 0) -> std::array<tensor::Tensor, 2>;
    auto attention_inputs(Gemma4State& state, std::span<Gemma4State*> batch_states, std::size_t step_count)
        -> Attention;
    auto per_layer_input(const tensor::Tensor& per_layer, int layer, std::int64_t length) -> tensor::Tensor;
    auto head(const tensor::Tensor& hidden, bool select, ops::TokenOutput output = ops::TokenOutput::HOST,
              tensor::Tensor* output_hidden = nullptr) -> tensor::Tensor;
    /// Decodes one token as a captured step: host values are written into `state.step`, caches by index.
    auto decode_step(const tensor::Tensor& token, Gemma4State& state, bool select,
                     ops::TokenOutput output = ops::TokenOutput::HOST) -> tensor::Tensor;
    /// Runs I32 `tokens` at the state position as a captured step: decoding runs every layer and the output head;
    /// prefill writes the producer layers' K/V and returns an undefined tensor.
    auto captured_step(const tensor::Tensor& tokens, Gemma4State& state, Gemma4StepInputs& inputs, bool prefill,
                       bool select, ops::TokenOutput output = ops::TokenOutput::HOST,
                       tensor::Tensor* output_hidden = nullptr) -> tensor::Tensor;
    auto can_decode_step(const Gemma4State& state) const -> bool;
    auto run_batch(std::span<const std::int32_t> tokens, std::span<Gemma4State*> states, bool select)
        -> Result<tensor::Tensor>;
    auto prefill_impl(std::span<const std::int32_t> tokens, Gemma4State& state) -> Result<void>;
    /// Runs every layer and the output head; `batch_states` decodes one token per request.
    auto project(std::span<const std::int32_t> tokens, Gemma4State& state, bool all_logits, bool select,
                 std::span<Gemma4State*> batch_states = {}, tensor::Tensor* output_hidden = nullptr)
        -> Result<tensor::Tensor>;
    std::unique_ptr<State> impl_;
};
} // namespace kidi::model