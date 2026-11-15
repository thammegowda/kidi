#pragma once

#include "kidi/core/precision.h"
#include "kidi/layers/gemma4.h"
#include "kidi/checkpoint/config.h"
#include "kidi/model/model.h"
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
struct Gemma4State : LanguageModelState {
    std::vector<layers::KeyValue> layers;
    std::vector<Gemma4ImageTokens> images;
    std::size_t position = 0, capacity = 0;
    bool crop_local_attention = true;
    bool capture_prefill = true;
    bool prefilling = true;
    bool profile_started = false;
    Gemma4StepInputs step, prefill_step;

    auto sequence_position() const noexcept -> std::size_t override { return position; }
    auto sequence_capacity() const noexcept -> std::size_t override { return capacity; }
    auto cache_bytes() const noexcept -> std::size_t override;
    auto attach_embeddings(std::vector<EmbeddingSpan> spans) -> Result<void> override;
};

KIDI_MODULE(Gemma4);
class Gemma4Impl : public CausalLanguageModel {
public:
    explicit Gemma4Impl(const YAML::Node& config);
    ~Gemma4Impl() override;
    static auto validate_config(const YAML::Node& config) -> Result<void>;
    static auto create(const YAML::Node& config) -> Result<Gemma4>;
    auto state_mapping_specs() const -> std::vector<checkpoint::StateMappingSpec> override;
    auto set_checkpoint(const checkpoint::Weights& weights) -> Result<void> override;
    auto set_checkpoint(const checkpoint::Weights& weights, RuntimeQuantization quantization) -> Result<void> override;
    auto set_checkpoint(const checkpoint::Weights& weights, std::int32_t weight_bits, std::int32_t group_size = 128,
                        bool packed_prefill = false) -> Result<void> {
        return set_checkpoint(weights,
                              {.weight_bits = weight_bits, .group_size = group_size, .packed_prefill = packed_prefill});
    }

    auto hidden_size() const noexcept -> std::int32_t override;
    auto vocabulary_size() const noexcept -> std::int32_t override;
    auto max_positions() const noexcept -> std::size_t override;
    auto configure_cache(const std::filesystem::path& directory, const std::filesystem::path& source) -> void override;
    auto set_precision(core::InferencePrecision precision) noexcept -> void override;
    auto set_kv_cache_precision(core::KVCachePrecision precision) noexcept -> void override;
    auto configure_lm_head(const std::filesystem::path& index) -> Result<void> override;
    auto create_state(std::size_t capacity) -> Result<std::unique_ptr<LanguageModelState>> override;
    auto configure_state(LanguageModelState& state, bool full_attention_cache, bool capture_prefill) -> void override;
    auto fork_state(const LanguageModelState& source, std::size_t prefix_length, std::size_t capacity)
        -> Result<std::unique_ptr<LanguageModelState>> override;
    auto rewind(LanguageModelState& state, std::size_t position) -> Result<void> override;
    auto prefill(std::span<const std::int32_t> tokens, LanguageModelState& state) -> Result<void> override;
    auto forward(std::span<const std::int32_t> tokens, LanguageModelState& state, bool all_logits = false,
                 tensor::Tensor* hidden = nullptr) -> Result<tensor::Tensor> override;
    auto forward_device_token(const tensor::Tensor& token, LanguageModelState& state)
        -> Result<tensor::Tensor> override;
    auto forward_batch(std::span<const std::int32_t> tokens, std::span<LanguageModelState* const> states)
        -> Result<tensor::Tensor>;
    auto select_token(std::span<const std::int32_t> tokens, LanguageModelState& state, bool all_tokens = false,
                      tensor::Tensor* hidden = nullptr) -> Result<tensor::Tensor> override;
    auto select_batch_tokens(std::span<const std::int32_t> tokens, std::span<LanguageModelState* const> states)
        -> Result<tensor::Tensor> override;
    auto preparation_ns() const noexcept -> std::uint64_t override;
    auto release_workspaces() -> void override;
    auto trim_workspaces() -> void override;
    auto accelerator() const -> std::string_view override;
    auto prefill_chunk_size(std::size_t requested) const -> std::size_t override;
    auto last_token_prefill() const -> bool override;
    auto shared_prefill_tail() const -> bool override;
    auto supports_device_tokens() const -> bool override;

private:
    friend class Gemma4AssistantImpl;
    struct State;
    struct Attention;
    auto assistant_embedding(ops::Context& context, std::int32_t token) const -> tensor::Tensor;
    auto assistant_cache(Gemma4State& state, int type) const -> layers::KeyValue&;
    auto assistant_positions(int type, std::size_t position) -> std::array<tensor::Tensor, 2>;
    auto embed(std::span<const std::int32_t> tokens, std::span<const Gemma4ImageTokens> images = {},
               std::size_t position = 0) -> std::array<tensor::Tensor, 2>;
    auto attention_inputs(Gemma4State& state, std::span<LanguageModelState* const> batch_states, std::size_t step_count)
        -> Attention;
    auto per_layer_input(const tensor::Tensor& per_layer, int layer, std::int64_t length) -> tensor::Tensor;
    auto head(const tensor::Tensor& hidden, bool select, ops::TokenOutput output = ops::TokenOutput::HOST,
              tensor::Tensor* output_hidden = nullptr) -> tensor::Tensor;
    auto decode_step(const tensor::Tensor& token, Gemma4State& state, bool select,
                     ops::TokenOutput output = ops::TokenOutput::HOST, tensor::Tensor* output_hidden = nullptr)
        -> tensor::Tensor;
    auto captured_step(const tensor::Tensor& tokens, Gemma4State& state, Gemma4StepInputs& inputs, bool prefill,
                       bool select, ops::TokenOutput output = ops::TokenOutput::HOST,
                       tensor::Tensor* output_hidden = nullptr) -> tensor::Tensor;
    auto can_decode_step(const Gemma4State& state) const -> bool;
    auto run_batch(std::span<const std::int32_t> tokens, std::span<LanguageModelState* const> states, bool select)
        -> Result<tensor::Tensor>;
    auto prefill_impl(std::span<const std::int32_t> tokens, Gemma4State& state) -> Result<void>;
    auto project(std::span<const std::int32_t> tokens, Gemma4State& state, bool all_logits, bool select,
                 std::span<LanguageModelState* const> batch_states = {}, tensor::Tensor* output_hidden = nullptr)
        -> Result<tensor::Tensor>;
    std::unique_ptr<State> impl_;
};
} // namespace kidi::model