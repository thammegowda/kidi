#pragma once

#include "kidi/layers/gemma4.h"
#include "kidi/model/weights.h"
#include <yaml-cpp/yaml.h>

namespace kidi::model {
struct Gemma4State {
    std::vector<layers::KeyValue> layers;
    std::size_t position = 0, capacity = 0;
    bool crop_local_attention = true;
    bool prefilling = true;
    bool profile_started = false;
};

KIDI_MODULE(Gemma4);
class Gemma4Impl : public Module {
public:
    explicit Gemma4Impl(const YAML::Node& config);
    ~Gemma4Impl();
    static auto validate_config(const YAML::Node& config) -> Result<void>;
    static auto create(const YAML::Node& config) -> Result<Gemma4>;
    auto set_checkpoint(const Weights& weights, std::int32_t weight_bits = 0, std::int32_t group_size = 128,
                        bool packed_prefill = false) -> Result<void>;
    auto create_state(std::size_t capacity) -> Result<Gemma4State>;
    auto fork_state(const Gemma4State& source, std::size_t prefix_length, std::size_t capacity) -> Result<Gemma4State>;
    auto prefill(std::span<const std::int32_t> tokens, Gemma4State& state) -> Result<void>;
    auto forward(std::span<const std::int32_t> tokens, Gemma4State& state, bool all_logits = false)
        -> Result<tensor::Tensor>;
    auto forward_token(std::span<const std::int32_t> tokens, Gemma4State& state) -> Result<std::int32_t>;
    auto forward_batch(std::span<const std::int32_t> tokens, std::span<Gemma4State*> states) -> Result<tensor::Tensor>;
    auto forward_batch_tokens(std::span<const std::int32_t> tokens, std::span<Gemma4State*> states)
        -> Result<std::vector<std::int32_t>>;
    auto preparation_ns() const -> std::uint64_t;
    auto last_token_prefill() const -> bool;
    auto shared_prefill_tail() const -> bool;

private:
    struct State;
    struct Attention;
    auto embed(std::span<const std::int32_t> tokens) -> std::array<tensor::Tensor, 2>;
    auto attention_inputs(Gemma4State& state, std::span<Gemma4State*> batch_states, std::size_t step_count)
        -> Attention;
    auto per_layer_input(const tensor::Tensor& per_layer, int layer, std::int64_t length) -> tensor::Tensor;
    auto head(const tensor::Tensor& hidden, bool select) -> tensor::Tensor;
    auto run_batch(std::span<const std::int32_t> tokens, std::span<Gemma4State*> states, bool select)
        -> Result<tensor::Tensor>;
    /// Runs every layer and the output head; `batch_states` decodes one token per request.
    auto project(std::span<const std::int32_t> tokens, Gemma4State& state, bool all_logits, bool select,
                 std::span<Gemma4State*> batch_states = {}) -> Result<tensor::Tensor>;
    std::unique_ptr<State> impl_;
};
} // namespace kidi::model