#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "kidi/model/model.h"
#include "kidi/ops/context.h"

namespace kidi::model {

/// Caches of one (layer, loop pass) visit: keys/values by absolute position and the last W - 1 inputs of each
/// short convolution (key, value, attention output, MoE output).
struct Inkling2Visit {
    tensor::Tensor key, value;
    tensor::Tensor key_conv, value_conv, attention_conv, mlp_conv;
};

struct Inkling2State : LanguageModelState {
    std::vector<Inkling2Visit> visits;
    std::size_t position = 0, capacity = 0;

    auto sequence_position() const noexcept -> std::size_t override { return position; }
    auto sequence_capacity() const noexcept -> std::size_t override { return capacity; }
    auto cache_bytes() const noexcept -> std::size_t override;
};

/// Thinking Machines Inkling-2 text model (`inkling2_loop_text`): block attention residuals, relative-position
/// attention with sinks, short convolutions, a sigmoid-routed MoE and a looped core of layers. CUDA only; experts
/// are dequantized from NVFP4/MXFP8 to BF16 at load. `tools/reference/inkling2.py` is the executable specification.
KIDI_MODULE(Inkling2);
class Inkling2Impl : public CausalLanguageModel {
public:
    explicit Inkling2Impl(const YAML::Node& config);
    static auto validate_config(const YAML::Node& config) -> Result<void>;
    static auto create(const YAML::Node& config) -> Result<Inkling2>;

    auto set_checkpoint(const checkpoint::Weights& weights) -> Result<void> override;
    auto set_checkpoint(const checkpoint::Weights& weights, RuntimeQuantization) -> Result<void> override {
        return set_checkpoint(weights);
    }
    auto hidden_size() const noexcept -> std::int32_t override { return hidden_; }
    auto vocabulary_size() const noexcept -> std::int32_t override { return vocabulary_; }
    auto max_positions() const noexcept -> std::size_t override { return max_positions_; }
    auto kv_cache_dtype(std::size_t) const -> Result<tensor::DType> override { return tensor::DType::BF16; }
    auto supports_state_recycling() const noexcept -> bool override { return true; }
    auto create_state(std::size_t capacity) -> Result<std::unique_ptr<LanguageModelState>> override;
    auto rewind(LanguageModelState& state, std::size_t position) -> Result<void> override;
    auto prefill(std::span<const std::int32_t> tokens, LanguageModelState& state) -> Result<void> override;
    auto forward(std::span<const std::int32_t> tokens, LanguageModelState& state, bool all_logits = false,
                 tensor::Tensor* hidden = nullptr) -> Result<tensor::Tensor> override;

private:
    /// Completed AttnRes blocks and the current partial block (BF16 [tokens, hidden]).
    struct Stream {
        std::vector<tensor::Tensor> blocks;
        tensor::Tensor partial;
    };
    auto weight(const std::string& name) const -> const tensor::Tensor&;
    auto visit(std::int32_t layer, Stream stream, Inkling2Visit& cache, std::size_t start,
               const std::vector<tensor::Tensor>* bank) -> Stream;
    auto attention(std::int32_t layer, const std::array<tensor::Tensor, 3>& inputs, Inkling2Visit& cache,
                   std::size_t start) -> tensor::Tensor;
    auto blend(std::int32_t layer, const std::string& stream, const tensor::Tensor& destination,
               const tensor::Tensor& message, const tensor::Tensor& norm) -> tensor::Tensor;
    auto moe(std::int32_t layer, const tensor::Tensor& input) -> tensor::Tensor;
    auto convolve(const tensor::Tensor& input, tensor::Tensor& state, const tensor::Tensor& weight,
                  const tensor::Tensor& taps, float scale) -> tensor::Tensor;
    auto local(std::int32_t layer) const -> bool;
    auto schedule() const -> std::vector<std::int32_t>;

    std::int32_t hidden_, layers_, heads_, key_heads_, head_width_, vocabulary_, padded_vocabulary_;
    std::int32_t experts_, experts_per_token_, intermediate_, block_, window_, relative_extent_;
    std::int32_t core_start_, core_end_, passes_;
    std::size_t max_positions_;
    float route_scale_, gate_cap_, up_cap_, logit_cap_, qk_clip_, relative_clip_, log_alpha_, log_floor_;
    std::vector<std::int32_t> local_layers_;
    std::map<std::string, tensor::Tensor, std::less<>> weights_;
    std::vector<std::vector<std::array<tensor::Tensor, 2>>> experts_weights_; // [layer][expert] = {w13, w2}
    std::unique_ptr<ops::Context> context_;
};

} // namespace kidi::model
