#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "kidi/model/model.h"
#include "kidi/checkpoint/prepare.h"
#include "kidi/ops/context.h"
#include "kidi/ops/quantization.h"

namespace kidi::model {

/// Caches of one (layer, loop pass) visit: keys/values by absolute position (on Metal and CUDA, sliding-window
/// layers keep key `p` in ring row `p % local_capacity`) and a ring buffer of the last W inputs of each short
/// convolution (key, value, attention output, MoE output).
struct Inkling2Visit {
    tensor::Tensor key, value;
    tensor::Tensor key_conv, value_conv, attention_conv, mlp_conv;
    tensor::Tensor key_scales, value_scales; // INT8 KV: F32 [capacity, key_heads * groups_per_head]
    std::array<tensor::Tensor, 4> rewind_inputs;
    bool record_rewind = false;
};

struct Inkling2State : LanguageModelState {
    struct StepInputs {
        tensor::Tensor token, positions, local_positions, conv_slot, tail_slots;
    };
    std::vector<Inkling2Visit> visits;
    std::vector<tensor::Tensor> replay_inputs;
    StepInputs decode_step, prefill_step;
    std::size_t position = 0, capacity = 0, local_capacity = 0;
    /// Convolution rings saved at `rewind_position` (0: none), so a prompt boundary can be restored.
    std::vector<tensor::Tensor> rewind_convs;
    std::size_t rewind_position = 0;
    /// CPU speculative verification: rings at `verify_position` plus each visit's `rewind_inputs` allow
    /// rewinding to any position in [verify_position, verify_end].
    std::vector<tensor::Tensor> verify_convs;
    std::size_t verify_position = 0, verify_end = 0;

    auto sequence_position() const noexcept -> std::size_t override { return position; }
    auto sequence_capacity() const noexcept -> std::size_t override { return capacity; }
    auto cache_bytes() const noexcept -> std::size_t override;
};

/// Thinking Machines Inkling-2 text model (`inkling2_loop_text`): block attention residuals, relative-position
/// attention with sinks, short convolutions, a sigmoid-routed MoE and a looped core of layers. CPU uses INT8 experts;
/// CUDA keeps native NVFP4/MXFP8 experts. KV caches are group-32 INT8 on CPU and BF16 (or opt-in INT8) on CUDA and
/// Metal; activations are BF16 (F32 on Metal).
/// `tools/reference/inkling2.py` is the executable specification.
KIDI_MODULE(Inkling2);
class Inkling2Impl : public CausalLanguageModel {
public:
    explicit Inkling2Impl(const YAML::Node& config);
    static auto validate_config(const YAML::Node& config) -> Result<void>;
    static auto create(const YAML::Node& config) -> Result<Inkling2>;
    static auto int8_preparation(const YAML::Node& config) -> Result<checkpoint::Preparation>;
    static auto vq_preparation(const YAML::Node& config) -> Result<checkpoint::Preparation>;

    auto set_checkpoint(const checkpoint::Weights& weights) -> Result<void> override;
    auto set_checkpoint(const checkpoint::Weights& weights, RuntimeQuantization quantization) -> Result<void> override;
    auto hidden_size() const noexcept -> std::int32_t override { return hidden_; }
    auto vocabulary_size() const noexcept -> std::int32_t override { return vocabulary_; }
    auto max_positions() const noexcept -> std::size_t override { return max_positions_; }
    auto set_kv_cache_precision(core::KVCachePrecision precision) noexcept -> void override {
        kv_cache_precision_ = precision;
    }
    auto kv_cache_dtype(std::size_t) const -> Result<tensor::DType> override {
        if (device().kind == tensor::DeviceKind::CPU) return tensor::DType::I8;
        if (device().kind == tensor::DeviceKind::A_GPU) {
            if (kv_cache_precision_ == core::KVCachePrecision::INT8) return tensor::DType::I8;
            if (kv_cache_precision_ == core::KVCachePrecision::FP32) return tensor::DType::F32;
            return tensor::DType::BF16;
        }
        if (kv_cache_precision_ == core::KVCachePrecision::INT8) return tensor::DType::I8;
        if (kv_cache_precision_ == core::KVCachePrecision::FP32)
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, "Inkling-2 CUDA KV caches support BF16 or INT8"});
        return tensor::DType::BF16;
    }
    auto preparation_ns() const noexcept -> std::uint64_t override;
    auto release_workspaces() -> void override;
    auto trim_workspaces() -> void override;
    auto accelerator() const -> std::string_view override;
    auto supports_state_recycling() const noexcept -> bool override { return true; }
    auto supports_device_tokens() const -> bool override;
    auto create_state(std::size_t capacity) -> Result<std::unique_ptr<LanguageModelState>> override;
    auto grow_state(LanguageModelState& state, std::size_t capacity) -> Result<void> override;
    auto rewind(LanguageModelState& state, std::size_t position) -> Result<void> override;
    auto mark_rewind_point(LanguageModelState& state) -> Result<void> override;
    auto rewind_floor(const LanguageModelState& state, std::size_t position) const -> std::size_t override;
    auto prefill(std::span<const std::int32_t> tokens, LanguageModelState& state) -> Result<void> override;
    auto forward(std::span<const std::int32_t> tokens, LanguageModelState& state, bool all_logits = false,
                 tensor::Tensor* hidden = nullptr) -> Result<tensor::Tensor> override;
    auto select_token(std::span<const std::int32_t> tokens, LanguageModelState& state, bool all_tokens = false,
                      tensor::Tensor* hidden = nullptr) -> Result<tensor::Tensor> override;
    auto forward_device_token(const tensor::Tensor& token, LanguageModelState& state)
        -> Result<tensor::Tensor> override;

private:
    static auto int8_checkpoint(const YAML::Node& config, const checkpoint::Weights& weights) -> Result<StateDict>;
    static auto vq_checkpoint(const YAML::Node& config, const checkpoint::Weights& weights) -> Result<StateDict>;
    auto load_prepared_checkpoint(const checkpoint::Weights& weights) -> Result<void>;
    auto load_prepared_vq_checkpoint(const checkpoint::Weights& weights) -> Result<void>;
    /// Completed AttnRes blocks and the current partial block (BF16 [tokens, hidden]).
    struct Stream {
        std::vector<tensor::Tensor> blocks;
        tensor::Tensor partial;
    };
    auto weight(const std::string& name) const -> const tensor::Tensor&;
    auto project_bf16(const tensor::Tensor& input, const std::string& name) -> tensor::Tensor;
    auto visit(std::int32_t layer, Stream stream, Inkling2Visit& cache, const tensor::Tensor& positions,
               const tensor::Tensor& local_positions, const tensor::Tensor& conv_slot, const tensor::Tensor& tail_slots,
               const std::vector<tensor::Tensor>* bank) -> Stream;
    auto attention(std::int32_t layer, const std::array<tensor::Tensor, 3>& inputs, Inkling2Visit& cache,
                   const tensor::Tensor& positions, const tensor::Tensor& local_positions,
                   const tensor::Tensor& conv_slot, const tensor::Tensor& tail_slots) -> tensor::Tensor;
    auto blend(std::int32_t layer, const std::string& stream, const tensor::Tensor& destination,
               const tensor::Tensor& message, const tensor::Tensor& norm) -> tensor::Tensor;
    auto moe(std::int32_t layer, const tensor::Tensor& input) -> tensor::Tensor;
    auto bind_local_kv_packed(const std::map<std::string, tensor::Tensor, std::less<>>& weights,
                              std::map<std::string, ops::PackedWeight, std::less<>>& packed) const -> void;
    auto convolve(const tensor::Tensor& input, tensor::Tensor& state, const tensor::Tensor& weight,
                  const tensor::Tensor& taps, const tensor::Tensor& conv_slot, const tensor::Tensor& tail_slots,
                  float scale, const tensor::Tensor& residual = {},
                  std::optional<tensor::DType> output_dtype = {}, tensor::Tensor* saved_input = nullptr,
                  bool fuse_single_update = true) -> tensor::Tensor;
    auto save_convolution_input(tensor::Tensor& saved, const tensor::Tensor& input) -> void;
    auto snapshot_verification(Inkling2State& state) -> void;
    auto stage_host_step(std::span<const std::int32_t> tokens, Inkling2State& state, bool all_logits,
                         std::optional<ops::TokenOutput> select, tensor::Tensor* hidden) -> Result<tensor::Tensor>;
    auto run_forward(const tensor::Tensor& ids, const tensor::Tensor& positions, const tensor::Tensor& local_positions,
                     const tensor::Tensor& conv_slot, const tensor::Tensor& tail_slots, Inkling2State& state,
                     std::int64_t T, bool all_logits, std::optional<ops::TokenOutput> select, tensor::Tensor* hidden)
        -> tensor::Tensor;
    auto execute_step(const tensor::Tensor& ids, const tensor::Tensor& positions, const tensor::Tensor& local_positions,
                      const tensor::Tensor& conv_slot, const tensor::Tensor& tail_slots, Inkling2State& state,
                      std::int64_t T, bool all_logits, std::optional<ops::TokenOutput> select, tensor::Tensor* hidden)
        -> tensor::Tensor;
    auto local(std::int32_t layer) const -> bool;
    auto schedule() const -> std::vector<std::int32_t>;
    auto cache_capacities(std::size_t capacity) const -> std::pair<std::size_t, std::size_t>;

    std::int32_t hidden_, layers_, heads_, key_heads_, head_width_, vocabulary_, padded_vocabulary_;
    std::int32_t experts_, experts_per_token_, intermediate_, block_, window_, relative_extent_;
    std::int32_t core_start_, core_end_, passes_;
    std::size_t max_positions_;
    float route_scale_, gate_cap_, up_cap_, logit_cap_, qk_clip_, relative_clip_, log_alpha_, log_floor_;
    std::vector<std::int32_t> local_layers_;
    std::map<std::string, tensor::Tensor, std::less<>> weights_;
    std::map<std::string, ops::PackedWeight, std::less<>> packed_; // CPU INT8 dense projections
    struct ExpertWeight {
        tensor::Tensor values, scales, global;
        ops::BlockScaledFormat format = ops::BlockScaledFormat::NVFP4;
        std::int32_t group_size = 0;
    };
    std::vector<std::vector<std::array<ExpertWeight, 2>>> experts_weights_; // [layer][expert] = {w13, w2}
    // Native checkpoint storage stacked over experts ([layer] = {w13, w2}) for device-routed decode; empty otherwise.
    std::vector<std::array<ExpertWeight, 2>> stacked_experts_;
    tensor::DType activation_ = tensor::DType::BF16;
    core::KVCachePrecision kv_cache_precision_ = core::KVCachePrecision::AUTO;
    std::unique_ptr<ops::Context> context_;
};

} // namespace kidi::model
