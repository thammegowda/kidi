#pragma once

#include "kidi/layers/gemma.h"
#include "kidi/model/weights.h"
#include <yaml-cpp/yaml.h>

namespace kidi::model {
struct GemmaState {
    std::vector<layers::KeyValue> layers;
    std::size_t position = 0, capacity = 0;
    bool crop_local_attention = true;
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
    auto create_state(std::size_t capacity) -> Result<GemmaState>;
    auto prefill(std::span<const std::int32_t> tokens, GemmaState& state) -> Result<void>;
    auto forward(std::span<const std::int32_t> tokens, GemmaState& state, bool all_logits = false)
        -> Result<tensor::Tensor>;
    auto preparation_ns() const -> std::uint64_t;

private:
    auto run(std::span<const std::int32_t> tokens, GemmaState& state, bool all_logits, bool project)
        -> Result<tensor::Tensor>;
    struct State;
    std::unique_ptr<State> impl_;
};
} // namespace kidi::model