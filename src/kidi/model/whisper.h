#pragma once

#include <yaml-cpp/yaml.h>

#include "kidi/audio/whisper.h"
#include "kidi/layers/whisper.h"
#include "kidi/checkpoint/weights.h"
#include "kidi/checkpoint/config.h"
#include "kidi/checkpoint/prepare.h"

namespace kidi::model {

struct WhisperEncoderState {
    tensor::Tensor convolution;
    tensor::Tensor hidden;
    std::vector<layers::KeyValue> layers;
};

/// Decoder state written in place each step; `tokens`, `mask`, and `index` are the per-token step inputs.
struct WhisperDecoderState {
    std::vector<layers::KeyValue> layers;
    tensor::Tensor tokens, mask, index;
    std::size_t position = 0, capacity = 0;
};

KIDI_MODULE(Whisper);
class WhisperImpl : public Module {
public:
    explicit WhisperImpl(const YAML::Node& config);
    ~WhisperImpl();
    static auto validate_config(const YAML::Node& config) -> Result<void>;
    static auto create(const YAML::Node& config) -> Result<Whisper>;
    static auto checkpoint_config() -> checkpoint::ConfigAdapter;
    static auto int8_preparation(const YAML::Node& config) -> Result<checkpoint::Preparation>;
    static auto int8_checkpoint(const YAML::Node& config, const checkpoint::Weights& weights) -> Result<StateDict>;
    auto set_checkpoint(const checkpoint::Weights& weights) -> Result<void>;
    /// Encodes the first `frames` feature frames (all when zero); an even count up to the 30-second window.
    auto encode(const audio::WhisperFeatures& features, std::size_t frames = 0) -> Result<WhisperEncoderState>;
    auto create_state(std::size_t capacity) -> Result<WhisperDecoderState>;
    auto forward(const WhisperEncoderState& source, std::span<const std::int32_t> tokens,
                 WhisperDecoderState& state) -> Result<tensor::Tensor>;
    auto prefill(const WhisperEncoderState& source, std::span<const std::int32_t> tokens,
                 WhisperDecoderState& state) -> Result<void>;
    auto preparation_ns() const -> std::uint64_t;

private:
    auto decode(const WhisperEncoderState& source, std::int32_t token, WhisperDecoderState& state,
                bool project) -> Result<tensor::Tensor>;
    struct State;
    std::unique_ptr<State> impl_;
};

} // namespace kidi::model