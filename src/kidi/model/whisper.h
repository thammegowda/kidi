#pragma once

#include <yaml-cpp/yaml.h>

#include "kidi/audio/whisper.h"
#include "kidi/layers/whisper.h"
#include "kidi/checkpoint/config.h"
#include "kidi/checkpoint/prepare.h"
#include "kidi/model/model.h"

namespace kidi::model {

struct WhisperEncoderState : EncoderMemory {
    tensor::Tensor convolution;
    tensor::Tensor hidden;
    std::vector<layers::KeyValue> layers;
};

/// Decoder state written in place each step; `tokens`, `mask`, and `index` are the per-token step inputs.
struct WhisperDecoderState : EncoderDecoderState {
    std::vector<layers::KeyValue> layers;
    tensor::Tensor tokens, mask, index;
    std::size_t position = 0, capacity = 0;
    auto sequence_position() const noexcept -> std::size_t override { return position; }
    auto sequence_capacity() const noexcept -> std::size_t override { return capacity; }
};

KIDI_MODULE(Whisper);
class WhisperImpl : public SpeechRecognitionModel {
public:
    explicit WhisperImpl(const YAML::Node& config);
    ~WhisperImpl() override;
    static auto validate_config(const YAML::Node& config) -> Result<void>;
    static auto create(const YAML::Node& config) -> Result<Whisper>;
    auto state_mapping_specs() const -> std::vector<checkpoint::StateMappingSpec> override;
    auto set_checkpoint(const checkpoint::Weights& weights) -> Result<void> override;
    static auto int8_preparation(const YAML::Node& config) -> Result<checkpoint::Preparation>;
    static auto int8_checkpoint(const YAML::Node& config, const checkpoint::Weights& weights) -> Result<StateDict>;

    auto hidden_size() const noexcept -> std::int32_t override;
    auto vocabulary_size() const noexcept -> std::int32_t override;
    auto max_positions() const noexcept -> std::size_t override;
    auto mel_bins() const noexcept -> std::int32_t override;
    auto preparation_ns() const noexcept -> std::uint64_t override;
    auto release_workspaces() -> void override;

    /// Encodes the first `frames` feature frames (all when zero); an even count up to the 30-second window.
    /// `cancelled` is polled between encoder blocks.
    auto encode_audio(const audio::WhisperFeatures& features, std::size_t frames = 0,
                      const std::function<bool()>& cancelled = {}) -> Result<std::shared_ptr<EncoderMemory>> override;
    auto create_decoder_state(std::shared_ptr<const EncoderMemory> memory, std::size_t batch, std::size_t capacity)
        -> Result<std::unique_ptr<LanguageModelState>> override;
    auto prefill(std::span<const std::int32_t> tokens, LanguageModelState& state) -> Result<void> override;
    auto decode_step(const EncoderMemory& memory, std::span<const std::int32_t> tokens, std::size_t batch,
                     LanguageModelState* state = nullptr, inference::InferenceStats* stats = nullptr)
        -> Result<tensor::Tensor> override;
    using SpeechRecognitionModel::select_token;
    auto select_token(const tensor::Tensor& logits, const tensor::Tensor& mask) -> tensor::Tensor override;
    auto select_language(const tensor::Tensor& logits, const tensor::Tensor& language_ids) -> tensor::Tensor override;

private:
    auto decode(const WhisperEncoderState& source, std::int32_t token, WhisperDecoderState& state, bool project)
        -> Result<tensor::Tensor>;
    struct State;
    std::unique_ptr<State> impl_;
};

} // namespace kidi::model