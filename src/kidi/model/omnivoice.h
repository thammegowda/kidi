#pragma once

#include <span>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "kidi/checkpoint/weights.h"
#include "kidi/inference/synthesizer.h"
#include "kidi/layers/higgs_audio.h"
#include "kidi/model/qwen3.h"

namespace kidi::model {

KIDI_MODULE(OmniVoice);
class OmniVoiceImpl : public Module, public inference::SynthesizerModel {
public:
    explicit OmniVoiceImpl(const YAML::Node& config);
    ~OmniVoiceImpl();

    static auto validate_config(const YAML::Node& config) -> Result<void>;
    static auto create(const YAML::Node& config) -> Result<OmniVoice>;
    static auto load(YAML::Node package, tensor::Device device) -> Result<OmniVoice>;
    static auto voice_instruction(const inference::VoiceAttributes& attributes, std::string_view text,
                                  std::string_view language) -> Result<std::string>;
    auto set_checkpoint(const checkpoint::Weights& weights) -> Result<void>;
    auto synthesize(std::string_view text, const inference::SynthesisOptions& options)
        -> Result<inference::Synthesis> override;
    auto execution() const -> std::string override { return "cpu"; }

    /// `input_ids` is codebook-major `[codebooks, sequence]`; `audio_mask` marks waveform-token positions.
    /// Returns target logits as `[1, target_length, codebooks * audio_vocabulary]`.
    auto forward(std::span<const std::int32_t> input_ids, std::span<const std::uint8_t> audio_mask,
                 std::size_t target_start, std::size_t target_length) -> Result<tensor::Tensor>;
    /// Decodes codebook-major `[codebooks, length]` audio tokens to a mono waveform.
    auto decode(std::span<const std::int32_t> tokens, std::size_t length) -> Result<std::vector<float>>;

    auto text_vocabulary_size() const noexcept -> std::int32_t;
    auto audio_vocabulary_size() const noexcept -> std::int32_t;
    auto audio_mask_id() const noexcept -> std::int32_t;
    auto codebook_count() const noexcept -> std::int32_t;
    auto sample_rate() const noexcept -> std::uint32_t;
    auto frame_rate() const noexcept -> std::uint32_t;
    auto preparation_ns() const noexcept -> std::uint64_t;

private:
    auto target_length(std::string_view text, const inference::SynthesisOptions& options) const -> std::size_t;
    struct State;
    std::unique_ptr<State> impl_;
};

} // namespace kidi::model
