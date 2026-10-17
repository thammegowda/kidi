#pragma once

#include <span>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "kidi/audio/wav.h"
#include "kidi/checkpoint/weights.h"
#include "kidi/inference/synthesizer.h"
#include "kidi/inference/coroutine.h"
#include "kidi/layers/higgs_audio.h"
#include "kidi/model/qwen3.h"

namespace kidi::model {

KIDI_MODULE(OmniVoice);
class OmniVoiceImpl : public Module, public inference::SynthesizerModel {
public:
    explicit OmniVoiceImpl(const YAML::Node& config);
    ~OmniVoiceImpl();

    static auto validate_config(const YAML::Node& config) -> Result<void>;
    static auto load(YAML::Node package, tensor::Device device) -> Result<OmniVoice>;
    static auto voice_instruction(const inference::VoiceAttributes& attributes, std::string_view text,
                                  std::string_view language) -> Result<std::string>;
    auto start_synthesis(std::string_view text, const inference::SynthesisOptions& options)
        -> Result<std::unique_ptr<inference::SynthesisSession>> override;
    auto voice_reference_chunk(std::span<const std::string> chunks, const inference::SynthesisOptions& options) const
        -> Result<std::optional<std::size_t>> override;
    auto supports_voice_prompt() const noexcept -> bool override { return true; }
    auto execution() const -> std::string override;
    auto release_workspaces() -> void override;

    /// Decodes host codebook-major audio tokens for browser voice-reference playback.
    auto decode_reference(std::span<const std::int32_t> tokens) -> Result<audio::Waveform>;

private:
    static auto create(const YAML::Node& config) -> Result<OmniVoice>;
    auto set_checkpoint(const checkpoint::Weights& weights) -> Result<void>;
    auto decode(const tensor::Tensor& tokens, bool normalize = false) -> Result<std::vector<float>>;
    auto target_length(std::string_view text, const inference::SynthesisOptions& options) const -> std::size_t;
    using Session = inference::CoroutineSession<inference::Synthesis, inference::SynthesisSession>;
    auto synthesis_session_(std::string text, inference::SynthesisOptions options) -> Session;
    struct State;
    std::unique_ptr<State> impl_;
};

} // namespace kidi::model
