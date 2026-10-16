#pragma once

#include <chrono>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "kidi/core/error.h"
#include "kidi/tensor/tensor.h"

namespace kidi::inference {

using VoiceAttributes = std::map<std::string, std::string, std::less<>>;

struct VoicePrompt {
    std::string text;
    /// Model-native, codebook-major audio tokens paired with `text`.
    std::vector<std::int32_t> tokens;
};

struct SynthesisOptions {
    std::optional<float> duration_seconds;
    std::string language;
    VoiceAttributes voice;
    std::shared_ptr<const VoicePrompt> voice_prompt;
    std::size_t steps = 32;
    float guidance_scale = 2.F;
    float time_shift = 0.1F;
    float layer_penalty = 5.F;
    float position_temperature = 5.F;
    /// Multiplies model-estimated automatic duration; greater than one speaks more slowly without resampling audio.
    float duration_scale = 1.F;
    std::uint64_t seed = 0;
    std::function<void(std::size_t, std::size_t)> on_progress;
    std::function<void(std::size_t, std::size_t)> on_chunk;
    /// Receives completed waveform chunks in playback order. The span is valid only for the duration of the callback.
    std::function<void(std::span<const float>, std::uint32_t)> on_audio_chunk;
    /// Disable when audio is consumed through `on_audio_chunk` to avoid retaining the complete waveform.
    bool retain_audio = true;
};

struct SynthesisStats {
    std::uint64_t generation_ns = 0;
    std::uint64_t decode_ns = 0;
    std::uint64_t preparation_ns = 0;
    std::size_t audio_tokens = 0;
    std::size_t chunks = 1;
};

struct Synthesis {
    std::vector<float> samples;
    std::uint32_t sample_rate;
    std::vector<std::int32_t> tokens;
    SynthesisStats stats;
    std::shared_ptr<const VoicePrompt> voice_prompt;
};

struct SynthesisReference {
    std::string_view text;
    std::span<const std::int32_t> tokens;
};

auto parse_synthesis_number(std::string_view value) -> Result<float>;
auto parse_voice_attributes(std::span<const std::string> specifications) -> Result<VoiceAttributes>;
auto synthesis_elapsed_ns(std::chrono::steady_clock::time_point start) -> std::uint64_t;

class SynthesizerModel {
public:
    virtual ~SynthesizerModel() = default;
    virtual auto synthesize(std::string_view text, const SynthesisOptions& options) -> Result<Synthesis> = 0;
    virtual auto synthesize_with_reference(std::string_view text, const SynthesisOptions& options,
                                           const SynthesisReference&) -> Result<Synthesis> {
        return synthesize(text, options);
    }
    virtual auto voice_reference_chunk(std::span<const std::string>, const SynthesisOptions&) const
        -> Result<std::optional<std::size_t>> {
        return std::optional<std::size_t>{};
    }
    virtual auto supports_voice_prompt() const noexcept -> bool { return false; }
    virtual auto execution() const -> std::string = 0;
    virtual auto release_workspaces() -> void {}
};

class Synthesizer {
public:
    Synthesizer(Synthesizer&&) noexcept;
    auto operator=(Synthesizer&&) noexcept -> Synthesizer&;
    ~Synthesizer();

    Synthesizer(const Synthesizer&) = delete;
    auto operator=(const Synthesizer&) -> Synthesizer& = delete;

    static auto load(const std::filesystem::path& directory, tensor::Device device = tensor::Device::cpu())
        -> Result<Synthesizer>;
    auto segment(std::string_view text, std::string_view language = {}) const -> Result<std::vector<std::string>>;
    auto synthesize(std::string_view text, const SynthesisOptions& options = {}) const -> Result<Synthesis>;
    auto execution() const -> std::string;
    auto supports_voice_prompt() const noexcept -> bool;

private:
    struct Impl;
    explicit Synthesizer(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

} // namespace kidi::inference
