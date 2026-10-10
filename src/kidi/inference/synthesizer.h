#pragma once

#include <chrono>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "kidi/core/error.h"
#include "kidi/tensor/tensor.h"

namespace kidi::inference {

using VoiceAttributes = std::map<std::string, std::string, std::less<>>;

struct SynthesisOptions {
    std::optional<float> duration_seconds;
    std::string language;
    VoiceAttributes voice;
    std::size_t steps = 32;
    float guidance_scale = 2.F;
    float time_shift = 0.1F;
    float layer_penalty = 5.F;
    float position_temperature = 5.F;
    std::uint64_t seed = 0;
    std::function<void(std::size_t, std::size_t)> on_progress;
};

struct SynthesisStats {
    std::uint64_t generation_ns = 0;
    std::uint64_t decode_ns = 0;
    std::uint64_t preparation_ns = 0;
    std::size_t audio_tokens = 0;
};

struct Synthesis {
    std::vector<float> samples;
    std::uint32_t sample_rate;
    std::vector<std::int32_t> tokens;
    SynthesisStats stats;
};

auto parse_synthesis_number(std::string_view value) -> Result<float>;
auto parse_voice_attributes(std::span<const std::string> specifications) -> Result<VoiceAttributes>;
auto synthesis_elapsed_ns(std::chrono::steady_clock::time_point start) -> std::uint64_t;

class SynthesizerModel {
public:
    virtual ~SynthesizerModel() = default;
    virtual auto synthesize(std::string_view text, const SynthesisOptions& options) -> Result<Synthesis> = 0;
    virtual auto execution() const -> std::string = 0;
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
    auto synthesize(std::string_view text, const SynthesisOptions& options = {}) const -> Result<Synthesis>;
    auto execution() const -> std::string;

private:
    struct Impl;
    explicit Synthesizer(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

} // namespace kidi::inference
