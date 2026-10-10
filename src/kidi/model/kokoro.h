#pragma once

#include <span>
#include <string_view>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "kidi/checkpoint/weights.h"
#include "kidi/inference/synthesizer.h"

namespace kidi::model {

struct KokoroOutput {
    std::vector<float> samples;
    std::vector<std::int32_t> durations;
};

KIDI_MODULE(Kokoro);
class KokoroImpl : public Module, public inference::SynthesizerModel {
public:
    KokoroImpl(YAML::Node config, checkpoint::Weights weights);
    ~KokoroImpl();

    static auto validate_config(const YAML::Node& config) -> Result<void>;
    static auto load(YAML::Node config, checkpoint::Weights weights) -> Result<Kokoro>;
    static auto load(YAML::Node package, tensor::Device device) -> Result<Kokoro>;

    auto synthesize(std::span<const std::int32_t> phoneme_ids, std::string_view voice, float speed, std::uint64_t seed)
        -> Result<KokoroOutput>;
    auto synthesize(std::string_view text, const inference::SynthesisOptions& options)
        -> Result<inference::Synthesis> override;
    auto execution() const -> std::string override { return "cpu"; }
    auto sample_rate() const noexcept -> std::uint32_t;
    auto preparation_ns() const noexcept -> std::uint64_t;

private:
    struct State;
    std::unique_ptr<State> impl_;
};

} // namespace kidi::model
