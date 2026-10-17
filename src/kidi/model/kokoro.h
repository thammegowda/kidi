#pragma once

#include <span>
#include <string_view>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "kidi/checkpoint/weights.h"
#include "kidi/inference/coroutine.h"
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
    auto start_synthesis(std::string_view text, const inference::SynthesisOptions& options)
        -> Result<std::unique_ptr<inference::SynthesisSession>> override;
    auto execution() const -> std::string override { return "cpu"; }
    auto release_workspaces() -> void override;
    auto sample_rate() const noexcept -> std::uint32_t;
    auto preparation_ns() const noexcept -> std::uint64_t;

private:
    using Session = inference::CoroutineSession<inference::Synthesis, inference::SynthesisSession>;
    auto synthesis_session_(std::string text, inference::SynthesisOptions options) -> Session;
    struct State;
    std::unique_ptr<State> impl_;
};

} // namespace kidi::model
