#pragma once

#include "kidi/inference/decoder.h"
#include "kidi/model/gemma4.h"

namespace kidi::model {
struct Gemma4Draft {
    tensor::Tensor hidden;
    std::int32_t token;
};

KIDI_MODULE(Gemma4Assistant);
class Gemma4AssistantImpl : public DraftLanguageModel {
public:
    Gemma4AssistantImpl(const YAML::Node& config, const YAML::Node& target_config);
    ~Gemma4AssistantImpl() override;
    static auto validate_config(const YAML::Node& config) -> Result<void>;
    static auto validate_config(const YAML::Node& config, const YAML::Node& target_config) -> Result<void>;
    static auto create(const YAML::Node& config, const YAML::Node& target_config) -> Result<Gemma4Assistant>;
    auto state_mapping_specs() const -> std::vector<checkpoint::StateMappingSpec> override;
    auto set_checkpoint(const checkpoint::Weights& weights) -> Result<void> override;

    auto preparation_ns() const noexcept -> std::uint64_t override;
    auto release_workspaces() -> void override;

    auto forward(Gemma4Impl& target, const tensor::Tensor& hidden, std::int32_t token, Gemma4State& state)
        -> Result<Gemma4Draft>;
    auto draft(LanguageModel& target, const tensor::Tensor& hidden, std::int32_t last_token, LanguageModelState& state,
               std::size_t max_drafts, const std::function<bool(std::int32_t)>& is_stop,
               std::vector<std::int32_t>& drafts) -> Result<void> override;

private:
    struct State;
    std::unique_ptr<State> impl_;
};
} // namespace kidi::model
