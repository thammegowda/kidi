#pragma once

#include "kidi/model/gemma4.h"

namespace kidi::model {
struct Gemma4Draft {
    tensor::Tensor hidden;
    std::int32_t token;
};

KIDI_MODULE(Gemma4Assistant);
class Gemma4AssistantImpl : public Module {
public:
    Gemma4AssistantImpl(const YAML::Node& config, const YAML::Node& target_config);
    ~Gemma4AssistantImpl();
    static auto checkpoint_config() -> checkpoint::ConfigAdapter;
    static auto validate_config(const YAML::Node& config, const YAML::Node& target_config) -> Result<void>;
    static auto create(const YAML::Node& config, const YAML::Node& target_config) -> Result<Gemma4Assistant>;
    auto set_checkpoint(const checkpoint::Weights& weights) -> Result<void>;
    auto forward(Gemma4Impl& target, const tensor::Tensor& hidden, std::int32_t token, Gemma4State& state)
        -> Result<Gemma4Draft>;
    auto preparation_ns() const -> std::uint64_t;
    auto release_workspaces() -> void;

private:
    struct State;
    std::unique_ptr<State> impl_;
};
} // namespace kidi::model
