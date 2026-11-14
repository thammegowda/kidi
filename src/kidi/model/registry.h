#pragma once

#include <memory>
#include <optional>
#include <span>
#include <string_view>

#include <yaml-cpp/yaml.h>

#include "kidi/model/model.h"

namespace kidi::model {

enum class Task {
    TRANSLATION,
    TEXT_GENERATION,
    SPEECH_RECOGNITION,
    TTS,
};

struct CreateOptions {
    tensor::Device device = tensor::Device::cpu();
    std::optional<tensor::DType> parameter_dtype;
};

struct Descriptor {
    std::string_view type;
    std::span<const Task> tasks;
    ModelKind kind;
    auto (*validate_config)(const YAML::Node& config) -> Result<void>;
    auto (*create)(const YAML::Node& config) -> Result<std::shared_ptr<Model>> = nullptr;
};

auto task_name(Task task) -> std::string_view;
auto registered_models() -> std::span<const Descriptor>;
auto find_model(std::string_view type) -> const Descriptor*;
auto supports_task(std::string_view type, Task task) -> bool;
auto validate_model_config(const YAML::Node& config) -> Result<void>;
auto create_model(const YAML::Node& config, CreateOptions options = {}) -> Result<std::shared_ptr<Model>>;

template <typename TaskInterface>
auto create_model_as(const YAML::Node& config, CreateOptions options = {}) -> Result<std::shared_ptr<TaskInterface>> {
    auto created = create_model(config, options);
    if (!created) return std::unexpected(std::move(created.error()));
    auto typed = std::dynamic_pointer_cast<TaskInterface>(*created);
    if (!typed)
        return std::unexpected(
            Error{ErrorCode::INVALID_MANIFEST, "model type does not implement the requested task interface"});
    return typed;
}

} // namespace kidi::model
