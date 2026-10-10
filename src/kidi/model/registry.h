#pragma once

#include <span>
#include <string_view>

namespace kidi::model {

enum class Task {
    TRANSLATION,
    TEXT_GENERATION,
    SPEECH_RECOGNITION,
    TTS,
};

struct Descriptor {
    std::string_view type;
    std::span<const Task> tasks;
};

auto task_name(Task task) -> std::string_view;
auto registered_models() -> std::span<const Descriptor>;
auto find_model(std::string_view type) -> const Descriptor*;
auto supports_task(std::string_view type, Task task) -> bool;

} // namespace kidi::model
