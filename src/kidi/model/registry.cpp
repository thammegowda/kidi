#include "kidi/model/registry.h"

#include <algorithm>
#include <array>

namespace kidi::model {
namespace {

constexpr std::array TRANSLATION_TASKS{Task::TRANSLATION};
constexpr std::array GENERATION_TASKS{Task::TEXT_GENERATION};
constexpr std::array RECOGNITION_TASKS{Task::SPEECH_RECOGNITION};
constexpr std::array TTS_TASKS{Task::TTS};
constexpr std::array MODELS{
    Descriptor{"rtg_transformer_nmt", TRANSLATION_TASKS},
    Descriptor{"gemma4_text", GENERATION_TASKS},
    Descriptor{"whisper", RECOGNITION_TASKS},
    Descriptor{"omnivoice", TTS_TASKS},
    Descriptor{"kokoro", TTS_TASKS},
};

} // namespace

auto task_name(Task task) -> std::string_view {
    switch (task) {
        case Task::TRANSLATION:
            return "translation";
        case Task::TEXT_GENERATION:
            return "text generation";
        case Task::SPEECH_RECOGNITION:
            return "speech recognition";
        case Task::TTS:
            return "TTS";
    }
    return "unknown";
}

auto registered_models() -> std::span<const Descriptor> { return MODELS; }

auto find_model(std::string_view type) -> const Descriptor* {
    const auto found = std::ranges::find(MODELS, type, &Descriptor::type);
    return found == MODELS.end() ? nullptr : &*found;
}

auto supports_task(std::string_view type, Task task) -> bool {
    const auto* descriptor = find_model(type);
    return descriptor && std::ranges::find(descriptor->tasks, task) != descriptor->tasks.end();
}

} // namespace kidi::model
