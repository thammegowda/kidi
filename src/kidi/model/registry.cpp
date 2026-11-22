#include "kidi/model/registry.h"

#include <algorithm>
#include <array>

#include "kidi/model/gemma4.h"
#include "kidi/model/gemma4_assistant.h"
#include "kidi/model/inkling2.h"
#include "kidi/model/kokoro.h"
#include "kidi/model/omnivoice.h"
#include "kidi/model/transformer.h"
#include "kidi/model/whisper.h"

namespace kidi::model {
namespace {

constexpr std::array TRANSLATION_TASKS{Task::TRANSLATION};
constexpr std::array GENERATION_TASKS{Task::TEXT_GENERATION};
constexpr std::array RECOGNITION_TASKS{Task::SPEECH_RECOGNITION};
constexpr std::array TTS_TASKS{Task::TTS};
constexpr std::span<const Task> NO_TASKS{};

template <typename Holder, auto Factory>
auto construct_model(const YAML::Node& config) -> Result<std::shared_ptr<Model>> {
    auto created = Factory(config);
    if (!created) return std::unexpected(std::move(created.error()));
    return std::shared_ptr<Model>(created->ptr());
}

const std::array MODELS{
    Descriptor{"rtg_transformer_nmt", TRANSLATION_TASKS, ModelKind::SEQ2SEQ_LM, TransformerImpl::validate_config,
               construct_model<Transformer, TransformerImpl::create>},
    Descriptor{"gemma4_text", GENERATION_TASKS, ModelKind::CAUSAL_LM, Gemma4Impl::validate_config,
               construct_model<Gemma4, Gemma4Impl::create>},
    Descriptor{"gemma4_assistant", NO_TASKS, ModelKind::DRAFT_LM,
               static_cast<Result<void> (*)(const YAML::Node&)>(Gemma4AssistantImpl::validate_config), nullptr},
    Descriptor{"inkling2_text", GENERATION_TASKS, ModelKind::CAUSAL_LM, Inkling2Impl::validate_config,
               construct_model<Inkling2, Inkling2Impl::create>},
    Descriptor{"whisper", RECOGNITION_TASKS, ModelKind::SPEECH_RECOGNITION_LM, WhisperImpl::validate_config,
               construct_model<Whisper, WhisperImpl::create>},
    Descriptor{"omnivoice", TTS_TASKS, ModelKind::SPEECH_SYNTHESIS, OmniVoiceImpl::validate_config, nullptr},
    Descriptor{"kokoro", TTS_TASKS, ModelKind::SPEECH_SYNTHESIS, KokoroImpl::validate_config, nullptr},
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

auto validate_model_config(const YAML::Node& config) -> Result<void> {
    try {
        if (!config.IsMap() || !config["type"] || !config["type"].IsScalar())
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "model config requires scalar type"});
        const auto type = config["type"].as<std::string>();
        const auto* descriptor = find_model(type);
        if (!descriptor) return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "unsupported model.type: " + type});
        return descriptor->validate_config(config);
    } catch (const YAML::Exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, error.what()});
    }
}

auto create_model(const YAML::Node& config, CreateOptions options) -> Result<std::shared_ptr<Model>> {
    try {
        if (auto valid = validate_model_config(config); !valid) return std::unexpected(std::move(valid.error()));
        const auto type = config["type"].as<std::string>();
        const auto* descriptor = find_model(type);
        if (!descriptor || !descriptor->create)
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, "model type cannot be constructed directly: " + type});
        const ModuleScope scope(options.parameter_dtype.value_or(module_dtype), false, options.device);
        return descriptor->create(config);
    } catch (const YAML::Exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, error.what()});
    }
}

} // namespace kidi::model
