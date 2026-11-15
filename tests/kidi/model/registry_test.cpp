#include "kidi/model/registry.h"

#include <cassert>

int main() {
    using kidi::model::supports_task;
    using kidi::model::Task;
    using kidi::model::validate_model_config;

    assert(supports_task("rtg_transformer_nmt", Task::TRANSLATION));
    assert(supports_task("gemma4_text", Task::TEXT_GENERATION));
    assert(supports_task("whisper", Task::SPEECH_RECOGNITION));
    assert(supports_task("omnivoice", Task::TTS));
    assert(supports_task("kokoro", Task::TTS));
    assert(!supports_task("gemma4_assistant", Task::TEXT_GENERATION));
    assert(!supports_task("omnivoice", Task::TEXT_GENERATION));
    assert(!supports_task("unknown", Task::TTS));

    YAML::Node moe_gemma;
    moe_gemma["type"] = "gemma4_text";
    moe_gemma["enable_moe_block"] = true;
    assert(!validate_model_config(moe_gemma));

    YAML::Node bad_assistant;
    bad_assistant["type"] = "gemma4_assistant";
    bad_assistant["text_config"]["enable_moe_block"] = true;
    assert(!validate_model_config(bad_assistant));

    YAML::Node bad_whisper;
    bad_whisper["type"] = "whisper";
    bad_whisper["model_type"] = "whisper";
    assert(!validate_model_config(bad_whisper));
}
