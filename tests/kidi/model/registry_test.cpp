#include "kidi/model/registry.h"

#include <cassert>

int main() {
    using kidi::model::supports_task;
    using kidi::model::Task;

    assert(supports_task("rtg_transformer_nmt", Task::TRANSLATION));
    assert(supports_task("gemma4_text", Task::TEXT_GENERATION));
    assert(supports_task("whisper", Task::SPEECH_RECOGNITION));
    assert(supports_task("omnivoice", Task::TTS));
    assert(supports_task("kokoro", Task::TTS));
    assert(!supports_task("omnivoice", Task::TEXT_GENERATION));
    assert(!supports_task("unknown", Task::TTS));
}
