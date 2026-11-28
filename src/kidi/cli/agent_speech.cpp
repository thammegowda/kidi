#include "kidi/cli/agent_speech.h"

#include "kidi/checkpoint/hub.h"
#include "kidi/cli/main.h"
#include "kidi/inference/generator.h"
#include "kidi/inference/synthesizer.h"
#include "kidi/inference/transcriber.h"
#include "kidi/ops/context.h"
#include "kidi/tensor/backend.h"

#include <memory>
#include <mutex>
#include <optional>

namespace kidi::cli {
namespace {

auto download(std::string_view model) -> Result<std::filesystem::path> {
    try {
        return checkpoint::Hub().download(model);
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::IO, error.what()});
    }
}

// The device the agent's --backend selects, as `kidi transcribe` and `kidi synthesize` choose it.
auto speech_device(const std::string& backend, bool allow_metal) -> Result<tensor::Device> {
    if (const auto device = tensor::parse_cuda_device(backend)) return *device;
    if (backend == "ynnpack") return tensor::Device::cpu();
    if (backend == "mps") return tensor::Device::apple_gpu();
    try {
        return auto_inference_device(tensor::BackendRegistry::instance().backends(),
                                     tensor::TransformerSupport::QUANTIZED, allow_metal);
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::UNSUPPORTED, error.what()});
    }
}

struct SpeechModels {
    std::string backend;
    agent::SpeechConfig config;
    std::mutex asr_mutex;
    std::optional<inference::Transcriber> transcriber;
    std::mutex tts_mutex;
    std::optional<inference::Synthesizer> synthesizer;
    std::shared_ptr<const inference::VoicePrompt> voice;

    auto transcribe(std::span<const float> samples, const agent::Speech::Cancelled& cancelled)
        -> Result<std::string> {
        const std::lock_guard lock(asr_mutex);
        if (!transcriber) {
            auto directory = download(config.transcribe_model);
            if (!directory) return std::unexpected(directory.error());
            auto device = speech_device(backend, false);
            if (!device) return std::unexpected(device.error());
            auto loaded = inference::Transcriber::load(*directory, *device);
            if (!loaded) return std::unexpected(loaded.error());
            transcriber.emplace(std::move(*loaded));
        }
        auto result = transcriber->transcribe(samples, 16000, {.fit_audio = true, .cancelled = cancelled});
        if (!result) return std::unexpected(result.error());
        return std::move(result->text);
    }

    auto speak(std::string_view text, const agent::Speech::Play& play, const agent::Speech::Cancelled& cancelled)
        -> Result<void> {
        const std::lock_guard lock(tts_mutex);
        if (!synthesizer) {
            auto directory = download(config.speak_model);
            if (!directory) return std::unexpected(directory.error());
            auto device = speech_device(backend, true);
            if (!device) return std::unexpected(device.error());
            auto loaded = inference::Synthesizer::load(*directory, *device);
            if (!loaded) return std::unexpected(loaded.error());
            synthesizer.emplace(std::move(*loaded));
        }
        inference::SynthesisOptions options;
        options.steps = config.steps;
        options.duration_scale = config.pace;
        options.voice = config.voice;
        options.retain_audio = false;
        options.on_progress = [&](std::size_t, std::size_t) {
            if (cancelled()) throw ops::Failure({ErrorCode::RUNTIME, "cancelled"});
        };
        // One synthesized reference keeps the same voice across sentences and replies.
        if (synthesizer->supports_voice_prompt() && !voice) {
            auto reference_options = options;
            reference_options.duration_scale = 1.F;
            auto reference = synthesizer->synthesize(config.reference_text, reference_options);
            if (!reference) return std::unexpected(reference.error());
            voice = std::move(reference->voice_prompt);
        }
        options.voice_prompt = voice;
        bool stopped = false;
        options.on_audio_chunk = [&](std::span<const float> samples, std::uint32_t sample_rate) {
            if (stopped || cancelled() || !play(samples, sample_rate)) {
                stopped = true;
                throw ops::Failure({ErrorCode::RUNTIME, "cancelled"});
            }
        };
        auto result = synthesizer->synthesize(text, options);
        if (!result && !stopped && !cancelled()) return std::unexpected(result.error());
        return {};
    }
};

} // namespace

auto agent_speech(std::string backend, const agent::SpeechConfig& config) -> agent::Speech {
    auto models = std::make_shared<SpeechModels>();
    models->backend = std::move(backend);
    models->config = config;
    agent::Speech speech;
    if (!config.transcribe_model.empty())
        speech.transcribe = [models](std::span<const float> samples, const agent::Speech::Cancelled& cancelled) {
            return models->transcribe(samples, cancelled);
        };
    if (!config.speak_model.empty())
        speech.speak = [models](std::string_view text, const agent::Speech::Play& play,
                                const agent::Speech::Cancelled& cancelled) {
            return models->speak(text, play, cancelled);
        };
    return speech;
}

} // namespace kidi::cli
