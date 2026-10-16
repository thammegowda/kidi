#include "kidi/inference/synthesizer.h"

#include <cctype>
#include <cmath>
#include <locale>
#include <sstream>

#include "kidi/checkpoint/config.h"
#include "kidi/model/kokoro.h"
#include "kidi/model/omnivoice.h"
#include "kidi/model/registry.h"
#include "kidi/ops/context.h"
#include "kidi/runtime/operator.h"
#include "kidi/runtime/ynn/graph.h"
#include "kidi/text/sentence.h"

namespace kidi::inference {
using ops::require;
namespace {

auto trim(std::string_view value) -> std::string_view {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) value.remove_prefix(1);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.remove_suffix(1);
    return value;
}

auto normalize_key(std::string_view value) -> Result<std::string> {
    value = trim(value);
    if (value.empty())
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "voice attribute key must not be empty"});
    std::string result;
    result.reserve(value.size());
    for (const auto character : value) {
        const auto byte = static_cast<unsigned char>(character);
        if (std::isalnum(byte))
            result.push_back(static_cast<char>(std::tolower(byte)));
        else if (character == '-' || character == '_')
            result.push_back('-');
        else
            return std::unexpected(
                Error{ErrorCode::INVALID_ARGUMENT, "voice attribute keys must use letters, numbers, '-' or '_'"});
    }
    return result;
}

} // namespace

auto parse_synthesis_number(std::string_view value) -> Result<float> {
    if (value.empty()) return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "expected a finite number"});
    std::istringstream input{std::string(value)};
    input.imbue(std::locale::classic());
    float result = 0;
    input >> std::noskipws >> result;
    if (!input || !input.eof() || !std::isfinite(result))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "expected a finite number"});
    return result;
}

auto parse_voice_attributes(std::span<const std::string> specifications) -> Result<VoiceAttributes> {
    VoiceAttributes result;
    for (const auto& specification : specifications) {
        const auto assignment = trim(specification);
        const auto separator = assignment.find('=');
        if (separator == std::string_view::npos || separator == 0 || separator + 1 == assignment.size())
            return std::unexpected(
                Error{ErrorCode::INVALID_ARGUMENT, "voice attributes must use nonempty key=value assignments"});
        auto key = normalize_key(assignment.substr(0, separator));
        if (!key) return std::unexpected(key.error());
        const auto value = trim(assignment.substr(separator + 1));
        if (value.empty())
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "voice attribute value must not be empty"});
        if (!result.emplace(*key, value).second)
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "duplicate voice attribute: " + *key});
    }
    return result;
}

auto synthesis_elapsed_ns(std::chrono::steady_clock::time_point start) -> std::uint64_t {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
}

struct Synthesizer::Impl {
    explicit Impl(text::SentenceSegmenter value) : sentences(std::move(value)) {}

    std::shared_ptr<SynthesizerModel> model;
    text::SentenceSegmenter sentences;
};

Synthesizer::Synthesizer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Synthesizer::Synthesizer(Synthesizer&&) noexcept = default;
auto Synthesizer::operator=(Synthesizer&&) noexcept -> Synthesizer& = default;
Synthesizer::~Synthesizer() = default;

auto Synthesizer::load(const std::filesystem::path& directory, tensor::Device device) -> Result<Synthesizer> {
    try {
        auto config = require(checkpoint::load_config(directory / "model.yaml"));
        const auto type = config["model"]["type"].as<std::string>();
        if (!model::supports_task(type, model::Task::TTS))
            throw ops::Failure({ErrorCode::INVALID_MANIFEST, type + " does not support the TTS task"});
        std::optional<ops::StepCompilerScope> accelerator;
        if (device.kind == tensor::DeviceKind::Q_NPU) {
            if (type != "omnivoice")
                throw ops::Failure({ErrorCode::UNSUPPORTED, type + " does not support Qualcomm NPU synthesis"});
            accelerator.emplace(require(runtime::npu_step_compiler()));
            device = tensor::Device::cpu();
        }
        require(runtime::ynn::reserve_thread_pool(runtime::ynn::thread_count()));
        auto sentences = require(text::SentenceSegmenter::load(config["sentence_prefixes_file"].as<std::string>()));
        auto impl = std::make_unique<Impl>(std::move(sentences));
        if (type == "omnivoice")
            impl->model = require(model::OmniVoiceImpl::load(std::move(config), device)).ptr();
        else if (type == "kokoro")
            impl->model = require(model::KokoroImpl::load(std::move(config), device)).ptr();
        else
            throw ops::Failure({ErrorCode::UNSUPPORTED, "no TTS loader for model type: " + type});
        return Synthesizer(std::move(impl));
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

auto Synthesizer::synthesize(std::string_view text, const SynthesisOptions& options) const -> Result<Synthesis> {
    if (options.voice_prompt && !impl_->model->supports_voice_prompt())
        return std::unexpected(Error{ErrorCode::UNSUPPORTED, "TTS model does not support generated voice prompts"});
    const auto synthesize_one = [&](std::string_view chunk) -> Result<Synthesis> {
        Result<Synthesis> synthesis = [&]() {
            if (!options.voice_prompt) return impl_->model->synthesize(chunk, options);
            const SynthesisReference reference{options.voice_prompt->text, options.voice_prompt->tokens};
            return impl_->model->synthesize_with_reference(chunk, options, reference);
        }();
        if (synthesis) {
            if (options.voice_prompt)
                synthesis->voice_prompt = options.voice_prompt;
            else if (impl_->model->supports_voice_prompt() && !synthesis->tokens.empty())
                synthesis->voice_prompt =
                    std::make_shared<VoicePrompt>(VoicePrompt{std::string(chunk), synthesis->tokens});
            if (options.on_audio_chunk) options.on_audio_chunk(synthesis->samples, synthesis->sample_rate);
            if (!options.retain_audio) synthesis->samples.clear();
        }
        return synthesis;
    };
    if (options.duration_seconds) {
        auto result = synthesize_one(text);
        impl_->model->release_workspaces();
        return result;
    }
    auto sentences = segment(text, options.language);
    if (!sentences) return std::unexpected(std::move(sentences.error()));
    if (sentences->size() == 1) {
        auto result = synthesize_one(sentences->front());
        impl_->model->release_workspaces();
        return result;
    }
    try {
        const auto selected_reference = options.voice_prompt
                                            ? std::optional<std::size_t>{}
                                            : require(impl_->model->voice_reference_chunk(*sentences, options));
        if (selected_reference && *selected_reference >= sentences->size())
            throw ops::Failure({ErrorCode::RUNTIME, "TTS model selected an invalid voice reference chunk"});

        Synthesis result{.sample_rate = 0};
        result.stats.chunks = sentences->size();
        std::size_t completed = 0;
        const auto generate = [&](std::size_t index, const SynthesisReference* reference) -> Result<Synthesis> {
            auto chunk_options = options;
            if (options.on_progress)
                chunk_options.on_progress = [callback = options.on_progress, completed, chunks = sentences->size()](
                                                std::size_t step, std::size_t steps) {
                    callback(completed * steps + step, chunks * steps);
                };
            chunk_options.on_chunk = {};
            auto chunk = reference
                             ? impl_->model->synthesize_with_reference((*sentences)[index], chunk_options, *reference)
                             : impl_->model->synthesize((*sentences)[index], chunk_options);
            if (!chunk) {
                impl_->model->release_workspaces();
                return std::unexpected(std::move(chunk.error()));
            }
            ++completed;
            if (options.on_chunk) options.on_chunk(completed, sentences->size());
            impl_->model->release_workspaces();
            return chunk;
        };

        std::optional<Synthesis> reference_chunk;
        auto voice_prompt = options.voice_prompt;
        if (selected_reference) {
            auto chunk = generate(*selected_reference, nullptr);
            if (!chunk) return std::unexpected(std::move(chunk.error()));
            voice_prompt = std::make_shared<VoicePrompt>(VoicePrompt{(*sentences)[*selected_reference], chunk->tokens});
            reference_chunk = std::move(*chunk);
        }

        for (std::size_t index = 0; index < sentences->size(); ++index) {
            Result<Synthesis> generated = [&]() -> Result<Synthesis> {
                if (selected_reference && index == *selected_reference) return std::move(*reference_chunk);
                if (!voice_prompt) return generate(index, nullptr);
                const SynthesisReference reference{voice_prompt->text, voice_prompt->tokens};
                return generate(index, &reference);
            }();
            if (!generated) return std::unexpected(std::move(generated.error()));
            auto chunk = std::move(*generated);
            if (result.sample_rate && result.sample_rate != chunk.sample_rate)
                throw ops::Failure({ErrorCode::RUNTIME, "TTS chunks produced different sample rates"});
            result.sample_rate = chunk.sample_rate;
            if (options.on_audio_chunk) options.on_audio_chunk(chunk.samples, chunk.sample_rate);
            if (options.retain_audio) result.samples.insert(result.samples.end(), chunk.samples.begin(), chunk.samples.end());
            result.tokens.insert(result.tokens.end(), chunk.tokens.begin(), chunk.tokens.end());
            result.stats.generation_ns += chunk.stats.generation_ns;
            result.stats.decode_ns += chunk.stats.decode_ns;
            result.stats.preparation_ns += chunk.stats.preparation_ns;
            result.stats.audio_tokens += chunk.stats.audio_tokens;
        }
        result.voice_prompt = std::move(voice_prompt);
        return result;
    } catch (const ops::Failure& error) {
        impl_->model->release_workspaces();
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        impl_->model->release_workspaces();
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

auto Synthesizer::segment(std::string_view text, std::string_view language) const -> Result<std::vector<std::string>> {
    return impl_->sentences.split(text, language);
}

auto Synthesizer::execution() const -> std::string { return impl_->model->execution(); }
auto Synthesizer::supports_voice_prompt() const noexcept -> bool { return impl_->model->supports_voice_prompt(); }

} // namespace kidi::inference
