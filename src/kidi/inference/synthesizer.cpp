#include "kidi/graph/compiler.h"
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
    std::shared_ptr<SynthesizerModel> model;
    text::SentenceSegmenter sentences;
    tensor::Device device;
};

namespace {
class SegmentedSynthesisSession final : public SynthesisSession {
public:
    SegmentedSynthesisSession(std::shared_ptr<SynthesizerModel> model, std::vector<std::string> chunks,
                              SynthesisOptions options, std::optional<std::size_t> selected_reference,
                              bool notify_single_chunk)
        : model_(std::move(model)),
          chunks_(std::move(chunks)),
          options_(std::move(options)),
          selected_reference_(selected_reference),
          completed_(chunks_.size()),
          voice_prompt_(options_.voice_prompt),
          notify_single_chunk_(notify_single_chunk) {
        result_.stats.chunks = chunks_.size();
        if (selected_reference_) order_.push_back(*selected_reference_);
        for (std::size_t index = 0; index < chunks_.size(); ++index)
            if (!selected_reference_ || index != *selected_reference_) order_.push_back(index);
    }

    auto step() -> Result<std::optional<Synthesis>> override {
        if (finished_)
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "synthesis session is already complete"});
        try {
            for (;;) {
                auto advanced = require(advance_chunk());
                if (!advanced) return std::optional<Synthesis>{};
                auto chunk = std::move(*advanced);
                active_.reset();
                model_->release_workspaces();

                const auto index = order_[completed_count_];
                if (!voice_prompt_ && model_->supports_voice_prompt() && !chunk.tokens.empty() &&
                    (chunks_.size() == 1 || selected_reference_ == index))
                    voice_prompt_ = std::make_shared<VoicePrompt>(VoicePrompt{chunks_[index], chunk.tokens});
                completed_[index] = std::move(chunk);
                ++completed_count_;
                if (options_.on_chunk && (notify_single_chunk_ || chunks_.size() > 1))
                    options_.on_chunk(completed_count_, chunks_.size());
                flush_ready();
                if (completed_count_ < order_.size()) continue;

                finished_ = true;
                result_.voice_prompt = std::move(voice_prompt_);
                return std::optional<Synthesis>{std::move(result_)};
            }
        } catch (const ops::Failure& error) {
            return fail(error.error());
        } catch (const std::exception& error) {
            return fail(Error{ErrorCode::RUNTIME, error.what()});
        }
    }

private:
    auto fail(Error error) -> Result<std::optional<Synthesis>> {
        finished_ = true;
        model_->release_workspaces();
        return std::unexpected(std::move(error));
    }

    auto advance_chunk() -> Result<std::optional<Synthesis>> {
        if (active_) return active_->step();
        const auto index = order_.at(completed_count_);
        auto chunk_options = options_;
        chunk_options.retain_audio = true;
        chunk_options.on_audio_chunk = {};
        chunk_options.on_chunk = {};
        if (options_.on_progress)
            chunk_options.on_progress = [callback = options_.on_progress, completed = completed_count_,
                                         chunks = chunks_.size()](std::size_t step, std::size_t steps) {
                callback(completed * steps + step, chunks * steps);
            };
        chunk_options.voice_prompt =
            voice_prompt_ && (!selected_reference_ || index != *selected_reference_) ? voice_prompt_ : nullptr;
        active_ = require(model_->start_synthesis(chunks_[index], chunk_options));
        return active_->step();
    }

    auto flush_ready() -> void {
        while (playback_position_ < completed_.size() && completed_[playback_position_]) {
            auto& chunk = *completed_[playback_position_];
            if (result_.sample_rate && result_.sample_rate != chunk.sample_rate)
                throw ops::Failure({ErrorCode::RUNTIME, "TTS chunks produced different sample rates"});
            result_.sample_rate = chunk.sample_rate;
            if (options_.on_audio_chunk) options_.on_audio_chunk(chunk.samples, chunk.sample_rate);
            if (options_.retain_audio)
                result_.samples.insert(result_.samples.end(), chunk.samples.begin(), chunk.samples.end());
            result_.tokens.insert(result_.tokens.end(), chunk.tokens.begin(), chunk.tokens.end());
            result_.stats.generation_ns += chunk.stats.generation_ns;
            result_.stats.decode_ns += chunk.stats.decode_ns;
            result_.stats.preparation_ns += chunk.stats.preparation_ns;
            result_.stats.audio_tokens += chunk.stats.audio_tokens;
            completed_[playback_position_++].reset();
        }
    }

    std::shared_ptr<SynthesizerModel> model_;
    std::vector<std::string> chunks_;
    SynthesisOptions options_;
    std::optional<std::size_t> selected_reference_;
    std::vector<std::size_t> order_;
    std::vector<std::optional<Synthesis>> completed_;
    std::shared_ptr<const VoicePrompt> voice_prompt_;
    std::unique_ptr<SynthesisSession> active_;
    Synthesis result_{.sample_rate = 0};
    std::size_t completed_count_ = 0, playback_position_ = 0;
    bool notify_single_chunk_, finished_ = false;
};
} // namespace

Synthesizer::Synthesizer(std::shared_ptr<SynthesizerModel> model, text::SentenceSegmenter sentences,
                         tensor::Device device)
    : impl_(std::make_unique<Impl>(std::move(model), std::move(sentences), device)) {
    if (!impl_->model) throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "a synthesizer requires a model"});
}
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
        std::shared_ptr<SynthesizerModel> model;
        if (type == "omnivoice")
            model = require(model::OmniVoiceImpl::load(std::move(config), device)).ptr();
        else if (type == "kokoro")
            model = require(model::KokoroImpl::load(std::move(config), device)).ptr();
        else
            throw ops::Failure({ErrorCode::UNSUPPORTED, "no TTS loader for model type: " + type});
        return Synthesizer(std::move(model), std::move(sentences), device);
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

auto Synthesizer::synthesize(std::string_view text, const SynthesisOptions& options) const -> Result<Synthesis> {
    if (impl_->device.capabilities().memory.deferred())
        return std::unexpected(Error{ErrorCode::UNSUPPORTED, "deferred synthesis requires a stepped session"});
    auto session = create_session(text, options, false);
    if (!session) return std::unexpected(std::move(session.error()));
    for (;;) {
        auto result = (*session)->step();
        if (!result) return std::unexpected(std::move(result.error()));
        if (*result) return std::move(**result);
    }
}

auto Synthesizer::start(std::string_view text, const SynthesisOptions& options) const
    -> Result<std::unique_ptr<SynthesisSession>> {
    return create_session(text, options, true);
}

auto Synthesizer::create_session(std::string_view text, const SynthesisOptions& options, bool stepped) const
    -> Result<std::unique_ptr<SynthesisSession>> {
    try {
        if (options.voice_prompt && !impl_->model->supports_voice_prompt())
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, "TTS model does not support generated voice prompts"});
        std::vector<std::string> chunks;
        if (options.duration_seconds)
            chunks.emplace_back(text);
        else
            chunks = require(segment(text, options.language));
        if (chunks.empty()) return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "TTS text must not be empty"});
        std::optional<std::size_t> selected_reference;
        if (chunks.size() > 1 && !options.voice_prompt)
            selected_reference = require(impl_->model->voice_reference_chunk(chunks, options));
        if (selected_reference && *selected_reference >= chunks.size())
            throw ops::Failure({ErrorCode::RUNTIME, "TTS model selected an invalid voice reference chunk"});
        return std::make_unique<SegmentedSynthesisSession>(impl_->model, std::move(chunks), options, selected_reference,
                                                           stepped);
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
