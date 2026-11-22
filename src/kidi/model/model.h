#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "kidi/checkpoint/weights.h"
#include "kidi/core/module.h"
#include "kidi/core/precision.h"

namespace kidi::audio {
struct WhisperFeatures;
}
namespace kidi::image {
struct Gemma4Image;
}
namespace kidi::inference {
struct InferenceStats;
class SynthesisSession;
struct SynthesisOptions;
} // namespace kidi::inference
namespace kidi::ops {
class Context;
}

namespace kidi::model {

enum class ModelKind {
    CAUSAL_LM,
    DRAFT_LM,
    SEQ2SEQ_LM,
    SPEECH_RECOGNITION_LM,
    VISION_ENCODER,
    SPEECH_SYNTHESIS,
};

struct RuntimeQuantization {
    std::int32_t weight_bits = 0; // 0 = checkpoint precision; 4 or 8 = load-time packing
    std::int32_t group_size = 128;
    bool packed_prefill = false;
};

/// Base class for all top-level inference models.
/// Owns checkpoint mapping (`state_mapping_specs`), loading (`set_checkpoint`),
/// compilation telemetry, and workspace cleanup without template-method callback soup.
class Model : public core::Module {
public:
    ~Model() override = default;

    virtual auto kind() const noexcept -> ModelKind = 0;

    /// Declarative checkpoint rename, projection fusion, scalar equality, and ignore rules.
    virtual auto state_mapping_specs() const -> std::vector<checkpoint::StateMappingSpec> { return {}; }

    auto map_checkpoint(const checkpoint::Weights& weights) const -> Result<checkpoint::Weights>;

    /// Loads an external checkpoint into this model.
    virtual auto set_checkpoint(const checkpoint::Weights& weights) -> Result<void>;

    virtual auto preparation_ns() const noexcept -> std::uint64_t { return 0; }
    virtual auto release_workspaces() -> void {}

protected:
    /// Concrete helper: applies `state_mapping_specs()`, matches against `parameter_specs()`
    /// (routing unmatched keys into `extras` when non-null), and normalizes I8->U8 and F16/BF16->F32.
    auto prepare_checkpoint(const checkpoint::Weights& weights, StateDict* extras = nullptr) const -> Result<StateDict>;
};

/// Modality-agnostic soft-token span (from a VisionModel or AudioEncoder) spliced at `position` during prefill.
struct EmbeddingSpan {
    std::size_t position = 0;
    tensor::Tensor embeddings; // [1, span_tokens, hidden_size]
};

/// Request-scoped autoregressive state allocated once per sequence.
struct LanguageModelState {
    virtual ~LanguageModelState() = default;
    virtual auto sequence_position() const noexcept -> std::size_t = 0;
    virtual auto sequence_capacity() const noexcept -> std::size_t = 0;
    virtual auto cache_bytes() const noexcept -> std::size_t { return 0; }
    virtual auto attach_embeddings(std::vector<EmbeddingSpan>) -> Result<void> {
        return std::unexpected(Error{ErrorCode::UNSUPPORTED, "model does not accept multimodal embedding spans"});
    }
};

/// Common base for all models that predict target-text tokens P(y_t | y_<t, state).
class LanguageModel : public Model {
public:
    virtual auto hidden_size() const noexcept -> std::int32_t = 0;
    virtual auto vocabulary_size() const noexcept -> std::int32_t = 0;
    virtual auto max_positions() const noexcept -> std::size_t = 0;

    virtual auto configure_lm_head(const std::filesystem::path&) -> Result<void> {
        return std::unexpected(Error{ErrorCode::UNSUPPORTED, "model does not support external lm_head index"});
    }
    virtual auto set_precision(core::InferencePrecision) noexcept -> void {}
    virtual auto set_kv_cache_precision(core::KVCachePrecision) noexcept -> void {}
    /// Storage dtype `create_state(capacity)` would allocate for the selected KV-cache precision.
    virtual auto kv_cache_dtype(std::size_t) const -> Result<tensor::DType> { return tensor::DType::F32; }

    virtual auto rewind(LanguageModelState& state, std::size_t position) -> Result<void>;
    virtual auto prefill(std::span<const std::int32_t> tokens, LanguageModelState& state) -> Result<void>;
    virtual auto forward(std::span<const std::int32_t> tokens, LanguageModelState& state, bool all_logits = false,
                         tensor::Tensor* hidden = nullptr) -> Result<tensor::Tensor> = 0;

    /// Returns I32 [1] or [tokens.size()] greedy token IDs (default: argmax over `forward`).
    virtual auto select_token(std::span<const std::int32_t> tokens, LanguageModelState& state, bool all_tokens = false,
                              tensor::Tensor* hidden = nullptr) -> Result<tensor::Tensor>;

    auto forward_token(std::span<const std::int32_t> tokens, LanguageModelState& state,
                       tensor::Tensor* hidden = nullptr) -> Result<std::int32_t>;
};

/// Decoder-only (and multimodal soft-token) causal language models -> driven by `inference::Generator`.
class CausalLanguageModel : public LanguageModel {
public:
    auto kind() const noexcept -> ModelKind final { return ModelKind::CAUSAL_LM; }

    using LanguageModel::set_checkpoint;
    virtual auto set_checkpoint(const checkpoint::Weights& weights, RuntimeQuantization quantization) -> Result<void> {
        return quantization.weight_bits == 0
                   ? set_checkpoint(weights)
                   : std::unexpected(Error{ErrorCode::UNSUPPORTED, "model does not support load-time weight packing"});
    }

    virtual auto configure_cache(const std::filesystem::path&, const std::filesystem::path&) -> void {}
    virtual auto trim_workspaces() -> void { release_workspaces(); }
    virtual auto accelerator() const -> std::string_view { return {}; }
    virtual auto prefill_chunk_size(std::size_t requested) const -> std::size_t { return requested; }
    virtual auto last_token_prefill() const -> bool { return false; }
    virtual auto shared_prefill_tail() const -> bool { return false; }
    virtual auto supports_device_tokens() const -> bool { return false; }

    virtual auto create_state(std::size_t capacity) -> Result<std::unique_ptr<LanguageModelState>> = 0;
    virtual auto configure_state(LanguageModelState&, bool /*full_attention_cache*/, bool /*capture_prefill*/) -> void {
    }
    virtual auto fork_state(const LanguageModelState& source, std::size_t prefix_length, std::size_t capacity)
        -> Result<std::unique_ptr<LanguageModelState>>;

    virtual auto select_batch_tokens(std::span<const std::int32_t> tokens, std::span<LanguageModelState* const> states)
        -> Result<tensor::Tensor>;
    virtual auto forward_device_token(const tensor::Tensor& token, LanguageModelState& state) -> Result<tensor::Tensor>;
    auto forward_batch_tokens(std::span<const std::int32_t> tokens, std::span<LanguageModelState* const> states)
        -> Result<std::vector<std::int32_t>>;
};

/// Speculative MTP / Eagle assistant paired with a `LanguageModel`.
class DraftLanguageModel : public Model {
public:
    auto kind() const noexcept -> ModelKind final { return ModelKind::DRAFT_LM; }

    virtual auto draft(LanguageModel& target, const tensor::Tensor& hidden, std::int32_t last_token,
                       LanguageModelState& state, std::size_t max_drafts,
                       const std::function<bool(std::int32_t)>& is_stop, std::vector<std::int32_t>& drafts)
        -> Result<void> = 0;
};

struct EncoderMemory {
    virtual ~EncoderMemory() = default;
};

struct EncoderDecoderState : public LanguageModelState {
    std::shared_ptr<const EncoderMemory> memory;
    std::size_t batch = 1;
};

/// Encoder-decoder autoregressive language models (Translation and Speech Recognition).
class EncoderDecoderModel : public LanguageModel {
public:
    virtual auto create_decoder_state(std::shared_ptr<const EncoderMemory> memory, std::size_t batch,
                                      std::size_t capacity) -> Result<std::unique_ptr<LanguageModelState>> = 0;

    virtual auto decode_step(const EncoderMemory& memory, std::span<const std::int32_t> tokens, std::size_t batch,
                             LanguageModelState* state = nullptr, inference::InferenceStats* stats = nullptr)
        -> Result<tensor::Tensor> = 0;

    auto forward(std::span<const std::int32_t> tokens, LanguageModelState& state, bool all_logits = false,
                 tensor::Tensor* hidden = nullptr) -> Result<tensor::Tensor> override;
};

/// Text-to-text encoder-decoder (`TRANSLATION`) -> driven by `inference::Translator`.
class Seq2SeqModel : public EncoderDecoderModel {
public:
    auto kind() const noexcept -> ModelKind final { return ModelKind::SEQ2SEQ_LM; }

    virtual auto source_vocabulary_size() const noexcept -> std::int32_t = 0;
    virtual auto encode_text(std::span<const std::vector<std::int32_t>> sources,
                             inference::InferenceStats* stats = nullptr) -> Result<std::shared_ptr<EncoderMemory>> = 0;
};

/// Audio-to-text encoder-decoder (`SPEECH_RECOGNITION`) -> driven by `inference::Transcriber`.
class SpeechRecognitionModel : public EncoderDecoderModel {
public:
    auto kind() const noexcept -> ModelKind final { return ModelKind::SPEECH_RECOGNITION_LM; }

    virtual auto mel_bins() const noexcept -> std::int32_t = 0;
    virtual auto encode_audio(const audio::WhisperFeatures& features, std::size_t frames = 0,
                              const std::function<bool()>& cancelled = {})
        -> Result<std::shared_ptr<EncoderMemory>> = 0;
    virtual auto select_token(const tensor::Tensor& logits, const tensor::Tensor& mask) -> tensor::Tensor;
    virtual auto select_language(const tensor::Tensor& logits, const tensor::Tensor& allowed_ids) -> tensor::Tensor;

    using LanguageModel::select_token;

protected:
    auto select_masked_token(ops::Context& context, const tensor::Tensor& logits, const tensor::Tensor& mask) const
        -> tensor::Tensor;
    auto select_allowed_token(ops::Context& context, const tensor::Tensor& logits,
                              const tensor::Tensor& allowed_ids) const -> tensor::Tensor;
};

/// Encodes non-text media into `[1, N, output_dim]` soft-token embeddings for `CausalLanguageModel`.
class ModalityEncoder : public Model {
public:
    virtual auto output_dim() const noexcept -> std::int32_t = 0;
    virtual auto set_precision(core::InferencePrecision) noexcept -> void {}
};

class VisionModel : public ModalityEncoder {
public:
    auto kind() const noexcept -> ModelKind final { return ModelKind::VISION_ENCODER; }
    virtual auto encode_image(const image::Gemma4Image& image) -> Result<tensor::Tensor> = 0;
};

/// Text-to-speech (`TTS`) -> driven by `inference::Synthesizer`.
class SpeechSynthesisModel : public Model {
public:
    auto kind() const noexcept -> ModelKind final { return ModelKind::SPEECH_SYNTHESIS; }

    virtual auto sample_rate() const noexcept -> std::uint32_t = 0;
    virtual auto supports_voice_prompt() const noexcept -> bool { return false; }
    virtual auto voice_reference_chunk(std::span<const std::string>, const inference::SynthesisOptions&) const
        -> Result<std::optional<std::size_t>> {
        return std::optional<std::size_t>{};
    }
    virtual auto start_synthesis(std::string_view text, const inference::SynthesisOptions& options)
        -> Result<std::unique_ptr<inference::SynthesisSession>> = 0;
    virtual auto execution() const -> std::string = 0;
};

} // namespace kidi::model
