# Model & Checkpoint Loading Specification

This document describes Kidi's model hierarchy, task interfaces, checkpoint mapping engine, and manifest contract as implemented across [core/](../src/kidi/core), [checkpoint/](../src/kidi/checkpoint), [model/](../src/kidi/model), and [inference/](../src/kidi/inference).

---

## 1. Layering & Ownership

```
┌──────────────────────────────────────────────────────────────────────────┐
│  kidi::inference   Task runners: tokenization, audio/image extraction,   │
│                    KV-cache scheduling, greedy/beam/MTP/TTS loops        │
│                    (Generator, Translator, Transcriber, Synthesizer)     │
├──────────────────────────────────────────────────────────────────────────┤
│  kidi::model       Model base, task interfaces, config validation,       │
│                    checkpoint state_mapping_specs, typed forward steps   │
├──────────────────────────────────────────────────────────────────────────┤
│  kidi::layers      Reusable Module blocks (Linear, RmsNorm,              │
│                    TokenEmbedding, GatedFeedForward, fused attention)    │
├──────────────────────────────────────────────────────────────────────────┤
│  kidi::checkpoint  Format-agnostic config.json, Hub dir, Safetensors &   │
│                    GGML/GGUF readers, zero-copy StateMappingSpec engine  │
├──────────────────────────────────────────────────────────────────────────┤
│  kidi::ops / graph / runtime / tensor                                    │
│                    Eager ops, replay capture, CPU/CUDA/Metal/Vulkan/WASM │
└──────────────────────────────────────────────────────────────────────────┘
```

### Core Design Rules

1. **Additive task hierarchy + multimodal composition**:
   - Models that predict target-text tokens $P(y_t \mid y_{<t}, \text{state})$ form the `LanguageModel` subtree (`CausalLanguageModel` and `EncoderDecoderModel` $\to$ `{Seq2SeqModel, SpeechRecognitionModel}`).
   - Auxiliary and non-token models (`DraftLanguageModel`, `ModalityEncoder` $\to$ `VisionModel`, and `SpeechSynthesisModel`) derive directly from `Model`.
   - Multimodal inputs feed `CausalLanguageModel` via generic `EmbeddingSpan` soft-token attachments on `LanguageModelState` rather than creating $2^M$ modality subclasses.
2. **Fused projections in `layers/`**:
   - Self-attention uses fused `qkv_proj` (or `q_proj` + `kv_proj` when QAT calibration scales, bit-widths, or KV-sharing differ between Q and KV); cross-attention uses `q_proj` + `kv_proj`; gated MLPs use `gate_up_proj`.
3. **Model-owned rules, general loader**:
   - [checkpoint/](../src/kidi/checkpoint) has zero dependencies on [model/](../src/kidi/model). Each model declares its own `state_mapping_specs()`; `Weights::mapped(...)` handles renaming, concatenation, scalar equality checks, and unused-key filtering uniformly for unfused, pre-fused, and pre-bound GPU checkpoints.
4. **Straight-line checkpoint preparation**:
   - `Module::set_state(...)` binds exact, already-mapped `StateDict` tensors to registered parameter slots.
   - `Model::set_checkpoint(...)` calls the non-virtual helper `prepare_checkpoint(weights)` and runs any model-specific quantization or validation in straight top-to-bottom code.

---

## 2. Unified `config.json` & Directory Contract

Every model directory (local or `<hub_cache>/<owner>/<repo>/`) stores `config.json`, weights, and companion assets as regular files:

| Task (`model::Task`) | `model.type` | Implementation | Companion Files in Directory | Fused Projections |
|---|---|---|---|---|
| `TEXT_GENERATION` | `gemma4_text` (or HF `gemma4`) | `Gemma4Impl` (+ optional `Gemma4VisionImpl`, `Gemma4AssistantImpl`) | `config.json`, `model.safetensors`, `tokenizer.json`, optional `lm_head.hnsw` | `kv_proj`, `gate_up_proj` (`q_proj` separate for QAT scales & shared KV) |
| *(Backbone LLM)* | `qwen3` | `Qwen3Impl` | Embedded under `llm_config` or standalone `config.json` + `model.safetensors` | `qkv_proj`, `gate_up_proj` |
| `TRANSLATION` | `rtg_transformer_nmt` | `TransformerImpl` | `config.json`, `model.safetensors`, `tokenizer.src.json`, `tokenizer.tgt.json` | `qkv` (self-attn), `source` (`kv` cross-attn) |
| `SPEECH_RECOGNITION` | `whisper` | `WhisperImpl` | `config.json`, `model.safetensors` (or GGML `.bin`), `tokenizer.json`, `preprocessor_config.json`, `generation_config.json` | `qkv_proj` (self-attn), `kv_proj` (cross-attn) |
| `TTS` | `omnivoice` | `OmniVoiceImpl` (composes `Qwen3Impl` + `HiggsDecoderImpl`) | `config.json`, `model.safetensors`, `tokenizer.json`, `synthesis_template.jinja` | `llm.*.qkv_proj`, `llm.*.gate_up_proj` |
| `TTS` | `kokoro` | `KokoroImpl` | `config.json`, `model.safetensors`, `lexicon-us.tsv` | Conv / LSTM / AdaIN blocks |

`checkpoint::load_config(directory)` reads `config.json`, normalizes Hugging Face `model_type` / `architectures` into `config["model"]["type"]`, and resolves companion file paths (`weights_file`, `tokenizer_file`, etc.).

---

## 3. Checkpoint State Mapping (`checkpoint::Weights`)

Declared in [weights.h](../src/kidi/checkpoint/weights.h):

```cpp
namespace kidi::checkpoint {

enum class StateMappingOp {
    CONCAT,    // 1 source: zero-copy rename; N sources: concatenate along concat_axis
    IDENTICAL, // Verify all sources have identical dtype, shape, and bytes; keep as destination
    IGNORE,    // Drop matching keys (e.g., unused modalities or shared-KV layer tensors)
};

struct StateMappingSpec {
    std::vector<std::string> sources; // Regex anchor + "$1", "$2" templates (or "0:<ref>" zero tensor)
    std::string destination;
    std::int64_t concat_axis = 0;
    StateMappingOp op = StateMappingOp::CONCAT;
};

auto prefix_state_mappings(std::string_view source_prefix, std::string_view destination_prefix,
                           std::span<const StateMappingSpec> mappings) -> std::vector<StateMappingSpec>;

class Weights {
public:
    static auto load(const std::filesystem::path& path,
                     std::span<const StateMappingSpec> mappings = {}) -> Result<Weights>;
    static auto save(const std::filesystem::path& path, const StateDict& state) -> Result<void>;

    auto mapped(std::span<const StateMappingSpec> mappings) const -> Result<Weights>;
    auto add(std::string name, tensor::Tensor value) -> Result<void>;

    auto contains(std::string_view name) const -> bool;
    auto size() const noexcept -> std::size_t;
    auto names() const -> std::vector<std::string>;
    auto tensor(std::string_view name) const -> Result<tensor::Tensor>;
    auto float32_tensor(std::string_view name) const -> Result<tensor::Tensor>;
    auto state_dict() const -> Result<StateDict>;
};

} // namespace kidi::checkpoint
```

Invariants of `Weights::mapped(specs)`:
1. **Idempotent on pre-fused checkpoints**: If `destination` already exists (and is not itself in `sources`), the rule is skipped. The same `state_mapping_specs()` handles raw Hugging Face unfused checkpoints, Kidi pre-fused checkpoints, and WebGPU pre-fused device buffers.
2. **Zero-copy contiguous concatenation**: When `op == CONCAT`, `concat_axis == 0`, and source byte spans are adjacent in memory, `Weights::mapped` creates a single zero-copy view. Non-contiguous sources allocate once and release their source mmap pages.
3. **Partial-bias synthesis**: Prefixing a source template with `"0:"` synthesizes a zero tensor matching the referenced tensor's shape and dtype (used for Whisper `k_proj`, which has no bias alongside `q_proj` and `v_proj` biases).
4. **Composable**: Composite architectures reuse child rules with `prefix_state_mappings("llm.", "llm.", child->state_mapping_specs())`.

---

## 4. Class Hierarchy

```text
core::Module                                  Parameter registration, submodules, tied aliases, set_state
  └── model::Model                            state_mapping_specs, set_checkpoint, prepare_checkpoint,
        │                                     preparation_ns, release_workspaces
        │
        ├── model::LanguageModel              Target-text token predictors P(y_t | y_<t, state)
        │     ├── model::CausalLanguageModel  Decoder-only + multimodal EmbeddingSpans -> Generator
        │     │     └── Gemma4Impl
        │     └── model::EncoderDecoderModel  Cross-attended EncoderMemory + decoder cache
        │           ├── model::Seq2SeqModel           [Text  -> Text]  TransformerImpl -> Translator
        │           └── model::SpeechRecognitionModel [Audio -> Text]  WhisperImpl     -> Transcriber
        │
        ├── model::DraftLanguageModel         Hidden-state speculative MTP drafter (Gemma4AssistantImpl)
        │
        ├── model::ModalityEncoder            Raw media -> [1, N, output_dim] soft tokens
        │     └── model::VisionModel          Gemma4VisionImpl -> Generator
        │
        └── model::SpeechSynthesisModel       Text + voice -> PCM audio sessions -> Synthesizer
              └── OmniVoiceImpl (composes Qwen3Impl + HiggsDecoder), KokoroImpl
```

Internal submodules composed inside a parent model (such as `Qwen3Impl` used as `OmniVoiceImpl`'s backbone without an `lm_head`) contribute their `state_mapping_specs()` to the enclosing `Model` via `prefix_state_mappings`.

---

## 5. Core & Model Contracts

### 5.1 `core::Module`

Declared in [module.h](../src/kidi/core/module.h). Layers and models register parameters and child modules during construction under a `ModuleScope`, and bind exact `StateDict` tensors afterward:

```cpp
namespace kidi::core {

struct ParameterSpec {
    std::vector<std::int64_t> shape;
    tensor::DType dtype;
    tensor::Device device;
    std::string alias;
};
using ParameterSpecMap = std::map<std::string, ParameterSpec, std::less<>>;

class Module {
public:
    virtual ~Module() = default;

    auto device() const noexcept -> tensor::Device;
    auto state_dict() const -> StateDict;
    auto parameter_specs() const -> ParameterSpecMap;

    /// Binds already-named, already-typed tensors to registered slots (transfers to each slot's Device).
    /// Set `copy = true` to copy into fresh storage instead of sharing same-device host buffers.
    auto set_state(const StateDict& state, bool strict = true, bool copy = false) -> Result<void>;

protected:
    auto register_parameter(std::string name, tensor::Tensor& parameter, std::vector<std::int64_t> shape,
                            tensor::DType dtype = module_dtype, bool allocate = allocate_parameters,
                            std::optional<tensor::Device> storage_device = {}) -> void;
    auto register_module(std::string name, std::shared_ptr<Module> module) -> void;
    auto tie_parameter(std::string alias, std::string target) -> void;
};

} // namespace kidi::core
```

### 5.2 `model::Model`

Declared in [model.h](../src/kidi/model/model.h):

```cpp
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

class Model : public core::Module {
public:
    ~Model() override = default;

    virtual auto kind() const noexcept -> ModelKind = 0;
    virtual auto state_mapping_specs() const -> std::vector<checkpoint::StateMappingSpec> { return {}; }

    auto map_checkpoint(const checkpoint::Weights& weights) const -> Result<checkpoint::Weights>;
    virtual auto set_checkpoint(const checkpoint::Weights& weights) -> Result<void>;

    virtual auto preparation_ns() const noexcept -> std::uint64_t { return 0; }
    virtual auto release_workspaces() -> void {}

protected:
    /// Applies `state_mapping_specs()`, matches against `parameter_specs()` (routing unmatched keys
    /// into `extras` if provided), and normalizes I8->U8 and F16/BF16->F32 for F32 slots.
    auto prepare_checkpoint(const checkpoint::Weights& weights, StateDict* extras = nullptr) const
        -> Result<StateDict>;
};

} // namespace kidi::model
```

How concrete models load checkpoints using `Model`:
- **`TransformerImpl`, `Gemma4VisionImpl`, `Qwen3Impl`**: Override `state_mapping_specs() const` and inherit `Model::set_checkpoint`.
- **`Gemma4AssistantImpl`, `OmniVoiceImpl`**: Call `prepare_checkpoint(weights)`, perform one model-specific check (`token_ordering` permutation or INT8 dtype invariant), and call `set_state(*state)`.
- **`WhisperImpl`**: Calls `prepare_checkpoint(weights)`, packs non-`I8` projections when in `I8` mode, verifies positive `.scale` tensors, and calls `set_state(*state)`.
- **`Gemma4Impl`**: Calls `prepare_checkpoint(weights, &extras)`, extracts `.weight_zero`/`.embedding_zero` from `extras`, applies QAT unsigned-to-signed nibble XOR and tied `lm_head` synthesis, calls `set_state(*state)`, binds zero points, and prepares runtime-packed weights when `RuntimeQuantization::weight_bits != 0`.

---

## 6. Task Interfaces ([model.h](../src/kidi/model/model.h))

### 6.1 `LanguageModel` & `CausalLanguageModel` (`CAUSAL_LM`)

`LanguageModel` defines single-sequence autoregressive token generation (`hidden_size`, `vocabulary_size`, `max_positions`, `configure_lm_head`, `rewind`, `prefill`, `forward`, `select_token`, `forward_token`), with per-request `LanguageModelState` (`sequence_position`, `sequence_capacity`, `cache_bytes`, `attach_embeddings`).

`CausalLanguageModel` extends `LanguageModel` for decoder-only (and multimodal soft-token) models with `set_checkpoint(weights, RuntimeQuantization)`, `create_state`, `configure_state`, `fork_state`, `select_batch_tokens`, `forward_device_token`, `forward_batch_tokens`, `configure_cache`, `trim_workspaces`, `accelerator`, and prefill policy queries.

### 6.2 `EncoderDecoderModel` $\to$ `Seq2SeqModel` (`SEQ2SEQ_LM`) & `SpeechRecognitionModel` (`SPEECH_RECOGNITION_LM`)

`EncoderDecoderModel` extends `LanguageModel` with `create_decoder_state(memory, batch, capacity)` and `decode_step(memory, tokens, batch, state, stats)`. Its `LanguageModel::forward` override delegates single-sequence (`batch == 1`) calls to `decode_step` and rejects batched states (`batch != 1`, which use `decode_step`).

- **`Seq2SeqModel`**: Adds `source_vocabulary_size()` and `encode_text(sources, stats) -> Result<std::shared_ptr<EncoderMemory>>`.
- **`SpeechRecognitionModel`**: Adds `mel_bins()`, `encode_audio(features, frames, cancelled) -> Result<std::shared_ptr<EncoderMemory>>`, `select_token(logits, mask)`, and `select_language(logits, allowed_ids)`.

### 6.3 `DraftLanguageModel` (`DRAFT_LM`)

`DraftLanguageModel : public Model` provides `draft(target, hidden, last_token, state, max_drafts, is_stop, drafts)` for MTP / hidden-state speculative assistants (`Gemma4AssistantImpl`). Standalone small draft LLMs reuse `CausalLanguageModel`.

### 6.4 `ModalityEncoder` / `VisionModel` (`VISION_ENCODER`)

`ModalityEncoder : public Model` defines `output_dim()` and `set_precision(...)`; `VisionModel` adds `encode_image(const image::Gemma4Image&) -> Result<tensor::Tensor>`.

### 6.5 `SpeechSynthesisModel` (`SPEECH_SYNTHESIS`)

`SpeechSynthesisModel : public Model` defines `sample_rate()`, `supports_voice_prompt()`, `voice_reference_chunk(...)`, `start_synthesis(...)`, and `execution()` for `OmniVoiceImpl` and `KokoroImpl`.

---

## 7. Model Registry ([registry.h](../src/kidi/model/registry.h))

Every top-level model type in `config["model"]["type"]` is registered in `MODELS` with its `Task` list, `ModelKind`, `validate_config`, and factory:

```cpp
namespace kidi::model {

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

auto validate_model_config(const YAML::Node& config) -> Result<void>;
auto create_model(const YAML::Node& config, CreateOptions options = {}) -> Result<std::shared_ptr<Model>>;

template <typename TaskInterface>
auto create_model_as(const YAML::Node& config, CreateOptions options = {}) -> Result<std::shared_ptr<TaskInterface>>;

} // namespace kidi::model
```

Task runners validate configs, construct models through `create_model_as<TaskInterface>`, and load weights via `model->set_checkpoint(...)`:

```cpp
auto weights = require(checkpoint::Weights::load(config["weights_file"].as<std::string>()));
auto model = require(model::create_model_as<model::Seq2SeqModel>(
    config["model"], {.device = device, .parameter_dtype = dtype}));
require(model->set_checkpoint(weights));
```

---

## 8. Checklist for Adding a New Model

1. **Layers**: Build with fused projections (`qkv_proj` or `q_proj` + `kv_proj`, and `gate_up_proj`) in [layers/](../src/kidi/layers).
2. **Model class**: Inherit the matching task base (`CausalLanguageModel`, `Seq2SeqModel`, `SpeechRecognitionModel`, `VisionModel`, `DraftLanguageModel`, or `SpeechSynthesisModel`) in [model/](../src/kidi/model).
3. **Checkpoint mapping**: Override `state_mapping_specs() const` to map external checkpoint names and fuse split projections while passing pre-fused checkpoints through unchanged. Inherit `Model::set_checkpoint` unless custom quantization or parameter validation is needed.
4. **Registration**: Register `validate_config` and `create` in [registry.cpp](../src/kidi/model/registry.cpp) and add `config.json` handling in [checkpoint/config.cpp](../src/kidi/checkpoint/config.cpp).

