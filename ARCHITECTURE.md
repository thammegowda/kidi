# Eager Architecture

Kidi has one model execution path: ordinary C++ functions operating on concrete
tensors. There is no symbolic value type, model compiler, graph partitioner, or
recorded control flow. A fixed-shape step may be captured from that same eager
code and replayed (see [Captured Steps](#captured-steps)).

```text
inference::{Decoder, Transcriber} -> model::{Transformer, Gemma4, Whisper} -> layers -> ops::Context -> backend
```

## Ownership

| Module | Responsibility |
|---|---|
| `tensor` | Storage, dtype, device, views, explicit transfers |
| `core::Module` | Shared module ownership, named parameter/child registration, state dictionaries |
| `ops::Context` | Immediate operator dispatch, validation, bounded prepared-operator cache, step replay |
| `graph` | Captured fixed-shape steps: recorded operators, structural validation, input rebinding, replay |
| `layers` | Bound parameters and reusable neural equations |
| `checkpoint` | Configuration/package I/O, format readers, serialization and generic cache preparation |
| `model` | Neural network topology, parameter binding, checkpoint policies, source and decoder state |
| `inference` | Generic generation/search, translation application, profiling |
| `runtime/ynn` | Private prepared CPU operators and YNNPACK ownership |
| `runtime/mps` | Private prepared Metal operators, command batches, INT8 kernels |
| `text`, `cli` | Tokenization and command/input/output policy |

RTG is an import/package format, not the owner of general modeling code. Do not
add independent CPU and GPU model implementations. Backend fusion belongs in
operators, not duplicated Transformer equations.

`kidi::model` contains Gemma 4, its vision tower, Transformer and Whisper model
implementations. Model-specific validation, parameter binding and checkpoint
customizations stay with the model. Models define metadata layout, required
sidecars, source compatibility, conversion/export rules and cache identity.
`kidi::checkpoint` invokes those hooks while owning filesystem access, source
inspection, locking, cache validation, staging, serialization and atomic writes.
Do not place format readers or filesystem/cache orchestration in model classes,
or add model-specific conversion branches to generic preparation code.
The format implementations have matching subdirectories and namespaces:
`checkpoint/ggml` (`kidi::checkpoint::ggml`) and `checkpoint/safetensors`
(`kidi::checkpoint::safetensors`). Shared `Weights`, `Package` and configuration
APIs live at the checkpoint root; applications normally use those APIs rather
than a particular format reader. There are no compatibility aliases in `model`.

## Configuration

Configuration stays in `YAML::Node`; there are no model-specific configuration
structs or mirrored manifest types. `checkpoint::load_config` reads `model.yaml`,
checks package structure and file paths, and resolves file paths relative to
the package. The `load_config(path, ConfigAdapter)` overload reads the adapter's
metadata filename and invokes its model-supplied configuration callback.
`ConfigSource` supplies the parsed YAML/JSON document and checked file/weight
resolution; it does not encode a model's required files or dimensions.
`WhisperImpl::checkpoint_config()` supplies the Hugging Face layout and GGML
header compatibility rules without writing a manifest. `Package::config()`
exposes RTG package documents.

`checkpoint::prepare` takes a config adapter and model-owned preparation hook.
The returned `Preparation` describes the source, cache directory/version,
sidecars, metadata labels and conversion callback. These are persistence
descriptors, not copies of model architecture fields. No registry or new model
base class is required to supply a different model's hooks:

```cpp
const auto adapter = model::WhisperImpl::checkpoint_config();
auto config = ops::require(checkpoint::load_config(directory, adapter));
auto prepared = ops::require(checkpoint::prepare(directory, adapter, model::WhisperImpl::int8_preparation));
```

`config["model"]` is self-contained: `type`, architecture fields,
and `source_tokens` live together. The package derives source padding and decoder
token IDs from the tokenizers, adding them to the in-memory `model` and `decode`
nodes. IDs are not duplicated in the configuration file, and source/target
tokenizers may assign different IDs. Model construction consumes only this node;
precision, allocation, and device are scoped construction settings. State is
assigned separately after registration:

```cpp
const auto embedding = ops::require(package.weights().tensor("target_embedding.weight"));
ModuleScope construction(embedding.dtype(), false);
auto model = ops::require(model::TransformerImpl::create(package.config()["model"]));
ops::require(model->set_state(package.weights()));
```

The model validates its own supported architecture and caches runtime dimensions
during construction. It does not read YAML during token execution. Translation
resolves defaults and CLI overrides from `config["decode"]`, without configuration
caps on beam size or extra tokens. It checks positive counts and finite,
non-negative penalties. Actual positional capacity and allocation failures remain
runtime constraints. Typed tensors and search requests remain runtime values,
not config types.
Safetensors is the native memory-mapped checkpoint format. `Weights` also indexes
little-endian GGUF v2/v3 and legacy Whisper GGML files, decoding supported tensors
to owned F32 buffers on demand. `names()` enumerates metadata without expanding
the checkpoint; Whisper uses it to bound import scratch space to one tensor.
The small adapted GGML reference codecs live beside the reader, with upstream
credits and the MIT license in their header. They do not introduce a runtime or a model
execution path. GGUF tensor names remain unchanged; architecture/tokenizer
mapping is still the model loader's responsibility, not a side effect of reading
a container. The inference loader derives the construction
dtype from the checkpoint. Layer constructors declare matching shapes/dtypes and
required INT8 scale parameters without reading weights. `set_state` validates them
before assignment. There is no separate encoding enum or YAML precision setting.
The execution policy follows stored weights; any future compute-precision override
is a separate runtime option, not a duplicate declaration of checkpoint metadata.
The old flat manifest layout is not supported or automatically migrated.

CLI diagnostics use spdlog on stderr. Translation results and inspection output
remain on stdout; machine-readable metric/profile records retain their unadorned
format on stderr. `SPDLOG_LEVEL` controls the human-readable logger.

The Python launcher optionally resolves `--model @owner/repo[@revision]` through
Hugging Face Hub. Native parsing and help run first; only a validated Hub reference
invokes the injected `ModelResolver`. `-c/--cache` defaults to
`~/.cache/kidi/model-hub` and is expanded by Python. Local model paths bypass the
resolver. The standalone executable has no Python/download dependency and reports
that Hub references require the Python launcher.

The `hf` extra supplies Hub, PyYAML and filelock. The resolver reads model metadata
first, then downloads required files at the same resolved commit using Hub's normal
snapshot/blob cache. Ready-made Kidi manifests retain their declared relative file
paths; unsupported or escaping paths are rejected. Dense Gemma 4 snapshots without
Kidi YAML use `kidi.converters.gemma4.configure`, under a per-snapshot lock, with an
atomic no-overwrite config install. A local generated config exactly matching the
old 256-output/2048-context defaults is atomically upgraded to 8192/16384 under
that lock; customized configs and downloaded manifests are left alone. Whisper conditional-generation snapshots
download their original config, Safetensors, tokenizer, preprocessor, and generation metadata without modification. Original
weights and tokenizer files remain unchanged. Incomplete
caches fail rather than being marked ready; `HF_HUB_OFFLINE=1` requires cached
files. No downloaded Python code or pickled checkpoints execute during resolution.
Explicit trusted RTG conversion is separate in `kidi.converters.rtg` with the
`convert` extra; its NLCodec helper is bundled in wheels.

Each model family has its own command rather than one command that branches on
`model.type`. `translate` serves RTG and reads Moses-tokenized text lines;
`generate` serves chat models and reads JSONL `messages` arrays with optional `id`
and `max_tokens`; `chat` is the interactive terminal session; `transcribe` serves Whisper WAV input. Text file commands
process lines in input order. A command run against the other family's package
fails with an explicit message, and options that belong to the other family are
rejected by the argument parser rather than by a runtime cross-check.

`Tokenizer::format_chat` uses the original checkpoint's
Jinja template and special-token configuration; `Generator::enqueue_chat` passes
the serialized result to the existing raw-prompt enqueue path. Text messages may
include system/developer instructions and user/assistant history. Multimodal content
and tools are rejected. No generic model execution abstraction is added.

The `chat` command uses stdin/stdout and the same
loaded `Generator` for a terminal chat session. The shell owns system/user/assistant
history and configures one serving slot. Successful replies append assistant text;
rejected or cancelled requests remove the pending user turn. Context overflow is
reported without silently truncating history. Terminal commands, ANSI color policy
and scoped SIGINT handling live in `cli/interactive.*`, outside inference code.
`generate` and `chat` share model, backend and generation options; system/color
options are chat-only, while file I/O and serving limits are generate-only.

`model::Whisper` registers the upstream encoder/decoder hierarchy directly, including separate biasless key projections,
learned decoder positions, fixed checkpoint encoder positions, and a tied output head. The 400-point Hann/STFT path uses
the checkpoint's embedded 80-bin mel bank. Conv1D is expressed as im2col plus the existing optimized linear operation;
attention, normalization, cache mutation, and projection remain ordinary eager operations. `inference::Transcriber` owns
language detection, task/no-timestamp prefixes, token suppression, and greedy ASR policy. Whisper processes one
padded/truncated 30-second segment per call. Encoder and decoder share one CPU context; Android uses the same YNNPACK
operators as native CPU inference. No Qualcomm SDK, calibration pass, or DSP graph cache is required.

Whisper INT8 binds per-output-channel signed weights and FP32 scales to the existing quantized linear operation, including
transposed checkpoint layouts, the im2col convolution, and a tied INT8 embedding/output table. The small input convolution
and numerically sensitive non-weight parameters remain FP32. CPU projection inputs are dynamically quantized per row.
`WhisperImpl::int8_preparation` defines its cache identity/sidecars and selects
`WhisperImpl::int8_checkpoint`, which constructs the INT8 model, binds source
weights and exports state without the duplicate tied output head. Generic
`checkpoint::prepare` writes the cache atomically, retains the source and checks
its size/mtime on reuse. Existing cache names, version strings and serialized
weights remain compatible. Tensor dtype in the encoder FFN determines stored precision; no duplicated manifest precision
flag is required. `Weights::save` writes exclusive Safetensors with aligned payload order and never overwrites a file.

Whisper can also load a legacy GGML file with matching HF sidecars in its parent
directory (or `ggml-model.bin` when a directory has no Safetensors). Dimensions
are checked against the GGML header. `Transcriber::load` prepares and reuses a
separate `<filename>.kidi-int8-v1` cache through the same native quantization
path. Unsupported legacy quantization versions fail explicitly. Source files
are retained, and cache metadata binds source size/mtime and importer version.
No GGML graph/backend or model equations are vendored; generic GGUF-to-Whisper
architecture mapping is not implemented.

`TranscriptionOptions::on_partial` optionally receives the accumulated UTF-8 text and detected language synchronously
on the inference caller's thread. It uses the tokenizer's existing incremental decoder and leaves final tokens/text
unchanged. Callbacks must be lightweight and must not reenter the same transcriber or JNI runtime. Android transports
partial snapshots as ASCII-escaped JSON, updates the composer only for the matching draft/refinement phase, and reports
callback failures through the existing error result. Streaming starts after encoding; it does not make the encoder causal.

## Image Input

`image::prepare_gemma4` owns bounded Tahoma Vision decoding, aspect-preserving antialiased bicubic resizing, and RGB
patch layout. `model::Gemma4Vision` binds the checkpoint vision tower/projector and composes existing eager operators
for axial rotary attention, RMS normalization, gated feed-forward layers, and spatial pooling. The default mobile
checkpoint uses trained INT8 vision projections; FP32 and QAT tiny fixtures are checked against Transformers.

Chat messages may supply local image paths. `Generator::enqueue_chat` lazily loads the vision module, expands the
checkpoint's image markers, and attaches positioned image vectors to the request. The text model uses PAD token lookup
for image positions' per-layer token embeddings, inserts projected image vectors before the per-layer model projection,
and uses the supported model's normal causal mask. Encoded images and features are bounded to the current request's
eight-image/32-MiB limit. Exact encoded content joins the serving prefix compatibility key; matching placeholder IDs
alone never permit image KV reuse. Image features and the vision module remain reusable while the generator is loaded.

Android owns camera/picker permissions and app-private image files; native model equations and preprocessing are not
implemented in Kotlin. Text-only APIs and unsupported image checkpoints retain explicit failure boundaries. The pinned
Tahoma submodule is unchanged; the consumer CMake hook supplies the external JPEG cross-toolchain and the NDK's
experimental stop-token feature required by pigzpp. Optional PDF/SVG dependencies are disabled.

## Eager Contract

`ops::Context` belongs to one device and one caller at a time. Inputs and outputs
are actual `tensor::Tensor` objects, not expressions awaiting evaluation.
Operations dispatch when called. Ordinary C++ branches, loops, local variables,
and debugger inspection can be used in model code.

CPU operators complete before returning. Metal operators encode into the current
command batch; call `context.synchronize()` before reading or modifying their
inputs/results on the host, moving them to another device, or passing them to
another execution context. CPU/GPU transfers are explicit. Model entry points
synchronize before returning source state or logits.

Shape/device and preparation failures throw `ops::Failure` at the operation.
Asynchronous device failures surface at synchronization. Public model and
translation methods convert these to `Result<T>`. Destruction drains outstanding
work but cannot report errors: explicit synchronization is the error boundary.
A context is not thread-safe; use separate model instances for concurrent calls.

Operators require contiguous tensors. `reshape` shares storage, and `slice`
returns a view when its contiguous layout and backend offset restrictions allow
it; otherwise it copies. Results retain ownership. Output pools reuse only
unaliased storage; queued Metal inputs/outputs stay alive until completion.

### Explicit Mutation

`func(...)` never writes to its inputs. `func_(Tensor&, ...)` mutates the first
tensor and returns that same `Tensor&` for chaining. Supported variants are
`add_`, `multiply_`, `gelu_`, `softmax_`, `layer_norm_`, and `scatter_`. They preserve
shape, dtype, and storage identity; broadcasting may not expand the destination.
Read-only storage is rejected. Shape-changing operations have no in-place variant.

```cpp
auto sum = context.add(input, residual);  // input is unchanged
context.add_(input, residual);           // input and its aliases see the sum
context.gelu_(input);
context.scatter_(cache.key, new_key, positions);
context.synchronize();
```

The thread-local `ops::is_inplace` flag describes the current dispatch. An RAII
scope sets it from the public operation and restores its previous value on normal
return or exception. Nested nonmutating calls select immutable execution; an
outer flag value never changes the meaning of a public function name. The flag
does not make a context thread-safe and should not be assigned by model code.

Views and shallow tensor copies alias storage, so they observe subsequent
explicit mutations. CPU and Metal scatter write only selected slots; overlapping
updates/indices are snapshotted first, and duplicate indices use the last update.
All indices are checked before any writes. CPU reports invalid indices at the
call; Metal reports them at synchronization without changing that scatter's
destination. Discard a context after an asynchronous device error.

Other CPU operators compute into a temporary before copy-back unless direct
aliasing is explicitly supported. Metal similarly computes into independent
storage and queues a blit back to the destination on the same command batch.
No host fence is introduced by an in-place call. This is an aliasing contract,
not a promise that every operation avoids scratch allocation or has a dedicated
in-place kernel. Prepared operator cache keys include the mutation mode, so a
mutable-only backend implementation cannot be reused for an immutable call.

Decoder K/V updates now use `scatter_`. A copied `DecoderState` shares the mutable
caches; create an independent state for another generation or branch. In-place
operations affect activations and caches, not registered inference parameters.

Gemma 4's sequential K/V updates use `copy_slice_(destination, source, axis, start)`.
This operation accepts contiguous, same-device tensors with matching dtype and
non-axis dimensions. Bounds and writability are checked before dispatch; overlaps
are snapshotted. CPU copies contiguous blocks; Metal queues offset blits and
retains their owners. The runtime offset does not create a prepared-cache entry.
General scatter behavior, including duplicate-index handling, is unchanged.

Linear and normalization parameters are fixed inference weights. Constant-weight
prepared operators key parameter identity as well as dtype and shapes and retain
owners. Same-device RMSNorm parameters and Metal linear parameters instead use
dynamic bindings, allowing one shape-specialized executable to serve many layers
without recompiling their weights. Queued bindings retain their tensor owners;
host-constant fallback and CPU packed projections still use identity-keyed entries.
Do not mutate their storage after first use; replace registered parameters with
`load_state_dict`, or create a new context/model. Activations, masks, and token indices are dynamic bindings and
may change between completed calls without recompilation. This is an inference
API, not an autograd or optimizer framework.

The prepared-operator cache defaults to 4,096 entries with LRU replacement. Reaching
the bound drains queued work, evicts the least-recent quarter, and releases pooled
buffers; each entry owns its constant-parameter references. This is an
entry limit, not a native-memory byte budget. `preparation_ns()` reports cumulative
operator preparation, including eviction synchronization when required.

### Allocation and Ownership

Dispatch borrows `TensorInputs` instead of constructing owning tensor arrays.
Attributes are stack-backed spans and the context reuses its dispatch-key buffer.
Tensor shape/stride metadata is inline through rank eight, with a heap fallback
for larger ranks. These changes avoid repeated heap allocations and shared-pointer
increments/decrements for transient arguments without weakening output ownership.

CPU output pools acquire slots from a backend-owned `tensor::Arena`. CPU reserves an
8 MiB slab at context construction and suballocates 256-byte-aligned regions;
growth adds slabs. Each region has an independent lifetime lease so pool reuse
does not mistake another live region for an alias. Escaped results and views keep
their slab alive after arena/context destruction. Slots are reusable only when
there are no caller or asynchronous-work owners. There is no blanket per-token
reset that could invalidate user tensors.

Metal uses a stream-owned output pool with separately backed, tracked buffers.
An unaliased output may be recycled after its last consumer has been encoded:
later writes are ordered after those reads on the same stream. The pool retains
storage until the prepared cache is evicted or the context is destroyed. Caller
aliases, including views, prevent reuse. Calibrated projection scratch uses the
same pool; expanded prefill weights remain separately cached by weight identity.
The pool must not service unordered queues or untracked resources without explicit
dependency handling. No general host-write permission follows from queued reuse.

External tensor owners are retained until successful completion and then released.
Prepared-cache eviction drains work before releasing pooled storage; escaped
tensors remain valid. Native binding objects may retain Metal buffers longer.
CPU arena capacity remains grow-only until destruction.
Neither backend promises a fixed-byte budget; diverse shapes and externally
retained outputs can increase memory. This is not slab allocation: all MPSGraph
bindings still use zero-offset buffers.

Decoder state preallocates its token embedding, causal mask, index, and K/V buffers
before the token loop. `EmbeddingImpl::forward_` fills a supplied host-accessible
device tensor; callers must ensure prior queued uses have completed. Generated
token vectors reserve their configured limits before batched greedy search.

Metal execution retains non-pool tensor owners through completion. Binding caches
hold buffer metadata and Metal data objects, not
additional tensor leases; up to 64 binding sets are retained per executable.
Command batches reuse completion tickets and scatter error storage after finishing.
The native command buffer itself remains one-shot. Callback lifetime ownership
and caller-visible tensor ownership still use reference counting where required.

This removes identified Kidi allocation/refcount churn, not every allocation inside
YNNPACK, MPSGraph, Objective-C descriptors, or cold preparation. Profiles distinguish
output-slot allocations from those internal allocations. See
[benchmarks/metal/ALLOCATION_REUSE.md](benchmarks/metal/ALLOCATION_REUSE.md) for results.
With `KIDI_PROFILE_MEMORY=1`, Metal also reports cumulative explicit output
allocations, prepared calibration scratch, expanded weights, external-owner slots,
and native `currentAllocatedSize` sampled at completion boundaries. The observed
native high-water mark can miss intra-batch transients and is not physical RSS or
system swap usage. Current results are in the
[Gemma 4 optimization journal](benchmarks/gemma4/JOURNAL.md).

## Backend Internals

Gemma 4 query/key RMSNorm and RoPE share a direct Metal dispatch with the same
reduction geometry and intermediate arithmetic order as the separate kernels.
CPU composes the existing operators.

Gemma 4 uses host-managed token lookup and positions with dense K/V storage on
the execution device. Device-state, direct-paged attention and projection-replay
experiments were removed after review; there is no alternate execution path or
benchmark callback in the runtime.

`Context::greedy_token` reduces each FP32 vocabulary row to an I32 token index.
Equal maxima choose the lowest index; NaN, positive infinity, or an entirely
negative-infinite row produces -1. Metal uses a two-stage reduction; CPU uses
the same selection contract. Gemma 4's `forward_token` applies this to the actual
post-softcap logits before the existing completion fence and rejects -1.
The generic decoder accepts preselected tokens only for unscored greedy search,
retaining the existing stop/length policy. GPU Gemma 4 generation always uses this
path; CPU selection is
unchanged. The profile records `device_selection` and `generation_ns` (request
start through text decoding), avoiding misleading comparisons when selection
moves across the model-only timing boundary. This does not eliminate per-token
synchronization or implement graph replay.

YNNPACK and MPSGraph are graph-oriented libraries. Small operator-private
executables are permitted, cached by input signatures, and reused without model
compilation. Their `runtime/.../graph.*` wrappers are backend infrastructure;
model/layer headers do not include them. Fused linear, normalization, and GELU
operators coexist with ordinary tensor operations. INT8 Metal projections use
the existing resident packed-GEMV/tiled-GEMM kernels.

CPU executables select their thread pool at preparation time; a zero thread
count inherits the configured default without changing it. On Apple Silicon,
single-row INT8 projections with at most 4,194,304 weight elements and batch-one,
single-query attention with at most 262,144 key elements use one thread. Larger
work retains the configured pool. This measured scheduling policy is backend
private, not a change to model equations or a globally toggled thread setting.
Other platforms retain default scheduling. See the
[matched scheduling comparison](benchmarks/metal/scheduling-20260919/README.md),
including preparation costs and quality differences from the old graph runtime.

No model graph, lazy fallback, compiler IR, or alternate generation policy is
retained; captured steps are recorded from eager code as described below. The former graph comparison benchmark was removed; its raw
measurements and report remain historical documentation. The superseded fusion
prototype benchmark was also removed; production fused operators and numerical
tests remain. CPU wrappers no longer maintain unused dynamic-shape or concurrency
query APIs; the eager cache prepares a separate executable for each signature.

## Captured Steps

`Context::replay(key, inputs, step)` runs one fixed-shape step, such as a decoder
token, as ordinary eager code. The first two calls with a key run `step` while
recording every dispatched operator, its prepared executable, and where each
operand's storage comes from: a step input, an earlier recorded output, or a fixed
external tensor such as a parameter or cache. The two recordings must match
structurally. This rejects temporaries created on the host each call, operator
arguments that change per call, and `copy_slice_` offsets; such values must be
written into step inputs or state that aliases them. Later calls skip `step`,
rebind any changed inputs (views are rebuilt at the same offsets), and run the
recorded operators into their retained, pointer-stable buffers. Outputs belong to
the captured step and are overwritten by the next call with the same key.

Model code stays eager: a step reads its inputs, parameters, and fixed-storage
state; host work that changes per call, such as token embedding lookups, masks, and
cache indices, is written in place before the call. Capture never changes device,
partitions a step, records host control flow, or falls back silently. Backends
declare support through `OperatorBackend::supports_replay` and implement
`Operator::run_into`; other backends, or `KIDI_REPLAY=0`, run `step` eagerly
every call with identical results.

Per-call values such as token IDs and positions are I32 tensors, so a step can
keep them on its device: `TokenEmbedding` looks up token tensors with the
`embedding` operator (CPU and WebGPU; Metal reads them on the host), and a step's
selected token can feed the next step without a host read
(`Gemma4Impl::forward_token(const Tensor&, ...)`).

Whisper decoding replays one step per decoder capacity: token, mask, and position
index inputs, with positions gathered by index and caches written by `scatter_`.
Gemma 4 single-request decoding replays one step per cache capacity, 128-position
key extent, and local-attention crop, using the same extents as eager decoding;
host-built masks and rotary angles are written into `Gemma4State::step`. WebGPU
uploads these small inputs explicitly and rebinds them on replay; captured
outputs and KV caches stay on the GPU. Batched decoding keeps the eager path.
WebGPU scatter updates cache rows without copying the whole cache and reports
invalid indices at synchronization. Replay preserves reference logits within
the backend's numerical tolerance and generated token IDs in the tested runs.

Serving can opt into `ServingOptions::compact_cache` to reserve prompt plus
maximum output length in 128-token buckets rather than the full context limit.
The reservation stays on the same side of the backend's INT8-cache threshold
as the requested limit, preserving its precision policy. Native callers retain
full reservations by default; the browser enables compact reservations to
reduce pressure on its native wasm64 linear heap (8 GiB growth ceiling).

The CPU backend replays prepared YNNPACK executables and custom kernels in order.
Eager dispatch was about 0.75% of Whisper Small decode time on an Apple M5 (about
0.4 us per operator), so CPU replay is time-neutral: Gemma 4 E2B QAT and Whisper
Small INT8 decode rates stayed within noise on the M5 and on an SM8750 phone with
identical token IDs. Captured steps exist so launch-bound accelerators can replay
a whole step at once. Lowering a captured step to one YNNPACK subgraph is deferred
until profiling shows a CPU benefit.

### Accelerators and Step Compilers

Devices with a full eager backend (CPU, Metal, WebGPU, Vulkan) run every operator
and replay captured steps on themselves. An accelerator without eager operators,
such as the Qualcomm NPU, is a `runtime::StepCompiler` attached to a CPU context
with `ops::StepCompilerScope` while a model is constructed. After a step's two
captures match, `Context::replay` asks the compiler to compile the recorded graph;
later calls run the returned `StepExecutable` with the step inputs and the
captured output buffers, and the CPU intermediates are released. A compiler may
decline a step, which keeps CPU replay. A compiler whose `compiles_in_background()`
is true compiles a copy of the graph on a worker thread while CPU replay keeps
serving the step, so a slow first compile (tens of seconds on the NPU) does not
stall generation; the executable takes over on the first call after it is ready,
and a failed background compile logs `kidi_step|...|fallback=cpu` and keeps CPU
replay. `KIDI_BACKGROUND_COMPILE=0` compiles synchronously, which benchmarks use
to measure the accelerated steady state. Its `key_extent` and
`crop_local_attention` policies let models choose few, coarse step shapes, for
example power-of-two attention extents. With a compiler attached, Gemma 4 also
captures full power-of-two prefill chunks (producer layers writing K/V, no
outputs); prompt tails stay eager on the CPU. A compiler may raise the preferred
prefill chunk; QNN uses 128 rows. Completed prefill executables are evicted before
decode so their HTP body weights do not coexist in memory, while their context
binaries remain cached on disk.

`inference::select_device` maps "auto", "cpu", "gpu", and "npu" to a device.
"auto" prefers the NPU, then Metal on Apple, then the CPU for Gemma, and the CPU
for Whisper; Vulkan is an explicit, experimental choice until it measures faster
than the CPU. Explicit choices fail when unavailable. `Generator::load` with
`Device::qualcomm_npu()` keeps Gemma on the CPU with the NPU step compiler, and
`Generator::execution()` reports the result (e.g. `cpu+qnn-htp`).

## Models and Generation

Neural implementations derive from `Module`, use `<Name>Impl` class names, and
expose a typed `forward(...)` method. `KIDI_MODULE(Name)` declares a
`ModuleHolder<NameImpl>` alias. The holder forwards constructor arguments to
`std::make_shared<NameImpl>` and owns that shared pointer. The base deliberately does not prescribe one
virtual forward signature: embeddings, attention, blocks, and models have
different typed inputs. It is an ownership/registration base, not a type-erased
execution engine.

```cpp
ModuleScope construction(tensor::DType::BF16, false, tensor::Device::cpu());
layers::Linear projection(768, 2048);
ops::require(projection->set_state(weights));
auto output = projection->forward(context, input);
const auto& implementation = *projection;
auto another = implementation.forward(context, input);
```

Model/layer composition owns holders; `->`, `*`, and `get()` borrow the implementation
without reference-count increments. `ptr()` exposes the underlying shared pointer
by const reference for explicit ownership interoperation. Copying a holder shares
the module; moving transfers ownership. `Name{nullptr}` creates an empty holder,
testable with `bool`. Default construction invokes the implementation's default
constructor, so layers requiring dimensions must receive them or explicit `nullptr`.
`ModuleList<Impl>` and `ModuleMap<Impl>` are holders for their typed container
implementations and default-construct usable empty containers. Omit `Impl`
to store heterogeneous `Module` pointers. Lists register numeric child names;
maps register supplied names. Iteration and `at()` return references rather
than copying pointers. The Transformer owns typed encoder/decoder module lists.

`register_parameter(name, tensor_member, shape, dtype)` and
`register_module(name, child)` are protected construction helpers. Parameter
shape/dtype declarations remain available when storage is deferred. Names must be nonempty, unique, and contain no
dot; null children and cycles are rejected. Module objects cannot be copied or
moved because parameter registration points to stable tensor members. Copy the
holder to share a module instead. `ModuleScope` saves and restores
the thread-local `module_dtype`, `allocate_parameters`, and `module_device`, including
on exception unwinding. Nested modules inherit the policy; new threads do not
inherit a caller's overrides. Defaults are FP32, allocation enabled, and the first
available GPU execution backend (Metal, then CUDA), otherwise CPU. Storage-only
backends do not qualify. Explicit unsupported devices are not silently substituted.

Each module captures its device at construction. The model's context and state
buffers use that device, independent of later scopes. Parameter allocation honors
it; with allocation disabled only parameter metadata is created, and derived
embedding positions are allocated lazily. Low-level tensor/ops APIs still accept
explicit devices. CLI `--backend auto` follows the GPU-first default; explicit
`ynnpack` and `mps` override it.

Allocated weights and biases are zeroed, LayerNorm weights and quantization scales
start at one. Callers can initialize writable floating tensors via `state_dict()`
before first use, including random initialization without any checkpoint. This
does not add automatic random initializers or an autograd/training implementation.

`state_dict()` returns an in-memory ordered map of qualified names to tensor
handles, recursively including registered children. It shares existing tensor
storage; it is not disk serialization. Derived position tables and generation
caches are not parameters. State keys follow module registration names, not
necessarily the original imported RTG checkpoint names. `TransformerImpl::state_mapping_specs()`
adapts RTG names once when loading the package; layer constructors never receive
checkpoint paths or prefixes.

`set_state(weights)` and `set_state(StateDict)` bind tensors after construction.
On the same device they retain incoming storage, including read-only mapped weights;
cross-device tensors are transferred to the module's captured device. Caller-retained
state shares same-device storage, so it must not be mutated during inference.
`load_state_dict(state, strict=true)` instead copies into fresh storage. Both validate all
keys, shapes, and dtypes before changes; strict mode requires exact keys, while
non-strict mode loads matching entries and ignores missing/unexpected ones.
All required transfers/copies finish before committing replacements. Copying loads
keep the supplied state independent of loaded parameters and change identities
used by prepared-operator caches so later forward calls see the new weights.
Registered ties preserve shared target-embedding/output storage in both loading
modes; conflicting values for tied keys are rejected. A canonical key suffices
when loading a tie. Old prepared entries remain bounded by normal eviction. No autograd, optimizer,
or buffer serialization framework is added.

Synchronize before exporting/loading tensors that may have pending device work.
Do not load state concurrently with forward calls. After changing model weights,
discard previously computed encoder/decoder caches and start a new generation.
Sharing a model pointer does not make its execution context thread-safe.

Layers register their parameters and children during construction, then load state
separately. For example, the decoder block
computes self-attention, cross-attention, and feed-forward residuals directly
using `Linear`, `LayerNorm`, `Attention`, and `FeedForward` objects.

`model::TransformerImpl::encode` produces projected encoder K/V tensors and a
padding mask. `create_state` allocates a fixed-capacity decoder cache. `forward`
embeds tokens, runs the bound decoder layers, and returns FP32 logits. Greedy
calls update explicit cache tensors; beam calls evaluate full prefixes without
cache reordering. Both devices use this implementation. Embedding lookup and
positional arithmetic currently execute on CPU, followed by an explicit transfer
when needed; neural projections and attention run on the selected device.

`inference::Decoder::generate` accepts a nonempty prompt and a score callback.
It supports greedy or beam search with no package, encoder, or device dependency.
`DecodeRequest::prefixes` is row-major `[active_beams, prefix_length]`, including
the prompt. `beam_indices` identifies active beam slots, not parent-cache rows;
adapters must reconcile caches from full prefixes if they implement beam caching.
Request spans are borrowed for the callback only.

`generate_batch` provides a host greedy loop with independent per-row EOS and
limits. Its callback receives one current token per row and the generated step
count. Finished rows remain in the batch with PAD inputs; their scores and output
lengths no longer change. `GreedyRequest::active_rows` lets adapters pack only live
rows while returning results at original row indices. Preselected token vectors
are accepted only for unscored greedy search with no simultaneous score values.
`GreedyState` owns incremental per-request stopping/scoring state and copies its
stop IDs, allowing independent retirement and admission. A fresh invocation resets
all search state.

Score callbacks return row-major FP32 logits or log probabilities, explicitly
tagged. Returned spans must remain alive until the next callback. Results exclude
the prompt, EOS, and PAD; limits count generated tokens. Optional unfinished-score
length preserves historical RTG truncation normalization. Translation preserves
bounded length-sorted batching and restores original input order.

`model::Gemma4` implements dense Gemma 4 text decoding using RMSNorm, gated
tanh-GELU MLPs, grouped-query attention, local/global masks, proportional RoPE,
shared K/V, per-layer embeddings, and logit softcapping. `inference::Generator`
loads the original Hugging Face checkpoint and single tokenizer through the
Gemma 4 configuration path. `Package` remains the RTG two-tokenizer package API.
Gemma 4 key mapping and small BF16-to-FP32 normalization conversions happen at
load time; there is no offline checkpoint conversion. Embeddings and the tied
output projection retain CPU-mapped weights, with looked-up rows allocated on
the execution device. Other projections execute on the selected backend.

Gemma 4 generation supports greedy text requests, including packed independent
decode rows through `forward_batch` and `forward_batch_tokens`. Projection and
pointwise work share packed rows; each attention segment retains its own cache,
length, position and mask. Scoped decode projection policy preserves packed-vector
arithmetic rather than selecting calibrated FP16 prefill at four or more rows.
Cache-only prefill stops at the final K/V-producing layer immediately after its
K/V writes: it omits that layer's query/attention/MLP and all subsequent shared-K/V
consumer layers. Those layers cannot affect retained history. Full all-token
logit evaluation still traverses every block, and tests require byte-identical
producer caches between both paths for the same input prefix. Native-QAT CPU
last-logit calls additionally prefill all preceding tokens cache-only and run the
full model for the final token.
The default native-QAT Metal matrix path
evaluates the entire projected chunk through its K/V-producing blocks, then retains
only the final four query rows for subsequent shared-K/V consumers when the chunk
has at least 64 tokens. Consumer masks and rotary rows are sliced together; cache
writes and final state positions remain unchanged. All-token logits, packed-only
prefill and other precisions retain full-consumer execution.
Long fixtures and
generated-token comparisons pass, though full-model logits are not universally
bit-identical across query shapes. Profiles distinguish `last_token_prefill`
and `shared_prefill_tail`. Prefill remains
chunked at the inference layer. Its
explicit fixed-capacity caches are shared by the configured later layers;
copying `Gemma4State` aliases mutable history. Start a new state for an independent
request. The generic decoder accepts additional stop IDs; disabling EOS stopping
is an explicit benchmark option that retains generated special tokens.

`Generator::generate_batch` accepts 1-16 prompts, prefills independently and retires
decode rows through the shared decoder. `configure_serving`, `enqueue`, `step` and
`cancel` expose a thread-confined incremental queue. Stable request IDs identify
streamed token IDs and final text/results; callers may enqueue between steps.
Outstanding requests, active slots (at most 16), and the sum of reserved dense-cache
capacities are bounded. FIFO admission observes both slots and cache-token budget;
ready requests decode each step before one bounded round-robin prefill chunk.
Completion and cancellation release reservations. Cancellation has no completion
event; unknown IDs are rejected. Drain or cancel requests before blocking generation
or reconfiguration. A failed model step invalidates serving and requires reload.

`GenerationOptions::stream_text` opts into `GenerationEvent::text` deltas alongside
the existing token and completion events. `Tokenizer::decode_delta` decodes the
accumulated token prefix, withholds incomplete UTF-8/replacement suffixes until
stable or final, and rejects decoders that revise already-emitted text. It uses
quadratic cumulative decode work in output length, a conservative tradeoff for
short interactive replies; JSONL does not enable it. Deltas concatenate to the
ordinary final text. Cancellation destroys the request's decoder state too.
The terminal flushes deltas after each model step. Cancellation is cooperative
between steps, not GPU preemption. Weights and prepared operators remain loaded;
cross-turn KV retention is opt-in through the existing prefix-cache byte budget.

Serving events may finish out of order. The CLI retains completed records until
their preceding input records are emitted, counting these retained completions
against the same admission window as unfinished requests. Thus the reorder buffer
is bounded by request count, not bytes, and a slow request backpressures input.
Responses include the echoed input ID and an assistant message. Invalid JSONL
stops processing with its physical input line number; JSON blank lines are invalid.

Serving may retain one completed or cancelled request's dense KV storage when
`prefix_cache_bytes` covers its full allocation. It records only tokens actually
processed by the model, including processed output tokens, not the last sampled
token that has not entered KV. Admission moves compatible storage into exactly one
new request and resumes its longest identical token prefix, leaving at least one
prompt token to produce logits. No KV copy or fresh allocation is needed on this
path. Other active requests retain independent storage; this is not paged attention
or mixed prefill/decode in one model invocation.

Capacity, effective prefill chunk, attention policy, or insufficient byte budget
invalidate serving reuse. Divergent text reuses only the identical prefix. Disabled
caching, reconfiguration, model unload, failed execution, or entry to blocking
generation discard the retained state. One idle allocation may exist in addition
to active cache-token reservations; it is bounded by the retiring request's byte
budget. Completion metrics distinguish valid-prefix bytes from the full retained
allocation. Android enables a 512 MiB cap; this retains the existing request
allocation rather than allocating a second copy of the conversation cache.

The separate blocking `generate()` prefix cache retains one dense snapshot, sized to a
complete-chunk prefix within the byte budget. `fork_state` copies the matching
prefix into independent request storage; subsequent writes cannot mutate the
snapshot. Chunk-size and attention-policy changes invalidate reuse. Reported
prefix reserved bytes equal the snapshot's K/V bytes, not the configured budget.

Serving step timings report shared decode/prefill/preparation work. Per-request
first-token and completion timestamps include queueing since enqueue; per-request
decode time is not apportioned from shared batches. Serial admission additionally
reports actual per-request decode and preparation time. Static-batch first-token times
are readiness during sequential prefill, not streamed delivery. None of these
batched/cache-hit rates substitutes for uncached batch-one latency comparisons.

Gemma 4 can prepare Q4/Q8 linear weights in memory through `set_checkpoint` runtime
options. The registered state still holds original tensors. `ops::Context` owns
derived packed bytes/scales and retains source handles, preventing address reuse;
packing is shared across prefill/decode shapes. Q4 mode uses grouped MLP weights
and per-channel Q8 for other projections. By default only single-row calls use
packed weights; `packed_prefill` enables packed GEMM for multi-row calls too.
Reloading a Gemma 4 checkpoint resets its prepared context, so disabling quantization
cannot reuse stale packed weights. There is no offline checkpoint rewrite.

CPU packed operators use YNNPACK's integer dot path with dynamic INT8 activation
quantization. Metal GEMV directly reads packed Q4/Q8 and FP32 activations; tiled
GEMM unpacks into bounded FP16 threadgroup tiles and accumulates in FP32. These
are different compute policies and do not guarantee identical logits. Supported
FP32 normalization, rotary, and pointwise operations use small Metal kernels;
unsupported layouts retain their existing private MPSGraph implementation.
Gemma 4 attention bounds reads by 128-token active-prefix buckets while retaining
the explicit full-capacity cache. CPU additionally crops old masked local history
inside prepared attention operators; global attention keeps the full active prefix.
The lower bound preserves every token visible to the first query of a prefill
chunk. GPU keeps its previous range because cropping did not improve measurements.
`Gemma4State::crop_local_attention = false` or CLI `--full-attention-cache` disables
the CPU crop, which can otherwise change floating-point reduction order and
quantized outputs. Local circular caches are not implemented.

`ops::Context::rms_norm_residual` computes post-normalization, residual addition,
and optional scalar output scaling in one operator. All operands are same-device
FP32 tensors; no input is mutated. Gemma 4 uses it for post-attention, post-MLP, and
post-per-layer-input residuals. CPU and Metal implement the same equation without
duplicating model topology or introducing model-level graphs.

CUDA is a registered placeholder: the device kind and integration points exist,
but operations fail with a "not implemented" error. QNN is implemented as the
captured-step compiler described below. Unsupported execution does not silently
fall back to CPU.

### Native Gemma 4 Mobile QAT

The original `gemma` quantization policy is retained under `model.quantization_config`.
Model construction resolves its supported per-module bit widths; registered packed
projections declare byte shapes, trained weight scales, and activation scales.
Packed embeddings declare their per-row or per-layer scale geometry. The loader
converts offset-packed Q2/Q4 bytes to signed packed encoding in memory and validates
scales before binding state. It does not quantize the trained model again, rewrite
the checkpoint, or load unused modality encoders. QAT precision overrides are rejected.

Calibrated CPU projections quantize activations directly with the trained scale
before integer dot products; uncalibrated projections retain dynamic activation
quantization. Metal calibrates input once into prepared scratch and fuses output
rounding into the packed projection. Static-range rounding uses ties-to-even,
INT8 clipping, and zero-scale bypass. K/V cache values are rounded using their
trained scales, while storage remains FP32.

Default calibrated Metal prefill may cache expanded FP16 matrices for native
matrix multiplication, sharing them across input shapes. Packed GEMV remains the
decode path. `Context(device, true)` and CLI `--packed-prefill` request packed
prefill instead. These derived caches retain owners and are released with the
context; there is no claim of a fixed-byte memory budget. Native QAT is a separately
identified checkpoint path, not a silent replacement of BF16 or PTQ behavior.

Expanded FP16 matrices are materialized by a GPU unpack kernel on the ordered
stream, with source/destination owners retained through completion. Native matrix
programs are shared by complete input shape, output width and feed dtype; matrix
identity and calibration stay in the individual operator bindings. The backend
keeps up to 128 geometry lookup entries, and live operators retain their program
when that lookup cache is cleared. No weight values are captured in these programs.

Metal can optionally bound retained expanded matrices with `KIDI_PREFILL_CACHE_BYTES`
(0..16 GiB). Matrices beyond that budget are unpacked from the original packed
weights into shared GPU scratch before the same FP16 matrix operator. This does
not bound the packed model, transient/native workspace, or total process memory.
The default retains all expanded matrices; the bounded path trades extra unpack
work for lower residency. Set the budget to zero to stream every matrix. Bounded
expansion requires the shared output pool. Each projection rounds its input into
scratch; no calibrated-value cache or write-invalidation protocol is needed.

## Compatibility and Verification

CLI flags remain stable. Packages require the nested YAML layout described above;
there is no flat-config compatibility path. Legacy machine-readable profile
field `graph_compile_ns` now reports backend operator preparation. The old detailed
graph counters and `last_hidden_ns` have been removed rather than reporting zeros.
Consumers should tolerate absent legacy keys. `translate_ns`
excludes measured preparation, whereas encompassing encoder/decoder timers can
include it. Use warmed request timing for throughput comparisons.

Tests retain established numerical fixtures for FP32/BF16/INT8, attention,
embedding offsets, quantized tails/constant rows/long contractions, plus eager
lifetimes, failures, and generic greedy/beam/batched search. Legacy CTest names
are retained for scripts, despite the removal of their old graph implementations.

See [benchmarks/metal/EAGER_MIGRATION.md](benchmarks/metal/EAGER_MIGRATION.md) for
actual corpus and performance results. The eager migration is not claimed to be
bit-identical for INT8. Non-Apple builds have not been run in this environment.

## Vulkan GPU Backend

When built with `KIDI_ENABLE_VULKAN=ON`, the Vulkan backend registers `Device::vulkan()` only on Vulkan 1.3 compute
devices that expose accelerated signed packed INT8 dot products and 64-lane arithmetic subgroups.  Tensor storage is
host-visible and persistently mapped so Gemma captured-step inputs (tokens, masks, rotary tables, and scatter indices)
remain writable by the host between replays.

The eager backend supports captured-step replay through `Operator::run_into` for every operation.  Operations without a
native Vulkan kernel execute through the CPU operator implementation over host-visible Vulkan storage and copy results
back to Vulkan tensors.  Native Vulkan kernels currently cover calibrated signed packed projections and fused calibrated
Gemma feed-forward blocks using Adreno integer dot-product compute shaders; weights are uploaded once by the prepared
operator, while activations and outputs use reusable mapped buffers for the fixed prepared shape.  This keeps Gemma text
decode functional while leaving broader batching, descriptor reuse, device-local allocation, and full attention/native
pointwise coverage as backend optimization work.

## Qualcomm NPU Captured-Step Compiler

When Kidi is built with `KIDI_ENABLE_QNN=ON` and a QAIRT SDK,
`runtime::npu_step_compiler()` loads the Qualcomm QNN HTP runtime with `dlopen`
and attaches a `StepCompiler` to CPU Gemma 4 contexts selected with
`Device::qualcomm_npu()`. Model code records its normal operators twice; the
compiler then lowers the entire captured prefill or decode step to chained QNN
graphs. There is no operator-by-operator CPU/NPU alternation in accelerated
steady state.

The whole-step lowering covers packed 2/4/8-bit embeddings, calibrated
projections and feed-forwards, RMS norms and residuals, rotary embeddings,
grouped-query causal attention, KV writes, the 262K 2-bit vocabulary head, and
greedy selection. Packed embedding tables reside in shared FastRPC memory and
are gathered, unpacked, and scaled on HTP. KV caches use their trained INT8
grids in registered shared memory; the host synchronizes an existing prefix
once when a new 512/1024/2048/... attention bucket is selected, and copies only
the newly produced KV rows back for model-state compatibility. Per-call host
work is limited to binding inputs, cache-prefix synchronization, graph launches,
and reading the selected token.

Low-bit FC weights use 8-bit containers with bit-width axis scales because this
HTP release rejects packed SFIXED2/SFIXED4 constants. Graphs are split by an
unpacked-weight budget (256 MiB by default) to keep `libQnnHtpPrepare` memory
bounded. On the tested Gemma 4 E2B model this produces five prefill graphs and
eleven decode graphs. Prefill and decode share a single FastRPC copy of the
packed embedding tables.

Whole-step context binaries and JSON binding metadata are cached under
`KIDI_QNN_CACHE_DIR`, keyed by step shape, sampled model content, lowering
version, and QNN build ID. The measured 128-row/512-key prefill context is
319 MiB and the 512-key decode context is 804 MiB; cold compilation takes about
114 and 92 seconds, while reload takes about 4 and 8 seconds. Compilation runs
in the background by default, so CPU replay remains usable until the context is
ready.

The Android APK packages the matching HTP runtime and prepare libraries, V79
stub/skel, and QNN System library. Prepare is 81 MiB installed but about 35 MiB
compressed in the APK; it is required only for an uncached shape. The skel is
extracted to the APK native-library directory because FastRPC cannot read it
from normal app data. Kidi requests the sustained-performance HTP power vote
when available.

### Kidi HTP Op Package

QNN does not re-fuse Kidi's fused operators: stock lowering expands each gated
feed-forward into about 18 QNN operators and each RMS norm into 12-14. A profile
of a 128-token INT8-KV prefill step found element-wise work dominating (GELU-tanh
chain 30%, other multiply/add 26%, quantize/dequantize 20%, RMS norms 10%), while
HMX matmuls were 3.6% and attention 1%.

`src/kidi/runtime/qnn/htp/KidiOpsPackage` is a QNN HTP op package (QHPI,
multithreaded HVX) that restores those fusions on the NPU. `GeluMultiply` maps
the UINT8 gate/up projection output directly to the UINT8 input of the down
projection in one pass, and `RmsNorm` covers plain and residual norms with an
output scale. The package ships as a host library for graph preparation
(`libQnnKidiOps.so`) and a V79 DSP library (`libQnnKidiOpsHtp.so`); the
compiler registers them when `KIDI_QNN_OP_PACKAGE` and
`KIDI_QNN_OP_PACKAGE_HTP` name them, keys cached contexts on whether fused ops
are in use, and keeps the stock lowering when registration fails
(`KIDI_QNN_FUSED_OPS=0` also disables them). On SM8750 at a 9216-token cache,
fused ops cut a steady 128-token prefill chunk from 179 ms to 64 ms and raised
decode from 28.3 to 33.2 tokens/s with identical greedy tokens; prefill
compilation dropped from 114 to 34-45 seconds.

The package also contains `StructuredAttention`, a causal/sliding-window
attention kernel that skips masked key tiles and reads INT8 caches. It measured
slower than QNN's attention, so it is opt-in (`KIDI_QNN_CUSTOM_ATTENTION=1`).
Experimental HMX and Crouton-layout helpers are kept unregistered.
