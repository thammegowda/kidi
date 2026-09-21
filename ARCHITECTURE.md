# Eager Architecture

Kidi has one model execution path: ordinary C++ functions operating on concrete
tensors. There is no Kidi model graph, symbolic value type, generic compiler,
lowerer, graph partitioner, or recorded control flow.

```text
inference::Decoder -> model::{Transformer, Gemma4} -> layers -> ops::Context -> backend
```

## Ownership

| Module | Responsibility |
|---|---|
| `tensor` | Storage, dtype, device, views, explicit transfers |
| `core::Module` | Shared module ownership, named parameter/child registration, state dictionaries |
| `ops::Context` | Immediate operator dispatch, validation, bounded prepared-operator cache |
| `layers` | Bound parameters and reusable neural equations |
| `model` | Package format, Transformer topology, source and decoder state |
| `inference` | Generic generation/search, translation application, profiling |
| `runtime/ynn` | Private prepared CPU operators and YNNPACK ownership |
| `runtime/mps` | Private prepared Metal operators, command batches, INT8 kernels |
| `text`, `cli` | Tokenization and command/input/output policy |

RTG is an import/package format, not the owner of general modeling code. Do not
add independent CPU and GPU model implementations. Backend fusion belongs in
operators, not duplicated Transformer equations.

## Configuration

Configuration stays in `YAML::Node`; there are no model-specific configuration
structs or mirrored manifest types. `model::load_config` reads `model.yaml`,
checks package structure and file paths, and resolves file paths relative to
the package. `Package::config()` exposes that document.

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
Safetensors is the checkpoint format. The inference loader derives the construction
dtype from the checkpoint. Layer constructors declare matching shapes/dtypes and
required INT8 scale parameters without reading weights. `set_state` validates them
before assignment. There is no separate encoding enum or YAML precision setting.
The execution policy follows stored weights; any future compute-precision override
is a separate runtime option, not a duplicate declaration of checkpoint metadata.
The old flat manifest layout is not supported or automatically migrated.

CLI diagnostics use spdlog on stderr. Translation results and inspection output
remain on stdout; machine-readable metric/profile records retain their unadorned
format on stderr. `SPDLOG_LEVEL` controls the human-readable logger.

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
buffers; each entry owns its constant-parameter references. The bounded override
`KIDI_OPERATOR_CACHE_CAPACITY=1..16384` supports working-set experiments. This is an
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
Set `KIDI_METAL_REUSE_OUTPUTS=0` before context creation for the per-operator
pool/scratch fallback. CPU arena capacity remains grow-only until destruction.
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
CPU composes the existing operators; `KIDI_SEPARATE_RMS_ROTARY=1` also restores
that composition on Metal.

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
retaining the existing stop/length policy. GPU Gemma 4 generation enables this
path by default; `KIDI_HOST_GREEDY=1` restores host selection. CPU selection is
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

No model graph, capture mode, lazy fallback, compiler IR, or alternate generation
policy is retained. The former graph comparison benchmark was removed; its raw
measurements and report remain historical documentation. The superseded fusion
prototype benchmark was also removed; production fused operators and numerical
tests remain. CPU wrappers no longer maintain unused dynamic-shape or concurrency
query APIs; the eager cache prepares a separate executable for each signature.

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

Gemma 4 generation supports greedy text-only requests, including packed independent
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
full model for the final token. `KIDI_FULL_LAST_CHUNK=1` disables this CPU policy.
The default native-QAT Metal matrix path
evaluates the entire projected chunk through its K/V-producing blocks, then retains
only the final four query rows for subsequent shared-K/V consumers when the chunk
has at least 64 tokens. Consumer masks and rotary rows are sliced together; cache
writes and final state positions remain unchanged. All-token logits, packed-only
prefill and other precisions retain full-consumer execution. Disable this policy
with `KIDI_SHARED_PREFILL_TAIL=0` or `KIDI_FULL_LAST_CHUNK=1`. Long fixtures and
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

Serving rejects prefix-cache options. Active K/V remains dense and independent;
this is not direct paged attention or mixed prefill/decode in one model invocation.
The optional single-request prefix cache retains one dense snapshot, sized to a
complete-chunk prefix within the byte budget. `fork_state` copies the matching
prefix into independent request storage; subsequent writes cannot mutate the
snapshot. Chunk-size and attention-policy changes invalidate reuse. Reported
prefix reserved bytes equal the snapshot's K/V bytes, not the configured budget.

Serving step timings report shared decode/prefill/preparation work. Per-request
first-token and completion timestamps include queueing since enqueue; per-request
decode time is not apportioned from shared batches. Static-batch first-token times
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

CUDA/QNN storage providers remain separate from execution; unsupported execution
does not silently fall back to CPU.

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