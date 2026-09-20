# Eager Architecture

Kidi has one model execution path: ordinary C++ functions operating on concrete
tensors. There is no Kidi model graph, symbolic value type, generic compiler,
lowerer, graph partitioner, or recorded control flow.

```text
inference::Decoder -> model::Transformer -> layers -> ops::Context -> backend
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

Linear and normalization parameters are fixed inference weights. Prepared
operators key parameter identity as well as dtype and shapes and retain owners.
Do not mutate their storage after first use; replace registered parameters with
`load_state_dict`, or create a new context/model. Activations, masks, and token indices are dynamic bindings and
may change between completed calls without recompilation. This is an inference
API, not an autograd or optimizer framework.

The prepared-operator cache is bounded at 1,024 entries; reaching the bound drains
work and clears it. `preparation_ns()` reports cumulative operator preparation.

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

Metal output pools directly own separately backed slots. A shared-slab prototype
using MPSNDArray offsets failed slice/attention correctness checks and was removed.
The later retain-only arena was also removed: it did not suballocate or recycle
Metal buffers and unnecessarily kept evicted operators' allocations alive.
The previous eight-output cap caused recurring allocations while GPU work retained
many outputs. Pools now grow to the observed high-water mark and retain all slots;
subsequent fixed-shape calls reuse them. Paired residual/normalized outputs share
one pool; a live first output prevents its slot from being selected for the second.
CPU arena capacity remains grow-only until context destruction. Metal allocations
can be released after pool eviction once pending work, reusable stream ownership
slots, native bindings, and caller tensors no longer retain them. Neither backend
promises a fixed-byte memory budget; externally retained outputs can grow memory.

Decoder state preallocates its token embedding, causal mask, index, and K/V buffers
before the token loop. `EmbeddingImpl::forward_` fills a supplied host-accessible
device tensor; callers must ensure prior queued uses have completed. Generated
token vectors reserve their configured limits before batched greedy search.

Metal execution retains reusable ownership slots, updating owners only when the
buffer changes. Binding caches hold buffer metadata and Metal data objects, not
additional tensor leases; up to 64 binding sets are retained per executable.
Command batches reuse completion tickets and scatter error storage after finishing.
The native command buffer itself remains one-shot. Callback lifetime ownership
and caller-visible tensor ownership still use reference counting where required.

This removes identified Kidi allocation/refcount churn, not every allocation inside
YNNPACK, MPSGraph, Objective-C descriptors, or cold preparation. Profiles distinguish
output-slot allocations from those internal allocations. See
[benchmarks/metal/ALLOCATION_REUSE.md](benchmarks/metal/ALLOCATION_REUSE.md) for results.

## Backend Internals

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
lengths no longer change. A fresh invocation resets all search state.

Score callbacks return row-major FP32 logits or log probabilities, explicitly
tagged. Returned spans must remain alive until the next callback. Results exclude
the prompt, EOS, and PAD; limits count generated tokens. Optional unfinished-score
length preserves historical RTG truncation normalization. Translation preserves
bounded length-sorted batching and restores original input order.

The shipped package loader remains RTG encoder-decoder-specific. Generic search
supports decoder-only callbacks, but no decoder-only model loader is implied.
CUDA/QNN storage providers remain separate from execution; unsupported execution
does not silently fall back to CPU.

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