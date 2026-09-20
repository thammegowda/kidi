# Asynchronous Metal decoding design

## Goal

Keep the autoregressive dependency chain, K/V caches, generator projection, and
token selection on the GPU. The CPU observes completed tokens asynchronously
through unified memory and stops future submission after EOS without blocking
each decode step.

Unified memory removes explicit CPU/GPU copies. It does not remove command
encoding, queue scheduling, cache-coherency fences, or completion waits. The
design therefore treats synchronization as the primary resource to eliminate.

## Required graph boundary

One compiled `decode_step` must perform all of the following:

1. Read the current token and position from device state.
2. Run target embedding and all incremental decoder layers.
3. Read and update self-attention K/V caches in device memory.
4. Attend to device-resident source K/V caches.
5. Run the vocabulary projection.
6. Select argmax for greedy decoding, or top-k plus parent reorder for beam.
7. Append the selected token to a preallocated device token buffer.
8. Update sticky `finished` and `first_eos_position` state.

Submitting only the vocabulary projection is insufficient because the next
step would still depend on host-visible hidden state and token selection.

## Device-owned state

All storage is allocated before inference and remains resident:

```text
current_token          int32   [batch, beam]
position               int32   scalar
finished               bool    [batch, beam]
all_finished           bool    scalar
first_eos_position     int32   [batch, beam], initialized to -1
tokens                  int32   [batch, beam, maximum_steps]
self_kv                 model dtype [layers, 2, batch, beam, heads, maximum_steps, head_dim]
source_kv               model dtype [layers, 2, batch, heads, source_length, head_dim]
hidden/logits/workspace backend-private, preallocated
completion              64-byte aligned shared control record
```

The CPU must not read a buffer while the GPU may still write it. It reads the
small shared completion record only after a Metal completion handler or shared
event establishes visibility. No logits or K/V tensors move to the CPU.

## Preferred execution: GPU while loop

MPSGraph provides `whileWithInitialInputs:before:after:name:`. The strongest
implementation compiles the complete decode loop:

```text
while position < maximum_steps and not all_finished:
    decode_step(state)
```

Loop-carried tensors contain tokens, position, finished flags, scores, and K/V
state. EOS terminates the loop on the GPU, so there is one command submission
and one final host synchronization. The CPU may wait asynchronously and decode
the resulting token sequence after completion.

This path has no burst size or queue-depth tuning. It should be attempted first
with MPSGraph. LiteRT/ML Drift is acceptable only if its delegate lowers the
control-flow and cache-update operations without CPU fallback.

## Streaming fallback: CPU trails a serial GPU queue

If the backend cannot lower the while loop, use a dedicated submission thread
and one serial `MTLCommandQueue`:

```text
submission thread:
    while not stop_requested and submitted < maximum_steps:
        enqueue decode_step using the previous device output as next input

Metal completion callback:
    publish completed_steps and first_eos_position
    notify CPU observer

CPU observer:
    consume newly completed tokens from shared memory
    if first_eos_position >= 0:
        stop_requested = true
```

There is no user-visible or model-specific `k`. Metal's default command queue
already bounds unfinished command buffers to 64. A backend may set a smaller
hardware policy through `newCommandQueueWithMaxCommandBufferCount`, but that is
resource backpressure, not a decoding parameter.

The observer receives a completion for every step. Queue capacity does not
change EOS polling frequency: the CPU can detect EOS on any completed step. It
only bounds how far submission may run ahead of observation. A capacity such as
eight is a backend scheduling policy and is not tuned per model or batch.

Committed command buffers cannot be cancelled reliably. Therefore `finished`
is sticky on the GPU. Every step begins with:

```text
if finished:
    preserve token, score, position, and K/V state
else:
    execute decode step and update state
```

Once EOS is selected, later queued work becomes a semantic no-op. The CPU stops
new submissions when it observes EOS. Overshoot is bounded by the command
queue, does not alter the returned hypothesis, and is truncated at
`first_eos_position`.

## Buffering

Use ping-pong buffers only where an operation cannot safely alias input and
output. A single serial Metal queue preserves dependencies:

```text
step 0: state A -> state B
step 1: state B -> state A
step 2: state A -> state B
```

Token history and K/V caches should use indexed in-place updates rather than
copies or concatenation. Completion records can use a small ring or one sticky
EOS record because tokens already have dedicated positions in the preallocated
sequence buffer.

## Measured scheduling probe

The Kidi Metal benchmark keeps weights and buffers resident, performs no timed
host copies, and places argmax on the GPU. With eight queued invocations and a
CPU observer trailing behind, five-process medians on Apple M5 were:

| Shape | YNNPACK sustained | Metal async trailing | Speedup |
|---|---:|---:|---:|
| Greedy row 1 | 1765.479 us/step | 1585.404 us/step | 1.122x |
| Beam rows 4 | 1988.268 us/step | 1627.138 us/step | 1.228x |

Exact selected-token parity was required. These numbers cover projection,
argmax, asynchronous command submission, persistent buffers, a completion
callback for every queued step, coherent reads of every token buffer, and a
trailing fence. They do not yet prove full-decoder token throughput because the
probe's steps are independent. The full implementation must preserve the
dependency chain described above.

## Acceptance gates

Enable a Metal decoder only when all gates pass:

1. The compiled graph is fully GPU-delegated; no per-step CPU fallback.
2. K/V caches, source memory, hidden state, logits, and token selection stay on
   the device for the full sentence.
3. Timed execution performs no allocation and no bulk host copy.
4. Greedy tokens match the CPU path; scores meet the existing precision bound.
5. EOS produces no cache or score mutations after `first_eos_position`.
6. Full-corpus target tokens/s, excluding model load and graph compilation,
   beats the YNNPACK backend.
7. Profiling confirms the CPU observer is not on the GPU critical path.
