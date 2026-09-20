# Gemma 4 E2B: Kidi and LiteRT-LM

## Result

Kidi now runs the original `google/gemma-4-E2B-it` Safetensors checkpoint on
YNNPACK CPU and Metal GPU. LiteRT-LM remains faster in these batch-one tests.
Reusing prepared executables improved Kidi's observed short-prompt decode rate
by 1.67x on CPU and 3.16x on GPU, without changing the generated token IDs.

This is **not a precision-matched runtime comparison**. Kidi uses the original
BF16 checkpoint, with FP32 activation interfaces and normalization. LiteRT-LM
uses its released mixed 2/4/8-bit mobile bundle and default activation policy.
The substantial difference in weight traffic is part of the measured result;
the remaining gap cannot be attributed exclusively to runtime overhead. No
offline quantization or rewritten checkpoint is required by Kidi.

## Method

- Hardware: Apple M5, 10 CPU cores, 16 GiB unified memory; macOS 26.6.2.
- CPU threads: four for both runtimes. GPU execution is native Metal for Kidi
  and LiteRT-LM's shipped GPU backend, retaining its default synchronization policy.
- Batch size: one throughout. Context capacity: 2048 tokens.
- Prompts: 128 and 1024 tokens, using synthetic repeated educational text.
  These are controlled workloads, not a representative task or quality suite.
- Greedy generation, one warmup and three measured repetitions per case, in a
  persistent loaded engine. Cases execute sequentially, without concurrent builds
  or inference. Tables report medians from one process per case, not confidence
  intervals across independent process trials.
- Original tokenizer IDs are checked against LiteRT's tokenizer. LiteRT inserts
  BOS during session setup, so its API receives the serialized prompt without
  the literal `<bos>` text. Effective input IDs and prefill counts agree exactly.
- Kidi prefill chunks: 128 tokens. LiteRT chooses its own prefill implementation.
  LiteRT GPU ring buffers are enabled; MTP is explicitly off or on as labelled.
- Both receive a 65-token output budget. Kidi produces the first token from
  prefill logits and times 64 recurrent decoder calls. LiteRT reports 65 tokens
  in its decode phase. Prefill/decode phase boundaries therefore differ slightly.
  Rates below use each runtime's actual measured counts, not a fabricated common
  denominator. Kidi decode timing excludes host token selection; LiteRT uses its
  own native benchmark counters. These are not end-to-end application throughput.
- Loading and cold preparation are outside the warm rates. Kidi's measured warm
  runs after optimization report zero additional operator preparation.
- Raw output, token counts, timings, and process peak RSS are in
  [results.json](results.json). RSS does not uniformly account for GPU allocations
  and must not be read as a comparable total unified-memory footprint.

## Warm Rates

All rates are tokens per second; larger is faster.

### 128 Input Tokens

| Runtime | CPU Prefill | CPU Decode | GPU Prefill | GPU Decode |
|---|---:|---:|---:|---:|
| Kidi, original BF16 | 246.47 | 13.74 | 989.62 | 25.11 |
| LiteRT-LM, mixed low-bit | 712.54 | 47.65 | 2187.49 | 95.61 |
| LiteRT-LM, mixed low-bit + MTP | 715.81 | 43.01 | 2253.66 | 127.56 |

### 1024 Input Tokens

| Runtime | CPU Prefill | CPU Decode | GPU Prefill | GPU Decode |
|---|---:|---:|---:|---:|
| Kidi, original BF16 | 252.13 | 13.67 | 1020.01 | 24.41 |
| LiteRT-LM, mixed low-bit | 573.92 | 47.76 | 2568.08 | 97.98 |
| LiteRT-LM, mixed low-bit + MTP | 570.98 | 49.01 | 2565.55 | 109.19 |

MTP helped GPU decoding for these prompts, but slightly hurt CPU decoding in the
short-prompt case. Its benefit is workload-dependent; it is not a universal
speedup and is not implemented in Kidi by this change.

## Applied Lesson

LiteRT-LM retains prepared execution and resident weights across requests.
Kidi's initial Gemma implementation prepared too many identity-keyed operators:
the combination of layer-specific normalization parameters and prefill/decode
shapes exceeded the bounded prepared cache. Repeated requests consequently
recompiled operators, especially costly on Metal.

Kidi now binds same-device RMSNorm parameters dynamically. Metal linear
operators also bind same-device weights, so layers with the same signature reuse
one executable rather than compiling their large weights as separate constants.
CPU projections and host-constant fallback retain their existing ownership and
identity-keying behavior. Pending Metal work still retains all required owners.
No model graph, lazy execution mode, or backend-specific model equations were added.

| 128-Token Case | Before Decode | After Decode | Observed Ratio |
|---|---:|---:|---:|
| CPU | 8.24 | 13.74 | 1.67x |
| GPU | 7.94 | 25.11 | 3.16x |

GPU prefill rose from 7.89 to 989.62 tokens/s. This large increase primarily
removes repeated preparation, not a claim of a similar matrix-kernel speedup.
Warm preparation fell from 2.1-4.4 seconds per CPU request and 16.0-21.6 seconds
per GPU request to zero in every measured repetition. All generated token IDs
matched before and after, for all three repetitions on each backend.

The pre-optimization binary was preserved locally as
`.cache/kidi-gemma-before-cache`, SHA256
`8cb413aba837e909d150553e1889c34bd619cb840f768f1526035203a261daba`.
Measurements were made on the working tree based on commit
`742dc1fbb9c85be0824d9f10f127f02c636ce14f`; no benchmark results imply an upstream release.

## Correctness and Remaining Work

The independent FP32 Transformers fixture checks grouped-query attention,
local/global masks, partial RoPE, K/V sharing, wider shared-layer MLPs, per-layer
embeddings, and logit softcapping. CPU and Metal agree with its logits for full
prefixes, incremental decoding, and chunked prefill. The real BF16 model produces
the expected Paris answer with identical CPU/GPU token IDs. These checks are not
a broad BF16 numerical or model-quality evaluation against the full reference.

RTG's existing CPU and Metal regression gates remain mandatory. The full unit
suite now has 16 targets, including the small offline Gemma fixture.

Further work should address packed low-bit execution and its quality tradeoffs,
then per-token dispatch/fusion and cache layout. A naive quantization of the
original checkpoint must not be labelled equivalent to LiteRT's released mobile
quantization scheme. Any optional packing should occur during loading or backend
preparation, preserve the original checkpoint, and have separate quality gates.
Speculative decoding should be evaluated separately from ordinary decoding.
No precision-matched comparison or claim that Kidi beats LiteRT-LM is made here.

## Reproduce

Use a Python 3.12 environment for the optional benchmark dependencies:

```sh
python3.12 -m venv .cache/gemma-venv
.cache/gemma-venv/bin/pip install -r benchmarks/gemma/requirements.txt
.cache/gemma-venv/bin/hf download google/gemma-4-E2B-it \
  config.json model.safetensors tokenizer.json \
  --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 \
  --local-dir ../models/gemma-4-E2B-it
.cache/gemma-venv/bin/python tools/configure_gemma4.py ../models/gemma-4-E2B-it
.cache/gemma-venv/bin/hf download litert-community/gemma-4-E2B-it-litert-lm \
  gemma-4-E2B-it.litertlm \
  --revision b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1 \
  --local-dir ../models/gemma-4-E2B-it-litert-lm
cmake --preset release
cmake --build --preset release --target kidi_cli
```

Run the configuration helper only once; it refuses to overwrite existing YAML.
Do not run the following measurements concurrently:

```sh
.cache/gemma-venv/bin/python benchmarks/gemma/run.py \
  --runtime kidi --backend cpu --prefill 128 --decode 64 \
  --output .cache/gemma-bench/kidi-cpu-128.json
```

Repeat with `--backend gpu`, `--runtime litert`, `--runtime litert-mtp`, and
`--prefill 1024`. Defaults are four CPU threads, one warmup, three measurements,
and a 2048-token context. The pinned LiteRT-LM source checkout is tag `v0.17.1`,
commit `5e58e9a0aef7abf7091207a8b1d1063a1c800f08`; execution uses its 0.17.1 Python
distribution and bundled native runtime.