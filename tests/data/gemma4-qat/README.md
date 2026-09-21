# Native Gemma QAT Fixture

Random four-layer model generated with PyTorch 2.14.0 and Transformers 5.17.0,
seed 173. It uses the official quantized Linear and Embedding implementations,
mixed Q2/Q4/Q8 weights, static activation calibration, packed per-layer embeddings,
and explicit cache rounding before the Transformers cache update. No pretrained
weights are included.

```sh
python tests/gemma4_reference.py tests/data/gemma4-qat --qat
```

The C++ test checks all 80 reference logits for prefill, incremental decoding,
chunked prefill, and the optional packed-prefill path on CPU and Metal.
Calibration constants avoid accidental exact half-step dot-product boundaries;
the eager operator test separately checks ties-to-even and saturation directly.
Normal CTest runs do not require Python, a network connection, or a full checkpoint.