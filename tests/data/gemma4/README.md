# Gemma 4 Numerical Fixture

This is a randomly initialized four-layer text model, not pretrained model data.
It exercises grouped-query attention, sliding-window masking, global partial
RoPE, shared K/V, double-width MLPs, per-layer embeddings, and logit softcapping.
The C++ test compares full-prefix and incremental logits on CPU and Metal with
independently computed Transformers FP32 outputs.

Generated using PyTorch 2.14.0 and Transformers 5.17.0, seed 173:

```sh
python tests/gemma4_reference.py tests/data/gemma4
```

Normal CTest execution reads the checked-in fixtures and does not require Python,
PyTorch, Transformers, a network connection, or a pretrained checkpoint.