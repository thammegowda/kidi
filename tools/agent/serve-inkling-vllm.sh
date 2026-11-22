#!/usr/bin/env bash
# Serve Inkling-2-Mini with vLLM as the reference endpoint for tools/agent/sandbox.sh.
#   VLLM_ENV=/path/to/venv CUDA_VISIBLE_DEVICES=6 tools/agent/serve-inkling-vllm.sh
# Install (Inkling needs a vLLM nightly; validated with 0.34.0rc1.dev1130+g8c17e4692):
#   uv venv -p 3.12 "$VLLM_ENV"
#   uv pip install -p "$VLLM_ENV" vllm --extra-index-url https://wheels.vllm.ai/nightly --torch-backend=auto
set -euo pipefail
: "${VLLM_ENV:?set VLLM_ENV to the vLLM virtual environment}"
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the GPU to use}"
export VLLM_NO_USAGE_STATS=1 HF_HUB_DISABLE_TELEMETRY=1

exec "$VLLM_ENV/bin/vllm" serve thinkingmachines/Inkling-2-Mini \
    --revision b85c3cfa615fc413cc9af3379f491b963f3b586a \
    --host 127.0.0.1 \
    --port "${INKLING_PORT:-18096}" \
    --max-model-len "${INKLING_MAX_LEN:-65536}" \
    --gpu-memory-utilization "${INKLING_GPU_UTIL:-0.92}" \
    --max-num-seqs "${INKLING_MAX_SEQS:-1}" \
    --reasoning-parser inkling \
    --enable-auto-tool-choice \
    --tool-call-parser inkling
