#!/usr/bin/env bash
# Serve a local model with kidi's OpenAI-compatible server for tools/agent/sandbox.sh.
#   KIDI_MODEL=/path/to/model CUDA_VISIBLE_DEVICES=7 tools/agent/serve-kidi.sh
#   KIDI_AGENT_BASE_URL=http://127.0.0.1:18097/v1 tools/agent/sandbox.sh ../kidi-task-42
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the GPU to use}"
: "${KIDI_MODEL:?set KIDI_MODEL to the model directory}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ -f $KIDI_MODEL/config.json ]] || { echo "missing model config: $KIDI_MODEL" >&2; exit 1; }
context="${KIDI_CONTEXT_SIZE:-32768}"

exec "${KIDI:-$root/build-release-cuda/kidi}" serve -m "$KIDI_MODEL" \
    --backend cuda \
    --host 127.0.0.1 \
    --port "${KIDI_PORT:-18097}" \
    --served-model-name "${KIDI_SERVED_MODEL_NAME:-kidi-local}" \
    --context-size "$context" \
    --cache-tokens "${KIDI_CACHE_TOKENS:-$context}" \
    --prefill-chunk-size "${KIDI_PREFILL_CHUNK_SIZE:-512}" \
    --max-new-tokens "${KIDI_MAX_NEW_TOKENS:-4096}"
