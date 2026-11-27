#!/usr/bin/env bash
# Explicitly download real-model test resources into the local hub, before running tests.
set -euo pipefail

cache="${KIDI_HUB:-$HOME/.cache/kidi/model-hub}"
if [[ "$cache" == "~/"* ]]; then cache="$HOME/${cache:2}"; fi
if ! command -v hf >/dev/null 2>&1; then
    printf 'hf is required; install huggingface-hub and log in for gated models.\n' >&2
    exit 1
fi

hf download google/gemma-4-E2B-it --local-dir "$cache/google/gemma-4-E2B-it"
hf download thinkingmachines/Inkling-2-Mini --local-dir "$cache/thinkingmachines/Inkling-2-Mini"
