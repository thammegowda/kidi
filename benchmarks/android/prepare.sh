#!/usr/bin/env bash
set -euo pipefail

for tool in say afconvert; do
    if ! command -v "$tool" >/dev/null; then
        printf 'Missing %s: speech fixture generation requires macOS.\n' "$tool" >&2
        exit 1
    fi
done

benchmark_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cache_dir="$benchmark_dir/.cache"
mkdir -p "$cache_dir"
temporary_dir="$(mktemp -d "$cache_dir/prepare.XXXXXX")"
trap 'rm -rf "$temporary_dir"' EXIT

say -v Samantha -r 145 -o "$temporary_dir/speech.aiff" \
    'The capital of France is Paris. Binary search repeatedly divides a sorted list in half. This is a speech recognition test on a mobile phone.'
afconvert -f WAVE -d LEI16@16000 -c 1 "$temporary_dir/speech.aiff" "$temporary_dir/speech.wav"
mv "$temporary_dir/speech.wav" "$cache_dir/speech.wav"
shasum -a 256 "$cache_dir/speech.wav"