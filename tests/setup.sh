#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

export KIDI_HUB="${KIDI_HUB:-$HOME/.cache/kidi/model-hub}"
environment="$root/.cache/test-venv"
requirements="$root/tests/requirements.txt"
flag="$environment/._OK"

if [[ -x "$environment/bin/python" && -f "$flag" ]] && cmp -s "$requirements" "$flag"; then
    printf 'Skipping Python test setup: %s\n' "$flag"
else
    mkdir -p "$environment"
    rm -f "$flag"
    "${PYTHON:-python3}" -m venv "$environment"
    "$environment/bin/python" -m pip install --disable-pip-version-check -r "$requirements"
    cp "$requirements" "$flag"
fi

"$environment/bin/python" tests/rtg_regression.py setup