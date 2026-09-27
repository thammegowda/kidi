#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

export KIDI_HUB="${KIDI_HUB:-$HOME/.cache/kidi/model-hub}"
environment="$root/.cache/test-venv"
requirements="$root/tests/requirements.txt"
flag="$environment/._OK"

if [[ -x "$environment/bin/python" && -f "$flag" ]] && cmp -s "$requirements" "$flag" &&
    "$environment/bin/python" -c 'import sys; raise SystemExit(sys.version_info < (3, 12))'; then
    printf 'Skipping Python test setup: %s\n' "$flag"
else
    rm -rf "$environment"
    "${PYTHON:-python3}" -m venv "$environment"
    "$environment/bin/python" -m pip install --disable-pip-version-check -r "$requirements"
    cp "$requirements" "$flag"
fi

"$environment/bin/python" tests/rtg_regression.py setup