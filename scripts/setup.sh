#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
tools="$root/.cache/tools"
export PATH="$tools/bin:$PATH"
export KIDI_HUB="${KIDI_HUB:-$HOME/.cache/kidi/model-hub}"

if [[ "$#" -gt 1 || ( "$#" -eq 1 && "${1:-}" != --tests-only && "${1:-}" != --node-only ) ]]; then
    printf 'Usage: bash scripts/setup.sh [--tests-only | --node-only]\n' >&2
    exit 2
fi

setup_tests() {
    local environment="$root/.cache/test-venv"
    case "${KIDI_TEST_PRESET:-debug}" in
        debug) ;;
        debug-cuda) environment+=-cuda ;;
        *) printf 'Test setup requires the debug or debug-cuda preset.\n' >&2; return 2 ;;
    esac
    local requirements="$root/tests/requirements.txt"
    local flag="$environment/._OK"

    "${PYTHON:-python3}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else "Python 3.12+ is required; set PYTHON to a compatible interpreter.")'
    if [[ -x "$environment/bin/python" && -f "$flag" ]] && cmp -s "$requirements" "$flag" &&
        "$environment/bin/python" -c 'import sys; raise SystemExit(sys.version_info < (3, 12))'; then
        printf 'Skipping Python test setup: %s\n' "$flag"
    else
        rm -rf "$environment"
        "${PYTHON:-python3}" -m venv "$environment"
        "$environment/bin/python" -m pip install --disable-pip-version-check -r "$requirements"
        cp "$requirements" "$flag"
    fi

    setup_node
    node web/tools/prepare-libs.mjs --tests
}

setup_node() (
    if command -v node >/dev/null 2>&1 &&
        node -e 'process.exit(Number(process.versions.node.split(".")[0]) < 24 ? 1 : 0)' &&
        command -v npm >/dev/null 2>&1; then
        return 0
    fi

    case "$(uname -m)" in
        x86_64) arch=x64 ;;
        aarch64|arm64) arch=arm64 ;;
        *) printf 'Node setup supports x64 and ARM64 only.\n' >&2; exit 1 ;;
    esac
    case "$(uname -s)" in
        Linux) platform=linux ;;
        Darwin) platform=darwin ;;
        *) printf 'Node setup supports Linux and macOS only.\n' >&2; exit 1 ;;
    esac

    mkdir -p "$tools/bin"
    temporary=$(mktemp -d)
    trap 'rm -rf "$temporary"' EXIT
    base=https://nodejs.org/dist/latest-v24.x
    curl --fail --location --retry 3 "$base/SHASUMS256.txt" -o "$temporary/SHASUMS256.txt"
    awk -v suffix="-$platform-$arch.tar.gz" '$2 ~ /^node-v[0-9.]+-/ && substr($2, length($2) - length(suffix) + 1) == suffix' \
        "$temporary/SHASUMS256.txt" > "$temporary/checksum"
    if ! read -r checksum archive < "$temporary/checksum" || [[ ! "$checksum" =~ ^[0-9a-f]{64}$ ]]; then
        printf 'No valid Node.js 24 download found for %s/%s.\n' "$platform" "$arch" >&2
        exit 1
    fi
    curl --fail --location --retry 3 "$base/$archive" -o "$temporary/$archive"
    if [[ "$platform" == linux ]]; then
        (cd "$temporary" && sha256sum --check checksum)
    else
        (cd "$temporary" && shasum -a 256 --check checksum)
    fi
    mkdir "$temporary/node"
    tar -xzf "$temporary/$archive" --strip-components=1 -C "$temporary/node"
    rm -rf "$tools/node"
    mv "$temporary/node" "$tools/node"
    ln -sfn "$tools/node/bin/node" "$tools/bin/node"
    ln -sfn "$tools/node/bin/npm" "$tools/bin/npm"
    ln -sfn "$tools/node/bin/npx" "$tools/bin/npx"
)

case "${1:-}" in
    --tests-only) setup_tests; exit 0 ;;
    --node-only) setup_node; exit 0 ;;
esac

mkdir -p "$tools/bin"

case "$(uname -s)" in
    Linux)
        source /etc/os-release
        if [[ "$ID" != ubuntu ]] || ! dpkg --compare-versions "$VERSION_ID" ge 24.04; then
            printf 'Setup requires Ubuntu 24.04 or newer, or macOS.\n' >&2
            exit 1
        fi
        elevate=()
        if [[ "$EUID" -ne 0 ]]; then
            elevate=(sudo -A)
        fi
        "${elevate[@]}" apt-get update
        "${elevate[@]}" apt-get install -y build-essential gcc-14 g++-14 cmake ninja-build ccache \
            git curl ca-certificates python3 python3-venv
        ln -sfn "$(command -v gcc-14)" "$tools/bin/cc"
        ln -sfn "$(command -v g++-14)" "$tools/bin/c++"
        ;;
    Darwin)
        if ! xcode-select -p >/dev/null 2>&1; then
            printf 'Install Xcode Command Line Tools with xcode-select --install, then rerun make setup.\n' >&2
            exit 1
        fi
        if ! command -v brew >/dev/null 2>&1; then
            printf 'Install Homebrew from https://brew.sh, then rerun make setup.\n' >&2
            exit 1
        fi
        brew install cmake ninja ccache python@3.12
        ln -sfn "$(brew --prefix python@3.12)/bin/python3.12" "$tools/bin/python3"
        ;;
    *)
        printf 'Setup supports Ubuntu and macOS only.\n' >&2
        exit 1
        ;;
esac

setup_node

"${PYTHON:-python3}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else "Python 3.12+ is required; set PYTHON to a compatible interpreter.")'
git submodule update --init --recursive --depth 10 --jobs 8
node web/tools/prepare-libs.mjs
printf '\nBuild setup complete. Run make debug or make release; make setup-test installs test tools.\n'