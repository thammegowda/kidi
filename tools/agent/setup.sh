#!/usr/bin/env bash
# Install the pinned Node.js and Pi runtime used by tools/agent/sandbox.sh.
# Installs outside the repository (default ~/.local/share/kidi-agent) so a sandboxed
# agent working in this checkout cannot modify the runtime it is launched with.
set -euo pipefail

node_version="v22.23.5"
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime="${KIDI_AGENT_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/kidi-agent}"

case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) platform="linux-x64" ;;
    Linux-aarch64) platform="linux-arm64" ;;
    *) echo "setup.sh: the Landlock sandbox requires Linux on x86_64 or aarch64" >&2; exit 1 ;;
esac

mkdir -p "$runtime"
if [[ "$("$runtime/node/bin/node" --version 2>/dev/null || true)" != "$node_version" ]]; then
    archive="node-$node_version-$platform.tar.xz"
    download="$(mktemp -d)"
    trap 'rm -rf "$download"' EXIT
    curl -fsSL "https://nodejs.org/dist/$node_version/$archive" -o "$download/$archive"
    curl -fsSL "https://nodejs.org/dist/$node_version/SHASUMS256.txt" -o "$download/SHASUMS256.txt"
    (cd "$download" && grep " $archive\$" SHASUMS256.txt | sha256sum -c --quiet -)
    tar -xJf "$download/$archive" -C "$download"
    rm -rf "$runtime/node"
    mv "$download/node-$node_version-$platform" "$runtime/node"
fi

mkdir -p "$runtime/pi"
cp "$here/package.json" "$here/package-lock.json" "$runtime/pi/"
(cd "$runtime/pi" && PATH="$runtime/node/bin:$PATH" npm ci --ignore-scripts --no-fund --no-audit --loglevel=error)

echo "node $("$runtime/node/bin/node" --version), pi $(PATH="$runtime/node/bin:$PATH" "$runtime/pi/node_modules/.bin/pi" --version) in $runtime"
