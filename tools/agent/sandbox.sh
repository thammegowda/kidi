#!/usr/bin/env bash
# Run the Pi coding agent, or any command, confined to one workspace directory.
#
#   tools/agent/sandbox.sh [options] WORKSPACE [-- PI_ARGS...]
#   tools/agent/sandbox.sh [options] --exec WORKSPACE -- COMMAND [ARGS...]
#
# Options:
#   --allow-https   allow outbound TCP 443 (PyPI, uv, pip); default allows only the model endpoint
#   --allow-port N  allow outbound TCP to port N (repeatable)
#   --ro PATH       extra read-only path (repeatable), e.g. --ro /opt/venv
#   --rw PATH       extra writable path (repeatable)
#   --no-sudo       do not use the private mount namespace even if passwordless sudo works
#   --exec          run COMMAND instead of Pi
#
# Environment: KIDI_AGENT_BASE_URL (default http://127.0.0.1:18096/v1), KIDI_AGENT_HOME,
# KIDI_AGENT_CACHE. See tools/agent/README.md for the policy and its limits.
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime="${KIDI_AGENT_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/kidi-agent}"
cache="${KIDI_AGENT_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/kidi-agent}"
base_url="${KIDI_AGENT_BASE_URL:-http://127.0.0.1:18096/v1}"
python=/usr/bin/python3

die() { echo "sandbox.sh: $*" >&2; exit 2; }
usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

https=0 exec_mode=0 use_sudo=auto
ports=() extra_ro=() extra_rw=()
while (($#)); do
    case "$1" in
        --allow-https) https=1 ;;
        --allow-port) [[ ${2:-} =~ ^[0-9]+$ ]] || die "--allow-port needs a number"; ports+=("$2"); shift ;;
        --ro) extra_ro+=("$(realpath -e "${2:?--ro needs a path}")"); shift ;;
        --rw) extra_rw+=("$(realpath -e "${2:?--rw needs a path}")"); shift ;;
        --no-sudo) use_sudo=0 ;;
        --exec) exec_mode=1 ;;
        -h | --help) usage ;;
        -*) echo "sandbox.sh: unknown option $1" >&2; usage 2 ;;
        *) break ;;
    esac
    shift
done
(($#)) || usage 2
ws="$(realpath -e "$1")"; shift
[[ -d $ws ]] || die "workspace is not a directory: $ws"
[[ ${1:-} == -- ]] && shift
if ((exec_mode)); then
    (($#)) || die "--exec needs a command"
else
    [[ -x $runtime/pi/node_modules/.bin/pi ]] || die "Pi is not installed; run tools/agent/setup.sh"
fi

inside_ws() { [[ $1 == "$ws" || $1 == "$ws/"* ]]; }

model_port="$("$python" -c 'import sys, urllib.parse as u; p = u.urlsplit(sys.argv[1]); print(p.port or (443 if p.scheme == "https" else 80))' "$base_url")"
ports+=("$model_port")
((https)) && ports+=(443)

# Git metadata is read-only: a writable .git, gitdir, commondir or submodule gitfile lets the
# agent point git at a config (e.g. core.fsmonitor) that runs when the user later reviews.
ro=() protect=()
if gitdir="$(git -C "$ws" rev-parse --absolute-git-dir 2>/dev/null)"; then
    common="$(realpath -e "$(git -C "$ws" rev-parse --path-format=absolute --git-common-dir)")"
    for dir in "$gitdir" "$common"; do inside_ws "$dir" || ro+=("$dir"); done
    toplevel="$(git -C "$ws" rev-parse --show-toplevel)"
    entries=("$toplevel/.git")
    while IFS= read -r path; do entries+=("$toplevel/$path/.git"); done \
        < <(git -C "$toplevel" submodule foreach --quiet --recursive 'echo "$displaypath"' 2>/dev/null)
    for entry in "${entries[@]}"; do
        [[ -L $entry ]] && die "refusing to run with a symlinked git entry: $entry"
        [[ -e $entry ]] && inside_ws "$entry" && protect+=("$entry")
    done
fi
# The launcher itself must not be editable by the agent it launches.
inside_ws "$here" && protect+=("$here")

if [[ $use_sudo == auto ]]; then
    use_sudo=0
    command -v setpriv >/dev/null && command -v unshare >/dev/null && sudo -n true 2>/dev/null && use_sudo=1
fi

mkdir -p "$cache/uv" "$cache/pip" "$cache/uv-python" "$runtime/run" "$runtime/sessions"
priv="$(mktemp -d "$runtime/run/XXXXXX")"
trap 'rm -rf "$priv"' EXIT
mkdir -p "$priv/home" "$priv/tmp" "$priv/agent"
for key in user.name user.email; do
    value="$(git config --global --get "$key" 2>/dev/null || true)"
    [[ -n $value ]] && git config --file "$priv/home/.gitconfig" "$key" "$value"
done

if ((https)); then
    network="outbound HTTPS is allowed (PyPI, uv, pip)."
else
    network="no internet; only the model endpoint is reachable. \`uv pip install --offline\` works for cached packages; otherwise ask the user to relaunch with --allow-https."
fi
if ((use_sudo)); then
    shm="\`multiprocessing\` works (private /dev/shm)."
else
    shm="/dev/shm is unavailable, so \`multiprocessing\` pools and locks fail; prefer single-process runs (e.g. no pytest -n)."
fi
prompt="$(<"$here/prompt.md")"
prompt="${prompt//'{{WORKSPACE}}'/$ws}"
prompt="${prompt//'{{NETWORK}}'/$network}"
printf '%s\n' "${prompt//'{{SHM}}'/$shm}" > "$priv/prompt.md"
"$python" - "$here/pi/models.json" "$priv/agent/models.json" "$base_url" <<'EOF'
import json, sys
config = json.load(open(sys.argv[1]))
for provider in config["providers"].values():
    provider["baseUrl"] = sys.argv[3]
json.dump(config, open(sys.argv[2], "w"), indent=2)
EOF
cp "$here/pi/settings.json" "$priv/agent/"

path="$runtime/node/bin:/usr/local/bin:/usr/bin:/bin"
env_vars=(
    HOME="$priv/home" TMPDIR="$priv/tmp" TERM="${TERM:-xterm-256color}" LANG="${LANG:-C.UTF-8}"
    PI_CODING_AGENT_DIR="$priv/agent" PI_OFFLINE=1
    UV_CACHE_DIR="$cache/uv" UV_PYTHON_INSTALL_DIR="$cache/uv-python" UV_PYTHON_PREFERENCE=system
    PIP_CACHE_DIR="$cache/pip" PIP_DISABLE_PIP_VERSION_CHECK=1
)
[[ -n ${COLORTERM:-} ]] && env_vars+=(COLORTERM="$COLORTERM")
((https)) || env_vars+=(UV_OFFLINE=1 PIP_RETRIES=0)
if [[ -x $ws/.venv/bin/python ]]; then
    env_vars+=(VIRTUAL_ENV="$ws/.venv")
    path="$ws/.venv/bin:$path"
fi
# uv hardlinks from its cache; across filesystems that only produces warnings.
[[ "$(stat -c %d "$cache/uv")" == "$(stat -c %d "$ws")" ]] || env_vars+=(UV_LINK_MODE=copy)
env_vars+=(PATH="$path")

rw=("$ws" "$priv" "$cache" "$runtime/sessions" /dev/null /dev/zero /dev/full "${extra_rw[@]}")
((use_sudo)) && rw+=(/dev/shm)
ro+=(/usr /bin /sbin /lib /lib32 /lib64 /etc /proc /sys /dev/urandom /dev/random
    "$runtime/node" "$runtime/pi" "${extra_ro[@]}")
landlock=("$python" "$here/landlock_run.py")
for p in "${rw[@]}"; do landlock+=(--rw "$p"); done
for p in "${ro[@]}"; do landlock+=(--ro "$p"); done
for p in "${ports[@]}"; do landlock+=(--tcp-connect "$p"); done

if ((exec_mode)); then
    command=("$@")
else
    command=("$runtime/pi/node_modules/.bin/pi" --session-dir "$runtime/sessions"
        --append-system-prompt "$priv/prompt.md" "$@")
fi
inner=(env -i "${env_vars[@]}" "${landlock[@]}" -- "${command[@]}")

echo "sandbox: rw $ws; tcp ${ports[*]}; $( ((use_sudo)) && echo "private /dev/shm, git metadata read-only" || echo "no mount namespace") " >&2
if ((!use_sudo && ${#protect[@]})); then
    echo "sandbox: warning: without sudo, ${#protect[@]} protected paths (git metadata such as ${protect[0]}, or this launcher) stay writable by the agent" >&2
fi

cd "$ws"
status=0
if ((use_sudo)); then
    # As root in a private mount namespace: fresh /dev/shm, read-only binds, then drop to the
    # calling user with no capabilities before applying Landlock.
    sudo -n unshare --mount --propagation private -- sh -c '
        set -e
        uid=$1 gid=$2 count=$3; shift 3
        mount -t tmpfs -o mode=1777,nosuid,nodev tmpfs /dev/shm
        while [ "$count" -gt 0 ]; do
            mount --bind "$1" "$1"
            mount -o remount,bind,ro "$1"
            shift; count=$((count - 1))
        done
        exec setpriv --reuid="$uid" --regid="$gid" --init-groups \
            --inh-caps=-all --bounding-set=-all -- "$@"
    ' sh "$(id -u)" "$(id -g)" "${#protect[@]}" "${protect[@]}" "${inner[@]}" || status=$?
else
    "${inner[@]}" || status=$?
fi
exit "$status"
