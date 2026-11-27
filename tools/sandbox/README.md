# Agent sandbox

`sandbox.sh` confines an agent, or any command, to one workspace directory with
[Landlock](https://docs.kernel.org/userspace-api/landlock.html). Linux only; works inside unprivileged
containers. Use it with the standalone [`kidi-agent`](../../kidi-agent/README.md) client when you
want kernel-level process confinement. `kidi agent` itself is not OS-sandboxed.

## Quick start

```bash
kidi serve -m @thinkingmachines/Inkling-2-Mini --backend cuda --port 18097   # outside the sandbox
git worktree add ../kidi-task-42                                            # one disposable worktree per task
bin=$(realpath .cache/kidi-agent-build/kidi-agent)                            # standalone client
tools/sandbox/sandbox.sh --ro "$(dirname "$bin")" ../kidi-task-42 -- "$bin"                     # interactive
tools/sandbox/sandbox.sh --ro "$(dirname "$bin")" ../kidi-task-42 -- "$bin" -p "fix the test"   # one-shot
tools/sandbox/sandbox.sh --allow-https ../kidi-task-42 -- bash               # same sandbox, a shell
```

The sandbox allows TCP to port 18097, the default `kidi-agent --base-url` endpoint
(`http://127.0.0.1:18097/v1`). For a server on another port, add `--allow-port N` and pass the
same URL to the agent with `--base-url`. The inference server runs outside the sandbox because the
sandbox has no GPU access.

## Policy

| | Access |
|---|---|
| Workspace, per-run `$HOME`/`$TMPDIR`, package caches (`--cache`, default `~/.cache/kidi-sandbox`) | read/write |
| `/usr`, `/etc`, `/lib*`, `/bin`, `/proc`, `/sys`, `--ro` paths (e.g. the agent binary) | read-only |
| Everything else (your `$HOME`, other repositories, `/tmp`, `/scratch`, host `/dev/shm`) | denied |
| Git metadata (`.git`, worktree gitdir, submodule `.git` files) | read-only: status/diff/log work, staging and commits fail |
| TCP | connect to the model port only (plus 443 with `--allow-https`, `--allow-port N`); no listening |
| Environment | cleared; only sandbox variables are set |

With passwordless sudo, the launcher also runs the command in a private mount namespace: a fresh
`/dev/shm` (so `multiprocessing` works without exposing other jobs' shared memory) and read-only
bind mounts over git entries and this launcher when they are inside the workspace. It then drops
to your user with no capabilities before applying Landlock. Without sudo (`--no-sudo`),
multiprocessing pools and locks fail and those paths stay writable; the launcher warns.

Git metadata is read-only because a writable `.git` file, gitdir, `commondir` or submodule
gitfile lets the agent point git at a config such as `core.fsmonitor` that would execute when
you later run `git diff` to review its work.

## Python

- An existing `WORKSPACE/.venv` is activated (`VIRTUAL_ENV`, `PATH`).
- Without a venv, `python` runs the system `python3` when the system has no `python`.
- `python3 -m venv` and `uv venv` both work; uv prefers the system Python, and `uv venv -p 3.13`
  installs a managed interpreter into the cache (needs `--allow-https`).
- uv and pip caches persist across runs, so `uv pip install --offline` works for packages
  fetched in an earlier `--allow-https` run.

## Limits

- Anything you later run from the workspace (scripts, builds, tests) is agent-written code.
- TCP rules are per port, not per host, and need Landlock ABI 4 (Linux 6.7); UDP and Unix
  sockets are not restricted.
- `/proc` is readable, so other processes are visible (not traceable or writable).
- No GPU access yet: `/dev/nvidia*` is not granted.
- Launch from a checkout the agent cannot modify, or review `tools/sandbox/` changes before
  relaunching: the launcher is only protected while a sandboxed run is active.

## Tests

```bash
python3 -m unittest tools/sandbox/sandbox_test.py
```
