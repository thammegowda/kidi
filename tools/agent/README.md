# Coding-agent sandbox

Runs the [Pi](https://github.com/earendil-works/pi) coding agent (pinned to v0.83.0) against an
OpenAI-compatible endpoint, confined
by [Landlock](https://docs.kernel.org/userspace-api/landlock.html) to one workspace directory.
Linux only; works inside unprivileged containers.

## Quick start (`kidi-code`)

From any project directory, `tools/agent/kidi-code` automatically installs Pi on first run, starts
`kidi serve` using your cached model (e.g. `Inkling-2-Mini`) or reuses an already-running server
on port `18097`, and launches Pi inside the Landlock sandbox:

```bash
cd /path/to/your/project
/path/to/kidi/tools/agent/kidi-code                        # interactive Pi in current dir
/path/to/kidi/tools/agent/kidi-code -p "fix the test"      # one-shot
/path/to/kidi/tools/agent/kidi-code --keep-server          # leave server running
```

Useful options: `-w/--workspace DIR`, `-m/--model MODEL`, `--backend cuda[:INDEX]|auto`,
`--port N`, `--context-size N`, `--max-new-tokens N`, `--no-sandbox` (also used on macOS).

## Manual launcher (`sandbox.sh`)

```bash
tools/agent/setup.sh                                  # Node 22 + Pi into ~/.local/share/kidi-agent
KIDI_MODEL=/path/to/model CUDA_VISIBLE_DEVICES=7 tools/agent/serve-kidi.sh # kidi endpoint (port 18097)
git worktree add ../kidi-task-42                      # one disposable worktree per task
KIDI_AGENT_MODEL_ID=your-served-name tools/agent/sandbox.sh ../kidi-task-42 # if model name differs
tools/agent/sandbox.sh ../kidi-task-42                # interactive Pi
tools/agent/sandbox.sh ../kidi-task-42 -- -p "fix the failing test"      # one-shot
tools/agent/sandbox.sh --allow-https ../kidi-task-42  # also allow PyPI downloads
tools/agent/sandbox.sh --exec ../kidi-task-42 -- bash # same sandbox, no agent
VLLM_ENV=... CUDA_VISIBLE_DEVICES=6 tools/agent/serve-inkling-vllm.sh    # optional Inkling reference (port 18096)
```

`KIDI_AGENT_BASE_URL` selects the endpoint (default `http://127.0.0.1:18097/v1`); set
`KIDI_AGENT_MODEL_ID` if the server advertises a different model name. The model path, served name,
context/cache size and token limits for `serve-kidi.sh` can be set with `KIDI_MODEL`,
`KIDI_SERVED_MODEL_NAME`, `KIDI_CONTEXT_SIZE`, `KIDI_CACHE_TOKENS`, and `KIDI_MAX_NEW_TOKENS`.
Pi's provider defaults are in [pi/models.json](pi/models.json). The CUDA build runs without CUTLASS;
CUTLASS is optional and used only for the SM90 int8 GEMM path.

## Policy

| | Access |
|---|---|
| Workspace, per-run `$HOME`/`$TMPDIR`, package caches (`~/.cache/kidi-agent`), Pi sessions | read/write |
| `/usr`, `/etc`, `/lib*`, `/bin`, `/proc`, `/sys`, Node and Pi runtime, `--ro` paths | read-only |
| Everything else (your `$HOME`, other repositories, `/tmp`, `/scratch`, host `/dev/shm`) | denied |
| Git metadata (`.git`, worktree gitdir, submodule `.git` files) | read-only: status/diff/log work, staging and commits fail |
| TCP | connect to the model port only (plus 443 with `--allow-https`, `--allow-port N`); no listening |
| Environment | cleared; only sandbox variables are set |

With passwordless sudo, the launcher also runs the agent in a private mount namespace: a fresh
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
- The appended system prompt ([prompt.md](prompt.md)) tells the model that each bash call is a
  fresh shell and how to use the environment.

## Limits

- Anything you later run from the workspace (scripts, builds, tests) is agent-written code.
- TCP rules are per port, not per host, and need Landlock ABI 4 (Linux 6.7); UDP and Unix
  sockets are not restricted.
- `/proc` is readable, so other processes are visible (not traceable or writable).
- No GPU access yet: `/dev/nvidia*` is not granted.
- Pi runs offline (`PI_OFFLINE=1`), so it cannot download its helper binaries; install `fd`
  and `rg` on the host for its file-search tools.
- Launch from a checkout the agent cannot modify, or review `tools/agent/` changes before
  relaunching: the launcher is only protected while a sandboxed run is active.

## Tests

```bash
python3 -m unittest tools/agent/sandbox_test.py
```
