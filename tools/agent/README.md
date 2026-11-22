# Coding-agent sandbox

Runs the [Pi](https://github.com/earendil-works/pi) coding agent (pinned to v0.83.0, the
version cited by the Inkling-2-Mini model card) against an OpenAI-compatible endpoint, confined
by [Landlock](https://docs.kernel.org/userspace-api/landlock.html) to one workspace directory.
Linux only; works inside unprivileged containers.

```bash
tools/agent/setup.sh                                  # Node 22 + Pi into ~/.local/share/kidi-agent
VLLM_ENV=... CUDA_VISIBLE_DEVICES=6 tools/agent/serve-inkling-vllm.sh   # reference endpoint
git worktree add ../kidi-task-42                      # one disposable worktree per task
tools/agent/sandbox.sh ../kidi-task-42                # interactive Pi
tools/agent/sandbox.sh ../kidi-task-42 -- -p "fix the failing test"      # one-shot
tools/agent/sandbox.sh --allow-https ../kidi-task-42  # also allow PyPI downloads
tools/agent/sandbox.sh --exec ../kidi-task-42 -- bash # same sandbox, no agent
```

`KIDI_AGENT_BASE_URL` selects the endpoint (default `http://127.0.0.1:18096/v1`). Pi model
settings are in [pi/models.json](pi/models.json). Pi thinking levels are sent as Inkling's
`reasoning_effort`; `off` is mapped to `none`, since omitting it would select vLLM's 0.9 default.

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
