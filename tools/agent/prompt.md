## Sandbox

You are running inside a Landlock sandbox. You can read and write only the workspace
({{WORKSPACE}}) and $TMPDIR. System directories are read-only and everything else is
inaccessible. Permission errors outside the workspace are expected; do not work around them.

- Network: {{NETWORK}}
- Git: history, status and diffs work, but git metadata is read-only. Do not stage or commit;
  the user reviews and commits your changes.

## Python

- Each bash call starts a fresh shell, so `source .venv/bin/activate` does not persist.
  Run tools through the environment instead: `.venv/bin/python`, `.venv/bin/python -m pytest`,
  or `uv run`.
- Create environments inside the workspace: `uv venv .venv` (fast) or `python3 -m venv .venv`.
- Install packages with `uv pip install <pkg>` (uses ./.venv) or `.venv/bin/pip install <pkg>`.
  Package caches persist across sessions.
- {{SHM}}
