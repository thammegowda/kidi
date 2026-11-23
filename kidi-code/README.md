# Kidi Code

A small C++23 coding-agent library and CLI. No Node, Pi, curl or TLS dependency.
Inference uses Kidi's Chat Completions API over plain HTTP; external model
providers and HTTPS are not supported.

## Quick start

```bash
kidi code
```

The first task starts a detached Kidi inference daemon; matching sessions reuse
it. The daemon handles up to two concurrent generations and exits 10 minutes
after the last client session closes. An open interactive session keeps it alive.

The default model is `@google/gemma-4-E2B-it-qat-mobile-transformers`, with a
cached `@google/gemma-4-E2B-it` as fallback. Weights are not bundled: missing
weights are downloaded through the Kidi model hub on first use.

```bash
kidi code --model /path/to/checkpoint --backend cuda:0
kidi code --workspace /path/to/project -p "Explain this project"
kidi code --config /trusted/agent.yaml
```

`KIDI_MODEL` supplies the default model; `--backend` defaults to `auto`.
The command is built by default on Linux/macOS (`KIDI_BUILD_CODE=OFF` disables
it). There is no in-process inference mode.

Daemon port and PID are published atomically in:

```text
$XDG_RUNTIME_DIR/kidi-code-<uid>/<server-key>/server.json
```

Without a usable private runtime directory, the root is `/tmp/kidi-code-<uid>`.
`serve.log` sits beside the state file. The key includes the binary, model,
model cache and backend, so a rebuilt executable starts a fresh daemon.
Clients check readiness and PID before reuse. A startup lock prevents duplicate
cold starts; stale metadata is replaced after a crash.

## Standalone client

`kidi-code` connects to an already running Kidi server; it does not start one.

```bash
cmake -S kidi-code -B .cache/kidi-code-build
cmake --build .cache/kidi-code-build -j

kidi serve --model /path/to/checkpoint --port 18097
# In another terminal:
.cache/kidi-code-build/kidi-code --base-url http://127.0.0.1:18097/v1
```

`KIDI_AGENT_BASE_URL` sets the default endpoint. There are no API keys or
external-provider model-selection options. Use only trusted Kidi endpoints:
prompts and tool results are sent unencrypted. Redirects and retries are disabled.
Kidi's ordinary `serve --max-active` remains available for manually managed servers.

## Tools and permissions

Both CLIs default to interactive mode, with `/exit` and `/reset`.
`-p` runs one task. Model and tool output is escaped before terminal display.
Failed turns report errors without closing an interactive session.

| Tool | Operation |
| --- | --- |
| `read_file` | Read a bounded UTF-8 file |
| `list_files` | List one directory; `.` is the workspace root |
| `write_file` | Atomically replace or create a UTF-8 file |
| `make_directory` | Create one directory |
| `run_command` | Run `/bin/sh -c` in the workspace root |

| `--permissions` | File writes | Commands |
| --- | --- | --- |
| `read-only` | Denied | Denied |
| `ask` (default) | Ask | Ask |
| `workspace-write` | Allowed | Ask |
| `full-access` | Allowed | Allowed |

One-shot tasks never read approvals from stdin: operations requiring approval
are denied. Use `full-access` for unattended commands only in a trusted,
disposable environment. Command results include exit status and combined
stdout/stderr, capped at 32 KiB. Commands time out after 60 seconds.

**Permissions are not an OS sandbox.** File handlers reject absolute paths,
`..`, `.git`, symlinks and special files using descriptor-relative access.
`run_command` has the user's full privileges and can access paths outside the
workspace. Remove it from the YAML registry to disable it. The host must control
workspace directory topology; use the existing [Pi/Landlock launcher](../tools/agent/README.md)
when kernel-level process confinement is required.

## Prompts and registry

[`agent.yaml`](agent.yaml) is embedded in the binary and used by default.
`--config PATH` replaces it completely; workspace configs are never discovered
automatically. Keep overrides outside agent-writable directories.

```yaml
version: 1
prompts:
  system: You are a coding assistant. Inspect files before editing.
  continue: Continue. Answer using the tool results above.
tools:
  - name: inspect
    handler: read_file
    description: Read a workspace file.
    parameters:
      type: object
      properties:
        path: {type: string}
      required: [path]
      additionalProperties: false
```

`handler` selects a compiled implementation; renaming a tool cannot change its
capability or grant permission. Remove entries to disable tools; use `tools: []`
for none. Unknown fields, duplicate names and unbound handlers are errors.
Handlers validate their arguments; the runtime is not a full JSON Schema validator.

Optional `prompts.continue` is sent once if a model replies silently after tool
results. It is not retained in history; omit it to disable this retry.

## Embedding

Add this directory with CMake and link `kidi::code` for the transport-independent
agent loop, YAML configuration and workspace handlers. Link `kidi::code_http`
for the plain-HTTP Kidi provider. Set `KIDI_CODE_BUILD_CLI=OFF` for library-only builds.

A `Provider` receives message/tool spans and a stop token, returning
`Result<Message>`. `run_turn` maintains history, executes tools serially,
enforces permission callbacks and bounds model calls, tool calls and output.
Failures preserve history and do not roll back side effects.

Hosts can register `Tool` values with their own capability and handler, then
bind YAML entries with `configure_tools`. Host callbacks must validate inputs
and honor cancellation. Process isolation, MCP, streaming UI, persistent
sessions, compaction and subagents are not implemented.

## Testing

```bash
ctest --test-dir .cache/kidi-code-build --output-on-failure
# In a root build with tests enabled:
ctest --test-dir .cache/kidi-code-host-build -R 'Code|ResponseTest|kidi_code_managed' --output-on-failure
```

Unit tests cover agent permissions/limits, YAML, workspace operations and HTTP
tool round trips. The managed-daemon test uses a tiny CPU model to verify
concurrent startup, reuse, config override and crash recovery without downloads.
Set `KIDI_TEST_REAL_IDLE=1` when running that test to check the full 10-minute
idle shutdown on the wall clock.

Codex's Apache-licensed registry and safety architecture informed the design.
This is original C++; no Codex source was copied.
