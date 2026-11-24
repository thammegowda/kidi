# Kidi Code

A small C++23 coding-agent library and CLI. No Node, curl or TLS dependency.
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

`KIDI_MODEL` supplies the default model; `--backend auto` prefers available
CUDA, then Metal, then CPU. Use `--backend ynnpack` to force CPU or
`--backend cuda:INDEX` to select a CUDA device. `kidi --version` reports
backend availability and the CUDA toolkit version in CUDA builds.
The command is built by default on Linux/macOS (`KIDI_BUILD_CODE=OFF` disables
it). There is no in-process inference mode.

Inkling-2-Mini also runs in a CPU-only build:

```bash
./build-release/kidi code --model @thinkingmachines/Inkling-2-Mini --backend ynnpack --restart
```

CPU execution automatically converts the checkpoint's NVFP4/MXFP8 experts and dense projections to
row-scaled INT8 (W8A8 matmuls); CPU KV caches use group-32 signed INT8 with FP32 scales.
`-j/--threads` sets the server's CPU threads
(default: up to 16). Conversion happens at each server start, not per request.
This needs more RAM than CUDA's native compact checkpoint mode. See
[CPU inference details](../docs/getting-started.md#inkling-2-on-cpu) for precision options.

Daemon port and PID are published atomically in:

```text
$XDG_RUNTIME_DIR/kidi-code-<uid>/<server-key>/server.json
```

Without a usable private runtime directory, the root is `/tmp/kidi-code-<uid>`.
`serve.log` sits beside the state file. The key includes the binary, model, model cache, backend and CUDA visibility
for `auto` and CUDA backends, plus the requested context size, so a rebuilt executable or a change to
`CUDA_VISIBLE_DEVICES` starts a fresh daemon.
Clients check readiness and PID before reuse. A startup lock prevents duplicate
cold starts; stale metadata is replaced after a crash.
`--restart` stops the matching daemon (SIGTERM, then SIGKILL after 30 seconds)
and starts a fresh one, e.g. after an inference failure. Other sessions using
that daemon lose their connection. Daemons with other keys are left alone.

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

Both CLIs default to interactive mode. On Linux/macOS, the pinned MIT-licensed
FTXUI terminal UI provides arrow-key editing, Home/End, task history, multiline
bracketed paste, PageUp/PageDown transcript scrolling and Tab completion for slash
commands. History stays in memory (up to 100 tasks); approval answers are never
stored in history. The UI uses the terminal's alternate screen. Piped input,
redirected output and `TERM=dumb` use plain line input.

`/help` lists commands. `/exit` quits; `/reset` clears the conversation but keeps
session approvals. Ctrl-C clears the input or cancels the current turn, including
pending HTTP requests and running shell commands; the session remains usable.
Ctrl-D exits on an empty input line, including at an approval prompt.
`--color auto|always|never` controls styling; auto respects `NO_COLOR` and
`TERM=dumb`. Valid UTF-8 is preserved; untrusted terminal controls are escaped.
`-p` runs one task. Failed turns report errors without closing an interactive session.

### Terminal presentation

User input is bright green, tools are cyan with compact argument previews, model
thinking is dim grey, and final answers are bright white. Commands are displayed
as `$ command`, not JSON; other tools show named arguments. Long activity and
thinking previews are abbreviated, but approval prompts show **all** arguments.
Blank lines separate user turns, not every tool or label. Short tool calls fit
on one line, thinking previews are limited to two lines/300 bytes, and completed
tools show their elapsed time. Approval prompts still show every argument.
Interactive model replies stream as they arrive, including actual model-provided
`reasoning_content` (within the preview limit); final answers are not printed
again at completion. Complete messages are still saved in the transcript.
One-shot/piped answers and compaction summaries remain non-streaming.

A grey, width-aware status bar is pinned to the bottom row of the interactive
screen, showing model name, context percentage/token count, token speed,
prefill time and the current phase. Numeric fields are fixed-width so the bar
does not jitter; the phase stays right-aligned and is never clipped. During
startup, context checks, generation, compaction, and tool calls it updates an
elapsed timer every 100 ms, including when no output is arriving.
Approval waits are explicitly labelled. Timers indicate activity, not measured
prefill progress or a prediction of time remaining.

```text
Inkling-2-Mini | ctx  12.0%  29860/248832 |  103.9 tok/s | prefill   0.2s | Ready
```

Context uses the server's existing exact preflight counts and response usage.
With a Kidi server, speed is the server's decode rate and `prefill` is the
time spent processing the prompt for the last model call. Other OpenAI servers
show `tok/s req`: generated tokens divided by request wall time, including
prefill and HTTP latency. Missing usage is shown as `--`, never estimated.

The managed server keeps the last request's KV state, so each agent turn
prefills only what was appended since the previous prompt rather than the whole
conversation. Plain `kidi serve` enables the same reuse with
`--prefix-cache-mib`. The transcript scrolls within the UI; use PageUp/PageDown to
review earlier output. The layout adapts to terminal resizes. Exiting restores
the terminal and returns to its prior screen.
Piped/one-shot output has no status bar or answer heading.

### Local transcripts

Both interactive and one-shot sessions automatically append JSONL records under:

```text
$XDG_CACHE_HOME/kidi/code/chats/<workspace-key>/session-<timestamp>.jsonl
```

The cache defaults to `~/.cache`. Each canonical workspace path has its own
stable key, so other projects' conversations are separate and symlink aliases
share the same workspace. The first interactive turn prints its transcript path.
Files are private (0600 on POSIX), flushed after every record, and never rewritten
as a growing JSON array. Records include session metadata, full messages and
reasoning, tool calls/results, errors, reset markers and compaction checkpoints.
Reset/compaction changes the live context, not the saved audit trail. A reader can
ignore an incomplete final line after a crash; complete earlier lines remain valid.
No chat picker or resume command is implemented yet.

Transcripts may contain source code, command output or secrets from tools.
They stay local but are not encrypted. Use `--no-save` to disable persistence;
save failures are reported and stop the session rather than silently losing data.

## Long sessions

Both clients automatically compact older conversation before it fills the
model context or the 4 MiB history buffer. A Kidi-specific
`POST /v1/chat/context` preflight counts the actual chat template, tool schemas
and tokenizer tokens without inference. This is not a bytes-to-tokens estimate.
Use an updated Kidi server with an updated standalone client.

Compaction keeps system instructions and the latest user request verbatim,
merges older messages into a continuation handoff, and retains the most recent
complete tool exchange when it fits. The handoff is one summary call when the
transcript fits, otherwise bounded fragments. Summarization exposes no tools,
requests no reasoning where supported, and accepts a length-truncated summary
that has content. It never executes tools or changes session permission grants.
If summarization fails, older history is dropped with a note telling the model
to re-read files; cancellation keeps the original history.

Summaries are lossy; important permanent rules belong in `prompts.system`.
Each turn allows at most three compactions. Beyond that, the turn stops with
"context is too small for this task" instead of looping, and suggests a larger
`--context-size`. Normal execution remains bounded by `--max-steps` and
`--max-tool-calls`. `/context` displays exact usage; `/compact` requests
compaction manually. Activity shows `[context] ... -> ... prompt tokens`.

### Context size

`kidi code` sizes the managed server context from free device memory
(three quarters of it, after weights load), up to the model's maximum position
limit. For example, Inkling-2-Mini on an 80 GB H100 gets about 248K tokens;
the log shows `context: N tokens (G GiB free on cuda:N)`. KV memory is reserved
in power-of-two buckets as the conversation grows, so short sessions stay small.
`--context-size` overrides the automatic choice. Contexts under 32K print a
`[context] warning` at startup, because real coding tasks compact constantly.
Long prompts cost prefill time: every model call re-reads the whole prompt.
For example, Inkling-2-Mini on H100 prefills about 2K tokens/s, so a 100K-token
prompt takes about 50 s per call. For the standalone client, configure context
on `kidi serve` instead.

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

Interactive approval offers **y** (once), **a** (any arguments to this named tool
for this session), or **n** / empty input (deny). A session approval covers later
commands only if explicitly granted for `run_command`; it does not approve other
tools or override denied capabilities. There is no automatic shell-command
allowlist. `/permissions` shows the mode and grants; `/permissions reset` revokes
session grants without changing the CLI mode. Grants are never saved to disk.

One-shot tasks never read approvals from stdin: operations requiring approval
are denied. Use `full-access` for unattended commands only in a trusted,
disposable environment. Command results include exit status and combined
stdout/stderr, capped at 32 KiB. Commands time out after 60 seconds.

**Permissions are not an OS sandbox.** File handlers reject absolute paths,
`..`, `.git`, symlinks and special files using descriptor-relative access.
`run_command` has the user's full privileges and can access paths outside the
workspace. Remove it from the YAML registry to disable it. The host must control
workspace directory topology; run the standalone client under the [Landlock sandbox](../tools/agent/README.md)
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
  compact: Summarize goals, constraints, findings, completed actions and outstanding work. Do not execute tools.
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
Optional `prompts.compact` controls the handoff instructions. The embedded
configuration enables it; omitting it from an explicit override disables
automatic and manual compaction.

### Sampling

Greedy decoding (temperature 0) can trap a model in a loop that repeats the same
tool call. The embedded configuration therefore samples agent turns:

```yaml
sampling:
  temperature: 0.6   # 0 = greedy; range [0, 2]
  top_p: 0.95        # (0, 1]
  top_k: 20          # 0 = disabled
```

Omitting `sampling:` means greedy. `--temperature T` overrides the configured
temperature (`--temperature 0` restores greedy). Compaction summaries are always
greedy. `top_p` is sent only when below 1 and `top_k` only when positive, because
some OpenAI-compatible servers reject `top_k`.

`kidi serve` accepts `temperature`, `top_p`, `top_k` and `seed` on
`/v1/chat/completions`; the same seed reproduces the same output. Sampled
requests skip speculative decoding and device-side token chaining, so they decode
roughly 10% slower than greedy requests.

## Embedding

Add this directory with CMake and link `kidi::code` for the transport-independent
agent loop, YAML configuration and workspace handlers. Link `kidi::code_http`
for the plain-HTTP Kidi provider. Set `KIDI_CODE_BUILD_CLI=OFF` for library-only builds.

A `Provider` wraps a completion callback receiving message/tool spans and a stop
token, returning `Result<Message>`. Its optional `context` and `summarize`
callbacks enable compaction; `summary_context` is needed when the summary uses
different template settings. The HTTP provider supplies all of them.
Set `RunOptions.compact_prompt` to enable automatic compaction, or call
`compact_history` directly. `run_turn` maintains history, executes tools serially,
enforces permission callbacks and bounds model calls, tool calls and output.
Failures preserve history and do not roll back side effects.

Hosts can register `Tool` values with their own capability and handler, then
bind YAML entries with `configure_tools`. Host callbacks must validate inputs
and honor cancellation. Process isolation, MCP, streaming UI, persistent
sessions and subagents are not implemented.

## Testing

```bash
ctest --test-dir .cache/kidi-code-build --output-on-failure
# In a root build with tests enabled:
ctest --test-dir .cache/kidi-code-host-build -R 'Code|ResponseTest|kidi_code_managed' --output-on-failure
```

Unit tests cover agent permissions/limits, YAML, workspace operations and HTTP
tool round trips. Pseudo-terminal tests exercise editing, Unicode, paste, history,
session approvals, Ctrl-C/Ctrl-D, terminal restoration and non-TTY input.
The managed-daemon test uses a tiny CPU model to verify
concurrent startup, reuse, config override and crash recovery without downloads.
It also verifies exact context HTTP responses, repeated automatic compaction,
manual compaction, the small-context warning and continued generation with a
16-token fixture context, so compaction is checked in seconds; production
runs use the automatically sized large context.
Set `KIDI_TEST_REAL_IDLE=1` when running that test to check the full 10-minute
idle shutdown on the wall clock. For the tiny-model CUDA smoke test, run the
managed test script against a CUDA build with `KIDI_TEST_AUTO_CUDA` set to an
idle GPU index; it checks that `auto` selects CUDA and explicit CPU still works.

Codex's Apache-licensed registry and safety architecture informed the design.
This is original C++; no Codex source was copied.
