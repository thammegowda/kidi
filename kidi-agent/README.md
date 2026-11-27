# Kidi Agent

A small C++23 local agent: a terminal shell for chat and workspace tasks, plus an embeddable library. Inference uses
Kidi's Chat Completions API over plain HTTP. There is no Node, curl or TLS dependency; external model providers and
HTTPS are not supported.

## Quick start

```bash
kidi agent                                            # chat and workspace tasks in the current directory
kidi agent --permissions none                         # plain chat without tools
kidi agent --permissions none --system "Answer in French."
kidi agent --workspace /path/to/project -p "Explain this project"
kidi agent --model /path/to/checkpoint --backend cuda:0
```

The first session starts a detached Kidi inference server; matching sessions reuse it. The server handles up to two
concurrent generations and exits 10 minutes after the last session closes. The default model is
`@google/gemma-4-E2B-it-qat-mobile-transformers` (a cached `@google/gemma-4-E2B-it` is the fallback); missing weights
are downloaded through the Kidi model hub on first use.

`--backend auto` prefers CUDA, then Metal, then CPU; `--backend ynnpack` forces CPU and `--backend cuda:INDEX` selects
a GPU. The command is built by default on Linux and macOS (`KIDI_BUILD_AGENT=OFF` disables it).

### Inkling-2-Mini

```bash
./build-release/kidi agent --model @thinkingmachines/Inkling-2-Mini --backend ynnpack --restart
```

On CPU the server converts the checkpoint's experts and dense projections to row-scaled INT8 at each start (W8A8
matmuls, group-32 INT8 KV cache), which needs more RAM than CUDA's compact checkpoint mode. `-j/--threads` sets its
CPU threads (default: up to 16). See [CPU inference details](../docs/getting-started.md#inkling-2-on-cpu).

For code-copy/edit workloads, `--prompt-lookup` enables speculative decoding without a second model. It copies
continuations of matching 2-8-token sequences from the prompt and history, verifies them in one batch and adapts the
draft budget, capped by `--draft-tokens` (default 3, at most 16). It requires `--temperature 0` and is off by default
because it can be slower on low-repetition tasks:

```bash
taskset -c 16-31 ./build-release/kidi agent -m @thinkingmachines/Inkling-2-Mini \
  --backend ynnpack --threads 16 --prompt-lookup --draft-tokens 8 --temperature 0 --restart
```

### Background server

The server publishes its port and PID in `$XDG_RUNTIME_DIR/kidi-agent-<uid>/<server-key>/server.json` (or under
`/tmp/kidi-agent-<uid>`), with `serve.log` beside it. The key covers the binary, model, model cache, backend, CUDA
visibility, context size, threads and prompt-lookup settings, so a rebuilt binary or a different
`CUDA_VISIBLE_DEVICES` starts a fresh server. `--restart` replaces the matching server, for example after an inference
failure; other sessions using it lose their connection.

The server sizes its context from free device memory (three quarters of it after weights load, up to the model's
limit; `--context-size` overrides) and keeps the last request's KV state, so each turn prefills only what was
appended. Contexts under 32K tokens print a warning because long tasks compact often. `kidi --log-level debug agent`
also logs the server at debug level.

## Shell

The terminal UI (FTXUI) provides line editing, task history, multiline paste, transcript scrolling and a status bar
with context usage, token speed and prefill time. Piped input, redirected output and `TERM=dumb` use plain line
input; `-p` runs one task. `--color auto|always|never` controls styling (`auto` respects `NO_COLOR`).

| Command | Action |
| --- | --- |
| `/help` | List commands |
| `/reset` | Clear the conversation (keeps session approvals) |
| `/context` | Show exact prompt and context token counts |
| `/compact` | Summarize older history |
| `/permissions` | Show the permission mode and session approvals; `/permissions reset` revokes the approvals |
| `/exit` | Quit |

Ctrl-C clears the input or cancels the current turn, including running commands; Ctrl-D exits at an empty prompt.

Sessions are saved as private JSONL transcripts under `$XDG_CACHE_HOME/kidi/agent/chats/<workspace-key>/` (default
`~/.cache`), including tool calls and results. They are local but not encrypted; `--no-save` disables them.

Before the conversation fills the context, the agent compacts older messages into a continuation handoff. It keeps
the system prompt, the latest request and the most recent complete tool exchange, using exact token counts from the
server (`POST /v1/chat/context`). Summaries are lossy, so put permanent rules in `prompts.system`.

## Tools and permissions

| Tool | Operation |
| --- | --- |
| `read_file` | Read UTF-8 text in pages (4 KiB by default; `offset` and `max_bytes` select a section) |
| `list_files` | List one directory; `.` is the workspace root |
| `write_file` | Atomically replace or create a UTF-8 file |
| `make_directory` | Create one directory |
| `run_command` | Run `/bin/sh -c` in the workspace root (60-second timeout, 32 KiB of output) |

| `--permissions` | Tools offered | File writes | Commands |
| --- | --- | --- | --- |
| `none` | None (plain chat) | - | - |
| `read-only` | `read_file`, `list_files` | Denied | Denied |
| `ask` (default) | All | Ask | Ask |
| `workspace-write` | All | Allowed | Ask |
| `full-access` | All | Allowed | Allowed |

Approval prompts show every argument and offer **y** (once), **a** (this tool for the rest of the session) or **n**.
One-shot tasks never read approvals from stdin, so operations that need approval are denied; use `full-access` for
unattended commands only in a trusted, disposable environment.

**Permissions are not an OS sandbox.** File handlers reject absolute paths, `..`, `.git`, symlinks and special files,
but `run_command` runs with your privileges. Remove it from the registry to disable it, or run the standalone client
under the [Landlock sandbox](../tools/sandbox/README.md).

## Prompts and registry

[`agent.yaml`](agent.yaml) is embedded in the binary and used by default. `--config PATH` replaces it completely
(workspace configurations are never discovered automatically); `--system TEXT` replaces only `prompts.system`.

```yaml
version: 1
prompts:
  system: You are a helpful assistant.
  tools: Inspect files before editing.
  continue: Continue. Answer using the tool results above.
  compact: Summarize goals, constraints, findings, completed actions and outstanding work. Do not execute tools.
sampling:
  temperature: 0.6   # 0 = greedy; range [0, 2]
  top_p: 0.95
  top_k: 20
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

`handler` selects a compiled implementation, so renaming a tool cannot change its capability. Remove entries to
disable tools (`tools: []` for none). `prompts.tools` is appended to the system prompt only when tools are offered
(never with `--permissions none`), `prompts.continue` is sent once if the model replies silently after tool results,
and `prompts.compact` enables compaction. Without `sampling:` decoding is greedy; `--temperature` overrides
the configured temperature, and compaction summaries are always greedy.

## Standalone client

`kidi-agent` connects to a running Kidi server; it does not start one:

```bash
cmake -S kidi-agent -B .cache/kidi-agent-build
cmake --build .cache/kidi-agent-build -j
kidi serve --model /path/to/checkpoint --port 18097
.cache/kidi-agent-build/kidi-agent --base-url http://127.0.0.1:18097/v1   # the default URL
```

Use only trusted Kidi endpoints: prompts and tool results are sent unencrypted and without API keys.

## Embedding

Link `kidi::agent` for the agent loop, YAML configuration and workspace tools, and `kidi::agent_http` for the HTTP
provider; `KIDI_AGENT_BUILD_CLI=OFF` builds only the libraries. `run_turn` maintains history, executes tools serially
and bounds model calls, tool calls and output; `compact_history` compacts on request. Hosts can register their own
`Tool` values and bind YAML entries with `configure_tools`.

## Testing

```bash
ctest --test-dir .cache/kidi-agent-build --output-on-failure
# In a root build with tests enabled:
ctest --preset debug -R 'Agent|ChatClient|kidi_agent_managed' --output-on-failure
```

Unit tests cover permissions and limits, YAML, workspace operations, HTTP round trips and the terminal. The managed
test ([`managed_agent_test.sh`](../tests/kidi/cli/managed_agent_test.sh)) uses a tiny CPU model to check concurrent
startup, reuse, plain chat, configuration overrides, crash recovery and compaction in seconds.
`KIDI_TEST_REAL_IDLE=1` also checks the full 10-minute idle shutdown, and `KIDI_TEST_AUTO_CUDA=GPU` adds a CUDA smoke
test.

Codex's Apache-licensed registry and safety architecture informed the design. This is original C++; no Codex source
was copied.
