# Kidi Agent

A small C++23 local agent: a terminal shell for chat and workspace tasks, plus an embeddable library. Inference uses
Kidi's Chat Completions API over plain HTTP; external model providers are not supported. The web tools and MCP
servers use the `curl` program for HTTPS; there is no Node or TLS library dependency.

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

The shell prints the conversation inline, so your terminal's own scrollback, search, selection and copying work as
usual. Replies are rendered as Markdown (headings, emphasis, code, lists, quotes and aligned tables) while they stream;
code blocks keep their exact text for copying. Below the transcript, a small live area shows the streaming reply or
running tool, the input line and a status bar with context usage, token speed and prefill time. The input supports
line editing, task history, multiline paste, and Alt-Enter or Ctrl-J for a new line. Piped input, redirected output
and `TERM=dumb` use plain line input; `-p` runs one task. `--color auto|always|never` controls styling (`auto` respects
`NO_COLOR`; without color, replies are printed as raw Markdown).

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

## Web UI

`kidi agent --web` serves the same agent as a lightweight local studio and prints a one-time link (`--port` picks
the port; over SSH, forward it and open the link yourself):

- **Chat view** centers the conversation; **Code view** adds a file tree, a CodeMirror editor with tabs, diffs of
  the agent's edits (Changes) and a terminal. Saved conversations are under **History**.
- Approvals appear inline with a diff for file writes (Allow / Always this session / Deny).
- **Settings** edits the system prompt for this server's conversations; `agent.yaml` is unchanged.
- Images can be attached, pasted or dropped into the composer for vision models (Gemma 4 E2B QAT); other models
  refuse them before they enter the conversation.
- The microphone button dictates into the composer (Whisper small), and the speaker button on a reply reads it
  aloud (OmniVoice). Both models download and load on first use, on the device `--backend` selects.
- Dark, light or system theme.

The server listens on 127.0.0.1 only and admits the browser that opened the link (an HTTP-only cookie). For UI work,
`--web-assets kidi-agent/web` serves the UI from a checkout instead of the embedded copy, so edits need only a reload.

The file tree, editor, diffs, terminal, themes and the workspace sandbox behind them (shared with the agent's file
and command tools) come from LiteApp's IDE components (`third_party/liteapp`: `liteapp::workspace` and `web/ide`).
Kidi adds the conversation, approvals, images, speech and settings in [`web/`](web/) and
[`ui/web.cpp`](src/kidi/agent/ui/web.cpp); generic UI building blocks belong in LiteApp.

## Tools and permissions

| Tool | Operation |
| --- | --- |
| `read_file` | Read UTF-8 text in pages (4 KiB by default; `offset` and `max_bytes` select a section) |
| `list_files` | List one directory; `.` is the workspace root |
| `write_file` | Atomically replace or create a UTF-8 file |
| `make_directory` | Create one directory |
| `run_command` | Run `/bin/sh -c` in the workspace root (60-second timeout, 32 KiB of output) |
| `web_search` | Search the web; returns up to 10 results as title, URL and snippet |
| `web_fetch` | Read a web page as compact Markdown, about 1,500 tokens per part, with an offset for the next part |

| `--permissions` | Tools offered | File writes | Commands | Web requests |
| --- | --- | --- | --- | --- |
| `none` | None (plain chat) | - | - | - |
| `read-only` | `read_file`, `list_files`, web tools | Denied | Denied | Ask |
| `ask` (default) | All | Ask | Ask | Ask |
| `workspace-write` | All | Allowed | Ask | Ask |
| `full-access` | All | Allowed | Allowed | Allowed |

Approval prompts show every argument and offer **y** (once), **a** (this tool for the rest of the session) or **n**.
One-shot tasks never read approvals from stdin, so operations that need approval are denied; use `full-access` for
unattended commands only in a trusted, disposable environment.

**Permissions are not an OS sandbox.** File handlers reject absolute paths, `..`, `.git`, symlinks and special files,
but `run_command` runs with your privileges. Remove it from the registry to disable it, or run the standalone client
under the [Landlock sandbox](../tools/sandbox/README.md).

### Web search and fetch

Search uses [TinyFish](https://docs.tinyfish.ai), which is free within a daily allowance. Nothing needs setting up
in advance: the first time the agent searches, it asks to connect TinyFish. Choose **Connect** (or answer **y** in the
terminal), sign in in the browser tab that opens, and the search continues on its own. Kidi talks to TinyFish through
its [MCP](https://modelcontextprotocol.io) server and signs in with OAuth (PKCE), so no key is copied anywhere; the
sign-in is kept in `~/.config/kidi/credentials.json` (readable only by you) and renewed automatically. **Not now**
leaves web search off for that request. Over SSH, the web UI works through its forwarded port; the terminal prints
the sign-in link and the port to forward.

Queries and page addresses leave your computer, so web requests ask for approval unless `--permissions
full-access`.

- `web_fetch` downloads pages itself with `curl` and converts them to compact Markdown: the main content only, with
  scripts, navigation, footers, forms, images and common clutter (cookie banners, share bars, sidebars) removed, and
  links within a site written as `/paths`. Pages never leave your computer for conversion; pages built by
  JavaScript come back with a note that content may be missing, and PDFs and other documents are refused.
- It opens only addresses that you gave or that earlier web results listed, so text planted in a page cannot make
  the model send data to an address it made up. It reads only the public internet: loopback, private, link-local and
  cloud metadata addresses are refused, and every redirect is checked again.
- Requests time out (15 s for search, 20 s for a page), network failures are retried once, and downloads are capped
  at 5 MiB. A failure tells the model what went wrong and what you can do.
- Any tool can declare a one-time setup like this sign-in (`ToolSetup`); the agent asks you before running it, and
  the model only learns whether the tool is available.
- Remove the tools from `agent.yaml` to disable web access.

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
reasoning_effort: medium   # none, minimal, low, medium, high or xhigh; omit to let the chat template choose
tools:
  - name: inspect
    handler: read_file
    description: Read a workspace file.
    parameters:
      type: object
      properties:
        path: {type: string}
      required: [path]
  - name: web_search
    handler: web_search
    description: Search the web.
    config: {server: tinyfish}
    parameters:
      type: object
      properties:
        query: {type: string}
      required: [query]
mcp_servers:
  tinyfish: {name: TinyFish, url: https://agent.tinyfish.ai/mcp}
```

`handler` selects a compiled implementation, so renaming a tool cannot change its capability. A tool's optional
`config` mapping is read by its handler: `web_search` takes `{server}`, naming an `mcp_servers` entry (with an
optional `tool`); other built-in handlers take none.

Arguments outside `parameters` are rejected unless the schema sets `additionalProperties: true`. Remove entries to
disable tools (`tools: []` for none). `prompts.tools` is appended to the system prompt only when tools are offered
(never with `--permissions none`), `prompts.continue` is sent once if the model replies silently after tool results,
and `prompts.compact` enables compaction. Without `sampling:` decoding is greedy; `--temperature` overrides
the configured temperature, and compaction summaries are always greedy. `reasoning_effort` is sent with every reply
request (`--reasoning-effort` overrides it; summaries use `none`). Inkling-2 thinks at `high` when it is unset and can
deliberate past the reply limit before acting, so the embedded config uses `medium`. `--max-tokens` (default 16384)
limits one reply, including reasoning and tool arguments such as a whole file passed to `write_file`.

### MCP servers

`mcp_servers` names remote [MCP](https://modelcontextprotocol.io) servers. The agent finds their tools by itself;
`include` decides which ones it may use:

```yaml
mcp_servers:
  tinyfish: {name: TinyFish, url: https://agent.tinyfish.ai/mcp}          # backs web_search and web_fetch only
  github:
    url: https://api.githubcopilot.com/mcp/
    include: [search_code, get_file_contents]                              # exactly these tools; or: all
```

- Included tools appear as `<server>_<tool>` (here `github_search_code`) with the server's own descriptions and
  parameters, so no schemas are written by hand. Without `include`, a server only backs built-in tools.
- `include: [names]` exposes exactly those tools, even if the server adds more later. `include: all` exposes
  whatever the server offers, now and in future; the agent notes it, since one changed server then changes your
  agent.
- Tool lists are cached for 12 hours in `~/.cache/kidi/mcp-tools.json`, so conversations start without asking the
  server; when the cache is stale and the server cannot be reached, the old list is used and you are told.
- MCP tools are network requests: they ask for approval under `--permissions ask`, results are untrusted data, and
  long results are cut at 32 KB. The first use asks you to connect the server (a browser sign-in, renewed
  automatically).

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
