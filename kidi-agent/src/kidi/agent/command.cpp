#include "kidi/agent/command.h"
#include "kidi/agent/chat_client.h"
#include "kidi/agent/session.h"
#include "kidi/agent/terminal.h"
#include "kidi/agent/transcript.h"
#include "kidi/agent/workspace.h"
#if defined(KIDI_AGENT_WEB)
#include "kidi/agent/web.h"
#endif

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <set>
#include <stop_token>
#include <thread>

namespace kidi::agent {

auto configure_parser(cli::ArgumentParser& parser, bool managed) -> void {
    if (managed) {
        parser.description(
            "Local agent for chat and workspace tasks. Starts or reuses a localhost background Kidi inference server.");
        parser.add_argument("-m", "--model")
            .default_value(std::string{})
            .help("local directory or @owner/model (default: Gemma 4 E2B)");
    } else {
        parser.description("Local agent for chat and workspace tasks. Connects to an existing Kidi server over HTTP.");
        parser.add_argument("--base-url").dest("base_url").default_value(std::string("http://127.0.0.1:18097/v1"));
    }
    parser.add_argument("-w", "--workspace").default_value(std::filesystem::path("."));
    parser.add_argument("--config")
        .default_value(std::filesystem::path{})
        .help("trusted agent YAML (default: embedded agent.yaml; never auto-loaded from the workspace)");
    parser.add_argument("--system")
        .default_value(std::string{})
        .help("replaces the config's prompts.system (prompts.tools is still appended when tools are offered)");
    parser.add_argument("-p", "--prompt").default_value(std::string{}).help("one-shot task; otherwise interactive");
    parser.add_argument("--color").default_value(std::string("auto")).choices({"auto", "always", "never"});
    parser.add_argument("--no-save")
        .dest("no_save")
        .action(cli::Action::STORE_TRUE)
        .help("do not save a local workspace-scoped JSONL transcript");
    parser.add_argument("--permissions")
        .default_value(std::string("ask"))
        .choices({"none", "read-only", "ask", "workspace-write", "full-access"})
        .help(
            "none: plain chat without tools; read-only: read files only; ask: ask before writes and commands "
            "(default); workspace-write: allow writes but ask before commands; full-access: allow both");
    parser.add_argument("--max-steps").dest("max_steps").default_value<std::size_t>(64);
    parser.add_argument("--max-tool-calls").dest("max_tool_calls").default_value<std::size_t>(256);
    parser.add_argument("--max-tokens")
        .dest("max_tokens")
        .default_value<std::size_t>(16384)
        .help("token limit for one model reply, including reasoning and tool arguments");
    parser.add_argument("--reasoning-effort")
        .dest("reasoning_effort")
        .default_value(std::string{})
        .choices({"none", "minimal", "low", "medium", "high", "xhigh"})
        .help("thinking effort for model replies (default: the agent config's reasoning_effort)");
    parser.add_argument("--temperature")
        .default_value<double>(-1)
        .help("sampling temperature; 0 is greedy (default: the agent config's sampling.temperature)");
    parser.add_argument("--timeout-ms")
        .dest("timeout_ms")
        .default_value<std::int64_t>(120000)
        .help("abort a model request after this long without any response bytes");
    parser.add_argument("--web")
        .action(cli::Action::STORE_TRUE)
        .help("open the agent in a local web UI with a file tree, editor, diffs and terminal");
    parser.add_argument("--port")
        .default_value<std::int32_t>(0)
        .help("loopback port for --web (default: any free port)");
    parser.add_argument("--web-assets")
        .dest("web_assets")
        .default_value(std::filesystem::path{})
        .help("serve the --web UI live from a checkout's kidi-agent/web, per its assets.txt (UI edits need a reload)");
}

auto agent_command(const cli::Namespace& arguments, ProviderFactory create_provider, SpeechFactory create_speech)
    -> int {
    const auto& config_path = arguments.get<std::filesystem::path>("config");
    auto config = config_path.empty() ? default_agent_config() : load_agent_config(config_path);
    if (!config) {
        std::cerr << "kidi-agent: " << terminal_text(config.error().message) << '\n';
        return 1;
    }
    if (const auto& system = arguments.get<std::string>("system"); !system.empty()) config->system_prompt = system;
    ChatClientOptions options_http{
        .model = arguments.contains("model") ? arguments.get<std::string>("model") : "kidi-local",
        .max_tokens = arguments.get<std::size_t>("max_tokens"),
        .timeout = std::chrono::milliseconds(arguments.get<std::int64_t>("timeout_ms")),
    };
    const auto temperature = arguments.get<double>("temperature");
    if (temperature > 2 || (temperature < 0 && temperature != -1)) {
        std::cerr << "kidi-agent: --temperature must be in [0, 2]\n";
        return 1;
    }
    options_http.temperature = temperature >= 0 ? static_cast<float>(temperature) : config->sampling.temperature;
    options_http.top_p = config->sampling.top_p;
    options_http.top_k = config->sampling.top_k;
    const auto& effort = arguments.get<std::string>("reasoning_effort");
    options_http.reasoning_effort = effort.empty() ? config->reasoning_effort : effort;
    if (arguments.contains("base_url")) {
        options_http.base_url = arguments.get<std::string>("base_url");
    }
    auto tools = workspace_tools(arguments.get<std::filesystem::path>("workspace"), *config);
    if (!tools) {
        std::cerr << "kidi-agent: " << terminal_text(tools.error().message) << '\n';
        return 1;
    }
    if (!options_http.max_tokens || options_http.timeout.count() <= 0) {
        std::cerr << "kidi-agent: positive --max-tokens and --timeout-ms are required\n";
        return 1;
    }
    RunOptions options{.max_model_calls = arguments.get<std::size_t>("max_steps"),
                       .max_tool_calls = arguments.get<std::size_t>("max_tool_calls"),
                       .continue_prompt = config->continue_prompt,
                       .compact_prompt = config->compact_prompt};
    const auto& mode = arguments.get<std::string>("permissions");
    const bool read_only = mode == "none" || mode == "read-only";
    options.permissions.read = mode == "none" ? Access::DENY : Access::ALLOW;
    options.permissions.write = read_only ? Access::DENY : mode == "ask" ? Access::ASK : Access::ALLOW;
    options.permissions.execute = read_only ? Access::DENY : mode == "full-access" ? Access::ALLOW : Access::ASK;
    const auto offered = [&](const Tool& tool) { return options.permissions.access(tool.capability) != Access::DENY; };
    if (!config->tools_prompt.empty() && std::ranges::any_of(*tools, offered)) {
        if (!config->system_prompt.ends_with('\n')) config->system_prompt += '\n';
        config->system_prompt += config->tools_prompt;
    }
    const auto& prompt = arguments.get<std::string>("prompt");
    const bool interactive = prompt.empty();
    SessionOptions session_options{.chat = std::move(options_http),
                                   .run = std::move(options),
                                   .tools = std::move(*tools),
                                   .system_prompt = config->system_prompt,
                                   .workspace = arguments.get<std::filesystem::path>("workspace"),
                                   .mode = mode,
                                   .save = !arguments.get<bool>("no_save"),
                                   .announce_transcript = interactive,
                                   .create_provider = std::move(create_provider)};
#if defined(KIDI_AGENT_WEB)
    if (arguments.get<bool>("web")) {
        if (!interactive) {
            std::cerr << "kidi-agent: --web and --prompt cannot be combined\n";
            return 1;
        }
        auto speech = create_speech ? create_speech(config->speech) : Speech{};
        return web_agent(std::move(session_options), std::move(speech), arguments.get<std::int32_t>("port"),
                         arguments.get<std::filesystem::path>("web_assets"));
    }
#else
    static_cast<void>(create_speech);
    if (arguments.get<bool>("web")) {
        std::cerr << "kidi-agent: this build has no web UI; use the kidi CLI (kidi agent --web)\n";
        return 1;
    }
#endif
    Terminal terminal(arguments.get<std::string>("color"));
    bool input_closed = false;
    std::unique_ptr<AgentSession> session;
    SessionView view;
    if (interactive && terminal.interactive())
        view.stream = [&](std::string_view text, bool thinking) { terminal.stream(text, thinking); };
    view.busy = [&](std::string_view phase, bool tool) { terminal.activity(phase, tool); };
    view.idle = [&](bool failed) {
        terminal.end_stream();
        terminal.finish_activity(failed);
    };
    view.reply = [&](const Message& reply, bool streamed_reasoning, bool streamed_content) {
        if (!streamed_reasoning) terminal.thinking(reply.reasoning_content);
        if (!streamed_content && !reply.tool_calls.empty() && !reply.content.empty())
            terminal.write(terminal_text(reply.content) + "\n", TerminalTone::ASSISTANT);
    };
    view.notice = [&](std::string_view text, Notice kind) {
        static constexpr TerminalTone TONES[] = {TerminalTone::DEFAULT, TerminalTone::CONTEXT, TerminalTone::SESSION,
                                                 TerminalTone::WARNING, TerminalTone::ERROR};
        terminal.write(std::string(text) + "\n", TONES[static_cast<int>(kind)]);
    };
    view.event = [&](const Event& event) {
        if (event.kind == EventKind::TOOL_START) {
            const auto arguments = Json::parse(event.detail, nullptr, false);
            if (arguments.is_object())
                terminal.tool(event.tool_name, arguments);
            else
                terminal.write("[tool] " + terminal_text(event.tool_name) + "\n" + terminal_text(event.detail) + "\n",
                               TerminalTone::TOOL);
        }
        if (event.kind == EventKind::TOOL_FINISH && event.is_error) {
            const auto error = Json::parse(event.detail, nullptr, false);
            const auto text = error.is_object() && error.contains("error") && error["error"].is_string()
                                  ? error["error"].get<std::string>()
                                  : event.detail;
            terminal.write("[tool error] " + terminal_text(text) + "\n", TerminalTone::ERROR);
        }
        if (event.kind == EventKind::COMPACT_START)
            terminal.write("[context] Compacting conversation...\n", TerminalTone::CONTEXT);
        if (event.kind == EventKind::COMPACT_FINISH)
            terminal.write("[context] " + terminal_text(event.detail) + "\n", TerminalTone::CONTEXT);
    };
    // Fixed-width fields first so live updates do not shift the bar; the changing phase goes last.
    const auto show_status = [&](const SessionStatus& status, std::string_view phase) {
        std::ostringstream text;
        text << std::filesystem::path(status.model).filename().string() << " | ctx " << std::fixed
             << std::setprecision(1);
        const auto& context = status.context;
        if (context && context->context_tokens)
            text << std::setw(5) << 100.0 * context->prompt_tokens / context->context_tokens << "% "
                 << std::setw(static_cast<int>(std::to_string(context->context_tokens).size()))
                 << context->prompt_tokens << '/' << context->context_tokens;
        else
            text << "   --";
        text << " | ";
        if (!status.speed)
            text << "    -- tok/s";
        else if (status.speed->decode_only)
            text << std::setw(6) << status.speed->tokens_per_second << " tok/s | prefill " << std::setw(5)
                 << status.speed->prefill_seconds << 's';
        else
            text << std::setw(6) << status.speed->tokens_per_second << " tok/s req";
        text << " | " << phase;
        terminal.status(text.str());
    };
    if (interactive) view.status = show_status;
    if (interactive)
        view.approve = [&](const ToolSpec& spec, const ToolCall& call, bool unsandboxed) -> Result<Approval> {
            terminal.write("Approval required: " + terminal_text(spec.name) + "\n", TerminalTone::WARNING);
            // Approval must expose every argument, even when the activity preview was abbreviated.
            terminal.tool(spec.name, call.arguments, true);
            terminal.write("Allow once [y], allow ANY arguments to this tool for this session [a], deny [N].\n",
                           TerminalTone::DIM);
            if (unsandboxed)
                terminal.write("Shell commands are not sandboxed; session approval permits any command as your user.\n",
                               TerminalTone::WARNING);
            for (;;) {
                auto response = terminal.read("Approve? [y/a/N] ", false);
                if (!response) return std::unexpected(response.error());
                if (response->state == InputState::INTERRUPTED)
                    return std::unexpected(Error{ErrorCode::RUNTIME, "approval cancelled"});
                if (response->state == InputState::END) {
                    input_closed = true;
                    return std::unexpected(Error{ErrorCode::IO, "cannot read approval: end of input"});
                }
                if (response->text == "y" || response->text == "yes") return Approval::ONCE;
                if (response->text == "a" || response->text == "always") return Approval::SESSION;
                if (response->text.empty() || response->text == "n" || response->text == "no") return Approval::DENY;
                terminal.write("Choose y, a or n; empty input denies.\n", TerminalTone::WARNING);
            }
        };
    session = std::make_unique<AgentSession>(std::move(session_options), std::move(view));
    const auto run = [&](std::string task) -> int {
        auto result = session->run(std::move(task));
        if (!result) return 1;
        if (!session->streamed_answer()) terminal.answer(result->text);
        return 0;
    };
    if (!interactive) return run(prompt);
    const auto interactive_loop = [&]() -> int {
        terminal.write("Kidi Agent (native; " + session->mode() +
                           "). /help for commands. Ctrl-C cancels; Ctrl-D exits.\n",
                       TerminalTone::TOOL);
        show_status(session->status(), "Ready");
        for (;;) {
            auto input = terminal.read("> ");
            if (!input) {
                terminal.write("kidi-agent: " + terminal_text(input.error().message) + "\n", TerminalTone::ERROR);
                return 1;
            }
            if (input->state == InputState::END) break;
            if (input->state == InputState::INTERRUPTED) continue;
            auto line = std::move(input->text);
            if (line == "/exit") break;
            if (line == "/help") {
                terminal.write(
                    "/exit               Quit\n/reset              Clear conversation (keep session approvals)\n"
                    "/permissions        Show mode and session approvals\n"
                    "/permissions reset  Revoke session approvals (keep the CLI permission mode)\n"
                    "/context            Show exact prompt and context token counts\n"
                    "/compact            Summarize older history (keep session approvals)\n"
                    "Arrows edit and recall tasks; Alt-Enter or Ctrl-J adds a line; Tab completes commands.\n"
                    "Scroll, select and copy the transcript with your terminal.\n",
                    TerminalTone::DIM);
                continue;
            }
            if (line == "/compact") {
                if (!session->compact() && session->failed()) return 1;
                continue;
            }
            if (line == "/context") {
                auto usage = session->context();
                if (usage)
                    terminal.write(std::to_string(usage->prompt_tokens) + " / " +
                                       std::to_string(usage->context_tokens) +
                                       " context tokens (including system prompt and tool schemas)\n",
                                   TerminalTone::CONTEXT);
                else if (session->failed())
                    return 1;
                continue;
            }
            if (line == "/permissions" || line == "/permissions reset") {
                if (line == "/permissions reset") session->revoke_grants();
                auto result = "Permission mode: " + session->mode() + "\nSession-approved tools:";
                for (const auto& name : session->grants()) result += ' ' + terminal_text(name);
                if (session->grants().empty()) result += " none";
                terminal.write(result + "\n", TerminalTone::CONTEXT);
                continue;
            }
            if (line == "/reset") {
                if (!session->reset()) return 1;
                terminal.write("Conversation cleared; session approvals unchanged.\n", TerminalTone::CONTEXT);
                continue;
            }
            if (line.empty()) continue;
            // A failed turn keeps the session; only an unavailable model or transcript ends it.
            if (run(std::move(line)) && session->failed()) return 1;
            if (input_closed) break;
        }
        return 0;
    };
    return terminal.run_interactive(interactive_loop, [&] { session->cancel(); });
}

} // namespace kidi::agent
