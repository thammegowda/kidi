#include "kidi/agent/command.h"
#include "kidi/agent/chat_client.h"
#include "kidi/agent/terminal.h"
#include "kidi/agent/transcript.h"
#include "kidi/agent/workspace.h"

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
    parser.add_argument("--max-tokens").dest("max_tokens").default_value<std::size_t>(4096);
    parser.add_argument("--temperature")
        .default_value<double>(-1)
        .help("sampling temperature; 0 is greedy (default: the agent config's sampling.temperature)");
    parser.add_argument("--timeout-ms")
        .dest("timeout_ms")
        .default_value<std::int64_t>(120000)
        .help("abort a model request after this long without any response bytes");
}

auto agent_command(const cli::Namespace& arguments, ProviderFactory create_provider) -> int {
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
    Terminal terminal(arguments.get<std::string>("color"));
    std::set<std::string> session_grants;
    std::mutex stop_mutex;
    std::stop_source stop;
    const auto current_stop = [&] {
        const std::lock_guard lock(stop_mutex);
        return stop;
    };
    const auto request_stop = [&] { current_stop().request_stop(); };
    bool input_closed = false;
    std::vector<Message> history{{.role = Role::SYSTEM, .content = config->system_prompt}};
    std::optional<Transcript> transcript;
    std::optional<Error> transcript_error;
    std::optional<ContextUsage> context;
    struct Speed {
        double tokens_per_second = 0, prefill_seconds = 0;
        bool decode_only = false;
    };
    std::optional<Speed> speed;
    bool streamed_content = false, streamed_reasoning = false;
    std::string model = options_http.model.empty() ? "local model" : options_http.model;
    const auto save = [&](Json record) {
        if (!transcript || transcript_error) return;
        auto result = transcript->append(std::move(record));
        if (!result) {
            transcript_error = result.error();
            request_stop();
        }
    };
    const auto show_status = [&](std::string_view phase) {
        if (!interactive) return;
        // Fixed-width fields first so live updates do not shift the bar; the changing phase goes last.
        std::ostringstream text;
        text << std::filesystem::path(model).filename().string() << " | ctx " << std::fixed << std::setprecision(1);
        if (context && context->context_tokens)
            text << std::setw(5) << 100.0 * context->prompt_tokens / context->context_tokens << "% "
                 << std::setw(static_cast<int>(std::to_string(context->context_tokens).size()))
                 << context->prompt_tokens << '/' << context->context_tokens;
        else
            text << "   --";
        text << " | ";
        if (!speed)
            text << "    -- tok/s";
        else if (speed->decode_only)
            text << std::setw(6) << speed->tokens_per_second << " tok/s | prefill " << std::setw(5)
                 << speed->prefill_seconds << 's';
        else
            text << std::setw(6) << speed->tokens_per_second << " tok/s req";
        text << " | " << phase;
        terminal.status(text.str());
    };
    Callbacks callbacks;
    if (interactive && terminal.interactive()) {
        options_http.on_delta = [&](std::string_view text, bool thinking) {
            auto& seen = thinking ? streamed_reasoning : streamed_content;
            if (!seen) {
                terminal.activity(thinking ? "Thinking" : "Answering");
                seen = true;
            }
            terminal.stream(text, thinking);
        };
    }
    if (interactive) {
        callbacks.approve = [&](const ToolSpec& spec, const ToolCall& call) -> Result<bool> {
            if (session_grants.contains(spec.name)) return true;
            terminal.write("Approval required: " + terminal_text(spec.name) + "\n", TerminalTone::WARNING);
            // Approval must expose every argument, even when the activity preview was abbreviated.
            terminal.tool(spec.name, call.arguments, true);
            terminal.write("Allow once [y], allow ANY arguments to this tool for this session [a], deny [N].\n",
                           TerminalTone::DIM);
            const auto tool =
                std::ranges::find_if(*tools, [&](const Tool& item) { return item.spec.name == spec.name; });
            if (tool != tools->end() && tool->capability == Capability::EXECUTE)
                terminal.write("Shell commands are not sandboxed; session approval permits any command as your user.\n",
                               TerminalTone::WARNING);
            for (;;) {
                auto response = terminal.read("Approve? [y/a/N] ", false);
                if (!response) return std::unexpected(response.error());
                if (response->state == InputState::INTERRUPTED) {
                    request_stop();
                    return std::unexpected(Error{ErrorCode::RUNTIME, "approval cancelled"});
                }
                if (response->state == InputState::END) {
                    input_closed = true;
                    request_stop();
                    return std::unexpected(Error{ErrorCode::IO, "cannot read approval: end of input"});
                }
                if (response->text == "y" || response->text == "yes") return true;
                if (response->text == "a" || response->text == "always") {
                    session_grants.insert(spec.name);
                    terminal.write("Session approval granted for " + terminal_text(spec.name) +
                                       ". /permissions reset revokes it.\n",
                                   TerminalTone::SESSION);
                    return true;
                }
                if (response->text.empty() || response->text == "n" || response->text == "no") return false;
                terminal.write("Choose y, a or n; empty input denies.\n", TerminalTone::WARNING);
            }
        };
    }
    callbacks.observe = [&](const Event& event) {
        if (event.kind == EventKind::MODEL_START) {
            show_status("Generating");
            terminal.activity(context ? "Processing " + std::to_string(context->prompt_tokens) + " prompt tokens"
                                      : "Waiting for model");
        }
        if (event.kind == EventKind::TOOL_START) {
            const auto arguments = Json::parse(event.detail, nullptr, false);
            if (arguments.is_object())
                terminal.tool(event.tool_name, arguments);
            else
                terminal.write("[tool] " + terminal_text(event.tool_name) + "\n" + terminal_text(event.detail) + "\n",
                               TerminalTone::TOOL);
            show_status("Working");
            terminal.activity(event.tool_name, true);
            save({{"type", "tool_start"},
                  {"name", event.tool_name},
                  {"id", event.call_id},
                  {"arguments", arguments.is_discarded() ? Json(event.detail) : arguments}});
        }
        if (event.kind == EventKind::TOOL_FINISH) {
            terminal.finish_activity(event.is_error);
            save(transcript_message({.role = Role::TOOL, .content = event.detail, .tool_call_id = event.call_id}));
            if (event.is_error) {
                const auto error = Json::parse(event.detail, nullptr, false);
                const auto text = error.is_object() && error.contains("error") && error["error"].is_string()
                                      ? error["error"].get<std::string>()
                                      : event.detail;
                terminal.write("[tool error] " + terminal_text(text) + "\n", TerminalTone::ERROR);
            }
        }
        if (event.kind == EventKind::COMPACT_START) {
            terminal.activity("Compacting");
            terminal.write("[context] Compacting conversation...\n", TerminalTone::CONTEXT);
        }
        if (event.kind == EventKind::COMPACT_FINISH) {
            terminal.finish_activity();
            terminal.write("[context] " + terminal_text(event.detail) + "\n", TerminalTone::CONTEXT);
            Json checkpoint{{"type", "compact"}, {"messages", Json::array()}};
            for (const auto& message : history) checkpoint["messages"].push_back(transcript_message(message));
            save(std::move(checkpoint));
        }
    };
    const auto visible_specs = [&] {
        std::vector<ToolSpec> specs;
        for (const auto& tool : *tools)
            if (options.permissions.access(tool.capability) != Access::DENY) specs.push_back(tool.spec);
        return specs;
    };
    Result<Provider> provider;
    // Tell the user up front when the context cannot hold real work, instead of compacting constantly.
    const auto check_context = [&] {
        constexpr std::size_t RECOMMENDED = 32768;
        if (!provider->context) return;
        auto usage = provider->context(std::span(history).first(1), visible_specs(), {});
        if (!usage || usage->context_tokens >= RECOMMENDED) return;
        terminal.write("[context] warning: " + std::to_string(usage->context_tokens) +
                           "-token context; the system prompt and tools already use " +
                           std::to_string(usage->prompt_tokens) +
                           ". Long tasks will compact often. Omit --context-size to size it from free memory, or "
                           "use " +
                           std::to_string(RECOMMENDED) + " or more.\n",
                       TerminalTone::WARNING);
    };
    const auto run = [&](std::string task, bool control = false) -> int {
        if (!*provider) {
            terminal.activity("Starting model");
            provider = create_provider(std::move(options_http));
            terminal.finish_activity(!provider);
            if (provider) {
                if (!provider->model.empty()) model = provider->model;
                auto complete = std::move(provider->complete);
                provider->complete = [&, complete = std::move(complete)](auto messages, auto specs,
                                                                         std::stop_token token) -> Result<Message> {
                    const auto started = std::chrono::steady_clock::now();
                    streamed_content = streamed_reasoning = false;
                    auto reply = complete(messages, specs, token);
                    terminal.end_stream();
                    terminal.finish_activity(!reply);
                    const auto seconds =
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
                    speed.reset();
                    if (reply) {
                        if (reply->usage) {
                            const auto& usage = *reply->usage;
                            if (usage.decode_ms > 0)
                                speed = Speed{usage.completion_tokens * 1000.0 / usage.decode_ms,
                                              usage.prefill_ms / 1000.0, true};
                            else if (seconds > 0)
                                speed = Speed{usage.completion_tokens / seconds};
                            if (context)
                                context->prompt_tokens = reply->usage->prompt_tokens + reply->usage->completion_tokens;
                        }
                        save(transcript_message(*reply));
                        if (transcript_error) return std::unexpected(*transcript_error);
                        if (!streamed_reasoning) terminal.thinking(reply->reasoning_content);
                        if (!streamed_content && !reply->tool_calls.empty() && !reply->content.empty())
                            terminal.write(terminal_text(reply->content) + "\n", TerminalTone::ASSISTANT);
                    }
                    return reply;
                };
                if (provider->context) {
                    auto inspect = std::move(provider->context);
                    provider->context = [&, inspect = std::move(inspect)](
                                            auto messages, auto specs, std::stop_token token) -> Result<ContextUsage> {
                        terminal.activity("Checking context");
                        auto usage = inspect(messages, specs, token);
                        terminal.finish_activity(!usage);
                        if (usage) context = *usage;
                        return usage;
                    };
                }
                check_context();
            }
        }
        if (!provider) {
            terminal.write("kidi-agent: " + terminal_text(provider.error().message) + "\n", TerminalTone::ERROR);
            return 1;
        }
        {
            const std::lock_guard lock(stop_mutex);
            stop = std::stop_source{};
        }
        if (transcript_error) {
            terminal.write("kidi-agent: " + terminal_text(transcript_error->message) + "\n", TerminalTone::ERROR);
            return 1;
        }
        if (!transcript && !arguments.get<bool>("no_save")) {
            auto opened = Transcript::open(arguments.get<std::filesystem::path>("workspace"), model);
            if (!opened) {
                transcript_error = opened.error();
                terminal.write("kidi-agent: " + terminal_text(opened.error().message) + " (use --no-save to opt out)\n",
                               TerminalTone::ERROR);
                return 1;
            }
            transcript.emplace(std::move(*opened));
            save(transcript_message(history.front()));
            if (interactive)
                terminal.write("[session] " + terminal_text(transcript->path().string()) + "\n", TerminalTone::SESSION);
        }
        if (transcript_error) {
            terminal.write("kidi-agent: " + terminal_text(transcript_error->message) + "\n", TerminalTone::ERROR);
            return 1;
        }
        const auto token = current_stop().get_token();
        if (control) {
            const auto specs = visible_specs();
            if (task == "/compact") {
                auto result = compact_history(*provider, specs, history, options, callbacks, token);
                terminal.finish_activity(!result);
                if (result && !transcript_error) {
                    show_status("Ready");
                    return 0;
                }
                if (transcript_error) result = std::unexpected(*transcript_error);
                terminal.write("kidi-agent: " + terminal_text(result.error().message) + "\n", TerminalTone::ERROR);
                return 1;
            }
            if (!provider->context) {
                terminal.write("kidi-agent: provider does not support context inspection\n", TerminalTone::ERROR);
                return 1;
            }
            auto usage = provider->context(history, specs, token);
            if (!usage) {
                terminal.write("kidi-agent: " + terminal_text(usage.error().message) + "\n", TerminalTone::ERROR);
                return 1;
            }
            terminal.write(std::to_string(usage->prompt_tokens) + " / " + std::to_string(usage->context_tokens) +
                               " context tokens (including system prompt and tool schemas)\n",
                           TerminalTone::CONTEXT);
            show_status("Ready");
            return 0;
        }
        save(transcript_message({.role = Role::USER, .content = task}));
        if (transcript_error) {
            terminal.write("kidi-agent: " + terminal_text(transcript_error->message) + "\n", TerminalTone::ERROR);
            return 1;
        }
        auto result = run_turn(*provider, *tools, history, std::move(task), options, callbacks, token);
        if (transcript_error) result = std::unexpected(*transcript_error);
        if (!result) {
            terminal.finish_activity(true);
            save({{"type", "error"}, {"message", result.error().message}});
            if (transcript_error) result = std::unexpected(*transcript_error);
            terminal.write("kidi-agent: " + terminal_text(result.error().message) + "\n", TerminalTone::ERROR);
            show_status("Stopped");
            return 1;
        }
        if (!streamed_content) terminal.answer(result->text);
        show_status("Ready");
        return 0;
    };
    if (!interactive) return run(prompt);
    const auto interactive_loop = [&]() -> int {
        terminal.write("Kidi Agent (native; " + mode + "). /help for commands. Ctrl-C cancels; Ctrl-D exits.\n",
                       TerminalTone::TOOL);
        show_status("Ready");
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
                    "Arrows edit and recall tasks; Page Up/Down scrolls the transcript; Tab completes commands.\n",
                    TerminalTone::DIM);
                continue;
            }
            if (line == "/compact" || line == "/context") {
                if (run(std::move(line), true) && transcript_error) return 1;
                continue;
            }
            if (line == "/permissions" || line == "/permissions reset") {
                if (line == "/permissions reset") session_grants.clear();
                auto result = "Permission mode: " + mode + "\nSession-approved tools:";
                for (const auto& name : session_grants) result += ' ' + terminal_text(name);
                if (session_grants.empty()) result += " none";
                terminal.write(result + "\n", TerminalTone::CONTEXT);
                continue;
            }
            if (line == "/reset") {
                history.resize(1);
                save({{"type", "reset"}});
                if (transcript_error) {
                    terminal.write("kidi-agent: " + terminal_text(transcript_error->message) + "\n",
                                   TerminalTone::ERROR);
                    return 1;
                }
                context.reset();
                speed.reset();
                terminal.write("Conversation cleared; session approvals unchanged.\n", TerminalTone::CONTEXT);
                continue;
            }
            if (line.empty()) continue;
            // A failed turn keeps the session; only an unavailable model ends it.
            if (run(std::move(line)) && (!provider || transcript_error)) return 1;
            if (input_closed) break;
        }
        return 0;
    };
    return terminal.run_interactive(interactive_loop, request_stop);
}

} // namespace kidi::agent
