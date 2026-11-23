#include "kidi/code/command.h"
#include "kidi/code/openai.h"
#include "kidi/code/terminal.h"
#include "kidi/code/workspace.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <set>
#include <thread>

#if defined(__unix__) || defined(__APPLE__)
#include <signal.h>
#endif

namespace kidi::code {
namespace {

auto env_or(const char* key, std::string fallback) -> std::string {
    const auto* value = std::getenv(key);
    return value && *value ? value : std::move(fallback);
}

#if defined(__unix__) || defined(__APPLE__)
std::atomic<bool> interrupted = false;
static_assert(std::atomic<bool>::is_always_lock_free);

auto interrupt(int) -> void { interrupted.store(true, std::memory_order_relaxed); }

class TurnInterrupt {
public:
    explicit TurnInterrupt(std::stop_source& stop) {
        interrupted.store(false, std::memory_order_relaxed);
        struct sigaction action{};
        action.sa_handler = interrupt;
        sigemptyset(&action.sa_mask);
        installed = ::sigaction(SIGINT, &action, &previous_) == 0;
        if (installed)
            watcher_ = std::jthread([&stop](std::stop_token done) {
                while (!done.stop_requested()) {
                    if (interrupted.load(std::memory_order_relaxed)) {
                        stop.request_stop();
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
            });
    }
    ~TurnInterrupt() {
        watcher_.request_stop();
        if (watcher_.joinable()) watcher_.join();
        if (installed && ::sigaction(SIGINT, &previous_, nullptr) < 0)
            std::cerr << "kidi-code: cannot restore SIGINT handler\n";
    }
    bool installed = false;

private:
    struct sigaction previous_{};
    std::jthread watcher_;
};
#endif

} // namespace

auto configure_parser(cli::ArgumentParser& parser, bool managed) -> void {
    if (managed) {
        parser.description("Native coding agent. Starts or reuses a localhost background Kidi inference server.");
        parser.add_argument("-m", "--model")
            .default_value(env_or("KIDI_MODEL", ""))
            .help("local directory or @owner/model (default: Gemma 4 E2B)");
    } else {
        parser.description("Native coding agent. Connects to an existing Kidi server over HTTP.");
        parser.add_argument("--base-url")
            .dest("base_url")
            .default_value(env_or("KIDI_AGENT_BASE_URL", "http://127.0.0.1:18097/v1"));
    }
    parser.add_argument("-w", "--workspace").default_value(std::filesystem::path("."));
    parser.add_argument("--config")
        .default_value(std::filesystem::path{})
        .help("trusted agent YAML (default: embedded agent.yaml; never auto-loaded from the workspace)");
    parser.add_argument("-p", "--prompt").default_value(std::string{}).help("one-shot task; otherwise interactive");
    parser.add_argument("--color").default_value(std::string("auto")).choices({"auto", "always", "never"});
    parser.add_argument("--permissions")
        .default_value(std::string("ask"))
        .choices({"read-only", "ask", "workspace-write", "full-access"})
        .help(
            "read only; ask before writes and commands (default); allow writes but ask before commands; "
            "or allow both");
    parser.add_argument("--max-steps").dest("max_steps").default_value<std::size_t>(64);
    parser.add_argument("--max-tool-calls").dest("max_tool_calls").default_value<std::size_t>(256);
    parser.add_argument("--max-tokens").dest("max_tokens").default_value<std::size_t>(4096);
    parser.add_argument("--timeout-ms").dest("timeout_ms").default_value<std::int64_t>(120000);
}

auto code_command(const cli::Namespace& arguments, ProviderFactory create_provider) -> int {
    const auto& config_path = arguments.get<std::filesystem::path>("config");
    auto config = config_path.empty() ? default_agent_config() : load_agent_config(config_path);
    if (!config) {
        std::cerr << "kidi-code: " << terminal_text(config.error().message) << '\n';
        return 1;
    }
    OpenAIOptions options_http{
        .model = arguments.contains("model") ? arguments.get<std::string>("model") : "kidi-local",
        .max_tokens = arguments.get<std::size_t>("max_tokens"),
        .timeout = std::chrono::milliseconds(arguments.get<std::int64_t>("timeout_ms")),
    };
    if (arguments.contains("base_url")) {
        options_http.base_url = arguments.get<std::string>("base_url");
    }
    auto tools = workspace_tools(arguments.get<std::filesystem::path>("workspace"), *config);
    if (!tools) {
        std::cerr << "kidi-code: " << terminal_text(tools.error().message) << '\n';
        return 1;
    }
    if (!options_http.max_tokens || options_http.timeout.count() <= 0) {
        std::cerr << "kidi-code: positive --max-tokens and --timeout-ms are required\n";
        return 1;
    }
    RunOptions options{.max_model_calls = arguments.get<std::size_t>("max_steps"),
                       .max_tool_calls = arguments.get<std::size_t>("max_tool_calls"),
                       .continue_prompt = config->continue_prompt,
                       .compact_prompt = config->compact_prompt};
    const auto& mode = arguments.get<std::string>("permissions");
    options.permissions.write = mode == "read-only" ? Access::DENY : mode == "ask" ? Access::ASK : Access::ALLOW;
    options.permissions.execute = mode == "read-only"     ? Access::DENY
                                  : mode == "full-access" ? Access::ALLOW
                                                          : Access::ASK;
    const auto& prompt = arguments.get<std::string>("prompt");
    const bool interactive = prompt.empty();
    Terminal terminal(arguments.get<std::string>("color"));
    std::set<std::string> session_grants;
    std::stop_source stop;
    bool input_closed = false;
    Callbacks callbacks;
    if (interactive) {
        callbacks.approve = [&](const ToolSpec& spec, const ToolCall& call) -> Result<bool> {
            if (session_grants.contains(spec.name)) return true;
            std::cerr << terminal.label("\nApproval required: ", "\033[33m") << terminal_text(spec.name) << ' '
                      << terminal_text(call.arguments.dump(-1, ' ', false, Json::error_handler_t::replace)) << '\n'
                      << "Allow once [y], allow ANY arguments to this tool for this session [a], deny [N].\n";
            const auto tool =
                std::ranges::find_if(*tools, [&](const Tool& item) { return item.spec.name == spec.name; });
            if (tool != tools->end() && tool->capability == Capability::EXECUTE)
                std::cerr << "Shell commands are not sandboxed; session approval permits any command as your user.\n";
            for (;;) {
                auto response = terminal.read(terminal.label("Approve? [y/a/N] ", "\033[33m"), false);
                if (!response) return std::unexpected(response.error());
                if (response->state == InputState::INTERRUPTED) {
                    stop.request_stop();
                    return std::unexpected(Error{ErrorCode::RUNTIME, "approval cancelled"});
                }
                if (response->state == InputState::END) {
                    input_closed = true;
                    stop.request_stop();
                    return std::unexpected(Error{ErrorCode::IO, "cannot read approval: end of input"});
                }
                if (response->text == "y" || response->text == "yes") return true;
                if (response->text == "a" || response->text == "always") {
                    session_grants.insert(spec.name);
                    std::cerr << "Session approval granted for " << terminal_text(spec.name)
                              << ". /permissions reset revokes it.\n";
                    return true;
                }
                if (response->text.empty() || response->text == "n" || response->text == "no") return false;
                std::cerr << "Choose y, a or n; empty input denies.\n";
            }
        };
    }
    callbacks.observe = [&](const Event& event) {
        if (event.kind == EventKind::TOOL_START)
            std::cerr << terminal.label("[tool] ", "\033[36m") << terminal_text(event.tool_name) << ' '
                      << terminal_text(event.detail) << '\n';
        if (event.kind == EventKind::TOOL_FINISH && event.is_error)
            std::cerr << terminal.label("[tool error] ", "\033[31m") << terminal_text(event.detail) << '\n';
        if (event.kind == EventKind::COMPACT_START)
            std::cerr << terminal.label("[context] ", "\033[36m") << "Compacting conversation...\n";
        if (event.kind == EventKind::COMPACT_FINISH)
            std::cerr << terminal.label("[context] ", "\033[36m") << terminal_text(event.detail) << '\n';
    };
    std::vector<Message> history{{.role = Role::SYSTEM, .content = config->system_prompt}};
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
        std::cerr << terminal.label("[context] ", "\033[33m") << "warning: " << usage->context_tokens
                  << "-token context; the system prompt and tools already use " << usage->prompt_tokens
                  << ". Long tasks will compact often. Omit --context-size to size it from free memory, or use "
                  << RECOMMENDED << " or more.\n";
    };
    const auto run = [&](std::string task, bool control = false) -> int {
        if (!*provider) {
            provider = create_provider(std::move(options_http));
            if (provider) check_context();
        }
        if (!provider) {
            std::cerr << "kidi-code: " << terminal_text(provider.error().message) << '\n';
            return 1;
        }
        stop = std::stop_source{};
#if defined(__unix__) || defined(__APPLE__)
        TurnInterrupt signals(stop);
        if (!signals.installed) {
            std::cerr << "kidi-code: cannot install SIGINT handler\n";
            return 1;
        }
#endif
        if (control) {
            const auto specs = visible_specs();
            if (task == "/compact") {
                auto result = compact_history(*provider, specs, history, options, callbacks, stop.get_token());
                if (result) return 0;
                std::cerr << "kidi-code: " << terminal_text(result.error().message) << '\n';
                return 1;
            }
            if (!provider->context) {
                std::cerr << "kidi-code: provider does not support context inspection\n";
                return 1;
            }
            auto usage = provider->context(history, specs, stop.get_token());
            if (!usage) {
                std::cerr << "kidi-code: " << terminal_text(usage.error().message) << '\n';
                return 1;
            }
            std::cerr << usage->prompt_tokens << " / " << usage->context_tokens
                      << " context tokens (including system prompt and tool schemas)\n";
            return 0;
        }
        auto result = run_turn(*provider, *tools, history, std::move(task), options, callbacks, stop.get_token());
        if (!result) {
            std::cerr << terminal.label("kidi-code: ", "\033[31m") << terminal_text(result.error().message) << '\n';
            return 1;
        }
        std::cout << terminal_text(result->text) << '\n' << std::flush;
        return 0;
    };
    if (!interactive) return run(prompt);
    std::cerr << terminal.label("Kidi Code", "\033[1;36m") << " (native; " << mode
              << "). /help for commands. Ctrl-C cancels; Ctrl-D exits on an empty line.\n";
    for (;;) {
        auto input = terminal.read(terminal.label("> ", "\033[1;32m"));
        if (!input) {
            std::cerr << "kidi-code: " << terminal_text(input.error().message) << '\n';
            return 1;
        }
        if (input->state == InputState::END) break;
        if (input->state == InputState::INTERRUPTED) continue;
        auto line = std::move(input->text);
        if (line == "/exit") break;
        if (line == "/help") {
            std::cerr
                << "/exit               Quit\n/reset              Clear conversation (keep session approvals)\n"
                   "/permissions        Show mode and session approvals\n"
                   "/permissions reset  Revoke session approvals (keep the CLI permission mode)\n"
                   "/context            Show exact prompt and context token counts\n"
                   "/compact            Summarize older history (keep session approvals)\n"
                   "Arrow keys edit and recall tasks; Tab completes commands; bracketed paste keeps multiple lines.\n";
            continue;
        }
        if (line == "/compact" || line == "/context") {
            (void)run(std::move(line), true);
            continue;
        }
        if (line == "/permissions" || line == "/permissions reset") {
            if (line == "/permissions reset") session_grants.clear();
            std::cerr << "Permission mode: " << mode << "\nSession-approved tools:";
            for (const auto& name : session_grants) std::cerr << ' ' << terminal_text(name);
            if (session_grants.empty()) std::cerr << " none";
            std::cerr << '\n';
            continue;
        }
        if (line == "/reset") {
            history.resize(1);
            std::cerr << "Conversation cleared; session approvals unchanged.\n";
            continue;
        }
        if (line.empty()) continue;
        // A failed turn keeps the session; only an unavailable model ends it.
        if (run(std::move(line)) && !provider) return 1;
        if (input_closed) break;
    }
    return 0;
}

} // namespace kidi::code
