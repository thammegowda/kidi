#include "kidi/code/command.h"
#include "kidi/code/openai.h"
#include "kidi/code/workspace.h"

#include <cstdlib>
#include <iostream>

namespace kidi::code {
namespace {

auto env_or(const char* key, std::string fallback) -> std::string {
    const auto* value = std::getenv(key);
    return value && *value ? value : std::move(fallback);
}

// Model and tool output is untrusted terminal data, including approval prompts.
auto terminal_text(std::string_view text) -> std::string {
    std::string result;
    for (unsigned char byte : text) {
        if (byte == '\n' || byte == '\t' || (byte >= 32 && byte < 127)) {
            result.push_back(static_cast<char>(byte));
        } else {
            constexpr char HEX[] = "0123456789abcdef";
            result += "\\x";
            result += HEX[byte >> 4];
            result += HEX[byte & 15];
        }
    }
    return result;
}

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
    parser.add_argument("--permissions")
        .default_value(std::string("ask"))
        .choices({"read-only", "ask", "workspace-write", "full-access"})
        .help(
            "read only; ask before writes and commands (default); allow writes but ask before commands; "
            "or allow both");
    parser.add_argument("--max-steps").dest("max_steps").default_value<std::size_t>(16);
    parser.add_argument("--max-tool-calls").dest("max_tool_calls").default_value<std::size_t>(64);
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
                       .continue_prompt = config->continue_prompt};
    const auto& mode = arguments.get<std::string>("permissions");
    options.permissions.write = mode == "read-only" ? Access::DENY : mode == "ask" ? Access::ASK : Access::ALLOW;
    options.permissions.execute = mode == "read-only"     ? Access::DENY
                                  : mode == "full-access" ? Access::ALLOW
                                                          : Access::ASK;
    const auto& prompt = arguments.get<std::string>("prompt");
    const bool interactive = prompt.empty();
    Callbacks callbacks;
    if (interactive) {
        callbacks.approve = [](const ToolSpec& spec, const ToolCall& call) -> Result<bool> {
            std::cerr << "\nApprove " << terminal_text(spec.name) << ' ' << terminal_text(call.arguments.dump())
                      << "? [y/N] " << std::flush;
            std::string response;
            if (!std::getline(std::cin, response)) return std::unexpected(Error{ErrorCode::IO, "cannot read approval"});
            return response == "y" || response == "yes";
        };
    }
    callbacks.observe = [](const Event& event) {
        if (event.kind == EventKind::TOOL_START) std::cerr << "[tool] " << terminal_text(event.tool_name) << '\n';
        if (event.kind == EventKind::TOOL_FINISH && event.is_error)
            std::cerr << "[tool error] " << terminal_text(event.detail) << '\n';
    };
    std::vector<Message> history{{.role = Role::SYSTEM, .content = config->system_prompt}};
    Result<Provider> provider;
    const auto run = [&](std::string task) -> int {
        if (!*provider) provider = create_provider(std::move(options_http));
        if (!provider) {
            std::cerr << "kidi-code: " << terminal_text(provider.error().message) << '\n';
            return 1;
        }
        auto result = run_turn(*provider, *tools, history, std::move(task), options, callbacks);
        if (!result) {
            std::cerr << "kidi-code: " << terminal_text(result.error().message) << '\n';
            return 1;
        }
        std::cout << terminal_text(result->text) << '\n';
        return 0;
    };
    if (!interactive) return run(prompt);
    std::cerr << "Kidi Code (native; " << mode << "). /exit to quit, /reset to clear history.\n";
    for (std::string line; std::cerr << "> " << std::flush, std::getline(std::cin, line);) {
        if (line == "/exit") break;
        if (line == "/reset") {
            history.resize(1);
            continue;
        }
        if (line.empty()) continue;
        // A failed turn keeps the session; only an unavailable model ends it.
        if (run(std::move(line)) && !provider) return 1;
    }
    return std::cin.bad() ? 1 : 0;
}

} // namespace kidi::code
