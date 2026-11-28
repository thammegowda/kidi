#pragma once

#include "kidi/agent/agent.h"
#include "kidi/agent/speech.h"

#include <filesystem>
#include <map>
#include <string_view>

namespace kidi::agent {

struct ToolRegistration {
    std::string handler;
    ToolSpec spec;
    Json config = Json::object(); // the entry's optional `config` mapping, interpreted by its handler
};

// An agent.yaml `mcp_servers` entry: a remote MCP server that tools refer to by its key.
struct McpServerConfig {
    std::string name;        // what users call it, such as "TinyFish"; the key when unset
    std::string url;         // its MCP endpoint (Streamable HTTP)
    // The server's tools the agent may use, by exact name (`include: [a, b]`), or all of them (`include: all`).
    // Without include, the server only backs built-in tools such as web_search.
    std::vector<std::string> include;
    bool include_all = false;
};

struct AgentConfig {
    std::string system_prompt;
    std::string tools_prompt; // appended to the system prompt when tools are offered
    std::string continue_prompt;
    std::string compact_prompt;
    // temperature 0 is greedy; summaries are always greedy.
    struct Sampling {
        float temperature = 0, top_p = 1;
        std::size_t top_k = 0;
    } sampling;
    // Sent as the request's reasoning_effort when nonempty; summaries always use "none".
    std::string reasoning_effort;
    SpeechConfig speech;
    std::map<std::string, McpServerConfig, std::less<>> mcp_servers;
    std::vector<ToolRegistration> tools;
};

auto parse_agent_config(std::string_view yaml) -> Result<AgentConfig>;
auto load_agent_config(const std::filesystem::path& path) -> Result<AgentConfig>;
auto default_agent_config() -> Result<AgentConfig>;

// Only listed tools are exposed. Names/schemas come from config; capabilities and code come from the host.
// These handlers take no `config`; an entry that sets one is rejected.
auto configure_tools(const AgentConfig& config, std::span<const Tool> handlers) -> Result<std::vector<Tool>>;

// A native implementation that builds one tool from its agent.yaml entry's `config` mapping (empty when unset).
struct ToolHandler {
    std::string name;
    std::function<Result<Tool>(const Json& config)> make;
};
auto configure_tools(const AgentConfig& config, std::span<const ToolHandler> handlers) -> Result<std::vector<Tool>>;
// A handler for a ready-made tool that takes no config.
auto fixed_handler(Tool tool) -> ToolHandler;

} // namespace kidi::agent
