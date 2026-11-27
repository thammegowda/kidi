#pragma once

#include "kidi/agent/agent.h"

#include <filesystem>
#include <string_view>

namespace kidi::agent {

struct ToolRegistration {
    std::string handler;
    ToolSpec spec;
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
    std::vector<ToolRegistration> tools;
};

auto parse_agent_config(std::string_view yaml) -> Result<AgentConfig>;
auto load_agent_config(const std::filesystem::path& path) -> Result<AgentConfig>;
auto default_agent_config() -> Result<AgentConfig>;

// Only listed tools are exposed. Names/schemas come from config; capabilities and code come from the host.
auto configure_tools(const AgentConfig& config, std::span<const Tool> handlers) -> Result<std::vector<Tool>>;

} // namespace kidi::agent
