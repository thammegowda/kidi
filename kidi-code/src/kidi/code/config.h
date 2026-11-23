#pragma once

#include "kidi/code/agent.h"

#include <filesystem>
#include <string_view>

namespace kidi::code {

struct ToolRegistration {
    std::string handler;
    ToolSpec spec;
};

struct AgentConfig {
    std::string system_prompt;
    std::string continue_prompt;
    std::vector<ToolRegistration> tools;
};

auto parse_agent_config(std::string_view yaml) -> Result<AgentConfig>;
auto load_agent_config(const std::filesystem::path& path) -> Result<AgentConfig>;
auto default_agent_config() -> Result<AgentConfig>;

// Only listed tools are exposed. Names/schemas come from config; capabilities and code come from the host.
auto configure_tools(const AgentConfig& config, std::span<const Tool> handlers) -> Result<std::vector<Tool>>;

} // namespace kidi::code
