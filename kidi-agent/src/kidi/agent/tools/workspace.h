#pragma once

#include "kidi/agent/config.h"
#include "kidi/agent/net/mcp.h"

#include <chrono>
#include <filesystem>

namespace kidi::agent::tools {

struct WorkspaceOptions {
    std::size_t max_file_bytes = 64 * 1024; // Maximum read page or file-write size.
    std::size_t max_directory_entries = 1024;
    std::size_t max_command_output = 32 * 1024;
    std::chrono::milliseconds command_timeout{60000};
};

// Tools over liteapp::workspace::Workspace: POSIX descriptor-relative access; no symlinks, parent traversal, absolute
// paths or .git entries. run_command runs /bin/sh in the workspace root with the user's privileges; it is not
// sandboxed. The root must remain under host control. Also binds the web_search and web_fetch entries, whose
// mcp_servers are signed in to on first use.
auto workspace_tools(const std::filesystem::path& root, WorkspaceOptions options = {}) -> Result<std::vector<Tool>>;
// `servers` are the agent's mcp_servers, shared with tool discovery; built from `config` when empty.
auto workspace_tools(const std::filesystem::path& root, const AgentConfig& config, WorkspaceOptions options = {},
                     net::McpServers servers = {}) -> Result<std::vector<Tool>>;

} // namespace kidi::agent::tools
