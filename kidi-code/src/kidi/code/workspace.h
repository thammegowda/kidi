#pragma once

#include "kidi/code/config.h"

#include <chrono>
#include <filesystem>

namespace kidi::code {

struct WorkspaceOptions {
    std::size_t max_file_bytes = 64 * 1024; // Maximum read page or file-write size.
    std::size_t max_directory_entries = 1024;
    std::size_t max_command_output = 32 * 1024;
    std::chrono::milliseconds command_timeout{60000};
};

// POSIX descriptor-relative access; no symlinks, parent traversal, absolute paths or .git entries.
// run_command runs /bin/sh in the workspace root with the user's privileges; it is not sandboxed.
// The root must remain under host control.
auto workspace_tools(const std::filesystem::path& root, WorkspaceOptions options = {}) -> Result<std::vector<Tool>>;
auto workspace_tools(const std::filesystem::path& root, const AgentConfig& config, WorkspaceOptions options = {})
    -> Result<std::vector<Tool>>;

} // namespace kidi::code
