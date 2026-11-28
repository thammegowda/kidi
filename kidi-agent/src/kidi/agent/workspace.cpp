#include "kidi/agent/workspace.h"

#include <liteapp/workspace/workspace.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace kidi::agent {
namespace {

using liteapp::workspace::Workspace;

// Temporary files of atomic writes; hidden from tools and never addressable.
constexpr auto TEMPORARY_PREFIX = ".kidi-agent-";

auto invalid(std::string message) -> std::unexpected<Error> {
    return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, std::move(message)});
}

auto from_liteapp(const liteapp::Error& error) -> std::unexpected<Error> {
    const auto code = error.code == "io"            ? ErrorCode::IO
                      : error.code == "unsupported" ? ErrorCode::UNSUPPORTED
                      : error.code == "cancelled" || error.code == "conflict" ? ErrorCode::RUNTIME
                                                                               : ErrorCode::INVALID_ARGUMENT;
    if (error.code == "binary") return invalid("expected UTF-8 text");
    return std::unexpected(Error{code, error.message});
}

auto string_arg(const Json& args, std::string_view key) -> Result<std::string> {
    const auto item = args.find(key);
    if (item == args.end() || !item->is_string()) return invalid(std::string(key) + " must be a string");
    return item->get<std::string>();
}

auto size_arg(const Json& args, std::string_view key, std::size_t fallback) -> Result<std::size_t> {
    const auto item = args.find(key);
    if (item == args.end()) return fallback;
    if (!item->is_number_unsigned() && (!item->is_number_integer() || item->get<std::int64_t>() < 0))
        return invalid(std::string(key) + " must be a nonnegative integer");
    const auto value = item->get<std::uint64_t>();
    if (value > std::numeric_limits<std::size_t>::max()) return invalid(std::string(key) + " is too large");
    return static_cast<std::size_t>(value);
}

// The JSON tool handlers; files and commands go through the shared liteapp workspace.
struct Tools {
    Workspace workspace;

    auto read(const Json& args, std::stop_token stop) const -> Result<std::string> {
        auto path = string_arg(args, "path");
        if (!path) return std::unexpected(path.error());
        const auto limit = workspace.options().limits.max_file_bytes;
        const auto offset = size_arg(args, "offset", 0);
        const auto max_bytes = size_arg(args, "max_bytes", std::min<std::size_t>(4096, limit));
        if (!offset) return std::unexpected(offset.error());
        if (!max_bytes) return std::unexpected(max_bytes.error());
        auto page = workspace.read_page(*path, *offset, *max_bytes, stop);
        if (!page) return from_liteapp(page.error());
        if (!*offset && !page->next_offset) return std::move(page->content);
        return Json{{"content", std::move(page->content)},
                    {"offset", page->offset},
                    {"next_offset", page->next_offset ? Json(*page->next_offset) : Json(nullptr)},
                    {"total_bytes", page->total_bytes}}.dump();
    }

    auto list(const Json& args, std::stop_token stop) const -> Result<std::string> {
        if (stop.stop_requested()) return std::unexpected(Error{ErrorCode::RUNTIME, "cancelled"});
        auto path = string_arg(args, "path");
        if (!path) return std::unexpected(path.error());
        if (path->empty()) return invalid("expected a nonempty relative workspace path");
        auto entries = workspace.list(*path);
        if (!entries) return from_liteapp(entries.error());
        std::vector<std::string> names;
        for (auto& entry : *entries) names.push_back(std::move(entry.name));
        std::ranges::sort(names);
        return Json(names).dump(-1, ' ', false, Json::error_handler_t::replace);
    }

    auto write(const Json& args, std::stop_token stop) const -> Result<std::string> {
        auto path = string_arg(args, "path");
        auto content = string_arg(args, "content");
        if (!path) return std::unexpected(path.error());
        if (!content) return std::unexpected(content.error());
        auto written = workspace.write(*path, *content, std::nullopt, stop);
        if (!written) return from_liteapp(written.error());
        return Json{{"written_bytes", content->size()}}.dump();
    }

    auto make_directory(const Json& args, std::stop_token stop) const -> Result<std::string> {
        if (stop.stop_requested()) return std::unexpected(Error{ErrorCode::RUNTIME, "cancelled"});
        auto path = string_arg(args, "path");
        if (!path) return std::unexpected(path.error());
        if (auto made = workspace.make_directory(*path); !made) return from_liteapp(made.error());
        return std::string("directory created");
    }

    auto run(const Json& args, std::stop_token stop) const -> Result<std::string> {
        auto command = string_arg(args, "command");
        if (!command) return std::unexpected(command.error());
        if (command->empty() || command->find('\0') != std::string::npos)
            return invalid("command must be nonempty text");
        auto result = workspace.execute({"/bin/sh", "-c", *command}, stop);
        if (!result) return from_liteapp(result.error());
        Json json{{"exit_code", result->exit_code}, {"output", std::move(result->output)}};
        if (result->truncated) json["truncated"] = true;
        if (result->timed_out) json["timed_out"] = true;
        return json.dump(-1, ' ', false, Json::error_handler_t::replace);
    }
};

} // namespace

auto workspace_tools(const std::filesystem::path& root, WorkspaceOptions options) -> Result<std::vector<Tool>> {
    auto config = default_agent_config();
    if (!config) return std::unexpected(config.error());
    return workspace_tools(root, *config, options);
}

auto workspace_tools(const std::filesystem::path& root, const AgentConfig& config, WorkspaceOptions options)
    -> Result<std::vector<Tool>> {
    auto workspace = Workspace::open(root, {.limits = {.max_file_bytes = options.max_file_bytes,
                                                       .max_directory_entries = options.max_directory_entries,
                                                       .max_command_output = options.max_command_output,
                                                       .command_timeout = options.command_timeout},
                                            .temporary_prefix = TEMPORARY_PREFIX});
    if (!workspace) return from_liteapp(workspace.error());
    auto tools = std::make_shared<const Tools>(Tools{std::move(*workspace)});
    const std::vector<Tool> handlers{
        {{.name = "read_file"},
         Capability::READ,
         [tools](const Json& args, std::stop_token stop) { return tools->read(args, stop); }},
        {{.name = "list_files"},
         Capability::READ,
         [tools](const Json& args, std::stop_token stop) { return tools->list(args, stop); }},
        {{.name = "write_file"},
         Capability::WRITE,
         [tools](const Json& args, std::stop_token stop) { return tools->write(args, stop); }},
        {{.name = "make_directory"},
         Capability::WRITE,
         [tools](const Json& args, std::stop_token stop) { return tools->make_directory(args, stop); }},
        {{.name = "run_command"},
         Capability::EXECUTE,
         [tools](const Json& args, std::stop_token stop) { return tools->run(args, stop); }},
    };
    return configure_tools(config, handlers);
}

} // namespace kidi::agent
