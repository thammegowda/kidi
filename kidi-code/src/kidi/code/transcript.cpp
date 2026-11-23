#include "kidi/code/transcript.h"

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace kidi::code {
namespace {

auto io_error(std::string message) -> std::unexpected<Error> {
    return std::unexpected(Error{ErrorCode::IO, "transcript: " + std::move(message)});
}

auto timestamp() -> std::int64_t {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

} // namespace

auto transcript_message(const Message& message) -> Json {
    const char* role = "system";
    switch (message.role) {
        case Role::SYSTEM:
            role = "system";
            break;
        case Role::USER:
            role = "user";
            break;
        case Role::ASSISTANT:
            role = "assistant";
            break;
        case Role::TOOL:
            role = "tool";
            break;
    }
    Json record{{"type", "message"}, {"role", role}, {"content", message.content}};
    if (!message.reasoning_content.empty()) record["reasoning_content"] = message.reasoning_content;
    if (!message.tool_call_id.empty()) record["tool_call_id"] = message.tool_call_id;
    if (message.is_summary) record["is_summary"] = true;
    if (!message.tool_calls.empty()) {
        record["tool_calls"] = Json::array();
        for (const auto& call : message.tool_calls)
            record["tool_calls"].push_back({{"id", call.id}, {"name", call.name}, {"arguments", call.arguments}});
    }
    if (message.usage)
        record["usage"] = {{"prompt_tokens", message.usage->prompt_tokens},
                           {"completion_tokens", message.usage->completion_tokens}};
    return record;
}

auto Transcript::open(const std::filesystem::path& workspace, std::string_view model) -> Result<Transcript> {
    std::error_code error;
    const auto canonical = std::filesystem::canonical(workspace, error);
    if (error) return io_error("resolve workspace: " + error.message());
    std::filesystem::path cache;
    if (const auto* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg)
        cache = xdg;
    else if (const auto* home = std::getenv("HOME"); home && *home)
        cache = std::filesystem::path(home) / ".cache";
#if defined(_WIN32)
    else if (const auto* local = std::getenv("LOCALAPPDATA"); local && *local)
        cache = local;
#endif
    if (cache.empty() || !cache.is_absolute())
        return io_error("set an absolute XDG_CACHE_HOME or HOME, or use --no-save");
    // Stable across processes and builds; canonical paths isolate workspaces, including symlink aliases.
    std::uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : canonical.generic_string()) hash = (hash ^ byte) * 1099511628211ULL;
    std::ostringstream key;
    key << std::hex << hash;
    const auto directory = cache / "kidi" / "code" / "chats" / key.str();
    std::filesystem::create_directories(directory, error);
    if (error) return io_error("create cache directory: " + error.message());
    std::filesystem::permissions(directory, std::filesystem::perms::owner_all, std::filesystem::perm_options::replace,
                                 error);
    if (error) return io_error("protect cache directory: " + error.message());
    Transcript result;
    result.path_ = directory / ("session-" + std::to_string(timestamp()) + ".jsonl");
    // Exclusive creation avoids overwriting another session; secure permissions precede the first record.
    result.file_.reset(std::fopen(result.path_.string().c_str(), "wx"));
    if (!result.file_) return io_error("create " + result.path_.string() + ": " + std::strerror(errno));
    std::filesystem::permissions(result.path_, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::replace, error);
    if (error) return io_error("protect session file: " + error.message());
    auto written = result.append(
        {{"type", "session"}, {"version", 1}, {"workspace", canonical.generic_string()}, {"model", model}});
    if (!written) return std::unexpected(written.error());
    return result;
}

auto Transcript::append(Json record) -> Result<void> {
    record["time_ns"] = timestamp();
    std::string line;
    try {
        line = record.dump() + '\n';
    } catch (const Json::type_error& error) {
        return io_error("encode record: " + std::string(error.what()));
    }
    if (std::fwrite(line.data(), 1, line.size(), file_.get()) != line.size() || std::fflush(file_.get()) != 0)
        return io_error("append " + path_.string() + ": " + std::strerror(errno));
    return {};
}

} // namespace kidi::code
