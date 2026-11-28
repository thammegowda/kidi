#include "kidi/agent/transcript.h"

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <array>
#include <cstdio>
#include <optional>
#include <string_view>
#include <sstream>
#include <utility>

namespace kidi::agent {
namespace {

auto io_error(std::string message) -> std::unexpected<Error> {
    return std::unexpected(Error{ErrorCode::IO, "transcript: " + std::move(message)});
}

auto timestamp() -> std::int64_t {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// The length of `file` up to and including its last newline, found by reading backwards from `size`.
auto complete_length(std::FILE* file, std::uintmax_t size) -> std::optional<std::uintmax_t> {
    std::array<char, 65536> block;
    for (auto end = size; end > 0;) {
        const auto start = end > block.size() ? end - block.size() : 0;
        const auto count = static_cast<std::size_t>(end - start);
        if (::fseeko(file, static_cast<off_t>(start), SEEK_SET) != 0 ||
            std::fread(block.data(), 1, count, file) != count)
            return std::nullopt;
        if (const auto newline = std::string_view(block.data(), count).rfind('\n'); newline != std::string_view::npos)
            return start + newline + 1;
        end = start;
    }
    return 0;
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
    if (!message.images.empty()) record["images"] = message.images;
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

auto Transcript::directory(const std::filesystem::path& workspace) -> Result<std::filesystem::path> {
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
    // Resolved like the workspace, so listed, opened and resumed transcripts share one path (macOS /tmp is a link).
    cache = std::filesystem::weakly_canonical(cache, error);
    if (error) return io_error("resolve cache directory: " + error.message());
    // Stable across processes and builds; canonical paths isolate workspaces, including symlink aliases.
    std::uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : canonical.generic_string()) hash = (hash ^ byte) * 1099511628211ULL;
    std::ostringstream key;
    key << std::hex << hash;
    return cache / "kidi" / "agent" / "chats" / key.str();
}

auto Transcript::open(const std::filesystem::path& workspace, std::string_view model) -> Result<Transcript> {
    std::error_code error;
    const auto canonical = std::filesystem::canonical(workspace, error);
    if (error) return io_error("resolve workspace: " + error.message());
    const auto directory = Transcript::directory(canonical);
    if (!directory) return std::unexpected(directory.error());
    std::filesystem::create_directories(*directory, error);
    if (error) return io_error("create cache directory: " + error.message());
    std::filesystem::permissions(*directory, std::filesystem::perms::owner_all, std::filesystem::perm_options::replace,
                                 error);
    if (error) return io_error("protect cache directory: " + error.message());
    Transcript result;
    result.path_ = *directory / ("session-" + std::to_string(timestamp()) + ".jsonl");
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

auto Transcript::reopen(const std::filesystem::path& path) -> Result<Transcript> {
    // A crash can leave a cut-off last record; drop it so the next record starts on a line of its own.
    {
        std::unique_ptr<std::FILE, CloseFile> file(std::fopen(path.string().c_str(), "r"));
        std::error_code error;
        const auto size = std::filesystem::file_size(path, error);
        if (!file || error) return io_error("read " + path.string());
        const auto complete = complete_length(file.get(), size);
        if (!complete) return io_error("read " + path.string() + ": " + std::strerror(errno));
        if (*complete != size) std::filesystem::resize_file(path, *complete, error);
        if (error) return io_error("repair " + path.string() + ": " + error.message());
    }
    Transcript result;
    result.path_ = path;
    result.file_.reset(std::fopen(path.string().c_str(), "a"));
    if (!result.file_) return io_error("open " + path.string() + ": " + std::strerror(errno));
    return result;
}

auto Transcript::read(const std::filesystem::path& path, std::vector<Json>& records) -> Result<void> {
    std::unique_ptr<std::FILE, CloseFile> file(std::fopen(path.string().c_str(), "r"));
    if (!file) return io_error("open " + path.string() + ": " + std::strerror(errno));
    std::string line;
    for (int c; (c = std::fgetc(file.get())) != EOF;) {
        if (c != '\n') {
            line += static_cast<char>(c);
            continue;
        }
        auto record = Json::parse(line, nullptr, false);
        line.clear();
        // A record cut short by a crash is skipped rather than failing the whole conversation.
        if (record.is_object()) records.push_back(std::move(record));
    }
    return {};
}

auto message_from_record(const Json& record) -> Result<Message> {
    static constexpr std::pair<std::string_view, Role> ROLES[] = {
        {"system", Role::SYSTEM}, {"user", Role::USER}, {"assistant", Role::ASSISTANT}, {"tool", Role::TOOL}};
    const auto role = record.value("role", "");
    const auto found = std::ranges::find(ROLES, role, &std::pair<std::string_view, Role>::first);
    if (found == std::end(ROLES) || !record.contains("content") || !record["content"].is_string())
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "transcript: invalid message record"});
    Message message{.role = found->second, .content = record["content"].get<std::string>()};
    message.reasoning_content = record.value("reasoning_content", "");
    message.tool_call_id = record.value("tool_call_id", "");
    message.is_summary = record.value("is_summary", false);
    if (const auto images = record.find("images"); images != record.end()) {
        if (!images->is_array() || !std::ranges::all_of(*images, [](const Json& image) { return image.is_string(); }))
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "transcript: invalid images"});
        message.images = images->get<std::vector<std::string>>();
    }
    if (const auto calls = record.find("tool_calls"); calls != record.end() && calls->is_array())
        for (const auto& call : *calls) {
            if (!call.is_object() || !call.contains("arguments") || !call["arguments"].is_object())
                return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "transcript: invalid tool call"});
            message.tool_calls.push_back({call.value("id", ""), call.value("name", ""), call["arguments"]});
        }
    if (const auto usage = record.find("usage"); usage != record.end() && usage->is_object())
        message.usage = TokenUsage{usage->value("prompt_tokens", std::size_t{0}),
                                   usage->value("completion_tokens", std::size_t{0})};
    return message;
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

} // namespace kidi::agent
