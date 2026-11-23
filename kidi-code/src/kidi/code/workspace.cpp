#include "kidi/code/workspace.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cctype>
#include <cstring>
#include <memory>
#include <utility>

#if defined(__unix__) || defined(__APPLE__)
#include <dirent.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace kidi::code {
namespace {

auto invalid(std::string message) -> std::unexpected<Error> {
    return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, std::move(message)});
}

#if defined(__unix__) || defined(__APPLE__)
class Fd {
public:
    explicit Fd(int value = -1) : value_(value) {}
    ~Fd() {
        if (value_ >= 0) ::close(value_);
    }
    Fd(const Fd&) = delete;
    auto operator=(const Fd&) -> Fd& = delete;
    Fd(Fd&& other) noexcept : value_(std::exchange(other.value_, -1)) {}
    auto operator=(Fd&& other) noexcept -> Fd& {
        if (this != &other) {
            if (value_ >= 0) ::close(value_);
            value_ = std::exchange(other.value_, -1);
        }
        return *this;
    }
    auto get() const -> int { return value_; }

private:
    int value_;
};

auto io_error(std::string operation) -> std::unexpected<Error> {
    return std::unexpected(Error{ErrorCode::IO, std::move(operation) + ": " + std::strerror(errno)});
}

auto string_arg(const Json& args, std::string_view key) -> Result<std::string> {
    const auto item = args.find(key);
    if (item == args.end() || !item->is_string()) return invalid(std::string(key) + " must be a string");
    return item->get<std::string>();
}

auto validate_text(const std::string& content) -> Result<void> {
    if (content.find('\0') != std::string::npos) return invalid("expected UTF-8 text without NUL bytes");
    try {
        (void)Json(content).dump();
    } catch (const Json::type_error&) {
        return invalid("expected UTF-8 text");
    }
    return {};
}

auto components(const std::string& path, bool allow_root) -> Result<std::vector<std::string>> {
    if (path.empty() || path.find('\0') != std::string::npos || std::filesystem::path(path).is_absolute())
        return invalid("expected a nonempty relative workspace path");
    std::vector<std::string> parts;
    for (const auto& part : std::filesystem::path(path)) {
        const auto text = part.string();
        auto folded = text;
        std::ranges::transform(folded, folded.begin(), [](unsigned char byte) { return std::tolower(byte); });
        if (text == ".." || folded == ".git" || text.starts_with(".kidi-code-"))
            return invalid("parent traversal, .git and reserved temporary paths are denied");
        if (text != "." && !text.empty()) parts.push_back(text);
    }
    if (parts.empty() && !allow_root) return invalid("path must name a file or subdirectory");
    return parts;
}

struct Workspace {
    Fd root;
    WorkspaceOptions options;

    auto directory(std::span<const std::string> parts) const -> Result<Fd> {
        Fd current(::openat(root.get(), ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC));
        if (current.get() < 0) return io_error("open workspace");
        for (const auto& part : parts) {
            Fd next(::openat(current.get(), part.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
            if (next.get() < 0) return io_error("open directory " + part);
            current = std::move(next);
        }
        return current;
    }

    auto read(const Json& args, std::stop_token stop) const -> Result<std::string> {
        auto path = string_arg(args, "path");
        if (!path) return std::unexpected(path.error());
        auto parts = components(*path, false);
        if (!parts) return std::unexpected(parts.error());
        auto parent = directory(std::span(*parts).first(parts->size() - 1));
        if (!parent) return std::unexpected(parent.error());
        Fd file(::openat(parent->get(), parts->back().c_str(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC));
        if (file.get() < 0) return io_error("open file");
        struct stat info{};
        if (::fstat(file.get(), &info) < 0) return io_error("stat file");
        if (!S_ISREG(info.st_mode)) return invalid("read_file requires a regular file");
        if (info.st_size < 0 || static_cast<std::uintmax_t>(info.st_size) > options.max_file_bytes)
            return invalid("file exceeds max_file_bytes");
        std::string content;
        std::array<char, 4096> buffer;
        while (true) {
            if (stop.stop_requested()) return std::unexpected(Error{ErrorCode::RUNTIME, "cancelled"});
            const auto count = ::read(file.get(), buffer.data(), buffer.size());
            if (count < 0) {
                if (errno == EINTR) continue;
                return io_error("read file");
            }
            if (count == 0) break;
            if (content.size() + static_cast<std::size_t>(count) > options.max_file_bytes)
                return invalid("file exceeds max_file_bytes");
            content.append(buffer.data(), static_cast<std::size_t>(count));
        }
        if (auto valid = validate_text(content); !valid) return std::unexpected(valid.error());
        return content;
    }

    auto list(const Json& args, std::stop_token stop) const -> Result<std::string> {
        auto path = string_arg(args, "path");
        if (!path) return std::unexpected(path.error());
        auto parts = components(*path, true);
        if (!parts) return std::unexpected(parts.error());
        auto fd = directory(*parts);
        if (!fd) return std::unexpected(fd.error());
        const auto duplicate = ::fcntl(fd->get(), F_DUPFD_CLOEXEC, 0);
        if (duplicate < 0) return io_error("duplicate directory");
        auto* raw = ::fdopendir(duplicate);
        if (!raw) {
            ::close(duplicate);
            return io_error("list directory");
        }
        const auto close_dir = [](DIR* value) { ::closedir(value); };
        std::unique_ptr<DIR, decltype(close_dir)> dir(raw, close_dir);
        std::vector<std::string> entries;
        while (true) {
            if (stop.stop_requested()) return std::unexpected(Error{ErrorCode::RUNTIME, "cancelled"});
            errno = 0;
            const auto* entry = ::readdir(dir.get());
            if (!entry) {
                if (errno) return io_error("read directory");
                break;
            }
            const std::string name(entry->d_name);
            if (name == "." || name == ".." || name == ".git" || name.starts_with(".kidi-code-")) continue;
            if (entries.size() >= options.max_directory_entries) return invalid("directory exceeds entry limit");
            entries.push_back(name);
        }
        std::ranges::sort(entries);
        return Json(entries).dump(-1, ' ', false, Json::error_handler_t::replace);
    }

    auto write(const Json& args, std::stop_token stop) const -> Result<std::string> {
        if (stop.stop_requested()) return std::unexpected(Error{ErrorCode::RUNTIME, "cancelled"});
        auto path = string_arg(args, "path");
        auto content = string_arg(args, "content");
        if (!path) return std::unexpected(path.error());
        if (!content) return std::unexpected(content.error());
        if (content->size() > options.max_file_bytes) return invalid("content exceeds max_file_bytes");
        if (auto valid = validate_text(*content); !valid) return std::unexpected(valid.error());
        auto parts = components(*path, false);
        if (!parts) return std::unexpected(parts.error());
        auto parent = directory(std::span(*parts).first(parts->size() - 1));
        if (!parent) return std::unexpected(parent.error());
        struct stat existing{};
        mode_t mode = 0600;
        if (::fstatat(parent->get(), parts->back().c_str(), &existing, AT_SYMLINK_NOFOLLOW) == 0) {
            if (!S_ISREG(existing.st_mode)) return invalid("write_file requires a regular file or a new path");
            mode = existing.st_mode & 0777;
        } else if (errno != ENOENT) {
            return io_error("stat destination");
        }
        static std::atomic<std::uint64_t> sequence{0};
        std::string temporary;
        Fd file;
        for (int attempt = 0; attempt < 32; ++attempt) {
            temporary = ".kidi-code-" + std::to_string(::getpid()) + "-" + std::to_string(sequence.fetch_add(1));
            file = Fd(
                ::openat(parent->get(), temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600));
            if (file.get() >= 0) break;
            if (errno != EEXIST) return io_error("create temporary file");
        }
        if (file.get() < 0) return io_error("create unique temporary file");
        struct Cleanup {
            int parent;
            const std::string& name;
            ~Cleanup() { ::unlinkat(parent, name.c_str(), 0); }
        } cleanup{parent->get(), temporary};
        for (std::size_t offset = 0; offset < content->size();) {
            if (stop.stop_requested()) return std::unexpected(Error{ErrorCode::RUNTIME, "cancelled"});
            const auto count = ::write(file.get(), content->data() + offset, content->size() - offset);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) return io_error("write file");
            offset += static_cast<std::size_t>(count);
        }
        if (::fchmod(file.get(), mode) < 0 || ::fsync(file.get()) < 0) return io_error("flush file");
        if (stop.stop_requested()) return std::unexpected(Error{ErrorCode::RUNTIME, "cancelled"});
        // Atomic replacement never writes through an existing hard link or a raced symlink.
        if (::renameat(parent->get(), temporary.c_str(), parent->get(), parts->back().c_str()) < 0)
            return io_error("replace file");
        return Json{{"written_bytes", content->size()}}.dump();
    }

    auto mkdir(const Json& args, std::stop_token stop) const -> Result<std::string> {
        if (stop.stop_requested()) return std::unexpected(Error{ErrorCode::RUNTIME, "cancelled"});
        auto path = string_arg(args, "path");
        if (!path) return std::unexpected(path.error());
        auto parts = components(*path, false);
        if (!parts) return std::unexpected(parts.error());
        auto parent = directory(std::span(*parts).first(parts->size() - 1));
        if (!parent) return std::unexpected(parent.error());
        if (::mkdirat(parent->get(), parts->back().c_str(), 0700) < 0) return io_error("create directory");
        return std::string("directory created");
    }

    auto run(const Json& args, std::stop_token stop) const -> Result<std::string> {
        auto command = string_arg(args, "command");
        if (!command) return std::unexpected(command.error());
        if (command->empty() || command->find('\0') != std::string::npos)
            return invalid("command must be nonempty text");
        std::array<int, 2> pipe_fds{};
        if (::pipe(pipe_fds.data()) < 0) return io_error("create command pipe");
        Fd read_end(pipe_fds[0]), write_end(pipe_fds[1]);
        Fd null_input(::open("/dev/null", O_RDONLY | O_CLOEXEC));
        if (null_input.get() < 0) return io_error("open /dev/null");
        const char* argv[] = {"/bin/sh", "-c", command->c_str(), nullptr};
        const auto pid = ::fork();
        if (pid < 0) return io_error("start command");
        if (pid == 0) {
            ::setpgid(0, 0);
            if (::fchdir(root.get()) < 0 || ::dup2(null_input.get(), STDIN_FILENO) < 0 ||
                ::dup2(write_end.get(), STDOUT_FILENO) < 0 || ::dup2(write_end.get(), STDERR_FILENO) < 0)
                ::_exit(126);
            ::execv(argv[0], const_cast<char* const*>(argv));
            ::_exit(127);
        }
        ::setpgid(pid, pid);
        write_end = Fd();
        std::string output;
        bool truncated = false, timed_out = false, cancelled = false;
        const auto deadline = std::chrono::steady_clock::now() + options.command_timeout;
        for (std::array<char, 4096> buffer{};;) {
            if (stop.stop_requested()) {
                cancelled = true;
                break;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                timed_out = true;
                break;
            }
            pollfd ready{.fd = read_end.get(), .events = POLLIN};
            const auto polled = ::poll(&ready, 1, 100);
            if (polled < 0 && errno != EINTR) break;
            if (polled <= 0) continue;
            const auto count = ::read(read_end.get(), buffer.data(), buffer.size());
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) break;
            const auto room = options.max_command_output - std::min(options.max_command_output, output.size());
            output.append(buffer.data(), std::min(room, static_cast<std::size_t>(count)));
            truncated = truncated || static_cast<std::size_t>(count) > room;
        }
        // Kill the whole process group so background children do not outlive a finished or stopped call.
        if (timed_out || cancelled) ::kill(-pid, SIGKILL);
        int status = 0;
        while (::waitpid(pid, &status, 0) < 0)
            if (errno != EINTR) return io_error("wait for command");
        if (!timed_out && !cancelled) ::kill(-pid, SIGKILL);
        if (cancelled) return std::unexpected(Error{ErrorCode::RUNTIME, "cancelled"});
        Json result{{"exit_code", WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status)},
                    {"output", std::move(output)}};
        if (truncated) result["truncated"] = true;
        if (timed_out) result["timed_out"] = true;
        return result.dump(-1, ' ', false, Json::error_handler_t::replace);
    }
};
#endif

} // namespace

auto workspace_tools(const std::filesystem::path& root, WorkspaceOptions options) -> Result<std::vector<Tool>> {
    auto config = default_agent_config();
    if (!config) return std::unexpected(config.error());
    return workspace_tools(root, *config, options);
}

auto workspace_tools(const std::filesystem::path& root, const AgentConfig& config, WorkspaceOptions options)
    -> Result<std::vector<Tool>> {
    if (!options.max_file_bytes || !options.max_directory_entries || !options.max_command_output ||
        options.command_timeout.count() <= 0)
        return invalid("workspace limits must be positive");
#if defined(__unix__) || defined(__APPLE__)
    Fd fd(::open(root.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
    if (fd.get() < 0) return io_error("open workspace root");
    auto workspace = std::make_shared<Workspace>(std::move(fd), options);
    const std::vector<Tool> handlers{
        {{.name = "read_file"},
         Capability::READ,
         [workspace](const Json& args, std::stop_token stop) { return workspace->read(args, stop); }},
        {{.name = "list_files"},
         Capability::READ,
         [workspace](const Json& args, std::stop_token stop) { return workspace->list(args, stop); }},
        {{.name = "write_file"},
         Capability::WRITE,
         [workspace](const Json& args, std::stop_token stop) { return workspace->write(args, stop); }},
        {{.name = "make_directory"},
         Capability::WRITE,
         [workspace](const Json& args, std::stop_token stop) { return workspace->mkdir(args, stop); }},
        {{.name = "run_command"},
         Capability::EXECUTE,
         [workspace](const Json& args, std::stop_token stop) { return workspace->run(args, stop); }},
    };
    return configure_tools(config, handlers);
#else
    return std::unexpected(
        Error{ErrorCode::UNSUPPORTED, "workspace tools require POSIX descriptor-relative file APIs"});
#endif
}

} // namespace kidi::code
