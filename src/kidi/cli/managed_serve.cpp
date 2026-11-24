#include "kidi/cli/managed_serve.h"

#include "kidi/checkpoint/hub.h"
#include "kidi/cli/log.h"
#include "kidi/code/chat_transport.h"
#include "kidi/inference/decoder.h"
#include "kidi/server/routes.h"

#include <httplib.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <system_error>
#include <thread>
#include <utility>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

namespace kidi::cli {
namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

constexpr auto POLL_INTERVAL = 100ms;
constexpr auto IDLE_POLL_INTERVAL = 1s;
constexpr auto IDLE_TIMEOUT = 10min;
constexpr auto STATE_NAME = "server.json";
constexpr auto DEFAULT_MODEL = "@google/gemma-4-E2B-it-qat-mobile-transformers";
constexpr auto FULL_MODEL = "@google/gemma-4-E2B-it";

class File {
public:
    explicit File(int fd) : fd_(fd) {}
    ~File() {
        if (fd_ >= 0) close(fd_);
    }
    File(File&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
    File(const File&) = delete;
    auto operator=(const File&) -> File& = delete;
    auto get() const -> int { return fd_; }

private:
    int fd_;
};

struct ClientLease {
    File file;
    ~ClientLease() {
        if (futimens(file.get(), nullptr) < 0) log::error("cannot record managed server session end");
    }
};

[[noreturn]] auto fail_io(std::string message) -> void {
    throw std::system_error(errno, std::generic_category(), std::move(message));
}

auto open_file(const fs::path& path, int flags) -> File {
    const auto fd = open(path.c_str(), flags | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) fail_io("cannot open " + path.string());
    if (fd >= 3) return File(fd);
    const auto copy = fcntl(fd, F_DUPFD_CLOEXEC, 3);
    const auto error = errno;
    close(fd);
    if (copy < 0) throw std::system_error(error, std::generic_category(), "cannot duplicate descriptor");
    return File(copy);
}

auto unlock(int fd) -> void {
    while (flock(fd, LOCK_UN) < 0)
        if (errno != EINTR) fail_io("cannot unlock managed server state");
}

auto try_lock(const File& file, int operation) -> bool {
    while (flock(file.get(), operation | LOCK_NB) < 0) {
        if (errno == EWOULDBLOCK) return false;
        if (errno != EINTR) fail_io("cannot lock managed server state");
    }
    return true;
}

auto private_directory(const fs::path& path, bool create) -> bool {
    if (create && mkdir(path.c_str(), 0700) < 0 && errno != EEXIST) fail_io("cannot create " + path.string());
    struct stat status{};
    return lstat(path.c_str(), &status) == 0 && S_ISDIR(status.st_mode) && status.st_uid == geteuid() &&
           (status.st_mode & 0777) == 0700;
}

auto runtime_directory() -> fs::path {
    auto parent = fs::path("/tmp");
    if (const auto* runtime = std::getenv("XDG_RUNTIME_DIR");
        runtime && fs::path(runtime).is_absolute() && private_directory(runtime, false))
        parent = runtime;
    const auto root = parent / ("kidi-code-" + std::to_string(geteuid()));
    if (!private_directory(root, true)) throw std::runtime_error("not a private directory: " + root.string());
    return root;
}

auto executable() -> fs::path {
#ifdef __APPLE__
    std::uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string path(size, '\0');
    if (_NSGetExecutablePath(path.data(), &size) != 0) throw std::runtime_error("cannot locate kidi executable");
    return fs::canonical(path.c_str());
#else
    return "/proc/self/exe";
#endif
}

auto choose_model(std::string model) -> std::string {
    if (model.empty()) {
        const checkpoint::Hub hub;
        if (!hub.resolve(DEFAULT_MODEL) && hub.resolve(FULL_MODEL)) return FULL_MODEL;
        return DEFAULT_MODEL;
    }
    return model.starts_with('@') ? model : fs::canonical(model).string();
}

auto start_server(const fs::path& exe, const fs::path& dir, const File& startup, const std::string& model,
                  const fs::path& cache, const Namespace& arguments) -> void {
    const auto log = open_file(dir / "serve.log", O_WRONLY | O_CREAT | O_TRUNC);
    const auto input = open_file("/dev/null", O_RDONLY);
    std::vector<std::string> flags{
        exe.string(),
        "serve",
        "--model=" + model,
        "--cache=" + cache.string(),
        "--backend=" + arguments.get<std::string>("backend"),
        "--threads=" + std::to_string(arguments.get<std::int32_t>("threads")),
        "--context-size=" + std::to_string(arguments.get<std::size_t>("context_size")),
        "--cache-tokens=" + std::to_string(std::max<std::size_t>(16384, arguments.get<std::size_t>("context_size"))),
        arguments.get<bool>("prompt_lookup") ? "--max-active=1" : "--max-active=2",
        "--managed-dir=" + dir.string(),
        "--managed-lock-fd=" + std::to_string(startup.get()),
    };
    if (arguments.get<bool>("prompt_lookup")) {
        flags.push_back("--prompt-lookup");
        flags.push_back("--draft-tokens=" + std::to_string(arguments.get<std::size_t>("draft_tokens")));
    }
    std::vector<char*> argv;
    for (auto& flag : flags) argv.push_back(flag.data());
    argv.push_back(nullptr);
    const auto launcher = fork();
    if (launcher < 0) fail_io("cannot start managed server");
    if (launcher == 0) {
        // The intermediate child is reaped by the client; the daemon survives the client's exit.
        if (setsid() < 0) _exit(1);
        const auto daemon = fork();
        if (daemon < 0) _exit(1);
        if (daemon > 0) _exit(0);
        if (dup2(input.get(), STDIN_FILENO) < 0 || dup2(log.get(), STDOUT_FILENO) < 0 ||
            dup2(log.get(), STDERR_FILENO) < 0 || fcntl(startup.get(), F_SETFD, 0) < 0)
            _exit(1);
        execv(argv[0], argv.data());
        constexpr char message[] = "cannot exec kidi serve\n";
        [[maybe_unused]] const auto written = write(STDERR_FILENO, message, sizeof(message) - 1);
        _exit(127);
    }
    int status;
    while (waitpid(launcher, &status, 0) < 0)
        if (errno != EINTR) fail_io("cannot wait for managed server launcher");
    if (!WIFEXITED(status) || WEXITSTATUS(status)) throw std::runtime_error("cannot detach managed server");
}

struct Endpoint {
    int port;
    pid_t pid;
};

auto read_endpoint(const fs::path& state) -> std::optional<Endpoint> {
    std::ifstream in(state);
    std::array<char, 256> bytes{};
    in.read(bytes.data(), bytes.size());
    if (in.gcount() == 0 || !in.eof()) return {};
    const auto json = code::Json::parse(bytes.data(), bytes.data() + in.gcount(), nullptr, false);
    if (!json.is_object() || !json.contains("port") || !json["port"].is_number_unsigned() || !json.contains("pid") ||
        !json["pid"].is_number_unsigned())
        return {};
    const auto port = json["port"].get<std::uint64_t>();
    const auto pid = json["pid"].get<std::uint64_t>();
    if (!port || port > 65535 || pid <= 1 || pid > std::numeric_limits<pid_t>::max()) return {};
    return Endpoint{static_cast<int>(port), static_cast<pid_t>(pid)};
}

auto accessible(Endpoint endpoint) -> bool {
    httplib::Client http("127.0.0.1", endpoint.port);
    http.set_connection_timeout(500ms);
    http.set_max_timeout(500ms);
    http.set_payload_max_length(256);
    const auto health = http.Get("/kidi-code/ready");
    return health && health->status == 200 && health->body == std::to_string(endpoint.pid);
}

// Guards against signalling an unrelated process that reused a stale PID.
auto managed_process(pid_t pid, const fs::path& dir) -> bool {
#ifdef __linux__
    std::ifstream in("/proc/" + std::to_string(pid) + "/cmdline", std::ios::binary);
    const auto expected = "--managed-dir=" + dir.string();
    for (std::string argument; std::getline(in, argument, '\0');)
        if (argument == expected) return true;
    return false;
#else
    (void)dir;
    return kill(pid, 0) == 0;
#endif
}

auto stop_server(const fs::path& dir) -> Result<void> {
    const auto endpoint = read_endpoint(dir / STATE_NAME);
    if (!endpoint || !(managed_process(endpoint->pid, dir) || accessible(*endpoint))) return {};
    log::info("stopping background inference (PID {})", endpoint->pid);
    for (const auto [signal, wait] : {std::pair{SIGTERM, 30s}, std::pair{SIGKILL, 10s}}) {
        if (kill(endpoint->pid, signal) < 0 && errno != ESRCH) fail_io("cannot stop managed server");
        for (const auto deadline = Clock::now() + wait; Clock::now() < deadline;) {
            if (!managed_process(endpoint->pid, dir)) return {};
            std::this_thread::sleep_for(POLL_INTERVAL);
        }
    }
    return std::unexpected(
        Error{ErrorCode::RUNTIME, "background inference PID " + std::to_string(endpoint->pid) + " did not exit"});
}

auto ensure_server(const fs::path& exe, const fs::path& dir, const std::string& model, const fs::path& cache,
                   const Namespace& arguments, bool restart) -> Result<Endpoint> {
    bool waiting_for_server = false;
    const auto state = dir / STATE_NAME;
    const auto deadline = Clock::now() + 30min;
    while (Clock::now() < deadline) {
        {
            const auto startup = open_file(dir / "server.lock", O_RDWR | O_CREAT);
            if (try_lock(startup, LOCK_EX)) {
                if (std::exchange(restart, false))
                    if (auto stopped = stop_server(dir); !stopped) return std::unexpected(stopped.error());
                if (const auto endpoint = read_endpoint(state); endpoint && accessible(*endpoint)) return *endpoint;
                if (waiting_for_server && !fs::exists(state))
                    return std::unexpected(
                        Error{ErrorCode::RUNTIME,
                              "background inference exited during startup; see " + (dir / "serve.log").string()});
                fs::remove(state);
                log::info("starting background inference for {}; log: {}", model, (dir / "serve.log").string());
                start_server(exe, dir, startup, model, cache, arguments);
            }
        }
        waiting_for_server = true;
        std::this_thread::sleep_for(POLL_INTERVAL);
    }
    return std::unexpected(
        Error{ErrorCode::IO, "background inference did not become ready; see " + (dir / "serve.log").string()});
}

auto managed_provider(code::ChatClientOptions options, const Namespace& arguments) -> Result<code::Provider> {
    const auto lookup = arguments.get<bool>("prompt_lookup");
    const auto drafts = arguments.get<std::size_t>("draft_tokens");
    if ((!lookup && drafts != 3) || !drafts || drafts > inference::MAX_DRAFT_TOKENS)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT,
                                     "--draft-tokens requires --prompt-lookup and a cap from 1 to 16"});
    options.model = choose_model(std::move(options.model));
    const auto exe = executable();
    struct stat binary{};
    if (stat(exe.c_str(), &binary) < 0) fail_io("cannot identify kidi executable");
    const auto cache = fs::absolute(checkpoint::Hub::default_cache()).lexically_normal();
    const auto& backend = arguments.get<std::string>("backend");
    const auto* visible_devices =
        (backend == "auto" || backend.starts_with("cuda")) ? std::getenv("CUDA_VISIBLE_DEVICES") : nullptr;
    const auto identity =
        code::Json::array({binary.st_dev, binary.st_ino, binary.st_mtime, binary.st_size, options.model, cache.string(),
                           backend, visible_devices ? code::Json(visible_devices) : code::Json(nullptr),
                           arguments.get<std::size_t>("context_size"), arguments.get<std::int32_t>("threads"),
                           arguments.get<bool>("prompt_lookup"), arguments.get<std::size_t>("draft_tokens")})
            .dump();
    const auto dir = runtime_directory() / std::to_string(std::hash<std::string>{}(identity));
    if (!private_directory(dir, true)) throw std::runtime_error("not a private directory: " + dir.string());
    auto lease = std::make_shared<ClientLease>(open_file(dir / "clients.lock", O_RDWR | O_CREAT));
    const auto deadline = Clock::now() + IDLE_TIMEOUT;
    while (!try_lock(lease->file, LOCK_SH)) {
        if (Clock::now() >= deadline)
            return std::unexpected(Error{ErrorCode::IO, "background inference did not finish shutting down"});
        std::this_thread::sleep_for(POLL_INTERVAL);
    }
    if (futimens(lease->file.get(), nullptr) < 0) fail_io("cannot record managed server session start");
    auto endpoint = ensure_server(exe, dir, options.model, cache, arguments, arguments.get<bool>("restart"));
    if (!endpoint) return std::unexpected(endpoint.error());
    options.base_url = "http://127.0.0.1:" + std::to_string(endpoint->port) + "/v1";

    return code::completion_provider(
        std::move(options),
        [lease = std::move(lease), dir, endpoint = *endpoint](const code::ChatClientOptions& http_options,
                                                              std::string payload, std::stop_token stop,
                                                              std::string_view route) -> Result<std::string> {
            auto result = code::http_completion(http_options, std::move(payload), stop, endpoint.pid, route);
            if (!result) result.error().message += "; see " + (dir / "serve.log").string();
            return result;
        });
}

} // namespace

auto configure_managed_code_parser(ArgumentParser& parser) -> void {
    code::configure_parser(parser, true);
    parser.add_argument("--backend")
        .default_value(std::string("auto"))
        .help("inference backend; auto prefers CUDA, then Metal, then CPU");
    parser.add_argument("--context-size")
        .dest("context_size")
        .default_value<std::size_t>(0)
        .help("background server context tokens (default: model configuration); larger contexts use more memory");
    // CPU decode scales to roughly 16 threads; beyond that the per-token dispatch overhead dominates.
    constexpr std::int32_t DEFAULT_THREADS = 16;
    parser.add_argument("-j", "--threads")
        .default_value(std::clamp(static_cast<std::int32_t>(std::thread::hardware_concurrency()), 1, DEFAULT_THREADS))
        .help("CPU inference threads for the background server");
    parser.add_argument("--restart")
        .action(Action::STORE_TRUE)
        .help("stop the background server for this model/backend/context and start a fresh one");
    parser.add_argument("--prompt-lookup")
        .dest("prompt_lookup")
        .action(Action::STORE_TRUE)
        .help("enable CPU Inkling-2 prompt-lookup speculation; requires --temperature 0");
    parser.add_argument("--draft-tokens")
        .dest("draft_tokens")
        .default_value<std::size_t>(3)
        .help("adaptive prompt-lookup draft cap, 1 to 16 (starts at 1)");
}

auto configure_managed_serve_parser(ArgumentParser& parser) -> void {
    parser.add_argument("--managed-dir").dest("managed_dir").default_value(fs::path{}).help("internal");
    parser.add_argument("--managed-lock-fd").dest("managed_lock_fd").default_value(-1).help("internal");
}

auto managed_code_command(const Namespace& arguments) -> int {
    return code::code_command(
        arguments, [&](code::ChatClientOptions options) { return managed_provider(std::move(options), arguments); });
}

auto protect_managed_lock(const Namespace& arguments) -> void {
    if (arguments.get<fs::path>("managed_dir").empty()) return;
    const auto fd = arguments.get<int>("managed_lock_fd");
    if (fd < 3) throw std::invalid_argument("managed server requires an inherited server lock");
    struct stat inherited{}, expected{};
    const auto path = arguments.get<fs::path>("managed_dir") / "server.lock";
    if (fstat(fd, &inherited) < 0 || lstat(path.c_str(), &expected) < 0 || inherited.st_dev != expected.st_dev ||
        inherited.st_ino != expected.st_ino || !S_ISREG(expected.st_mode))
        throw std::invalid_argument("invalid inherited managed server lock");
    if (fcntl(fd, F_SETFD, FD_CLOEXEC) < 0) fail_io("cannot protect managed server lock");
}

auto serve_managed(inference::Generator& generator, const ServeOptions& options, const Namespace& arguments) -> int {
    const auto& dir = arguments.get<fs::path>("managed_dir");
    const auto state = dir / STATE_NAME;
    const auto pid = getpid();
    const File startup(arguments.get<int>("managed_lock_fd"));
    const auto clients = dir / "clients.lock";
    const auto lease = open_file(clients, O_RDWR);
    auto http = liteapp::HttpServer::create();
    if (!http) {
        log::error("cannot create managed server: {}", http.error().message);
        return 1;
    }
    server::ChatEngine engine(generator);
    server::mount_chat_routes(*http, engine, {.model_name = options.model_name, .defaults = options.generation});
    http->get("/kidi-code/ready", [pid](const auto&) { return liteapp::Response::text(std::to_string(pid)); });
    const auto port = http->bind();
    if (!port) {
        log::error("cannot bind managed server: {}", port.error().message);
        return 1;
    }
    auto running = http->start();
    if (!running) {
        log::error("cannot start managed server: {}", running.error().message);
        return 1;
    }
    const auto staging = dir / ("server." + std::to_string(pid) + ".tmp");
    std::ofstream out(staging, std::ios::trunc);
    out << code::Json{{"port", *port}, {"pid", pid}}.dump() << '\n';
    out.close();
    if (!out) throw std::runtime_error("cannot write managed server state: " + staging.string());
    fs::rename(staging, state);
    unlock(startup.get());
    log::info("serving {} on http://127.0.0.1:{}/v1 (PID {}); state: {}", options.model_name, *port, pid,
              state.string());

    const auto owns_state = [&] {
        const auto endpoint = read_endpoint(state);
        return endpoint && endpoint->pid == pid && endpoint->port == *port;
    };
    std::jthread idle([&](std::stop_token stop) {
        try {
            auto last_client = Clock::now();
            auto last_release = fs::last_write_time(clients);
            bool had_clients = false;
            while (!stop.stop_requested()) {
                std::this_thread::sleep_for(IDLE_POLL_INTERVAL);
                if (stop.stop_requested()) return;
                if (!owns_state()) break;
                if (try_lock(lease, LOCK_EX)) {
                    // The file timestamp also captures sessions shorter than the polling interval.
                    const auto released = fs::last_write_time(clients);
                    if (had_clients || released != last_release) {
                        last_release = released;
                        last_client = Clock::now();
                    }
                    had_clients = false;
                    if (Clock::now() - last_client >= IDLE_TIMEOUT) break;
                    unlock(lease.get());
                } else {
                    had_clients = true;
                }
            }
        } catch (const std::system_error& error) {
            log::error("managed server idle check: {}", error.what());
        }
        http->stop();
    });
    const auto listened = running->wait();
    idle.request_stop();
    idle.join();
    if (try_lock(startup, LOCK_EX) && owns_state()) fs::remove(state);
    if (listened) return 0;
    log::error("managed server listener failed: {}", listened.error().message);
    return 1;
}

} // namespace kidi::cli
