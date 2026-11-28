#include "kidi/agent/net/http.h"

#include "kidi/agent/agent.h"
#include "kidi/agent/text.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string_view>

#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <crt_externs.h>
#else
extern char** environ;
#endif

namespace kidi::agent::net {
namespace {

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

constexpr auto USER_AGENT = "Mozilla/5.0 (compatible; KidiAgent/1.0; +https://github.com/thammegowda/kidi)";

// A curl config-file value: quoted, with backslashes, quotes and control characters escaped.
auto curl_quote(std::string_view text) -> std::string {
    std::string out = "\"";
    for (const auto character : text) {
        if (character == '\\' || character == '"') {
            out += '\\';
            out += character;
        } else if (character == '\n') {
            out += "\\n";
        } else if (character == '\r') {
            out += "\\r";
        } else if (character == '\t') {
            out += "\\t";
        } else {
            out += character;
        }
    }
    return out + '"';
}

constexpr std::string_view WRITE_OUT_MARK = "\n@@kidi-curl@@";

auto curl_config(const HttpRequest& request) -> std::string {
    const auto seconds = std::max<long long>(1, request.timeout.count());
    std::string config;
    const auto option = [&](std::string_view name, std::string_view value) {
        config += name;
        config += " = ";
        config += curl_quote(value);
        config += '\n';
    };
    // Redirects are returned, not followed, so that each hop's address can be checked.
    config += "silent\nshow-error\n";
    config += "globoff\n"; // URLs keep their [] and {} characters
    option("proto", "=http,https");
    option("max-time", std::to_string(seconds));
    option("connect-timeout", std::to_string(std::min<long long>(seconds, 10)));
    option("user-agent", USER_AGENT);
    for (const auto& header : request.headers) option("header", header);
    if (request.body) option("data-binary", *request.body);
    if (!request.resolve.empty()) option("resolve", request.resolve);
    option("write-out", std::string(WRITE_OUT_MARK) + "%{http_code}\t%{redirect_url}\t%{content_type}\t%{header_json}");
    option("url", request.url);
    return config;
}

// A curl exit status as a sentence; network failures are IO errors, which callers may retry.
auto curl_failure(int status, std::string_view stderr_text, std::chrono::seconds timeout) -> Error {
    const auto detail = one_line(stderr_text, 200);
    const auto with_detail = [&](std::string text) { return detail.empty() ? text : text + " (" + detail + ")"; };
    switch (status) {
        case 6:
            return {ErrorCode::IO, with_detail("could not resolve the host name")};
        case 7:
            return {ErrorCode::IO, with_detail("could not connect to the server")};
        case 28:
            return {ErrorCode::IO, "timed out after " + std::to_string(timeout.count()) + " s"};
        case 35:
        case 51:
        case 58:
        case 59:
        case 60:
        case 77:
        case 83:
            return {ErrorCode::IO, with_detail("the secure (TLS) connection failed")};
        case 52:
            return {ErrorCode::IO, "the server closed the connection without replying"};
        case 16:
        case 18:
        case 55:
        case 56:
        case 92:
            return {ErrorCode::IO, with_detail("the connection broke while transferring")};
        case 1:
            return {ErrorCode::INVALID_ARGUMENT, with_detail("unsupported address scheme")};
        case 3:
            return {ErrorCode::INVALID_ARGUMENT, with_detail("malformed URL")};
        case 127:
            return {ErrorCode::UNSUPPORTED, "the curl program is not installed"};
        default:
            return {ErrorCode::IO, with_detail("curl failed with exit status " + std::to_string(status))};
    }
}

struct Pipe {
    std::array<int, 2> fds{-1, -1};
    Pipe() = default;
    Pipe(const Pipe&) = delete;
    Pipe& operator=(const Pipe&) = delete;
    ~Pipe() {
        close(0);
        close(1);
    }
    auto open() -> bool {
        if (::pipe(fds.data()) != 0) return false;
        for (const auto fd : fds) ::fcntl(fd, F_SETFD, FD_CLOEXEC);
        return true;
    }
    auto close(int end) -> void {
        if (fds[static_cast<std::size_t>(end)] >= 0) ::close(fds[static_cast<std::size_t>(end)]);
        fds[static_cast<std::size_t>(end)] = -1;
    }
};

// Writes `text` with SIGPIPE held back in this thread, so a curl that exited early cannot kill the process.
auto write_quietly(int fd, std::string_view text) -> void {
    sigset_t pipe_signal, previous;
    sigemptyset(&pipe_signal);
    sigaddset(&pipe_signal, SIGPIPE);
    ::pthread_sigmask(SIG_BLOCK, &pipe_signal, &previous);
    while (!text.empty()) {
        const auto written = ::write(fd, text.data(), text.size());
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) break;
        text.remove_prefix(static_cast<std::size_t>(written));
    }
    sigset_t pending;
    sigpending(&pending);
    if (sigismember(&pending, SIGPIPE)) {
        int signal = 0;
        sigwait(&pipe_signal, &signal);
    }
    ::pthread_sigmask(SIG_SETMASK, &previous, nullptr);
}

auto run_curl(const HttpRequest& request, std::stop_token stop) -> Result<HttpResponse> {
    Pipe input, output, errors;
    if (!input.open() || !output.open() || !errors.open())
        return failure(ErrorCode::IO, std::string("cannot create pipes for curl: ") + std::strerror(errno));
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, input.fds[0], STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&actions, output.fds[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, errors.fds[1], STDERR_FILENO);
    // Hosts block or ignore signals (for example to sigwait for Ctrl-C); curl gets the defaults back.
    posix_spawnattr_t attributes;
    posix_spawnattr_init(&attributes);
    sigset_t none, defaults;
    sigemptyset(&none);
    sigemptyset(&defaults);
    for (const auto signal : {SIGPIPE, SIGINT, SIGTERM, SIGHUP}) sigaddset(&defaults, signal);
    posix_spawnattr_setsigmask(&attributes, &none);
    posix_spawnattr_setsigdefault(&attributes, &defaults);
    posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF);
    // -q skips ~/.curlrc; the request itself arrives on standard input, keeping API keys off the command line.
    std::array<std::string, 4> arguments{"curl", "-q", "--config", "-"};
    std::array<char*, 5> argv{arguments[0].data(), arguments[1].data(), arguments[2].data(), arguments[3].data(),
                              nullptr};
#if defined(__APPLE__)
    char** environment = *_NSGetEnviron();
#else
    char** environment = environ;
#endif
    pid_t pid = 0;
    const int spawned = posix_spawnp(&pid, "curl", &actions, &attributes, argv.data(), environment);
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attributes);
    if (spawned == ENOENT) return failure(ErrorCode::UNSUPPORTED, "the curl program is not installed");
    if (spawned != 0) return failure(ErrorCode::IO, std::string("cannot start curl: ") + std::strerror(spawned));
    input.close(0);
    output.close(1);
    errors.close(1);
    write_quietly(input.fds[1], curl_config(request));
    input.close(1);

    enum class Stop { NONE, CANCELLED, TIMED_OUT, TOO_LARGE } stopped = Stop::NONE;
    std::string body, diagnostics;
    const auto deadline = Clock::now() + request.timeout + 5s; // curl enforces the timeout; this is a backstop
    std::array<char, 65536> buffer;
    while (output.fds[0] >= 0 || errors.fds[0] >= 0) {
        if (stop.stop_requested()) stopped = Stop::CANCELLED;
        else if (Clock::now() > deadline) stopped = Stop::TIMED_OUT;
        else if (body.size() > request.max_bytes + 4096) stopped = Stop::TOO_LARGE;
        if (stopped != Stop::NONE) {
            ::kill(pid, SIGKILL);
            break;
        }
        std::array<pollfd, 2> polled{pollfd{output.fds[0], POLLIN, 0}, pollfd{errors.fds[0], POLLIN, 0}};
        if (::poll(polled.data(), polled.size(), 100) < 0 && errno != EINTR) break;
        for (std::size_t index = 0; index < polled.size(); ++index) {
            if (polled[index].fd < 0 || !(polled[index].revents & (POLLIN | POLLHUP | POLLERR))) continue;
            const auto count = ::read(polled[index].fd, buffer.data(), buffer.size());
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) {
                (index == 0 ? output : errors).close(0);
                continue;
            }
            auto& sink = index == 0 ? body : diagnostics;
            if (index == 0 || sink.size() < 4096) sink.append(buffer.data(), static_cast<std::size_t>(count));
        }
    }
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    switch (stopped) {
        case Stop::CANCELLED:
            return cancelled();
        case Stop::TIMED_OUT:
            return failure(ErrorCode::IO, "timed out after " + std::to_string(request.timeout.count()) + " s");
        case Stop::TOO_LARGE:
            return failure(ErrorCode::INVALID_ARGUMENT,
                           "the response is larger than " + std::to_string(request.max_bytes / (1024 * 1024)) + " MiB");
        case Stop::NONE:
            break;
    }
    if (!WIFEXITED(status)) return failure(ErrorCode::IO, "curl stopped unexpectedly");
    if (WEXITSTATUS(status) != 0)
        return std::unexpected(curl_failure(WEXITSTATUS(status), diagnostics, request.timeout));
    const auto mark = body.rfind(WRITE_OUT_MARK);
    if (mark == std::string::npos) return failure(ErrorCode::IO, "curl returned an incomplete response");
    const auto meta = std::string_view(body).substr(mark + WRITE_OUT_MARK.size());
    HttpResponse response;
    const auto first_tab = meta.find('\t');
    const auto second_tab = first_tab == std::string_view::npos ? first_tab : meta.find('\t', first_tab + 1);
    response.status = std::atoi(std::string(meta.substr(0, first_tab)).c_str());
    if (second_tab != std::string_view::npos) {
        response.redirect_url = std::string(meta.substr(first_tab + 1, second_tab - first_tab - 1));
        const auto third_tab = meta.find('\t', second_tab + 1);
        response.content_type = std::string(meta.substr(second_tab + 1, third_tab - second_tab - 1));
        if (third_tab != std::string_view::npos) {
            const auto headers = nlohmann::json::parse(meta.substr(third_tab + 1), nullptr, false);
            if (headers.is_object())
                for (const auto& [name, values] : headers.items())
                    if (values.is_array() && !values.empty() && values[0].is_string())
                        response.headers.emplace(lower(name), values[0].get<std::string>());
        }
    }
    body.resize(mark);
    response.body = std::move(body);
    return response;
}


} // namespace

auto curl_transport() -> HttpTransport { return run_curl; }

auto url_encode(std::string_view text) -> std::string {
    static constexpr std::string_view HEX = "0123456789ABCDEF";
    std::string out;
    for (const auto byte : text) {
        const auto value = static_cast<unsigned char>(byte);
        if ((value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') || (value >= '0' && value <= '9') ||
            value == '-' || value == '_' || value == '.' || value == '~') {
            out += static_cast<char>(value);
        } else {
            out += '%';
            out += HEX[value >> 4];
            out += HEX[value & 0xF];
        }
    }
    return out;
}

} // namespace kidi::agent::net
