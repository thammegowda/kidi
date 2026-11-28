#include "kidi/agent/net/sign_in.h"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <crt_externs.h>
#else
extern char** environ;
#endif

namespace kidi::agent::net {
namespace {

#if !defined(MSG_NOSIGNAL)
#define MSG_NOSIGNAL 0 // macOS: sockets set SO_NOSIGPIPE instead
#endif

// A descriptor closed on exec and, where the platform needs it, kept from raising SIGPIPE.
auto prepare(int fd) -> int {
    if (fd < 0) return fd;
    ::fcntl(fd, F_SETFD, FD_CLOEXEC);
#if defined(SO_NOSIGPIPE)
    const int on = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof(on));
#endif
    return fd;
}

struct Socket {
    int fd = -1;
    explicit Socket(int value) : fd(value) {}
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
    ~Socket() {
        if (fd >= 0) ::close(fd);
    }
};

auto send_all(int fd, std::string_view text) -> void {
    while (!text.empty()) {
        const auto sent = ::send(fd, text.data(), text.size(), MSG_NOSIGNAL);
        if (sent < 0 && errno == EINTR) continue;
        if (sent <= 0) return;
        text.remove_prefix(static_cast<std::size_t>(sent));
    }
}

// The request target of one HTTP request on `fd`, read within a few seconds.
auto request_target(int fd) -> std::string {
    std::string request;
    std::array<char, 4096> buffer;
    for (int waits = 0; request.find("\r\n") == std::string::npos && request.size() < 16384 && waits < 50; ++waits) {
        pollfd ready{fd, POLLIN, 0};
        if (::poll(&ready, 1, 100) <= 0) continue;
        const auto count = ::recv(fd, buffer.data(), buffer.size(), 0);
        if (count <= 0) break;
        request.append(buffer.data(), static_cast<std::size_t>(count));
    }
    const auto line = request.substr(0, request.find("\r\n"));
    const auto first = line.find(' ');
    const auto second = line.find(' ', first + 1);
    if (line.compare(0, first, "GET") != 0 || second == std::string::npos) return {};
    return line.substr(first + 1, second - first - 1);
}

} // namespace

auto sign_in_done_page() -> std::string {
    return "<!doctype html><meta charset=utf-8><title>Signed in</title>"
           "<body style=\"font:16px system-ui;margin:3em;text-align:center\"><h2>Signed in</h2>"
           "<p>You can close this tab and return to Kidi.</p><script>setTimeout(() => window.close(), 1500)</script>";
}

auto open_in_browser(const std::string& url) -> bool {
    if (std::getenv("SSH_CONNECTION") || std::getenv("SSH_TTY")) return false;
#if defined(__APPLE__)
    const char* program = "open";
    char** environment = *_NSGetEnviron();
#else
    if (!std::getenv("DISPLAY") && !std::getenv("WAYLAND_DISPLAY")) return false;
    const char* program = "xdg-open";
    char** environment = environ;
#endif
    std::string program_name = program, target = url;
    std::array<char*, 3> argv{program_name.data(), target.data(), nullptr};
    pid_t pid = 0;
    if (posix_spawnp(&pid, program, nullptr, nullptr, argv.data(), environment) != 0) return false;
    std::thread([pid] { ::waitpid(pid, nullptr, 0); }).detach();
    return true;
}

auto loopback_sign_in(std::function<void(const std::string& url, int port)> show, std::chrono::seconds timeout)
    -> Result<BrowserSignIn> {
    auto listener = std::make_shared<Socket>(prepare(::socket(AF_INET, SOCK_STREAM, 0)));
    if (listener->fd < 0) return std::unexpected(Error{ErrorCode::IO, std::string("socket: ") + std::strerror(errno)});
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t size = sizeof(address);
    if (::bind(listener->fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
        ::listen(listener->fd, 4) != 0 ||
        ::getsockname(listener->fd, reinterpret_cast<sockaddr*>(&address), &size) != 0)
        return std::unexpected(
            Error{ErrorCode::IO, std::string("cannot listen for the sign-in: ") + std::strerror(errno)});
    const int port = ntohs(address.sin_port);
    BrowserSignIn sign_in;
    sign_in.redirect_uri = "http://127.0.0.1:" + std::to_string(port) + "/oauth/callback";
    sign_in.visit = [listener, port, timeout, show = std::move(show)](const std::string& url,
                                                                      std::stop_token stop) -> Result<std::string> {
        show(url, port);
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (!stop.stop_requested() && std::chrono::steady_clock::now() < deadline) {
            pollfd ready{listener->fd, POLLIN, 0};
            if (::poll(&ready, 1, 200) <= 0) continue;
            const Socket connection(prepare(::accept(listener->fd, nullptr, nullptr)));
            if (connection.fd < 0) continue;
            const auto target = request_target(connection.fd);
            if (!target.starts_with("/oauth/callback")) {
                send_all(connection.fd, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                continue;
            }
            const auto page = sign_in_done_page();
            send_all(connection.fd, "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: " +
                                        std::to_string(page.size()) + "\r\nConnection: close\r\n\r\n" + page);
            return "http://127.0.0.1:" + std::to_string(port) + target;
        }
        if (stop.stop_requested()) return std::unexpected(Error{ErrorCode::RUNTIME, "cancelled"});
        return std::unexpected(Error{ErrorCode::RUNTIME, "no sign-in arrived within " +
                                                             std::to_string(timeout.count() / 60) + " minutes"});
    };
    return sign_in;
}

} // namespace kidi::agent::net
