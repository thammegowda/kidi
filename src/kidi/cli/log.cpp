#include "kidi/cli/log.h"

#include <atomic>
#include <cctype>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace kidi::cli::log {
namespace {

std::atomic<int> active_level{static_cast<int>(Level::INFO)};

auto stderr_initial_rdbuf() -> std::streambuf* {
    static std::streambuf* const initial = std::cerr.rdbuf();
    return initial;
}

auto stderr_colors_enabled() -> bool {
    if (std::cerr.rdbuf() != stderr_initial_rdbuf()) return false;
    const char* no_color = std::getenv("NO_COLOR");
    if (no_color && *no_color != '\0') return false;
    const char* term = std::getenv("TERM");
    if (term && std::string_view(term) == "dumb") return false;
#if defined(_WIN32)
    return _isatty(_fileno(stderr)) != 0;
#else
    return isatty(STDERR_FILENO) != 0;
#endif
}

auto parse_level_token(std::string_view raw) -> std::optional<Level> {
    while (!raw.empty() && std::isspace(static_cast<unsigned char>(raw.front()))) raw.remove_prefix(1);
    while (!raw.empty() && std::isspace(static_cast<unsigned char>(raw.back()))) raw.remove_suffix(1);
    std::string lower;
    lower.reserve(raw.size());
    for (unsigned char ch : raw) lower.push_back(static_cast<char>(std::tolower(ch)));
    if (lower == "trace") return Level::TRACE;
    if (lower == "debug") return Level::DEBUG;
    if (lower == "info") return Level::INFO;
    if (lower == "warn" || lower == "warning") return Level::WARN;
    if (lower == "err" || lower == "error") return Level::ERROR;
    if (lower == "critical") return Level::CRITICAL;
    if (lower == "off") return Level::OFF;
    return std::nullopt;
}

auto level_label(Level level) -> std::string_view {
    switch (level) {
        case Level::TRACE:
            return "trace";
        case Level::DEBUG:
            return "debug";
        case Level::INFO:
            return "info";
        case Level::WARN:
            return "warning";
        case Level::ERROR:
            return "error";
        case Level::CRITICAL:
            return "critical";
        case Level::OFF:
            return "off";
    }
    return "info";
}

auto level_ansi(Level level) -> std::string_view {
    switch (level) {
        case Level::INFO:
            return "\033[32m";
        case Level::WARN:
            return "\033[33m\033[1m";
        case Level::ERROR:
        case Level::CRITICAL:
            return "\033[31m\033[1m";
        default:
            return "";
    }
}

auto log_mutex() -> std::mutex& {
    static std::mutex mutex;
    return mutex;
}

} // namespace

void init_from_env() {
    (void)stderr_initial_rdbuf();
    Level configured = Level::INFO;
    const char* env = std::getenv("KIDI_LOG_LEVEL");
    if (!env || *env == '\0') env = std::getenv("SPDLOG_LEVEL");
    if (env && *env != '\0') {
        std::string_view spec(env);
        while (!spec.empty()) {
            const auto comma = spec.find(',');
            const auto entry = spec.substr(0, comma);
            const auto eq = entry.find('=');
            if (eq == std::string_view::npos) {
                if (const auto parsed = parse_level_token(entry)) configured = *parsed;
            } else if (entry.substr(0, eq) == "kidi") {
                if (const auto parsed = parse_level_token(entry.substr(eq + 1))) configured = *parsed;
            }
            if (comma == std::string_view::npos) break;
            spec.remove_prefix(comma + 1);
        }
    }
    active_level.store(static_cast<int>(configured), std::memory_order_relaxed);
}

void write(Level level, std::string_view message) {
    if (static_cast<int>(level) < active_level.load(std::memory_order_relaxed)) return;
    std::lock_guard lock(log_mutex());
    const bool color = stderr_colors_enabled();
    const auto ansi = color ? level_ansi(level) : std::string_view{};
    std::cerr << "[kidi] [";
    if (!ansi.empty()) std::cerr << ansi;
    std::cerr << level_label(level);
    if (!ansi.empty()) std::cerr << "\033[m";
    std::cerr << "] " << message << '\n';
}

void write_args(Level level, std::string_view fmt, std::span<const Arg> args) {
    if (static_cast<int>(level) < active_level.load(std::memory_order_relaxed)) return;
    std::string out;
    out.reserve(fmt.size() + args.size() * 16);
    std::size_t arg_index = 0;
    for (std::size_t i = 0; i < fmt.size();) {
        if (i + 1 < fmt.size() && fmt[i] == '{' && fmt[i + 1] == '}' && arg_index < args.size()) {
            const Arg& a = args[arg_index++];
            char buf[64];
            switch (a.kind) {
                case Arg::Kind::STRING:
                    out.append(a.str);
                    break;
                case Arg::Kind::INT: {
                    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), a.i64);
                    if (ec == std::errc{}) out.append(buf, ptr);
                    break;
                }
                case Arg::Kind::UINT: {
                    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), a.u64);
                    if (ec == std::errc{}) out.append(buf, ptr);
                    break;
                }
                case Arg::Kind::DOUBLE: {
                    const int n = std::snprintf(buf, sizeof(buf), "%g", a.f64);
                    if (n > 0) out.append(buf, static_cast<std::size_t>(n));
                    break;
                }
            }
            i += 2;
        } else {
            out.push_back(fmt[i++]);
        }
    }
    write(level, out);
}

} // namespace kidi::cli::log
