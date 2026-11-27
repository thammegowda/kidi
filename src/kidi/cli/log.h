#pragma once

#include <concepts>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

namespace kidi::cli::log {

enum class Level { TRACE = 0, DEBUG = 1, INFO = 2, WARN = 3, ERROR = 4, CRITICAL = 5, OFF = 6 };

struct Arg {
    enum class Kind { STRING, INT, UINT, DOUBLE } kind;
    std::string_view str{};
    std::int64_t i64 = 0;
    std::uint64_t u64 = 0;
    double f64 = 0.0;

    Arg(const char* value) noexcept : kind(Kind::STRING), str(value ? value : "") {}
    Arg(std::string_view value) noexcept : kind(Kind::STRING), str(value) {}
    Arg(const std::string& value) noexcept : kind(Kind::STRING), str(value) {}

    template <std::signed_integral T>
        requires(!std::same_as<std::remove_cv_t<T>, char>)
    Arg(T value) noexcept : kind(Kind::INT), i64(static_cast<std::int64_t>(value)) {}

    template <std::unsigned_integral T>
        requires(!std::same_as<std::remove_cv_t<T>, bool> && !std::same_as<std::remove_cv_t<T>, char>)
    Arg(T value) noexcept : kind(Kind::UINT), u64(static_cast<std::uint64_t>(value)) {}

    Arg(bool value) noexcept : kind(Kind::STRING), str(value ? "true" : "false") {}
    Arg(double value) noexcept : kind(Kind::DOUBLE), f64(value) {}
    Arg(float value) noexcept : kind(Kind::DOUBLE), f64(static_cast<double>(value)) {}
};

/// Accepts trace, debug, info, warning, error, critical and off.
auto parse_level(std::string_view name) -> std::optional<Level>;
void set_level(Level level);
void write(Level level, std::string_view message);
void write_args(Level level, std::string_view fmt, std::span<const Arg> args);

template <typename... Args>
inline void info(std::string_view fmt, const Args&... args) {
    if constexpr (sizeof...(Args) == 0) {
        write(Level::INFO, fmt);
    } else {
        const Arg packed[]{Arg(args)...};
        write_args(Level::INFO, fmt, packed);
    }
}

template <typename... Args>
inline void warn(std::string_view fmt, const Args&... args) {
    if constexpr (sizeof...(Args) == 0) {
        write(Level::WARN, fmt);
    } else {
        const Arg packed[]{Arg(args)...};
        write_args(Level::WARN, fmt, packed);
    }
}

template <typename... Args>
inline void error(std::string_view fmt, const Args&... args) {
    if constexpr (sizeof...(Args) == 0) {
        write(Level::ERROR, fmt);
    } else {
        const Arg packed[]{Arg(args)...};
        write_args(Level::ERROR, fmt, packed);
    }
}

} // namespace kidi::cli::log
