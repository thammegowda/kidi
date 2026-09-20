#pragma once

#include <algorithm>
#include <any>
#include <charconv>
#include <concepts>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace kidi::cli {
namespace detail {

template <typename>
inline constexpr bool ALWAYS_FALSE = false;

template <typename T>
auto parse_argument(std::string_view value) -> T {
    if constexpr (std::same_as<T, std::string>) {
        return std::string(value);
    } else if constexpr (std::same_as<T, bool>) {
        if (value == "true" || value == "1") return true;
        if (value == "false" || value == "0") return false;
        throw std::invalid_argument("expected a boolean");
    } else if constexpr (std::integral<T>) {
        T result{};
        const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
        if (error != std::errc{} || end != value.data() + value.size()) {
            throw std::invalid_argument("expected an integer");
        }
        return result;
    } else if constexpr (std::floating_point<T>) {
        T result{};
        const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
        if (error != std::errc{} || end != value.data() + value.size()) {
            throw std::invalid_argument("expected a number");
        }
        return result;
    } else if constexpr (std::constructible_from<T, std::string>) {
        return T(std::string(value));
    } else {
        static_assert(ALWAYS_FALSE<T>, "unsupported argument type");
    }
}

template <typename T>
auto display_argument(const T& value) -> std::string {
    if constexpr (std::same_as<T, std::string>) {
        return value;
    } else if constexpr (std::same_as<T, bool>) {
        return value ? "true" : "false";
    } else if constexpr (requires(std::ostringstream& output) { output << value; }) {
        std::ostringstream output;
        output << value;
        return std::move(output).str();
    } else {
        return {};
    }
}

} // namespace detail

enum class Action {
    STORE,
    STORE_TRUE,
    STORE_FALSE,
};

enum class ParseStatus {
    READY,
    HELP,
    VERSION,
};

class ParseError : public std::runtime_error {
public:
    ParseError(std::string message, std::string usage);

    auto usage() const noexcept -> std::string_view;

private:
    std::string usage_;
};

class Namespace {
public:
    template <typename T>
    auto get(std::string_view destination) const -> const T& {
        const auto iterator = values_.find(std::string(destination));
        if (iterator == values_.end()) {
            throw std::logic_error("argument has no value: " + std::string(destination));
        }
        const auto* value = std::any_cast<T>(&iterator->second);
        if (value == nullptr) {
            throw std::logic_error("argument has a different type: " + std::string(destination));
        }
        return *value;
    }

    auto contains(std::string_view destination) const -> bool;
    auto status() const noexcept -> ParseStatus;
    auto should_exit() const noexcept -> bool;
    auto output() const noexcept -> std::string_view;

private:
    friend class ArgumentParser;

    auto set(std::string destination, std::any value) -> void;

    std::unordered_map<std::string, std::any> values_;
    ParseStatus status_ = ParseStatus::READY;
    std::string output_;
};

class Argument {
public:
    auto help(std::string value) -> Argument&;
    auto metavar(std::string value) -> Argument&;
    auto dest(std::string value) -> Argument&;
    auto required(bool value = true) noexcept -> Argument&;
    auto action(Action value) -> Argument&;
    auto choices(std::initializer_list<std::string_view> values) -> Argument&;

    template <typename T>
    auto type() -> Argument& {
        if (action_ != Action::STORE) throw std::logic_error("flag actions cannot have a value type");
        converter_ = [](std::string_view value) { return std::any(detail::parse_argument<T>(value)); };
        return *this;
    }

    template <typename T>
    auto default_value(T value) -> Argument& {
        type<T>();
        default_display_ = detail::display_argument(value);
        default_value_ = std::move(value);
        return *this;
    }

private:
    friend class ArgumentParser;

    explicit Argument(std::vector<std::string> names);

    auto is_flag() const noexcept -> bool;
    auto display_name() const -> std::string;
    auto value_name() const -> std::string;
    auto parse(std::string_view value) const -> std::any;

    std::vector<std::string> names_;
    std::string destination_;
    std::string help_;
    std::string metavar_;
    std::vector<std::string> choices_;
    bool optional_;
    bool required_;
    Action action_ = Action::STORE;
    std::function<std::any(std::string_view)> converter_ = [](std::string_view value) {
        return std::any(std::string(value));
    };
    std::optional<std::any> default_value_;
    std::string default_display_;
};

class ArgumentParser;

class Subparsers {
public:
    ~Subparsers();

    Subparsers(const Subparsers&) = delete;
    auto operator=(const Subparsers&) -> Subparsers& = delete;

    auto required(bool value = true) noexcept -> Subparsers&;
    auto dest(std::string value) -> Subparsers&;
    auto add_parser(std::string name, std::string help = {}) -> ArgumentParser&;

private:
    friend class ArgumentParser;

    struct Entry {
        std::string name;
        std::string help;
        std::unique_ptr<ArgumentParser> parser;
    };

    explicit Subparsers(ArgumentParser& owner, std::string destination);

    auto find(std::string_view name) const -> const Entry*;

    ArgumentParser& owner_;
    std::string destination_;
    bool required_ = false;
    std::vector<Entry> entries_;
};

class ArgumentParser {
public:
    explicit ArgumentParser(std::string program, std::string description = {});
    ~ArgumentParser();

    ArgumentParser(const ArgumentParser&) = delete;
    auto operator=(const ArgumentParser&) -> ArgumentParser& = delete;
    ArgumentParser(ArgumentParser&&) = delete;
    auto operator=(ArgumentParser&&) -> ArgumentParser& = delete;

    auto description(std::string value) -> ArgumentParser&;
    auto version(std::string value) -> ArgumentParser&;

    template <typename... Names>
    auto add_argument(std::string name, Names&&... aliases) -> Argument& {
        std::vector<std::string> names;
        names.reserve(1 + sizeof...(Names));
        names.push_back(std::move(name));
        (names.emplace_back(std::forward<Names>(aliases)), ...);
        return add_argument(std::move(names));
    }

    auto add_subparsers(std::string destination = "command") -> Subparsers&;

    auto parse_args(int argc, const char* const argv[]) const -> Namespace;
    auto parse_args(std::span<const std::string_view> arguments) const -> Namespace;
    auto format_usage() const -> std::string;
    auto format_help() const -> std::string;

private:
    friend class Subparsers;

    auto add_argument(std::vector<std::string> names) -> Argument&;
    auto set_value(Namespace& result, const Argument& argument, std::string_view value) const -> void;
    auto validate_arguments(const std::unordered_set<const Argument*>& seen) const -> void;
    [[noreturn]] auto fail(std::string message) const -> void;

    static auto argument_help(const Argument& argument) -> std::string;
    static auto append_rows(std::string& output, std::string_view heading,
                            const std::vector<std::pair<std::string, std::string>>& rows) -> void;

    std::string program_;
    std::string description_;
    std::optional<std::string> version_;
    std::vector<std::unique_ptr<Argument>> arguments_;
    std::unordered_map<std::string, Argument*> options_;
    std::vector<Argument*> positionals_;
    std::unique_ptr<Subparsers> subparsers_;
};

} // namespace kidi::cli