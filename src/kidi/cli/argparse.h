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
T parse_argument(std::string_view value) {
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
std::string display_argument(const T& value) {
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

    [[nodiscard]] std::string_view usage() const noexcept;

private:
    std::string usage_;
};

class Namespace {
public:
    template <typename T>
    [[nodiscard]] const T& get(std::string_view destination) const {
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

    [[nodiscard]] bool contains(std::string_view destination) const;
    [[nodiscard]] ParseStatus status() const noexcept;
    [[nodiscard]] bool should_exit() const noexcept;
    [[nodiscard]] std::string_view output() const noexcept;

private:
    friend class ArgumentParser;

    void set(std::string destination, std::any value);

    std::unordered_map<std::string, std::any> values_;
    ParseStatus status_ = ParseStatus::READY;
    std::string output_;
};

class Argument {
public:
    Argument& help(std::string value);
    Argument& metavar(std::string value);
    Argument& dest(std::string value);
    Argument& required(bool value = true) noexcept;
    Argument& action(Action value);
    Argument& choices(std::initializer_list<std::string_view> values);

    template <typename T>
    Argument& type() {
        if (action_ != Action::STORE) throw std::logic_error("flag actions cannot have a value type");
        converter_ = [](std::string_view value) { return std::any(detail::parse_argument<T>(value)); };
        return *this;
    }

    template <typename T>
    Argument& default_value(T value) {
        type<T>();
        default_display_ = detail::display_argument(value);
        default_value_ = std::move(value);
        return *this;
    }

private:
    friend class ArgumentParser;

    explicit Argument(std::vector<std::string> names);

    [[nodiscard]] bool is_flag() const noexcept;
    [[nodiscard]] std::string display_name() const;
    [[nodiscard]] std::string value_name() const;
    [[nodiscard]] std::any parse(std::string_view value) const;

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
    Subparsers& operator=(const Subparsers&) = delete;

    Subparsers& required(bool value = true) noexcept;
    Subparsers& dest(std::string value);
    ArgumentParser& add_parser(std::string name, std::string help = {});

private:
    friend class ArgumentParser;

    struct Entry {
        std::string name;
        std::string help;
        std::unique_ptr<ArgumentParser> parser;
    };

    explicit Subparsers(ArgumentParser& owner, std::string destination);

    [[nodiscard]] const Entry* find(std::string_view name) const;

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
    ArgumentParser& operator=(const ArgumentParser&) = delete;
    ArgumentParser(ArgumentParser&&) = delete;
    ArgumentParser& operator=(ArgumentParser&&) = delete;

    ArgumentParser& description(std::string value);
    ArgumentParser& version(std::string value);

    template <typename... Names>
    Argument& add_argument(std::string name, Names&&... aliases) {
        std::vector<std::string> names;
        names.reserve(1 + sizeof...(Names));
        names.push_back(std::move(name));
        (names.emplace_back(std::forward<Names>(aliases)), ...);
        return add_argument(std::move(names));
    }

    Subparsers& add_subparsers(std::string destination = "command");

    [[nodiscard]] Namespace parse_args(int argc, const char* const argv[]) const;
    [[nodiscard]] Namespace parse_args(std::span<const std::string_view> arguments) const;
    [[nodiscard]] std::string format_usage() const;
    [[nodiscard]] std::string format_help() const;

private:
    friend class Subparsers;

    Argument& add_argument(std::vector<std::string> names);
    void set_value(Namespace& result, const Argument& argument, std::string_view value) const;
    void validate_arguments(const std::unordered_set<const Argument*>& seen) const;
    [[noreturn]] void fail(std::string message) const;

    static std::string argument_help(const Argument& argument);
    static void append_rows(std::string& output, std::string_view heading,
                            const std::vector<std::pair<std::string, std::string>>& rows);

    std::string program_;
    std::string description_;
    std::optional<std::string> version_;
    std::vector<std::unique_ptr<Argument>> arguments_;
    std::unordered_map<std::string, Argument*> options_;
    std::vector<Argument*> positionals_;
    std::unique_ptr<Subparsers> subparsers_;
};

} // namespace kidi::cli