#include "kidi/cli/argparse.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <memory>
#include <optional>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

namespace kidi::cli {
namespace {

bool is_option(std::string_view value) { return value.size() > 1 && value.front() == '-'; }

std::string option_destination(std::span<const std::string> names) {
    auto selected = names.front();
    for (const auto& name : names) {
        if (name.starts_with("--")) {
            selected = name;
            break;
        }
    }
    const auto first_character = selected.find_first_not_of('-');
    auto result = selected.substr(first_character);
    std::ranges::replace(result, '-', '_');
    return result;
}

std::string uppercase(std::string value) {
    std::ranges::transform(value, value.begin(),
                           [](unsigned char character) { return static_cast<char>(std::toupper(character)); });
    return value;
}

std::string join(std::span<const std::string> values, std::string_view separator) {
    std::string result;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) result += separator;
        result += values[index];
    }
    return result;
}

} // namespace

ParseError::ParseError(std::string message, std::string usage)
    : std::runtime_error(std::move(message)), usage_(std::move(usage)) {}

std::string_view ParseError::usage() const noexcept { return usage_; }

bool Namespace::contains(std::string_view destination) const { return values_.contains(std::string(destination)); }

ParseStatus Namespace::status() const noexcept { return status_; }

bool Namespace::should_exit() const noexcept { return status_ != ParseStatus::READY; }

std::string_view Namespace::output() const noexcept { return output_; }

void Namespace::set(std::string destination, std::any value) {
    values_.insert_or_assign(std::move(destination), std::move(value));
}

Argument::Argument(std::vector<std::string> names)
    : names_(std::move(names)), optional_(is_option(names_.front())), required_(!optional_) {
    destination_ = optional_ ? option_destination(names_) : names_.front();
}

Argument& Argument::help(std::string value) {
    help_ = std::move(value);
    return *this;
}

Argument& Argument::metavar(std::string value) {
    metavar_ = std::move(value);
    return *this;
}

Argument& Argument::dest(std::string value) {
    if (value.empty()) throw std::logic_error("argument destination cannot be empty");
    destination_ = std::move(value);
    return *this;
}

Argument& Argument::required(bool value) noexcept {
    required_ = value;
    return *this;
}

Argument& Argument::action(Action value) {
    action_ = value;
    if (action_ == Action::STORE_TRUE) {
        converter_ = [](std::string_view) { return std::any(true); };
        default_value_ = std::any(false);
        default_display_ = "false";
    } else if (action_ == Action::STORE_FALSE) {
        converter_ = [](std::string_view) { return std::any(false); };
        default_value_ = std::any(true);
        default_display_ = "true";
    } else {
        converter_ = [](std::string_view argument) { return std::any(std::string(argument)); };
        default_value_.reset();
        default_display_.clear();
    }
    return *this;
}

Argument& Argument::choices(std::initializer_list<std::string_view> values) {
    choices_.clear();
    choices_.reserve(values.size());
    for (const auto value : values) choices_.emplace_back(value);
    return *this;
}

bool Argument::is_flag() const noexcept { return action_ != Action::STORE; }

std::string Argument::display_name() const { return optional_ ? join(names_, ", ") : value_name(); }

std::string Argument::value_name() const {
    if (!metavar_.empty()) return metavar_;
    if (!choices_.empty()) return "{" + join(choices_, ",") + "}";
    return optional_ ? uppercase(destination_) : destination_;
}

std::any Argument::parse(std::string_view value) const {
    if (!choices_.empty() && std::ranges::find(choices_, value) == choices_.end()) {
        throw std::invalid_argument("invalid choice '" + std::string(value) + "' (choose from " + join(choices_, ", ") +
                                    ")");
    }
    return converter_(value);
}

Subparsers::Subparsers(ArgumentParser& owner, std::string destination)
    : owner_(owner), destination_(std::move(destination)) {}

Subparsers::~Subparsers() = default;

Subparsers& Subparsers::required(bool value) noexcept {
    required_ = value;
    return *this;
}

Subparsers& Subparsers::dest(std::string value) {
    if (value.empty()) throw std::logic_error("subcommand destination cannot be empty");
    destination_ = std::move(value);
    return *this;
}

ArgumentParser& Subparsers::add_parser(std::string name, std::string help) {
    if (name.empty() || is_option(name)) throw std::logic_error("invalid command name: " + name);
    if (find(name) != nullptr) throw std::logic_error("duplicate command: " + name);
    auto parser = std::make_unique<ArgumentParser>(owner_.program_ + " " + name);
    auto& result = *parser;
    entries_.push_back(Entry{.name = std::move(name), .help = std::move(help), .parser = std::move(parser)});
    return result;
}

const Subparsers::Entry* Subparsers::find(std::string_view name) const {
    const auto iterator = std::ranges::find(entries_, name, &Entry::name);
    return iterator == entries_.end() ? nullptr : std::addressof(*iterator);
}

ArgumentParser::ArgumentParser(std::string program, std::string description)
    : program_(std::move(program)), description_(std::move(description)) {
    if (program_.empty()) throw std::logic_error("program name cannot be empty");
}

ArgumentParser::~ArgumentParser() = default;

ArgumentParser& ArgumentParser::description(std::string value) {
    description_ = std::move(value);
    return *this;
}

ArgumentParser& ArgumentParser::version(std::string value) {
    version_ = std::move(value);
    return *this;
}

Subparsers& ArgumentParser::add_subparsers(std::string destination) {
    if (!positionals_.empty()) throw std::logic_error("subparsers cannot follow positional arguments");
    if (subparsers_ != nullptr) throw std::logic_error("only one subparser group is supported");
    subparsers_ = std::unique_ptr<Subparsers>(new Subparsers(*this, std::move(destination)));
    return *subparsers_;
}

Namespace ArgumentParser::parse_args(int argc, const char* const argv[]) const {
    if (argc < 1 || argv == nullptr) throw std::logic_error("invalid argc/argv");
    std::vector<std::string_view> arguments;
    arguments.reserve(static_cast<std::size_t>(argc - 1));
    for (int index = 1; index < argc; ++index) arguments.emplace_back(argv[index]);
    return parse_args(arguments);
}

Namespace ArgumentParser::parse_args(std::span<const std::string_view> arguments) const {
    Namespace result;
    std::unordered_set<const Argument*> seen;
    for (const auto& argument : arguments_) {
        if (argument->default_value_) result.set(argument->destination_, *argument->default_value_);
    }

    std::size_t positional_index = 0;
    bool positional_only = false;
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        auto token = arguments[index];
        if (!positional_only && token == "--") {
            positional_only = true;
            continue;
        }
        if (!positional_only && (token == "-h" || token == "--help")) {
            result.status_ = ParseStatus::HELP;
            result.output_ = format_help();
            return result;
        }
        if (!positional_only && token == "--version" && version_) {
            result.status_ = ParseStatus::VERSION;
            result.output_ = *version_ + "\n";
            return result;
        }
        if (!positional_only && subparsers_ != nullptr) {
            if (const auto* entry = subparsers_->find(token)) {
                validate_arguments(seen);
                auto subcommand_result = entry->parser->parse_args(arguments.subspan(index + 1));
                if (subcommand_result.should_exit()) return subcommand_result;
                for (const auto& [name, value] : subcommand_result.values_) result.set(name, value);
                result.set(subparsers_->destination_, entry->name);
                return result;
            }
        }
        if (!positional_only && is_option(token)) {
            auto option_name = token;
            std::optional<std::string_view> inline_value;
            if (const auto separator = token.find('='); separator != std::string_view::npos) {
                option_name = token.substr(0, separator);
                inline_value = token.substr(separator + 1);
            }
            const auto iterator = options_.find(std::string(option_name));
            if (iterator == options_.end()) fail("unrecognized argument '" + std::string(option_name) + "'");
            const auto* argument = iterator->second;
            seen.insert(argument);
            if (argument->is_flag()) {
                if (inline_value) fail("argument " + argument->display_name() + " does not take a value");
                result.set(argument->destination_, argument->action_ == Action::STORE_TRUE);
                continue;
            }
            if (!inline_value) {
                if (++index == arguments.size()) fail("argument " + argument->display_name() + " expects a value");
                inline_value = arguments[index];
            }
            set_value(result, *argument, *inline_value);
            continue;
        }
        if (positional_index == positionals_.size()) {
            if (subparsers_ != nullptr) {
                std::vector<std::string> choices;
                choices.reserve(subparsers_->entries_.size());
                for (const auto& entry : subparsers_->entries_) choices.push_back(entry.name);
                fail("invalid command '" + std::string(token) + "' (choose from " + join(choices, ", ") + ")");
            }
            fail("unexpected argument '" + std::string(token) + "'");
        }
        const auto* argument = positionals_[positional_index++];
        seen.insert(argument);
        set_value(result, *argument, token);
    }

    validate_arguments(seen);
    if (subparsers_ != nullptr && subparsers_->required_) fail("a command is required");
    return result;
}

std::string ArgumentParser::format_usage() const {
    std::string result = "usage: " + program_ + " [-h]";
    if (version_) result += " [--version]";
    for (const auto& argument : arguments_) {
        std::string value;
        if (argument->optional_) {
            value = argument->names_.back();
            if (!argument->is_flag()) value += " " + argument->value_name();
            if (!argument->required_) value = "[" + value + "]";
        } else {
            value = argument->value_name();
            if (!argument->required_) value = "[" + value + "]";
        }
        result += " " + value;
    }
    if (subparsers_ != nullptr) {
        std::vector<std::string> choices;
        choices.reserve(subparsers_->entries_.size());
        for (const auto& entry : subparsers_->entries_) choices.push_back(entry.name);
        result += " {" + join(choices, ",") + "} ...";
    }
    return result + "\n";
}

std::string ArgumentParser::format_help() const {
    std::string result = format_usage();
    if (!description_.empty()) result += "\n" + description_ + "\n";

    std::vector<std::pair<std::string, std::string>> positional_rows;
    for (const auto* argument : positionals_) {
        positional_rows.emplace_back(argument->value_name(), argument_help(*argument));
    }
    append_rows(result, "positional arguments", positional_rows);

    std::vector<std::pair<std::string, std::string>> option_rows = {
        {"-h, --help", "show this help message and exit"},
    };
    if (version_) option_rows.emplace_back("--version", "show program version and exit");
    for (const auto& argument : arguments_) {
        if (!argument->optional_) continue;
        auto label = join(argument->names_, ", ");
        if (!argument->is_flag()) label += " " + argument->value_name();
        option_rows.emplace_back(std::move(label), argument_help(*argument));
    }
    append_rows(result, "options", option_rows);

    if (subparsers_ != nullptr) {
        std::vector<std::pair<std::string, std::string>> command_rows;
        command_rows.reserve(subparsers_->entries_.size());
        for (const auto& entry : subparsers_->entries_) command_rows.emplace_back(entry.name, entry.help);
        append_rows(result, "commands", command_rows);
    }
    return result;
}

Argument& ArgumentParser::add_argument(std::vector<std::string> names) {
    if (names.empty() || names.front().empty()) throw std::logic_error("argument name cannot be empty");
    const bool optional = is_option(names.front());
    for (const auto& name : names) {
        if (name.empty() || is_option(name) != optional) {
            throw std::logic_error("argument aliases must all be positional or optional");
        }
        if (name == "-h" || name == "--help" || name == "--version") {
            throw std::logic_error("argument name is reserved: " + name);
        }
    }
    if (!optional && names.size() != 1) throw std::logic_error("positional arguments cannot have aliases");
    if (!optional && subparsers_ != nullptr) throw std::logic_error("positional arguments cannot follow subparsers");

    auto argument = std::unique_ptr<Argument>(new Argument(std::move(names)));
    auto* result = argument.get();
    if (optional) {
        for (const auto& name : result->names_) {
            if (options_.contains(name)) throw std::logic_error("duplicate option: " + name);
            options_.emplace(name, result);
        }
    } else {
        positionals_.push_back(result);
    }
    arguments_.push_back(std::move(argument));
    return *result;
}

void ArgumentParser::set_value(Namespace& result, const Argument& argument, std::string_view value) const {
    try {
        result.set(argument.destination_, argument.parse(value));
    } catch (const std::invalid_argument& error) {
        fail("argument " + argument.display_name() + ": " + error.what() + ": '" + std::string(value) + "'");
    }
}

void ArgumentParser::validate_arguments(const std::unordered_set<const Argument*>& seen) const {
    std::vector<std::string> missing;
    for (const auto& argument : arguments_) {
        if (argument->required_ && !seen.contains(argument.get())) missing.push_back(argument->display_name());
    }
    if (!missing.empty()) fail("the following arguments are required: " + join(missing, ", "));
}

void ArgumentParser::fail(std::string message) const { throw ParseError(std::move(message), format_usage()); }

std::string ArgumentParser::argument_help(const Argument& argument) {
    auto result = argument.help_;
    if (argument.default_value_ && argument.action_ == Action::STORE && !argument.default_display_.empty()) {
        if (!result.empty()) result += " ";
        result += "(default: " + argument.default_display_ + ")";
    }
    return result;
}

void ArgumentParser::append_rows(std::string& output, std::string_view heading,
                                 const std::vector<std::pair<std::string, std::string>>& rows) {
    if (rows.empty()) return;
    constexpr std::size_t LABEL_WIDTH = 26;
    output += "\n" + std::string(heading) + ":\n";
    for (const auto& [label, help] : rows) {
        output += "  " + label;
        if (label.size() + 2 >= LABEL_WIDTH) {
            output += "\n" + std::string(LABEL_WIDTH, ' ');
        } else {
            output += std::string(LABEL_WIDTH - label.size() - 2, ' ');
        }
        output += help + "\n";
    }
}

} // namespace kidi::cli