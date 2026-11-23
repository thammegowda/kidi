#include "kidi/code/config.h"
#include "default_config.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <set>

namespace kidi::code {
namespace {

constexpr std::size_t MAX_CONFIG_BYTES = 256 * 1024;

auto invalid(std::string message) -> std::unexpected<Error> {
    return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "agent config: " + std::move(message)});
}

auto json_value(const YAML::Node& node, std::size_t depth, std::size_t& remaining) -> Result<Json> {
    if (depth > 32 || remaining == 0) return invalid("YAML nesting or node limit exceeded");
    --remaining;
    if (node.IsNull()) return Json(nullptr);
    if (node.IsScalar()) {
        const auto text = node.Scalar();
        if (node.Tag() != "!" && node.Tag() != "tag:yaml.org,2002:str") {
            auto scalar = Json::parse(text, nullptr, false);
            if (!scalar.is_discarded() && scalar.is_primitive()) return scalar;
        }
        return Json(text);
    }
    if (node.IsSequence()) {
        auto result = Json::array();
        for (const auto& entry : node) {
            auto value = json_value(entry, depth + 1, remaining);
            if (!value) return std::unexpected(value.error());
            result.push_back(std::move(*value));
        }
        return result;
    }
    if (node.IsMap()) {
        auto result = Json::object();
        for (const auto& entry : node) {
            if (!entry.first.IsScalar()) return invalid("mapping keys must be strings");
            const auto key = entry.first.Scalar();
            if (result.contains(key)) return invalid("duplicate key: " + key);
            auto value = json_value(entry.second, depth + 1, remaining);
            if (!value) return std::unexpected(value.error());
            result[key] = std::move(*value);
        }
        return result;
    }
    return invalid("unsupported YAML value");
}

auto fields(const Json& value, std::initializer_list<std::string_view> expected,
            std::initializer_list<std::string_view> optional = {}) -> Result<void> {
    if (!value.is_object()) return invalid("expected a mapping");
    for (const auto& entry : value.items())
        if (std::ranges::find(expected, entry.key()) == expected.end() &&
            std::ranges::find(optional, entry.key()) == optional.end())
            return invalid("unknown field: " + entry.key());
    for (const auto key : expected)
        if (!value.contains(key)) return invalid("missing field: " + std::string(key));
    return {};
}

auto nonempty_string(const Json& value) -> bool {
    return value.is_string() && value.get_ref<const std::string&>().find_first_not_of(" \t\r\n") != std::string::npos;
}

auto valid_name(std::string_view name) -> bool {
    return !name.empty() && name.size() <= 64 && std::ranges::all_of(name, [](unsigned char byte) {
        return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') || (byte >= '0' && byte <= '9') ||
               byte == '_' || byte == '-';
    });
}

auto validate_registration(const ToolRegistration& tool) -> Result<void> {
    if (!valid_name(tool.spec.name) || !valid_name(tool.handler))
        return invalid("tool names and handlers must be 1-64 ASCII letters, digits, '_' or '-'");
    if (tool.spec.description.find_first_not_of(" \t\r\n") == std::string::npos)
        return invalid("tool description must be nonempty");
    if (!tool.spec.parameters.is_object() || !tool.spec.parameters.contains("type") ||
        tool.spec.parameters["type"] != "object")
        return invalid("tool parameters must declare type: object");
    return {};
}

} // namespace

auto parse_agent_config(std::string_view yaml) -> Result<AgentConfig> {
    if (yaml.size() > MAX_CONFIG_BYTES) return invalid("file exceeds 256 KiB");
    try {
        const auto documents = YAML::LoadAll(std::string(yaml));
        if (documents.size() != 1) return invalid("expected exactly one YAML document");
        std::size_t remaining = 16384;
        auto document = json_value(documents.front(), 0, remaining);
        if (!document) return std::unexpected(document.error());
        if (auto valid = fields(*document, {"version", "prompts", "tools"}); !valid)
            return std::unexpected(valid.error());
        if (!(*document)["version"].is_number_integer() || (*document)["version"] != 1)
            return invalid("version must be integer 1");
        const auto& prompts = (*document)["prompts"];
        if (auto valid = fields(prompts, {"system"}, {"continue"}); !valid) return std::unexpected(valid.error());
        if (!nonempty_string(prompts["system"])) return invalid("prompts.system must be a nonempty string");
        AgentConfig config{.system_prompt = prompts["system"].get<std::string>()};
        if (prompts.contains("continue")) {
            if (!nonempty_string(prompts["continue"])) return invalid("prompts.continue must be a nonempty string");
            config.continue_prompt = prompts["continue"].get<std::string>();
        }
        const auto& tools = (*document)["tools"];
        if (!tools.is_array()) return invalid("tools must be a sequence (use [] for no tools)");
        std::set<std::string> names;
        for (const auto& entry : tools) {
            if (auto valid = fields(entry, {"name", "handler", "description", "parameters"}); !valid)
                return std::unexpected(valid.error());
            for (const auto* key : {"name", "handler", "description"})
                if (!nonempty_string(entry[key])) return invalid(std::string(key) + " must be a nonempty string");
            ToolRegistration tool{
                entry["handler"].get<std::string>(),
                {entry["name"].get<std::string>(), entry["description"].get<std::string>(), entry["parameters"]}};
            if (auto valid = validate_registration(tool); !valid) return std::unexpected(valid.error());
            if (!names.insert(tool.spec.name).second) return invalid("duplicate tool name: " + tool.spec.name);
            config.tools.push_back(std::move(tool));
        }
        return config;
    } catch (const YAML::Exception& error) {
        return invalid(error.what());
    }
}

auto load_agent_config(const std::filesystem::path& path) -> Result<AgentConfig> {
    std::ifstream input(path, std::ios::binary);
    if (!input) return std::unexpected(Error{ErrorCode::IO, "cannot open agent config: " + path.string()});
    std::string text;
    std::array<char, 4096> buffer;
    while (input.read(buffer.data(), buffer.size()) || input.gcount()) {
        text.append(buffer.data(), static_cast<std::size_t>(input.gcount()));
        if (text.size() > MAX_CONFIG_BYTES) return invalid("file exceeds 256 KiB");
    }
    if (!input.eof()) return std::unexpected(Error{ErrorCode::IO, "cannot read agent config: " + path.string()});
    auto config = parse_agent_config(text);
    if (!config) config.error().message = path.string() + ": " + config.error().message;
    return config;
}

auto default_agent_config() -> Result<AgentConfig> { return parse_agent_config(DEFAULT_AGENT_YAML); }

auto configure_tools(const AgentConfig& config, std::span<const Tool> handlers) -> Result<std::vector<Tool>> {
    std::set<std::string> handler_names;
    for (const auto& handler : handlers)
        if (!handler.invoke || !handler_names.insert(handler.spec.name).second)
            return invalid("native handlers require unique names and callable implementations");
    std::set<std::string> names;
    std::vector<Tool> tools;
    for (const auto& entry : config.tools) {
        if (auto valid = validate_registration(entry); !valid) return std::unexpected(valid.error());
        if (!names.insert(entry.spec.name).second) return invalid("duplicate tool name: " + entry.spec.name);
        const auto handler =
            std::ranges::find_if(handlers, [&](const Tool& tool) { return tool.spec.name == entry.handler; });
        if (handler == handlers.end()) return invalid("unknown native handler: " + entry.handler);
        tools.push_back({entry.spec, handler->capability, handler->invoke});
    }
    return tools;
}

} // namespace kidi::code
