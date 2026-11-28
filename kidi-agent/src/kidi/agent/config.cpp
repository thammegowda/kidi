#include "kidi/agent/config.h"
#include "default_config.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <set>

namespace kidi::agent {
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

auto parse_speech(const Json& speech) -> Result<SpeechConfig> {
    if (auto valid = fields(speech, {}, {"transcribe", "speak"}); !valid) return std::unexpected(valid.error());
    SpeechConfig config;
    if (speech.contains("transcribe")) {
        const auto& transcribe = speech["transcribe"];
        if (auto valid = fields(transcribe, {"model"}); !valid) return std::unexpected(valid.error());
        if (!nonempty_string(transcribe["model"])) return invalid("speech.transcribe.model must be a nonempty string");
        config.transcribe_model = transcribe["model"].get<std::string>();
    }
    if (speech.contains("speak")) {
        const auto& speak = speech["speak"];
        if (auto valid = fields(speak, {"model", "reference_text"}, {"steps", "pace", "voice"}); !valid)
            return std::unexpected(valid.error());
        if (!nonempty_string(speak["model"]) || !nonempty_string(speak["reference_text"]))
            return invalid("speech.speak.model and reference_text must be nonempty strings");
        config.speak_model = speak["model"].get<std::string>();
        config.reference_text = speak["reference_text"].get<std::string>();
        if (speak.contains("steps")) {
            if (!speak["steps"].is_number_unsigned() || speak["steps"] == 0 || speak["steps"] > 64)
                return invalid("speech.speak.steps must be an integer in [1, 64]");
            config.steps = speak["steps"].get<std::size_t>();
        }
        if (speak.contains("pace")) {
            if (!speak["pace"].is_number() || speak["pace"].get<double>() < 0.25 || speak["pace"].get<double>() > 4)
                return invalid("speech.speak.pace must be a number in [0.25, 4]");
            config.pace = speak["pace"].get<float>();
        }
        if (speak.contains("voice")) {
            if (!speak["voice"].is_object()) return invalid("speech.speak.voice must be a mapping");
            for (const auto& [key, value] : speak["voice"].items()) {
                if (!value.is_string()) return invalid("speech.speak.voice values must be strings");
                config.voice.emplace(key, value.get<std::string>());
            }
        }
    }
    return config;
}

auto parse_mcp_servers(const Json& servers) -> Result<std::map<std::string, McpServerConfig, std::less<>>> {
    if (!servers.is_object()) return invalid("mcp_servers must be a mapping from names to servers");
    std::map<std::string, McpServerConfig, std::less<>> result;
    for (const auto& [key, server] : servers.items()) {
        if (!valid_name(key)) return invalid("mcp_servers names must be 1-64 ASCII letters, digits, '_' or '-'");
        if (auto valid = fields(server, {"url"}, {"name", "include"}); !valid)
            return invalid("mcp_servers." + key + ": " + valid.error().message.substr(14));
        for (const auto* field : {"url", "name"})
            if (server.contains(field) && !nonempty_string(server[field]))
                return invalid("mcp_servers." + key + "." + field + " must be a nonempty string");
        McpServerConfig entry{server.value("name", key), server["url"].get<std::string>(), {}, false};
        if (server.contains("include")) {
            const auto& include = server["include"];
            if (include == "all") {
                entry.include_all = true;
            } else if (include.is_array()) {
                for (const auto& name : include) {
                    if (!name.is_string() || !valid_name(name.get<std::string>()))
                        return invalid("mcp_servers." + key + ".include must list tool names");
                    entry.include.push_back(name.get<std::string>());
                }
            } else {
                return invalid("mcp_servers." + key + ".include must be all or a list of tool names");
            }
        }
        // Sign-ins travel only over HTTPS, except to a server on this computer.
        if (!entry.url.starts_with("https://") && !entry.url.starts_with("http://127.0.0.1") &&
            !entry.url.starts_with("http://localhost"))
            return invalid("mcp_servers." + key + ".url must be an https:// address");
        result.emplace(key, std::move(entry));
    }
    return result;
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
        if (auto valid = fields(*document, {"version", "prompts", "tools"},
                                {"sampling", "reasoning_effort", "speech", "mcp_servers"});
            !valid)
            return std::unexpected(valid.error());
        if (!(*document)["version"].is_number_integer() || (*document)["version"] != 1)
            return invalid("version must be integer 1");
        const auto& prompts = (*document)["prompts"];
        if (auto valid = fields(prompts, {"system"}, {"tools", "continue", "compact"}); !valid)
            return std::unexpected(valid.error());
        if (!nonempty_string(prompts["system"])) return invalid("prompts.system must be a nonempty string");
        AgentConfig config{.system_prompt = prompts["system"].get<std::string>()};
        if (prompts.contains("tools")) {
            if (!nonempty_string(prompts["tools"])) return invalid("prompts.tools must be a nonempty string");
            config.tools_prompt = prompts["tools"].get<std::string>();
        }
        if (prompts.contains("continue")) {
            if (!nonempty_string(prompts["continue"])) return invalid("prompts.continue must be a nonempty string");
            config.continue_prompt = prompts["continue"].get<std::string>();
        }
        if (prompts.contains("compact")) {
            if (!nonempty_string(prompts["compact"])) return invalid("prompts.compact must be a nonempty string");
            config.compact_prompt = prompts["compact"].get<std::string>();
        }
        if (document->contains("sampling")) {
            const auto& sampling = (*document)["sampling"];
            if (auto valid = fields(sampling, {}, {"temperature", "top_p", "top_k"}); !valid)
                return std::unexpected(valid.error());
            const auto number = [&](const char* key, double minimum, double maximum, float& value) {
                if (!sampling.contains(key)) return true;
                if (!sampling[key].is_number()) return false;
                const auto parsed = sampling[key].get<double>();
                value = static_cast<float>(parsed);
                return parsed >= minimum && parsed <= maximum;
            };
            if (!number("temperature", 0, 2, config.sampling.temperature))
                return invalid("sampling.temperature must be a number in [0, 2]");
            if (!number("top_p", 0, 1, config.sampling.top_p) || config.sampling.top_p <= 0)
                return invalid("sampling.top_p must be a number in (0, 1]");
            if (sampling.contains("top_k")) {
                if (!sampling["top_k"].is_number_unsigned()) return invalid("sampling.top_k must be a non-negative integer");
                config.sampling.top_k = sampling["top_k"].get<std::size_t>();
            }
        }
        if (document->contains("reasoning_effort")) {
            if (!nonempty_string((*document)["reasoning_effort"]))
                return invalid("reasoning_effort must be a nonempty string");
            config.reasoning_effort = (*document)["reasoning_effort"].get<std::string>();
        }
        if (document->contains("speech")) {
            auto speech = parse_speech((*document)["speech"]);
            if (!speech) return std::unexpected(speech.error());
            config.speech = std::move(*speech);
        }
        if (document->contains("mcp_servers")) {
            auto servers = parse_mcp_servers((*document)["mcp_servers"]);
            if (!servers) return std::unexpected(servers.error());
            config.mcp_servers = std::move(*servers);
        }
        const auto& tools = (*document)["tools"];
        if (!tools.is_array()) return invalid("tools must be a sequence (use [] for no tools)");
        std::set<std::string> names;
        for (const auto& entry : tools) {
            if (auto valid = fields(entry, {"name", "handler", "description", "parameters"}, {"config"}); !valid)
                return std::unexpected(valid.error());
            for (const auto* key : {"name", "handler", "description"})
                if (!nonempty_string(entry[key])) return invalid(std::string(key) + " must be a nonempty string");
            ToolRegistration tool{
                entry["handler"].get<std::string>(),
                {entry["name"].get<std::string>(), entry["description"].get<std::string>(), entry["parameters"]}};
            if (entry.contains("config")) {
                if (!entry["config"].is_object())
                    return invalid("config of tool " + tool.spec.name + " must be a mapping");
                tool.config = entry["config"];
            }
            // Tool arguments are strict unless a schema opts in to extra properties.
            if (tool.spec.parameters.is_object() && !tool.spec.parameters.contains("additionalProperties"))
                tool.spec.parameters["additionalProperties"] = false;
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
    std::vector<ToolHandler> fixed;
    for (const auto& handler : handlers) {
        if (!handler.invoke) return invalid("native handlers require unique names and callable implementations");
        fixed.push_back(fixed_handler(handler));
    }
    return configure_tools(config, fixed);
}

auto fixed_handler(Tool tool) -> ToolHandler {
    auto name = tool.spec.name;
    return {std::move(name), [tool = std::move(tool)](const Json& config) -> Result<Tool> {
                if (!config.empty())
                    return invalid("tool handler " + tool.spec.name + " takes no config; remove its config mapping");
                return tool;
            }};
}

auto configure_tools(const AgentConfig& config, std::span<const ToolHandler> handlers) -> Result<std::vector<Tool>> {
    std::set<std::string> handler_names;
    for (const auto& handler : handlers)
        if (!handler.make || !handler_names.insert(handler.name).second)
            return invalid("native handlers require unique names and callable implementations");
    std::set<std::string> names;
    std::vector<Tool> tools;
    for (const auto& entry : config.tools) {
        if (auto valid = validate_registration(entry); !valid) return std::unexpected(valid.error());
        if (!names.insert(entry.spec.name).second) return invalid("duplicate tool name: " + entry.spec.name);
        const auto handler =
            std::ranges::find_if(handlers, [&](const ToolHandler& item) { return item.name == entry.handler; });
        if (handler == handlers.end()) return invalid("unknown native handler: " + entry.handler);
        auto tool = handler->make(entry.config);
        if (!tool) return invalid("tool " + entry.spec.name + ": " + tool.error().message);
        if (!tool->invoke) return invalid("native handler " + entry.handler + " built no implementation");
        tool->spec = entry.spec;
        tools.push_back(std::move(*tool));
    }
    return tools;
}

} // namespace kidi::agent
