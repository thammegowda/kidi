#include "kidi/app/prompts.h"

#include <cstdint>
#include <yaml-cpp/yaml.h>

namespace kidi::app {
namespace {
auto invalid(std::string message) -> std::unexpected<Error> {
    return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, std::move(message)});
}

auto model_local_prompts(const YAML::Node& config) -> bool {
    const auto models = config["models"];
    if (!models || !models.IsMap()) return false;
    const auto chat = models["chat"];
    if (!chat || !chat.IsMap()) return false;
    const auto editor = chat["editor"];
    return chat["system_prompt"] || chat["translation_prompt"] || (editor && editor.IsMap() && editor["prompts"]);
}

auto text_scalar(const YAML::Node& value) -> bool {
    if (!value.IsScalar()) return false;
    const auto tag = value.Tag();
    if (tag == "!" || tag == "tag:yaml.org,2002:str") return true;
    if (tag != "?") return false;
    const auto text = value.Scalar();
    if (text == "true" || text == "True" || text == "TRUE" || text == "false" || text == "False" || text == "FALSE")
        return false;
    double number;
    std::int64_t integer;
    return !YAML::convert<double>::decode(value, number) && !YAML::convert<std::int64_t>::decode(value, integer);
}
} // namespace

auto application_prompts(const YAML::Node& config, std::string_view runtime) -> Result<Prompts> {
    try {
        if (!config.IsMap()) return invalid("Application configuration must be a mapping");
        if (model_local_prompts(config))
            return invalid(
                "Move model-local prompts to top-level prompts.system, prompts.translate, and prompts.<action>");
        auto entries = YAML::Clone(config["prompts"]);
        if (!entries.IsMap()) return invalid("prompts must be a top-level mapping");
        if (!runtime.empty()) {
            const auto settings = config["runtime"];
            if (settings && !settings.IsMap()) return invalid("runtime must be a mapping");
            const auto override = settings && settings.IsMap() ? settings[std::string(runtime)] : YAML::Node{};
            if (override && !override.IsNull()) {
                if (!override.IsMap()) return invalid("runtime." + std::string(runtime) + " must be a mapping");
                if (model_local_prompts(override)) return invalid("Runtime prompt overrides must use prompts.<name>");
                const auto prompts = override["prompts"];
                if (prompts) {
                    if (!prompts.IsMap()) return invalid("Runtime prompts must be a mapping");
                    for (const auto& entry : prompts)
                        entries[entry.first.as<std::string>()] = YAML::Clone(entry.second);
                }
            }
        }
        Prompts result;
        for (const auto& entry : entries) {
            if (!entry.first.IsScalar() || !text_scalar(entry.second))
                return invalid("Prompt names and values must be text scalars");
            const auto name = entry.first.as<std::string>();
            const auto text = entry.second.as<std::string>();
            if (name.empty() || text.find_first_not_of(" \t\r\n") == std::string::npos)
                return invalid("Prompt names and values must not be empty");
            if (!result.emplace(name, text).second) return invalid("Duplicate prompt: " + name);
        }
        for (const auto* name : {"system", "translate"})
            if (!result.contains(name)) return invalid("prompts." + std::string(name) + " is required");
        if (result.contains("chat")) return invalid("Use prompts.system for chat instead of prompts.chat");
        return result;
    } catch (const YAML::Exception& error) {
        return invalid("Invalid prompt configuration: " + std::string(error.what()));
    }
}

} // namespace kidi::app
