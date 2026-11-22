#include "kidi/text/response.h"

#include <algorithm>
#include <array>
#include <optional>
#include <string>

namespace kidi::text {
namespace {
using json = nlohmann::ordered_json;

enum class Block { TEXT, REASONING, TOOL, HEADER };

struct Marker {
    std::string_view text;
    enum Kind { MESSAGE, END_MESSAGE, THINK, END_THINK, TOOL_JSON, END_TOOL_JSON, GEMMA_THINK, GEMMA_END,
                GEMMA_TOOL, GEMMA_END_TOOL, FUNCTION } kind;
};

constexpr std::array markers{
    Marker{"<|message_model|>", Marker::MESSAGE},
    Marker{"<|end_message|>", Marker::END_MESSAGE},
    Marker{"<|content_model_end_sampling|>", Marker::END_MESSAGE},
    Marker{"<|content_thinking|>", Marker::THINK},
    Marker{"<|content_invoke_tool_json|>", Marker::TOOL_JSON},
    Marker{"<|content_text|>", Marker::END_THINK},
    Marker{"<|content_invoke_tool_text|>", Marker::END_THINK},
    Marker{"<|content_tool_error|>", Marker::END_THINK},
    Marker{"<think>", Marker::THINK},
    Marker{"</think>", Marker::END_THINK},
    Marker{"<analysis>", Marker::THINK},
    Marker{"</analysis>", Marker::END_THINK},
    Marker{"<tool_call>", Marker::TOOL_JSON},
    Marker{"</tool_call>", Marker::END_TOOL_JSON},
    Marker{"<|channel>thought\n", Marker::GEMMA_THINK},
    Marker{"<|channel>analysis\n", Marker::GEMMA_THINK},
    Marker{"<channel|>", Marker::GEMMA_END},
    Marker{"<|tool_call>", Marker::GEMMA_TOOL},
    Marker{"<tool_call|>", Marker::GEMMA_END_TOOL},
    Marker{"<function=", Marker::FUNCTION},
};

auto trim(std::string_view value) -> std::string_view {
    while (!value.empty() && (value.front() == ' ' || value.front() == '\n' || value.front() == '\r' ||
                              value.front() == '\t'))
        value.remove_prefix(1);
    while (!value.empty() && (value.back() == ' ' || value.back() == '\n' || value.back() == '\r' ||
                              value.back() == '\t'))
        value.remove_suffix(1);
    return value;
}

auto tool_call(std::string_view body, std::string_view header = {}) -> std::optional<ToolCall> {
    body = trim(body);
    std::string name(trim(header));
    json arguments = json::object();
    bool implicit_arguments = false;
    if (body.starts_with("call:")) {
        implicit_arguments = true;
        body.remove_prefix(5);
        const auto brace = body.find('{');
        if (brace == std::string_view::npos) return std::nullopt;
        if (name.empty()) name = std::string(trim(body.substr(0, brace)));
        body = body.substr(brace);
    }

    const auto payload = json::parse(body, nullptr, false);
    if (!payload.is_discarded()) {
        if (!payload.is_object()) return std::nullopt;
        if (payload.contains("function") && payload["function"].is_object()) {
            const auto& function = payload["function"];
            if (name.empty() && function.contains("name") && function["name"].is_string())
                name = function["name"].get<std::string>();
            if (function.contains("arguments")) {
                const auto& value = function["arguments"];
                if (value.is_object())
                    arguments = value;
                else if (value.is_string()) {
                    arguments = json::parse(value.get<std::string>(), nullptr, false);
                    if (arguments.is_discarded() || !arguments.is_object()) return std::nullopt;
                } else {
                    return std::nullopt;
                }
            }
        } else if (implicit_arguments) {
            arguments = payload;
        } else {
            if (name.empty() && payload.contains("name") && payload["name"].is_string())
                name = payload["name"].get<std::string>();
            const char* key = payload.contains("arguments") ? "arguments" : "args";
            if (payload.contains(key)) {
                const auto& value = payload[key];
                if (value.is_object())
                    arguments = value;
                else if (value.is_string()) {
                    arguments = json::parse(value.get<std::string>(), nullptr, false);
                    if (arguments.is_discarded() || !arguments.is_object()) return std::nullopt;
                } else {
                    return std::nullopt;
                }
            }
        }
        if (name.empty()) return std::nullopt;
        return ToolCall{std::move(name), std::move(arguments)};
    }

    if (name.empty()) return std::nullopt;
    const auto open = body.find('{');
    const auto close = body.rfind('}');
    if (open == std::string_view::npos || close <= open) return std::nullopt;
    auto content = trim(body.substr(open + 1, close - open - 1));
    if (content.empty()) return ToolCall{std::move(name), json::object()};
    json parsed = json::parse("{" + std::string(content) + "}", nullptr, false);
    if (!parsed.is_discarded() && parsed.is_object()) return ToolCall{std::move(name), std::move(parsed)};

    json values = json::object();
    while (!content.empty()) {
        const auto comma = content.find(',');
        const auto entry = trim(content.substr(0, comma));
        const auto colon = entry.find(':');
        if (colon == std::string_view::npos) return std::nullopt;
        const auto key_value = trim(entry.substr(0, colon));
        auto value = trim(entry.substr(colon + 1));
        std::string key(key_value);
        if (key.size() >= 2 && ((key.front() == '"' && key.back() == '"') ||
                                (key.front() == '\'' && key.back() == '\'')))
            key = key.substr(1, key.size() - 2);
        if (key.empty()) return std::nullopt;
        auto json_value = json::parse(value, nullptr, false);
        if (!json_value.is_discarded())
            values[key] = std::move(json_value);
        else {
            if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                                      (value.front() == '\'' && value.back() == '\'')))
                value = value.substr(1, value.size() - 2);
            values[key] = std::string(value);
        }
        if (comma == std::string_view::npos) break;
        content.remove_prefix(comma + 1);
    }
    return ToolCall{std::move(name), std::move(values)};
}

auto parse_function(std::string_view opening, std::string_view body) -> std::optional<ToolCall> {
    constexpr std::string_view prefix = "<function=";
    if (!opening.starts_with(prefix) || opening.back() != '>') return std::nullopt;
    auto name = trim(opening.substr(prefix.size(), opening.size() - prefix.size() - 1));
    if (name.empty()) return std::nullopt;
    json arguments = json::object();
    while (!body.empty()) {
        const auto start = body.find("<parameter=");
        if (start == std::string_view::npos) break;
        const auto name_end = body.find('>', start);
        const auto value_end = body.find("</parameter>", name_end);
        if (name_end == std::string_view::npos || value_end == std::string_view::npos) return std::nullopt;
        auto key = trim(body.substr(start + 11, name_end - start - 11));
        auto value = trim(body.substr(name_end + 1, value_end - name_end - 1));
        if (key.empty()) return std::nullopt;
        auto parsed = json::parse(value, nullptr, false);
        arguments[std::string(key)] = parsed.is_discarded() ? json(std::string(value)) : std::move(parsed);
        body.remove_prefix(value_end + 12);
    }
    return ToolCall{std::string(name), std::move(arguments)};
}

auto partial_marker(std::string_view value) -> bool;

auto parse_message_blocks(std::string_view text, bool complete) -> AssistantResponse {
    AssistantResponse response;
    auto block = Block::TEXT;
    std::string header, body;
    const auto finish = [&] {
        if (block == Block::REASONING) response.reasoning += body;
        if (block == Block::TEXT) response.content += body;
        if (block == Block::TOOL) {
            if (auto call = tool_call(body, header))
                response.tool_calls.push_back(std::move(*call));
            else
                response.content += body;
        }
        block = Block::TEXT;
        header.clear();
        body.clear();
    };
    if (!complete && text.ends_with('<')) text.remove_suffix(1);
    std::size_t position = 0;
    while (position < text.size()) {
        const auto open = text.find("<|", position);
        const auto plain = text.substr(position, open == std::string_view::npos ? open : open - position);
        (block == Block::HEADER ? header : body) += plain;
        if (open == std::string_view::npos) break;
        const auto close = text.find("|>", open + 2);
        if (close == std::string_view::npos && !complete) {
            if (partial_marker(text.substr(open))) break;
            body += text.substr(open, 2);
            position = open + 2;
            continue;
        }
        const auto length = close == std::string_view::npos ? 0 : close + 2 - open;
        position = open + std::max<std::size_t>(length, 2);
        const auto marker = text.substr(open, std::max<std::size_t>(length, 2));
        if (!length) {
            (block == Block::HEADER ? header : body) += marker;
        } else if (marker == "<|message_model|>") {
            finish();
            block = Block::HEADER;
        } else if (marker == "<|end_message|>" || marker == "<|content_model_end_sampling|>") {
            finish();
        } else if (marker == "<|content_thinking|>") {
            if (block != Block::HEADER) finish();
            block = Block::REASONING;
        } else if (marker == "<|content_invoke_tool_json|>") {
            if (block != Block::HEADER) finish();
            block = Block::TOOL;
        } else if (marker == "<|content_text|>" || marker == "<|content_invoke_tool_text|>" ||
                   marker == "<|content_tool_error|>") {
            if (block != Block::HEADER) finish();
            block = Block::TEXT;
        } else {
            (block == Block::HEADER ? header : body) += marker;
        }
    }
    if (block != Block::TOOL || complete) finish();
    return response;
}

auto partial_marker(std::string_view value) -> bool {
    return std::ranges::any_of(markers, [&](const Marker& marker) {
        return value.size() < marker.text.size() && marker.text.starts_with(value);
    });
}

} // namespace

auto parse_assistant_response(std::string_view text, bool complete) -> AssistantResponse {
    if (text.find("<|message_model|>") != std::string_view::npos ||
        (!complete && text.starts_with("<|") && std::string_view("<|message_model|>").starts_with(text)))
        return parse_message_blocks(text, complete);

    AssistantResponse response;
    Block block = Block::TEXT;
    std::string tool_body;
    const auto append = [&](std::string_view value) {
        if (block == Block::REASONING)
            response.reasoning += value;
        else if (block == Block::TOOL)
            tool_body += value;
        else
            response.content += value;
    };
    const auto finish_tool = [&] {
        if (auto call = tool_call(tool_body))
            response.tool_calls.push_back(std::move(*call));
        else
            response.content += tool_body;
        tool_body.clear();
        block = Block::TEXT;
    };

    std::size_t position = 0;
    while (position < text.size()) {
        const Marker* found = nullptr;
        std::size_t found_at = std::string_view::npos;
        for (const auto& marker : markers) {
            const auto index = text.find(marker.text, position);
            if (index < found_at) {
                found = &marker;
                found_at = index;
            }
        }
        if (!found) {
            auto end = text.size();
            if (!complete) {
                const auto open = text.rfind('<');
                if (open != std::string_view::npos && partial_marker(text.substr(open))) end = open;
            }
            append(text.substr(position, end - position));
            break;
        }
        append(text.substr(position, found_at - position));
        position = found_at + found->text.size();
        switch (found->kind) {
            case Marker::THINK:
            case Marker::GEMMA_THINK:
                block = Block::REASONING;
                break;
            case Marker::END_THINK:
            case Marker::GEMMA_END:
                if (block == Block::TOOL) append(found->text);
                else block = Block::TEXT;
                break;
            case Marker::TOOL_JSON:
            case Marker::GEMMA_TOOL:
                block = Block::TOOL;
                tool_body.clear();
                break;
            case Marker::END_TOOL_JSON:
            case Marker::GEMMA_END_TOOL:
                if (block == Block::TOOL) finish_tool();
                else append(found->text);
                break;
            case Marker::FUNCTION: {
                const auto open_end = text.find('>', found_at + found->text.size());
                const auto close = open_end == std::string_view::npos
                                       ? std::string_view::npos
                                       : text.find("</function>", open_end + 1);
                if (open_end == std::string_view::npos || close == std::string_view::npos) {
                    if (!complete) {
                        position = found_at;
                        break;
                    }
                    append(text.substr(found_at, found->text.size()));
                    break;
                }
                const auto opening = text.substr(found_at, open_end - found_at + 1);
                const auto body = text.substr(open_end + 1, close - open_end - 1);
                if (auto call = parse_function(opening, body))
                    response.tool_calls.push_back(std::move(*call));
                else
                    append(text.substr(found_at, close + 11 - found_at));
                position = close + 11;
                break;
            }
            case Marker::MESSAGE:
            case Marker::END_MESSAGE:
                append(found->text);
                break;
        }
        if (!complete && position == found_at) break;
    }
    if (block == Block::TOOL) {
        if (complete) finish_tool();
    }
    return response;
}

} // namespace kidi::text
