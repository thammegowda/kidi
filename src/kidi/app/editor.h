#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "kidi/core/error.h"
#include "kidi/text/tokenizer.h"

namespace kidi::app {

enum class EditorTask {
    POLISH,
    EXPLAIN,
    SUMMARIZE,
    FORMALIZE,
    FRIENDLY,
    CONCISE,
    CUSTOM,
    CHAT,
    EMAIL_COMPOSE,
    EMAIL_POLISH,
    EMAIL_REPLY,
    TRANSLATE,
};

enum class LengthPreference {
    PRESERVE,
    SHORTER,
    LONGER,
};

struct EditorTurn {
    enum class Role { USER, ASSISTANT };
    Role role;
    std::string content;
};

struct EditorRequest {
    EditorTask task;
    std::string text;
    std::optional<std::string> system_prompt;
    std::optional<std::string> instruction;
    std::optional<std::string> tone;
    std::optional<std::string> target_language;
    LengthPreference length{LengthPreference::PRESERVE};
    std::vector<EditorTurn> history;
};

struct EditorLimits {
    std::size_t maximum_text_bytes{64 * 1024};
    std::size_t maximum_system_prompt_bytes{16 * 1024};
    std::size_t maximum_instruction_bytes{2048};
    std::size_t maximum_tone_bytes{128};
    std::size_t maximum_history_turns{20};
    std::size_t maximum_history_bytes{64 * 1024};
};

auto parse_editor_task(std::string_view value) -> Result<EditorTask>;
auto parse_length_preference(std::string_view value) -> Result<LengthPreference>;
auto validate_editor_request(EditorRequest request, EditorLimits limits = {}) -> Result<EditorRequest>;
auto editor_messages(const EditorRequest& request) -> std::vector<text::ChatMessage>;

} // namespace kidi::app
