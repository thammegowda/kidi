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
};

enum class LengthPreference {
    PRESERVE,
    SHORTER,
    LONGER,
};

struct EditorRequest {
    EditorTask task;
    std::string text;
    std::optional<std::string> system_prompt;
    std::optional<std::string> instruction;
    std::optional<std::string> tone;
    LengthPreference length{LengthPreference::PRESERVE};
};

struct EditorLimits {
    std::size_t maximum_text_bytes{64 * 1024};
    std::size_t maximum_system_prompt_bytes{16 * 1024};
    std::size_t maximum_instruction_bytes{2048};
    std::size_t maximum_tone_bytes{128};
};

auto parse_editor_task(std::string_view value) -> Result<EditorTask>;
auto parse_length_preference(std::string_view value) -> Result<LengthPreference>;
auto validate_editor_request(EditorRequest request, EditorLimits limits = {}) -> Result<EditorRequest>;
auto editor_messages(const EditorRequest& request) -> std::vector<text::ChatMessage>;

} // namespace kidi::app
