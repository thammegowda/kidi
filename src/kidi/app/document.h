#pragma once

#include <chrono>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "kidi/app/editor.h"
#include "kidi/inference/generator.h"

namespace kidi::app {

enum class DocumentTask { SUMMARIZE, TRANSLATE };

struct DocumentRequest {
    DocumentTask task;
    std::string text;
    std::optional<std::string> system_prompt;
    std::optional<std::string> target_language;
    std::optional<std::string> instruction;
};

struct DocumentPart {
    std::string text;
    bool paragraph_end;
};

struct DocumentProgress {
    std::size_t part, parts, round;
    bool translating, completed;
};

auto document_parts(std::string_view text, std::size_t maximum_characters, bool preserve_paragraphs)
    -> Result<std::vector<DocumentPart>>;

class DocumentSession {
public:
    static auto start(inference::Generator& generator, DocumentRequest request, inference::GenerationOptions options)
        -> Result<DocumentSession>;
    auto step() -> Result<inference::GenerationStep>;
    auto cancel() -> Result<void>;
    auto id() const -> std::uint64_t { return id_; }
    auto progress() const -> DocumentProgress;

private:
    DocumentSession(inference::Generator& generator, DocumentRequest request, inference::GenerationOptions options,
                    std::vector<DocumentPart> parts);
    auto enqueue_part() -> Result<void>;

    std::reference_wrapper<inference::Generator> generator_;
    DocumentRequest request_;
    inference::GenerationOptions options_;
    std::vector<DocumentPart> parts_;
    std::vector<std::string> outputs_;
    std::string summary_;
    std::size_t part_ = 0;
    std::uint64_t id_ = 0, current_id_ = 0;
    bool completed_ = false;
    inference::GenerationStats stats_{};
    std::vector<std::int32_t> tokens_;
    std::chrono::steady_clock::time_point started_ = std::chrono::steady_clock::now();
};

} // namespace kidi::app
