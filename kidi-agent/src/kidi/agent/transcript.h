#pragma once

#include "kidi/agent/agent.h"

#include <cstdio>
#include <filesystem>
#include <memory>
#include <vector>

namespace kidi::agent {

auto transcript_message(const Message& message) -> Json;
auto message_from_record(const Json& record) -> Result<Message>;

class Transcript {
public:
    static auto open(const std::filesystem::path& workspace, std::string_view model) -> Result<Transcript>;
    // Appends to an existing transcript.
    static auto reopen(const std::filesystem::path& path) -> Result<Transcript>;
    // The private cache directory that holds a workspace's transcripts.
    static auto directory(const std::filesystem::path& workspace) -> Result<std::filesystem::path>;
    // Every well-formed record of a transcript, in order.
    static auto read(const std::filesystem::path& path, std::vector<Json>& records) -> Result<void>;
    auto append(Json record) -> Result<void>;
    auto path() const -> const std::filesystem::path& { return path_; }

private:
    struct CloseFile {
        auto operator()(std::FILE* file) const noexcept -> void { std::fclose(file); }
    };
    std::filesystem::path path_;
    std::unique_ptr<std::FILE, CloseFile> file_;
};

} // namespace kidi::agent
