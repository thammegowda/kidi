#pragma once

#include "kidi/code/agent.h"

#include <cstdio>
#include <filesystem>
#include <memory>

namespace kidi::code {

auto transcript_message(const Message& message) -> Json;

class Transcript {
public:
    static auto open(const std::filesystem::path& workspace, std::string_view model) -> Result<Transcript>;
    auto append(Json record) -> Result<void>;
    auto path() const -> const std::filesystem::path& { return path_; }

private:
    struct CloseFile {
        auto operator()(std::FILE* file) const noexcept -> void { std::fclose(file); }
    };
    std::filesystem::path path_;
    std::unique_ptr<std::FILE, CloseFile> file_;
};

} // namespace kidi::code
