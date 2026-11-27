#pragma once
// Model hub cache lookup and explicit downloads; default cache is configured by KIDI_HUB.

#include <filesystem>
#include <string>
#include <string_view>

#include "kidi/core/error.h"

namespace kidi::checkpoint {

class Hub {
public:
    Hub();
    Hub(std::string remote, std::filesystem::path local);

    static auto expand_home(std::filesystem::path path) -> std::filesystem::path;
    static auto default_cache() -> std::filesystem::path;

    // Local-only lookup of [@]owner/model or [@]owner/model/path/to/file.
    auto resolve(std::string_view reference, std::string_view revision = "main") const -> Result<std::filesystem::path>;
    auto get_file(std::string_view model_id, std::string_view file, std::string_view revision = "main") const
        -> Result<std::filesystem::path>;
    auto download(std::string_view reference) const -> Result<std::filesystem::path>;

    auto remote() const noexcept -> const std::string& { return remote_; }
    auto local() const noexcept -> const std::filesystem::path& { return local_; }

private:
    auto repository(std::string_view model_id) const -> Result<std::filesystem::path>;

    std::string remote_;
    std::filesystem::path local_;
};

} // namespace kidi::checkpoint
