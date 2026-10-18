#pragma once

#include <filesystem>
#include <string>
#include <string_view>

#include "kidi/core/error.h"

namespace kidi::checkpoint {

class Hub {
public:
    Hub(std::string remote, std::filesystem::path local);

    auto resolve(std::string_view model_id, std::string_view revision = "main") const
        -> Result<std::filesystem::path>;
    auto get_file(std::string_view model_id, std::string_view file, std::string_view revision = "main") const
        -> Result<std::filesystem::path>;

    auto remote() const noexcept -> const std::string& { return remote_; }
    auto local() const noexcept -> const std::filesystem::path& { return local_; }

private:
    auto repository(std::string_view model_id) const -> Result<std::filesystem::path>;

    std::string remote_;
    std::filesystem::path local_;
};

} // namespace kidi::checkpoint
