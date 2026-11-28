#pragma once

#include "kidi/agent/agent.h"

#include <filesystem>
#include <functional>
#include <optional>
#include <string>

namespace kidi::agent::net {

// Credentials that tool setup saved (OAuth sign-ins), as JSON objects keyed by service URL in one file that only
// the user can read: $XDG_CONFIG_HOME/kidi/credentials.json, else ~/.config/kidi/credentials.json.
class CredentialStore {
public:
    explicit CredentialStore(std::filesystem::path path = default_path());
    static auto default_path() -> std::filesystem::path;

    auto get(const std::string& key) const -> std::optional<Json>;
    auto put(const std::string& key, Json value) -> Result<void>;
    auto erase(const std::string& key) -> Result<void>;

private:
    static auto parse(std::string_view text) -> Json;
    auto load() const -> Json;
    // Applies `change` to the stored entries under a lock shared by all processes; saves when it returns true.
    auto update(const std::function<bool(Json&)>& change) -> Result<void>;

    std::filesystem::path path_;
};

} // namespace kidi::agent::net
