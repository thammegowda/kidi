#pragma once

#include "kidi/agent/agent.h"

#include <filesystem>
#include <mutex>
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
    auto load() const -> Json;
    auto save(const Json& entries) -> Result<void>;

    std::filesystem::path path_;
    mutable std::mutex mutex_;
};

} // namespace kidi::agent::net
