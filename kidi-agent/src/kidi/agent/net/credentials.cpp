#include "kidi/agent/net/credentials.h"

#include "kidi/core/files.h"

#include <cstdlib>
#include <fstream>
#include <sstream>

#include <sys/stat.h>

namespace kidi::agent::net {

CredentialStore::CredentialStore(std::filesystem::path path) : path_(std::move(path)) {}

auto CredentialStore::default_path() -> std::filesystem::path {
    if (const auto* config = std::getenv("XDG_CONFIG_HOME"); config && *config)
        return std::filesystem::path(config) / "kidi" / "credentials.json";
    const auto* home = std::getenv("HOME");
    return std::filesystem::path(home && *home ? home : ".") / ".config" / "kidi" / "credentials.json";
}

auto CredentialStore::parse(std::string_view text) -> Json {
    auto entries = Json::parse(text, nullptr, false);
    return entries.is_object() ? entries : Json::object();
}

auto CredentialStore::load() const -> Json {
    std::ifstream input(path_, std::ios::binary);
    if (!input) return Json::object();
    std::stringstream text;
    text << input.rdbuf();
    return parse(text.str());
}

auto CredentialStore::update(const std::function<bool(Json&)>& change) -> Result<void> {
    std::error_code error;
    std::filesystem::create_directories(path_.parent_path(), error);
    if (error)
        return std::unexpected(Error{ErrorCode::IO, "cannot save credentials to " + path_.string() + ": " +
                                                        error.message()});
    ::chmod(path_.parent_path().c_str(), 0700);
    // Agent processes share the file; write_atomic locks it across the read-modify-write, so one process's sign-in
    // cannot overwrite another's.
    return core::write_atomic(path_, [&](std::string current) -> std::optional<std::string> {
        auto entries = parse(current);
        if (!change(entries)) return std::nullopt;
        return entries.dump(2);
    });
}

auto CredentialStore::get(const std::string& key) const -> std::optional<Json> {
    const auto entries = load();
    const auto entry = entries.find(key);
    if (entry == entries.end() || !entry->is_object()) return std::nullopt;
    return *entry;
}

auto CredentialStore::put(const std::string& key, Json value) -> Result<void> {
    return update([&](Json& entries) {
        entries[key] = std::move(value);
        return true;
    });
}

auto CredentialStore::erase(const std::string& key) -> Result<void> {
    return update([&](Json& entries) { return entries.erase(key) > 0; });
}

} // namespace kidi::agent::net
