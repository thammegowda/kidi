#include "kidi/agent/net/credentials.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace kidi::agent::net {

CredentialStore::CredentialStore(std::filesystem::path path) : path_(std::move(path)) {}

auto CredentialStore::default_path() -> std::filesystem::path {
    if (const auto* config = std::getenv("XDG_CONFIG_HOME"); config && *config)
        return std::filesystem::path(config) / "kidi" / "credentials.json";
    const auto* home = std::getenv("HOME");
    return std::filesystem::path(home && *home ? home : ".") / ".config" / "kidi" / "credentials.json";
}

auto CredentialStore::load() const -> Json {
    std::ifstream input(path_, std::ios::binary);
    if (!input) return Json::object();
    std::stringstream text;
    text << input.rdbuf();
    auto entries = Json::parse(text.str(), nullptr, false);
    return entries.is_object() ? entries : Json::object();
}

auto CredentialStore::save(const Json& entries) -> Result<void> {
    const auto fail = [&](std::string what) {
        return std::unexpected(Error{ErrorCode::IO, "cannot save credentials to " + path_.string() + ": " + what});
    };
    std::error_code error;
    std::filesystem::create_directories(path_.parent_path(), error);
    if (error) return fail(error.message());
    ::chmod(path_.parent_path().c_str(), 0700);
    // Written beside the file with owner-only permissions, then renamed over it.
    const auto temporary = path_.string() + ".tmp";
    ::unlink(temporary.c_str());
    const int fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) return fail(std::strerror(errno));
    const auto text = entries.dump(2);
    std::string_view rest = text;
    while (!rest.empty()) {
        const auto written = ::write(fd, rest.data(), rest.size());
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) {
            const auto message = std::string(std::strerror(errno));
            ::close(fd);
            ::unlink(temporary.c_str());
            return fail(message);
        }
        rest.remove_prefix(static_cast<std::size_t>(written));
    }
    if (::fsync(fd) != 0 || ::close(fd) != 0 || ::rename(temporary.c_str(), path_.c_str()) != 0) {
        const auto message = std::string(std::strerror(errno));
        ::unlink(temporary.c_str());
        return fail(message);
    }
    return {};
}

auto CredentialStore::get(const std::string& key) const -> std::optional<Json> {
    const std::lock_guard lock(mutex_);
    const auto entries = load();
    const auto entry = entries.find(key);
    if (entry == entries.end() || !entry->is_object()) return std::nullopt;
    return *entry;
}

auto CredentialStore::put(const std::string& key, Json value) -> Result<void> {
    const std::lock_guard lock(mutex_);
    auto entries = load();
    entries[key] = std::move(value);
    return save(entries);
}

auto CredentialStore::erase(const std::string& key) -> Result<void> {
    const std::lock_guard lock(mutex_);
    auto entries = load();
    if (!entries.contains(key)) return {};
    entries.erase(key);
    return save(entries);
}

} // namespace kidi::agent::net
