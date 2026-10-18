#include "kidi/checkpoint/hub.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace kidi::checkpoint {
namespace {

auto expand_home(std::filesystem::path path) -> std::filesystem::path {
    const auto value = path.string();
    if (value.empty() || value.front() != '~') return path;
    const auto* home = std::getenv(
#if defined(_WIN32)
        "USERPROFILE"
#else
        "HOME"
#endif
    );
    if (!home || !*home) throw std::invalid_argument("cannot expand Hub cache without a home directory");
    if (value == "~") return home;
    if (value.size() < 2 || (value[1] != '/' && value[1] != '\\'))
        throw std::invalid_argument("Hub cache uses an invalid home-relative path");
    return std::filesystem::path(home) / value.substr(2);
}

auto safe_revision(std::string_view value) -> bool {
    if (value.empty() || value.front() == '/' || value.back() == '/') return false;
    for (std::size_t start = 0; start <= value.size();) {
        const auto end = value.find('/', start);
        const auto part = value.substr(start, end - start);
        if (part.empty() || part == "." || part == ".." ||
            !std::ranges::all_of(part, [](unsigned char byte) {
                return std::isalnum(byte) || byte == '_' || byte == '.' || byte == '-';
            }))
            return false;
        if (end == std::string_view::npos) break;
        start = end + 1;
    }
    return true;
}

} // namespace

Hub::Hub(std::string remote, std::filesystem::path local)
    : remote_(std::move(remote)), local_(expand_home(std::move(local)).lexically_normal()) {
    while (remote_.ends_with('/')) remote_.pop_back();
    if (!remote_.starts_with("https://") || remote_.size() <= std::string_view{"https://"}.size())
        throw std::invalid_argument("Hub remote must be an HTTPS origin");
    if (local_.empty()) throw std::invalid_argument("Hub local cache must not be empty");
}

auto Hub::repository(std::string_view model_id) const -> Result<std::filesystem::path> {
    const auto slash = model_id.find('/');
    if (slash == std::string_view::npos || !slash || slash + 1 == model_id.size() ||
        model_id.find('/', slash + 1) != std::string_view::npos ||
        !std::ranges::all_of(model_id, [](unsigned char byte) {
            return std::isalnum(byte) || byte == '_' || byte == '.' || byte == '-' || byte == '/';
        }))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Hub model ID"});
    auto key = std::string(model_id);
    key.replace(slash, 1, "--");
    return local_ / ("models--" + key);
}

auto Hub::resolve(std::string_view model_id, std::string_view revision) const -> Result<std::filesystem::path> {
    if (!safe_revision(revision))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Hub model revision"});
    auto root = repository(model_id);
    if (!root) return std::unexpected(root.error());
    auto resolved = std::string(revision);
    if (revision.size() != 40) {
        std::ifstream reference(*root / "refs" / std::filesystem::path(resolved));
        if (!reference || !std::getline(reference, resolved) || !safe_revision(resolved))
            return std::unexpected(Error{ErrorCode::IO, "Hub model revision is not cached: " +
                                                               std::string(model_id) + '@' + std::string(revision)});
    }
    const auto snapshot = *root / "snapshots" / resolved;
    std::error_code error;
    if (!std::filesystem::is_directory(snapshot, error))
        return std::unexpected(
            Error{ErrorCode::IO, "Hub model snapshot is not cached: " + std::string(model_id) + '@' + resolved});
    return snapshot;
}

auto Hub::get_file(std::string_view model_id, std::string_view file, std::string_view revision) const
    -> Result<std::filesystem::path> {
    const auto relative = std::filesystem::path(file);
    if (relative.empty() || relative.is_absolute() ||
        std::ranges::any_of(relative, [](const auto& part) { return part == ".."; }))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Hub model file path"});
    auto snapshot = resolve(model_id, revision);
    if (!snapshot) return std::unexpected(snapshot.error());
    const auto path = *snapshot / relative;
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error))
        return std::unexpected(Error{ErrorCode::IO, "Hub model file is not cached: " + std::string(file)});
    return path;
}

} // namespace kidi::checkpoint
