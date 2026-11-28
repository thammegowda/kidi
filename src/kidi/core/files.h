#pragma once

#include "kidi/core/error.h"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace kidi::core {

// An advisory lock on a file, shared between processes. Every LockFile opens the file anew, so two LockFiles
// exclude each other even within one process. A lock is released by unlock(), by destruction, or when the last
// process holding the file closes it; a child process that inherits the handle shares the lock.
class LockFile {
public:
    enum class Mode { SHARED, EXCLUSIVE };
#if defined(_WIN32)
    using Handle = void*;
#else
    using Handle = int;
#endif

    // Opens `path` (created with owner-only permissions when `create`); symbolic links are refused.
    static auto open(const std::filesystem::path& path, bool create = true) -> Result<LockFile>;
    // Takes ownership of an already open handle, such as one inherited from a parent process.
    static auto adopt(Handle handle) -> LockFile;

    // Waits for the lock; switching between SHARED and EXCLUSIVE on a held lock converts it.
    auto lock(Mode mode = Mode::EXCLUSIVE) -> Result<void>;
    // False when another holder's lock conflicts.
    auto try_lock(Mode mode = Mode::EXCLUSIVE) -> Result<bool>;
    auto unlock() -> Result<void>;
    auto native_handle() const -> Handle { return handle_; }

    LockFile(LockFile&& other) noexcept;
    auto operator=(LockFile&& other) noexcept -> LockFile&;
    LockFile(const LockFile&) = delete;
    auto operator=(const LockFile&) -> LockFile& = delete;
    ~LockFile();

private:
    explicit LockFile(Handle handle) : handle_(handle) {}
    Handle handle_;
    bool locked_ = false; // Windows only: LockFileEx needs an unlock before a conversion
};

// Replaces `path` with `parts`, written in order to a uniquely named owner-only file beside it, flushed to storage
// and renamed over it, so readers see the old or the new contents, never a mix. Writers of one path take turns
// through a lock on `path` + ".lock", which stays beside it.
auto write_atomic(const std::filesystem::path& path, std::span<const std::span<const std::byte>> parts)
    -> Result<void>;
auto write_atomic(const std::filesystem::path& path, std::string_view text) -> Result<void>;
// Read-modify-write under the same lock: `change` receives the current contents (empty when there is no file) and
// returns the new contents, or nullopt to leave the file as it is.
auto write_atomic(const std::filesystem::path& path,
                  const std::function<std::optional<std::string>(std::string current)>& change) -> Result<void>;

} // namespace kidi::core
