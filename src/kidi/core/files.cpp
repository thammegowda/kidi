#include "kidi/core/files.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace kidi::core {
namespace {

auto io_error(std::string message) -> std::unexpected<Error> {
    return std::unexpected(Error{ErrorCode::IO, std::move(message)});
}

#if defined(_WIN32)
const LockFile::Handle CLOSED = INVALID_HANDLE_VALUE;

auto system_message(DWORD code) -> std::string {
    std::array<char, 256> text{};
    const auto size = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code, 0,
                                     text.data(), static_cast<DWORD>(text.size()), nullptr);
    return size ? std::string(text.data(), size) : "error " + std::to_string(code);
}
#else
constexpr LockFile::Handle CLOSED = -1;

auto system_message(int code) -> std::string { return std::strerror(code); }

// Retries a call interrupted by a signal.
template <class Call> auto retry(Call call) -> int {
    int result;
    do result = call();
    while (result < 0 && errno == EINTR);
    return result;
}
#endif

} // namespace

auto LockFile::open(const std::filesystem::path& path, bool create) -> Result<LockFile> {
#if defined(_WIN32)
    const auto handle = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
                                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                    create ? OPEN_ALWAYS : OPEN_EXISTING,
                                    FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (handle == INVALID_HANDLE_VALUE)
        return io_error("cannot open " + path.string() + ": " + system_message(GetLastError()));
    BY_HANDLE_FILE_INFORMATION information{};
    if (!GetFileInformationByHandle(handle, &information) ||
        (information.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) {
        CloseHandle(handle);
        return io_error("cannot open " + path.string() + ": not a regular file");
    }
    return LockFile(handle);
#else
    auto fd = retry([&] {
        return ::open(path.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW | (create ? O_CREAT : 0), 0600);
    });
    if (fd < 0) return io_error("cannot open " + path.string() + ": " + system_message(errno));
    // Never a standard stream, so the handle can be passed to a child process beside redirected stdio.
    if (fd < 3) {
        const auto copy = ::fcntl(fd, F_DUPFD_CLOEXEC, 3);
        const auto error = errno;
        ::close(fd);
        if (copy < 0) return io_error("cannot open " + path.string() + ": " + system_message(error));
        fd = copy;
    }
    return LockFile(fd);
#endif
}

auto LockFile::adopt(Handle handle) -> LockFile { return LockFile(handle); }

auto LockFile::lock(Mode mode) -> Result<void> {
#if defined(_WIN32)
    // LockFileEx does not convert a held lock, so release it first, as flock does on conversion.
    if (locked_) UnlockFileEx(handle_, 0, MAXDWORD, MAXDWORD, nullptr);
    OVERLAPPED overlapped{};
    locked_ = LockFileEx(handle_, mode == Mode::EXCLUSIVE ? LOCKFILE_EXCLUSIVE_LOCK : 0, 0, MAXDWORD, MAXDWORD,
                         &overlapped);
    if (!locked_) return io_error("cannot lock file: " + system_message(GetLastError()));
#else
    if (retry([&] { return ::flock(handle_, mode == Mode::EXCLUSIVE ? LOCK_EX : LOCK_SH); }) < 0)
        return io_error("cannot lock file: " + system_message(errno));
#endif
    return {};
}

auto LockFile::try_lock(Mode mode) -> Result<bool> {
#if defined(_WIN32)
    if (locked_) UnlockFileEx(handle_, 0, MAXDWORD, MAXDWORD, nullptr);
    OVERLAPPED overlapped{};
    locked_ = LockFileEx(handle_, (mode == Mode::EXCLUSIVE ? LOCKFILE_EXCLUSIVE_LOCK : 0) | LOCKFILE_FAIL_IMMEDIATELY,
                         0, MAXDWORD, MAXDWORD, &overlapped);
    if (locked_) return true;
    if (const auto error = GetLastError(); error != ERROR_LOCK_VIOLATION)
        return io_error("cannot lock file: " + system_message(error));
    return false;
#else
    if (retry([&] { return ::flock(handle_, (mode == Mode::EXCLUSIVE ? LOCK_EX : LOCK_SH) | LOCK_NB); }) == 0)
        return true;
    if (errno != EWOULDBLOCK) return io_error("cannot lock file: " + system_message(errno));
    return false;
#endif
}

auto LockFile::unlock() -> Result<void> {
#if defined(_WIN32)
    if (locked_ && !UnlockFileEx(handle_, 0, MAXDWORD, MAXDWORD, nullptr))
        return io_error("cannot unlock file: " + system_message(GetLastError()));
    locked_ = false;
#else
    if (retry([&] { return ::flock(handle_, LOCK_UN); }) < 0)
        return io_error("cannot unlock file: " + system_message(errno));
#endif
    return {};
}

LockFile::LockFile(LockFile&& other) noexcept
    : handle_(std::exchange(other.handle_, CLOSED)), locked_(std::exchange(other.locked_, false)) {}

auto LockFile::operator=(LockFile&& other) noexcept -> LockFile& {
    if (this != &other) {
        LockFile closing(std::move(*this));
        handle_ = std::exchange(other.handle_, CLOSED);
        locked_ = std::exchange(other.locked_, false);
    }
    return *this;
}

LockFile::~LockFile() {
    if (handle_ == CLOSED) return;
#if defined(_WIN32)
    CloseHandle(handle_);
#else
    ::close(handle_);
#endif
}

namespace {

// Writes `parts` beside `path` and renames the result over it; the caller holds the path's lock.
auto replace(const std::filesystem::path& path, std::span<const std::span<const std::byte>> parts) -> Result<void> {
    const auto fail = [&](const std::string& reason) {
        return io_error("cannot write " + path.string() + ": " + reason);
    };
#if defined(_WIN32)
    std::random_device random;
    std::filesystem::path temporary;
    HANDLE handle = INVALID_HANDLE_VALUE;
    for (int attempt = 0; attempt < 16 && handle == INVALID_HANDLE_VALUE; ++attempt) {
        temporary = path.wstring() + L"." + std::to_wstring(random()) + L".tmp";
        handle = CreateFileW(temporary.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle == INVALID_HANDLE_VALUE && GetLastError() != ERROR_FILE_EXISTS) break;
    }
    if (handle == INVALID_HANDLE_VALUE) return fail(system_message(GetLastError()));
    const auto abandon = [&](DWORD error) {
        CloseHandle(handle);
        DeleteFileW(temporary.c_str());
        return fail(system_message(error));
    };
    for (auto part : parts)
        while (!part.empty()) {
            const auto size = static_cast<DWORD>(std::min<std::size_t>(part.size(), 1U << 30));
            DWORD written = 0;
            if (!WriteFile(handle, part.data(), size, &written, nullptr) || written == 0)
                return abandon(GetLastError());
            part = part.subspan(written);
        }
    if (!FlushFileBuffers(handle)) return abandon(GetLastError());
    CloseHandle(handle);
    if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const auto error = GetLastError();
        DeleteFileW(temporary.c_str());
        return fail(system_message(error));
    }
#else
    auto temporary = path.string() + ".XXXXXX";
    const int fd = ::mkstemp(temporary.data());
    if (fd < 0) return fail(system_message(errno));
    ::fcntl(fd, F_SETFD, FD_CLOEXEC);
    const auto abandon = [&](int error) {
        ::close(fd);
        ::unlink(temporary.c_str());
        return fail(system_message(error));
    };
    for (auto part : parts)
        while (!part.empty()) {
            const auto written = ::write(fd, part.data(), part.size());
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) return abandon(written < 0 ? errno : EIO);
            part = part.subspan(static_cast<std::size_t>(written));
        }
    if (::fsync(fd) != 0) return abandon(errno);
    if (::close(fd) != 0 || ::rename(temporary.c_str(), path.c_str()) != 0) {
        const auto error = errno;
        ::unlink(temporary.c_str());
        return fail(system_message(error));
    }
#endif
    return {};
}

auto lock_for(const std::filesystem::path& path) -> Result<LockFile> {
    auto file = LockFile::open(path.string() + ".lock");
    if (!file) return std::unexpected(file.error());
    if (auto locked = file->lock(); !locked) return std::unexpected(locked.error());
    return file;
}

} // namespace

auto write_atomic(const std::filesystem::path& path, std::span<const std::span<const std::byte>> parts)
    -> Result<void> {
    auto lock = lock_for(path);
    if (!lock) return std::unexpected(lock.error());
    return replace(path, parts);
}

auto write_atomic(const std::filesystem::path& path, std::string_view text) -> Result<void> {
    const std::array parts{std::as_bytes(std::span(text))};
    return write_atomic(path, parts);
}

auto write_atomic(const std::filesystem::path& path,
                  const std::function<std::optional<std::string>(std::string current)>& change) -> Result<void> {
    auto lock = lock_for(path);
    if (!lock) return std::unexpected(lock.error());
    std::string current;
    if (std::ifstream input{path, std::ios::binary}) {
        std::ostringstream text;
        text << input.rdbuf();
        current = std::move(text).str();
    }
    const auto updated = change(std::move(current));
    if (!updated) return {};
    const std::array parts{std::as_bytes(std::span(*updated))};
    return replace(path, parts);
}

} // namespace kidi::core
