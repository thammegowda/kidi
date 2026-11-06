#include "kidi/runtime/ynn/feed_forward_cache.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <system_error>
#include <type_traits>

#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace kidi::runtime::ynn {
namespace {
constexpr std::array<char, 8> MAGIC{'K', 'I', 'D', 'I', 'G', 'F', 'F', '1'};
constexpr std::uint32_t FORMAT_VERSION = 1;
constexpr std::size_t PAYLOAD_OFFSET = 4096;

struct Header {
    std::array<char, 8> magic{};
    std::uint32_t version = 0;
    std::uint32_t header_size = 0;
    std::uint64_t source_size = 0;
    std::int64_t source_mtime = 0;
    FeedForwardCacheSpec spec;
    std::uint64_t payload_size = 0;
};
static_assert(std::is_trivially_copyable_v<Header>);
static_assert(sizeof(Header) <= PAYLOAD_OFFSET);

#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
class Mapping {
public:
    explicit Mapping(const std::filesystem::path& path) {
        const auto descriptor = ::open(path.c_str(), O_RDONLY);
        if (descriptor < 0) return;
        struct stat status {};
        if (::fstat(descriptor, &status) == 0 && status.st_size > 0) {
            size_ = static_cast<std::size_t>(status.st_size);
            auto* mapped = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, descriptor, 0);
            if (mapped != MAP_FAILED) data_ = static_cast<const std::byte*>(mapped);
        }
        ::close(descriptor);
    }
    ~Mapping() {
        if (data_) ::munmap(const_cast<std::byte*>(data_), size_);
    }
    Mapping(const Mapping&) = delete;
    auto operator=(const Mapping&) -> Mapping& = delete;
    auto data() const noexcept -> const std::byte* { return data_; }
    auto size() const noexcept -> std::size_t { return size_; }

private:
    const std::byte* data_ = nullptr;
    std::size_t size_ = 0;
};
#endif

auto same_spec(const FeedForwardCacheSpec& left, const FeedForwardCacheSpec& right) -> bool {
    return left.bits == right.bits && left.input_size == right.input_size &&
           left.intermediate_size == right.intermediate_size && left.source_hash == right.source_hash;
}

auto filename(const FeedForwardCacheSpec& spec) -> std::string {
    return "gff-" + std::to_string(spec.source_hash) + "-b" + std::to_string(spec.bits) + "-i" +
           std::to_string(spec.input_size) + "-m" + std::to_string(spec.intermediate_size) + ".bin";
}

#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
auto write_all(int descriptor, const void* data, std::size_t size) -> bool {
    const auto* bytes = static_cast<const std::byte*>(data);
    while (size) {
        const auto written = ::write(descriptor, bytes, size);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return false;
        bytes += written;
        size -= static_cast<std::size_t>(written);
    }
    return true;
}
#endif
} // namespace

auto FeedForwardCache::configure(const std::filesystem::path& directory,
                                 const std::filesystem::path& source) noexcept -> void {
    enabled_ = false;
#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
    try {
        if (!std::filesystem::is_regular_file(source)) return;
        directory_ = directory;
        source_size_ = std::filesystem::file_size(source);
        source_mtime_ = std::filesystem::last_write_time(source).time_since_epoch().count();
        enabled_ = true;
    } catch (...) {
    }
#else
    (void)directory;
    (void)source;
#endif
}

auto FeedForwardCache::load(const FeedForwardCacheSpec& spec, std::size_t payload_size) const noexcept
    -> FeedForwardCacheEntry {
#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
    try {
        if (!enabled_) return {};
        auto mapping = std::make_shared<Mapping>(directory_ / filename(spec));
        if (!mapping->data() || mapping->size() != PAYLOAD_OFFSET + payload_size) return {};
        Header header;
        std::memcpy(&header, mapping->data(), sizeof(header));
        if (header.magic != MAGIC || header.version != FORMAT_VERSION || header.header_size != PAYLOAD_OFFSET ||
            header.source_size != source_size_ || header.source_mtime != source_mtime_ ||
            header.payload_size != payload_size || !same_spec(header.spec, spec))
            return {};
        const auto* payload = reinterpret_cast<const std::uint8_t*>(mapping->data() + PAYLOAD_OFFSET);
        return {{payload, payload_size}, std::move(mapping)};
    } catch (...) {
        return {};
    }
#else
    (void)spec;
    (void)payload_size;
    return {};
#endif
}

auto FeedForwardCache::store(const FeedForwardCacheSpec& spec, std::span<const std::uint8_t> payload) const noexcept
    -> void {
#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
    std::filesystem::path temporary;
    try {
        if (!enabled_ || payload.empty()) return;
        std::filesystem::create_directories(directory_);
        const auto destination = directory_ / filename(spec);
        static std::atomic<std::uint64_t> sequence = 0;
        int descriptor = -1;
        while (descriptor < 0) {
            temporary = destination.string() + ".tmp." + std::to_string(::getpid()) + "." +
                        std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
            descriptor = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
            if (descriptor < 0 && errno != EEXIST) return;
        }
        Header header{.magic = MAGIC,
                      .version = FORMAT_VERSION,
                      .header_size = PAYLOAD_OFFSET,
                      .source_size = source_size_,
                      .source_mtime = source_mtime_,
                      .spec = spec,
                      .payload_size = payload.size()};
        std::array<std::byte, PAYLOAD_OFFSET> prefix{};
        std::memcpy(prefix.data(), &header, sizeof(header));
        const auto written = write_all(descriptor, prefix.data(), prefix.size()) &&
                             write_all(descriptor, payload.data(), payload.size());
        if (::close(descriptor) != 0 || !written) {
            std::filesystem::remove(temporary);
            return;
        }
        std::error_code error;
        std::filesystem::rename(temporary, destination, error);
        if (error) std::filesystem::remove(temporary);
    } catch (...) {
        if (!temporary.empty()) {
            std::error_code error;
            std::filesystem::remove(temporary, error);
        }
    }
#else
    (void)spec;
    (void)payload;
#endif
}

} // namespace kidi::runtime::ynn
