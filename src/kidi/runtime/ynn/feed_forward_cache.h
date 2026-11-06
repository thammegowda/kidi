#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

namespace kidi::runtime::ynn {

struct FeedForwardCacheSpec {
    std::uint32_t index = 0;
    std::uint32_t bits = 0;
    std::uint64_t input_size = 0;
    std::uint64_t intermediate_size = 0;
};

struct FeedForwardCacheEntry {
    std::span<const std::uint8_t> bytes;
    std::shared_ptr<const void> owner;
    explicit operator bool() const noexcept { return !bytes.empty(); }
};

class FeedForwardCache {
public:
    auto configure(const std::filesystem::path& directory, const std::filesystem::path& source) noexcept -> void;
    auto load(const FeedForwardCacheSpec& spec, std::size_t payload_size) const noexcept -> FeedForwardCacheEntry;
    auto store(const FeedForwardCacheSpec& spec, std::span<const std::uint8_t> payload) const noexcept -> void;

private:
    std::filesystem::path directory_;
    std::uint64_t source_size_ = 0;
    std::int64_t source_mtime_ = 0;
    bool enabled_ = false;
};

} // namespace kidi::runtime::ynn
