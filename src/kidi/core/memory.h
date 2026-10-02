#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace kidi::core {

/// Allocator counters. Values are -1 where the platform does not expose them.
struct HeapUsage {
    std::int64_t in_use = -1;    // bytes currently allocated through malloc
    std::int64_t free = -1;      // bytes held by the allocator but free (fragmentation and reuse)
    std::int64_t footprint = -1; // peak bytes obtained from the system by the allocator
};

/// Net allocation recorded for one category of work, such as loading the tokenizer.
struct MemoryEntry {
    std::string category;
    std::int64_t bytes = 0;
    std::uint64_t count = 0;
};

/// Walks the allocator's heap (tens of milliseconds with a model loaded), so keep it off hot paths.
auto heap_usage() noexcept -> HeapUsage;
/// Recorded categories, largest first.
auto memory_entries() -> std::vector<MemoryEntry>;

/// Records the net change in allocated bytes between construction and destruction under `category`.
class MemoryScope {
public:
    explicit MemoryScope(std::string_view category);
    ~MemoryScope();
    MemoryScope(const MemoryScope&) = delete;
    auto operator=(const MemoryScope&) -> MemoryScope& = delete;

private:
    std::string category_;
    std::int64_t start_;
};

} // namespace kidi::core
