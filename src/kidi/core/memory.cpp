#include "kidi/core/memory.h"

#include <algorithm>
#include <map>
#include <mutex>

#if defined(__EMSCRIPTEN__)
#include <malloc.h>
#endif

namespace kidi::core {
namespace {
struct Ledger {
    std::mutex mutex;
    std::map<std::string, MemoryEntry, std::less<>> entries;
};
auto ledger() -> Ledger& {
    static Ledger instance;
    return instance;
}
} // namespace

auto heap_usage() noexcept -> HeapUsage {
#if defined(__EMSCRIPTEN__)
    const auto info = mallinfo();
    return {static_cast<std::int64_t>(info.uordblks), static_cast<std::int64_t>(info.fordblks),
            static_cast<std::int64_t>(info.usmblks)};
#else
    return {};
#endif
}

auto memory_entries() -> std::vector<MemoryEntry> {
    auto& state = ledger();
    const std::scoped_lock lock(state.mutex);
    std::vector<MemoryEntry> result;
    for (const auto& [name, entry] : state.entries) result.push_back(entry);
    std::ranges::sort(result, std::greater<>{}, &MemoryEntry::bytes);
    return result;
}

MemoryScope::MemoryScope(std::string_view category) : category_(category), start_(heap_usage().in_use) {}
MemoryScope::~MemoryScope() {
    if (start_ < 0) return;
    try {
        auto& state = ledger();
        const auto bytes = heap_usage().in_use - start_;
        const std::scoped_lock lock(state.mutex);
        auto& entry = state.entries.try_emplace(category_, MemoryEntry{category_}).first->second;
        entry.bytes += bytes;
        ++entry.count;
    } catch (...) {
    }
}

} // namespace kidi::core
