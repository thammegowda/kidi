#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include <ynnpack.h>

#include "kidi/core/error.h"
#include "ynnpack/base/thread_pool.h"

namespace kidi::runtime::ynn {

using ::ynn::ParallelRangeSlots;
using ::ynn::PhaseBarrier;
using ::ynn::set_thread_count;
using ::ynn::supported_arch_flags;
using ::ynn::supported_arch_names;
using ::ynn::thread_count;
using ::ynn::WorkerContext;

auto check_status(ynn_status status, const char* operation) -> Result<void>;

inline auto reserve_thread_pool(std::size_t total_threads) -> Result<void> {
    return check_status(::ynn::reserve_thread_pool(total_threads), "reserve thread pool");
}

inline auto parallel_for(std::size_t size, const std::function<void(std::size_t)>& body) -> Result<void> {
    return check_status(::ynn::parallel_for(size, body), "parallel for");
}

inline auto parallel_tasks(std::size_t tasks, void* context, void (*body)(void*, std::size_t)) -> Result<void> {
    return check_status(::ynn::parallel_tasks(tasks, context, body), "parallel tasks");
}

inline auto parallel_ranges(std::size_t items, std::size_t min_grain, void* context,
                            void (*body)(void*, std::size_t, std::size_t)) -> Result<void> {
    return check_status(::ynn::parallel_ranges(items, min_grain, context, body), "parallel ranges");
}

inline auto parallel_ranges_with_worker(std::size_t items, std::size_t min_grain, std::size_t scratch_bytes,
                                        void* context,
                                        void (*body)(void*, const WorkerContext&, std::size_t, std::size_t))
    -> Result<void> {
    return check_status(::ynn::parallel_ranges_with_worker(items, min_grain, scratch_bytes, context, body),
                        "parallel ranges");
}

inline auto parallel_team(std::size_t max_threads, std::size_t scratch_bytes, void* context,
                          void (*body)(void*, const WorkerContext&, PhaseBarrier&)) -> Result<void> {
    return check_status(::ynn::parallel_team(max_threads, scratch_bytes, context, body), "parallel team");
}

template <typename Body>
auto parallel_tasks(std::size_t tasks, Body&& body) -> Result<void> {
    return check_status(::ynn::parallel_tasks(tasks, std::forward<Body>(body)), "parallel tasks");
}

template <typename Body>
auto parallel_ranges(std::size_t items, std::size_t min_grain, Body&& body) -> Result<void> {
    return check_status(::ynn::parallel_ranges(items, min_grain, std::forward<Body>(body)), "parallel ranges");
}

template <typename Body>
auto parallel_ranges_with_worker(std::size_t items, std::size_t min_grain, std::size_t scratch_bytes, Body&& body)
    -> Result<void> {
    return check_status(::ynn::parallel_ranges_with_worker(items, min_grain, scratch_bytes, std::forward<Body>(body)),
                        "parallel ranges");
}

template <typename Body>
auto parallel_team(std::size_t max_threads, std::size_t scratch_bytes, Body&& body) -> Result<void> {
    return check_status(::ynn::parallel_team(max_threads, scratch_bytes, std::forward<Body>(body)), "parallel team");
}

} // namespace kidi::runtime::ynn