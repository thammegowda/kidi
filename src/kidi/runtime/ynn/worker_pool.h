#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#include "slinky/base/function_ref.h"
#include "slinky/base/ref_count.h"
#include "slinky/base/thread_pool.h"
#include "slinky/base/thread_pool_impl.h"

namespace kidi::runtime::ynn {

/// Lock for the pool's few-instruction critical sections. Waiters poll a plain load instead of hammering the lock word,
/// and yield once the holder is slow to release it, for example because it was preempted on an oversubscribed machine.
class SpinLock {
public:
    auto lock() noexcept -> void;
    auto try_lock() noexcept -> bool;
    auto unlock() noexcept -> void;

private:
    std::atomic<bool> locked_ = false;
};

/// The thread pool behind YNNPACK's slinky runtime.
///
/// It keeps the `slinky::thread_pool` contract and loop scheduling of `slinky::thread_pool_impl` (shared work-stealing
/// loops, nested loops, `wait_for` helping, `atomic_call`), but changes how threads idle and wake. A decoding step
/// forks hundreds of short loops, so dispatch latency is a large part of its runtime. Idle threads poll an epoch
/// counter without taking the queue lock, so the enqueuing thread never competes with them for it, and only block on a
/// condition variable after a bounded spin. Completions only touch the lock when some thread is waiting for one.
class WorkerPool final : public slinky::thread_pool {
public:
    using Loop = slinky::thread_pool_impl::task_impl;

    /// Starts `workers` background threads, which inherit the calling thread's CPU affinity. The thread that runs a
    /// loop participates in it, so a pool with `workers` threads runs loops on `workers + 1` threads.
    explicit WorkerPool(int workers);
    ~WorkerPool() override;

    WorkerPool(const WorkerPool&) = delete;
    auto operator=(const WorkerPool&) -> WorkerPool& = delete;

    /// Number of background worker threads.
    auto thread_count() const -> int override { return workers_; }
    auto enqueue(std::size_t n, task_body body, int max_workers = std::numeric_limits<int>::max())
        -> slinky::ref_count<task> override;
    using thread_pool::enqueue;
    auto wait_for(task* t) -> void override;
    auto wait_for(predicate_ref condition) -> void override;
    auto atomic_call(slinky::function_ref<void()> t) -> void override;

private:
    /// Threads that found nothing to run and wait for the epoch to move.
    struct Waiters {
        std::condition_variable_any parked;
        int sleeping = 0;
    };
    /// Which blocked threads an epoch change wakes. Spinning threads always notice it.
    enum class Notify { EVERYONE, ONE, HELPERS };

    auto worker_main() -> void;
    auto dequeue(int& worker) -> slinky::ref_count<Loop>;
    static auto run(Loop& loop, int worker) -> bool;
    auto idle(std::unique_lock<SpinLock>& lock, std::uint64_t seen, Waiters& waiters) -> void;
    auto update(slinky::function_ref<void()> change, Notify notify) -> void;
    auto loop_completed() -> void;

    int workers_;
    std::vector<std::thread> threads_;
    std::atomic<bool> stop_ = false;

    alignas(slinky::cache_line_size) SpinLock lock_;
    std::deque<slinky::ref_count<Loop>> queue_;
    Waiters worker_waiters_;
    Waiters helper_waiters_;

    /// Moves, under `lock_`, whenever a waiting thread may have something new to do: a loop was queued, an atomic call
    /// ran, or a loop completed while a thread waited for one.
    alignas(slinky::cache_line_size) std::atomic<std::uint64_t> epoch_ = 0;
    /// Threads inside `wait_for(predicate)`. Loop completions are only announced while this is non-zero.
    alignas(slinky::cache_line_size) std::atomic<int> waiting_helpers_ = 0;
};

} // namespace kidi::runtime::ynn
