#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

namespace kidi::runtime::ynn {

inline constexpr std::size_t CACHE_LINE_BYTES = 64;

/// Lock for few-instruction critical sections (cold async fallback and exception capture).
class SpinLock {
public:
    auto lock() noexcept -> void;
    auto try_lock() noexcept -> bool;
    auto unlock() noexcept -> void;

private:
    std::atomic<bool> locked_ = false;
};

/// Per-worker execution context passed into parallel range and multi-phase team callbacks.
struct WorkerContext {
    std::size_t tid = 0;          // [0, thread_count); 0 is the calling thread
    std::size_t thread_count = 1; // active team size for this dispatch
    std::byte* scratch = nullptr; // cache-line-aligned persistent per-worker scratchpad
    std::size_t scratch_size = 0;
};

/// Sense-reversing user-space barrier synchronizing all participants of a `WorkerPool::run_team` launch.
class PhaseBarrier {
public:
    explicit PhaseBarrier(std::size_t participants = 1) noexcept;

    auto reset(std::size_t participants) noexcept -> void;
    auto abort() noexcept -> void;
    auto aborted() const noexcept -> bool { return aborted_.load(std::memory_order_acquire); }
    auto arrive_and_wait() noexcept -> bool;

private:
    std::size_t participants_ = 1;
    std::atomic<bool> aborted_ = false;
    alignas(CACHE_LINE_BYTES) std::atomic<std::size_t> remaining_ = 1;
    alignas(CACHE_LINE_BYTES) std::atomic<std::uint64_t> phase_ = 0;
};

/// Slinky-free CPU execution core for Kidi and YNNPACK composites.
///
/// Provides:
/// - Explicit `WorkerContext` (`tid`, `thread_count`, and persistent cache-aligned per-worker scratchpads).
/// - Deterministic per-worker home-range slots with contention-free ring stealing (`run_ranges`).
/// - Single-dispatch multi-phase team execution (`run_team` + `PhaseBarrier`) so multi-stage ops synchronize in L1/L2
///   without re-entering the thread pool.
/// - Hardware-timed `yield` spinning + C++20 `std::atomic::wait` (`ulock`/`futex`) parking with zero mutex wakeups.
class WorkerPool final {
public:
    using TeamFn = void (*)(void* context, const WorkerContext& worker, PhaseBarrier& barrier);
    using RangeFn = void (*)(void* context, const WorkerContext& worker, std::size_t begin, std::size_t end);
    using TaskFn = void (*)(void* context, std::size_t task);
    using AsyncFn = void (*)(void* task_context);

    /// Starts `workers` background threads (`tid = 1 .. workers`); the calling thread participates as `tid = 0`.
    explicit WorkerPool(int workers);
    ~WorkerPool();

    WorkerPool(const WorkerPool&) = delete;
    auto operator=(const WorkerPool&) -> WorkerPool& = delete;

    /// Number of background worker threads.
    auto thread_count() const noexcept -> int { return workers_; }
    /// Total execution threads including the calling thread (`workers + 1`).
    auto total_threads() const noexcept -> std::size_t {
        return static_cast<std::size_t>(workers_ > 0 ? workers_ + 1 : 1);
    }

    auto run_team(std::size_t max_threads, std::size_t scratch_bytes, void* context, TeamFn fn) -> void;
    auto run_ranges(std::size_t items, std::size_t min_grain, std::size_t scratch_bytes, void* context, RangeFn fn)
        -> void;
    auto run_tasks(std::size_t n, void* context, TaskFn fn) -> void;
    auto parallel_for(std::size_t n, const std::function<void(std::size_t)>& body) -> void;

    /// Asynchronous fire-and-forget scheduling hook used only by the legacy `ynn_scheduler` adapter.
    auto schedule_async(void* task_context, AsyncFn task) -> void;

private:
    struct ScratchBuffer {
        std::byte* data = nullptr;
        std::size_t capacity = 0;

        ScratchBuffer() = default;
        ~ScratchBuffer();
        ScratchBuffer(ScratchBuffer&& other) noexcept;
        auto operator=(ScratchBuffer&& other) noexcept -> ScratchBuffer&;
        ScratchBuffer(const ScratchBuffer&) = delete;
        auto operator=(const ScratchBuffer&) -> ScratchBuffer& = delete;

        auto ensure(std::size_t required_bytes) -> void;
    };

    struct alignas(CACHE_LINE_BYTES) RangeSlot {
        std::atomic<std::size_t> next = 0;
        std::size_t end = 0;
        std::size_t home_grain = 1;
        std::size_t steal_grain = 1;
    };

    struct AsyncItem {
        void* context = nullptr;
        AsyncFn fn = nullptr;
    };

    auto execute_team_member(std::size_t tid) noexcept -> void;
    auto record_exception(std::exception_ptr error) noexcept -> void;
    auto run_one_async() -> bool;
    auto worker_main(std::size_t tid) -> void;
    auto idle_worker(std::uint64_t seen) -> void;

    int workers_;
    std::vector<std::thread> threads_;
    std::vector<ScratchBuffer> scratches_;
    std::unique_ptr<RangeSlot[]> range_slots_;
    std::atomic<bool> stop_ = false;

    alignas(CACHE_LINE_BYTES) SpinLock caller_lock_;
    std::uint64_t team_seq_ = 0;
    std::size_t active_team_size_ = 0;
    TeamFn team_fn_ = nullptr;
    void* team_ctx_ = nullptr;
    std::size_t team_scratch_bytes_ = 0;
    SpinLock error_lock_;
    std::exception_ptr team_error_;
    PhaseBarrier team_barrier_;

    alignas(CACHE_LINE_BYTES) std::atomic<std::uint64_t> team_id_ = 0;
    alignas(CACHE_LINE_BYTES) std::atomic<std::size_t> team_remaining_ = 0;

    alignas(CACHE_LINE_BYTES) SpinLock async_lock_;
    std::deque<AsyncItem> async_queue_;
    std::atomic<bool> async_nonempty_ = false;
    std::atomic<std::size_t> async_active_ = 0;

    alignas(CACHE_LINE_BYTES) std::atomic<std::uint64_t> event_seq_ = 0;
    alignas(CACHE_LINE_BYTES) std::atomic<int> sleeping_ = 0;
};

} // namespace kidi::runtime::ynn
