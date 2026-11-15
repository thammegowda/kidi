#include "kidi/runtime/ynn/worker_pool.h"

#include <algorithm>
#include <cassert>
#include <mutex>
#include <new>
#include <utility>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif
#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

namespace kidi::runtime::ynn {
namespace {

constexpr int LOCK_PAUSES = 128;
constexpr int BARRIER_PAUSES = 4096;
constexpr std::uint64_t SPIN_MICROSECONDS = 180;
constexpr int X86_SPIN_PAUSES = 4096;
constexpr int GENERIC_SPIN_YIELDS = 64;

inline auto cpu_relax() noexcept -> void {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    _mm_pause();
#elif (defined(__aarch64__) || defined(__arm__)) && (defined(__GNUC__) || defined(__clang__))
    __asm__ __volatile__("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

#if (defined(__aarch64__) || defined(__arm64__)) && (defined(__GNUC__) || defined(__clang__))
inline auto arm_counter_ticks() noexcept -> std::uint64_t {
    std::uint64_t ticks = 0;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(ticks));
    return ticks;
}

inline auto arm_spin_budget_ticks(std::uint64_t microseconds) noexcept -> std::uint64_t {
    static const std::uint64_t freq = [] {
        std::uint64_t f = 0;
        __asm__ __volatile__("mrs %0, cntfrq_el0" : "=r"(f));
        return f ? f : std::uint64_t{24'000'000};
    }();
    return std::max<std::uint64_t>(1, (freq * microseconds) / 1'000'000);
}
#endif

/// Polls `ready` in user space (`yield` gated by `cntvct_el0` every 16 iterations on ARM64, `_mm_pause` on x86)
/// without pipeline-flushing `isb` instructions or premature kernel `sched_yield` syscalls.
template <typename Ready>
auto spin_until_ready(Ready&& ready) noexcept -> bool {
    if (ready()) return true;
#if (defined(__aarch64__) || defined(__arm64__)) && (defined(__GNUC__) || defined(__clang__))
    const auto budget = arm_spin_budget_ticks(SPIN_MICROSECONDS);
    const auto start = arm_counter_ticks();
    do {
        for (int i = 0; i < 16; ++i) {
            if (ready()) return true;
            cpu_relax();
        }
    } while (arm_counter_ticks() - start < budget);
#elif defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    for (int i = 0; i < X86_SPIN_PAUSES; ++i) {
        if (ready()) return true;
        _mm_pause();
    }
#else
    for (int i = 0; i < GENERIC_SPIN_YIELDS; ++i) {
        if (ready()) return true;
        std::this_thread::yield();
    }
#endif
    return ready();
}

thread_local const WorkerPool* active_team_pool = nullptr;

class ActiveTeamScope {
public:
    explicit ActiveTeamScope(const WorkerPool* pool) noexcept : previous_(std::exchange(active_team_pool, pool)) {}
    ~ActiveTeamScope() { active_team_pool = previous_; }
    ActiveTeamScope(const ActiveTeamScope&) = delete;
    auto operator=(const ActiveTeamScope&) -> ActiveTeamScope& = delete;

private:
    const WorkerPool* previous_;
};

struct InlineScratchStack {
    std::vector<std::unique_ptr<std::byte[], void (*)(std::byte*)>> slots;
    std::vector<std::size_t> capacities;
    std::size_t depth = 0;
};

thread_local InlineScratchStack tls_inline_scratch;

class InlineScratchScope {
public:
    explicit InlineScratchScope(std::size_t required_bytes) {
        if (required_bytes == 0) return;
        auto& stack = tls_inline_scratch;
        index_ = stack.depth++;
        active_ = true;
        if (index_ >= stack.slots.size()) {
            stack.slots.emplace_back(nullptr, [](std::byte* ptr) {
                if (ptr != nullptr) ::operator delete(ptr, std::align_val_t{CACHE_LINE_BYTES});
            });
            stack.capacities.push_back(0);
        }
        if (stack.capacities[index_] < required_bytes) {
            const auto alloc_bytes = std::max(required_bytes, stack.capacities[index_] * 2);
            auto* fresh = static_cast<std::byte*>(::operator new(alloc_bytes, std::align_val_t{CACHE_LINE_BYTES}));
            stack.slots[index_].reset(fresh);
            stack.capacities[index_] = alloc_bytes;
        }
        data_ = stack.slots[index_].get();
    }

    ~InlineScratchScope() {
        if (active_) --tls_inline_scratch.depth;
    }

    InlineScratchScope(const InlineScratchScope&) = delete;
    auto operator=(const InlineScratchScope&) -> InlineScratchScope& = delete;

    auto data() const noexcept -> std::byte* { return data_; }

private:
    std::byte* data_ = nullptr;
    std::size_t index_ = 0;
    bool active_ = false;
};

} // namespace

auto SpinLock::lock() noexcept -> void {
    int spins = 0;
    while (locked_.exchange(true, std::memory_order_acquire)) {
        while (locked_.load(std::memory_order_relaxed)) {
            if (spins < LOCK_PAUSES) {
                ++spins;
                cpu_relax();
            } else {
                std::this_thread::yield();
            }
        }
    }
}

auto SpinLock::try_lock() noexcept -> bool {
    return !locked_.load(std::memory_order_relaxed) && !locked_.exchange(true, std::memory_order_acquire);
}

auto SpinLock::unlock() noexcept -> void { locked_.store(false, std::memory_order_release); }

PhaseBarrier::PhaseBarrier(std::size_t participants) noexcept
    : participants_(std::max<std::size_t>(1, participants)), remaining_(participants_) {}

auto PhaseBarrier::reset(std::size_t participants) noexcept -> void {
    participants_ = std::max<std::size_t>(1, participants);
    aborted_.store(false, std::memory_order_relaxed);
    remaining_.store(participants_, std::memory_order_relaxed);
    phase_.store(0, std::memory_order_relaxed);
}

auto PhaseBarrier::abort() noexcept -> void {
    aborted_.store(true, std::memory_order_release);
    phase_.fetch_add(1, std::memory_order_release);
}

auto PhaseBarrier::arrive_and_wait() noexcept -> bool {
    if (aborted_.load(std::memory_order_acquire)) return false;
    if (participants_ <= 1) return true;
    const auto observed = phase_.load(std::memory_order_acquire);
    if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        remaining_.store(participants_, std::memory_order_relaxed);
        phase_.fetch_add(1, std::memory_order_release);
        return !aborted_.load(std::memory_order_acquire);
    }
    int spins = 0;
    while (phase_.load(std::memory_order_acquire) == observed && !aborted_.load(std::memory_order_acquire)) {
        if (spins < BARRIER_PAUSES) {
            ++spins;
            cpu_relax();
        } else {
            std::this_thread::yield();
        }
    }
    return !aborted_.load(std::memory_order_acquire);
}

WorkerPool::ScratchBuffer::~ScratchBuffer() {
    if (data != nullptr) ::operator delete(data, std::align_val_t{CACHE_LINE_BYTES});
}

WorkerPool::ScratchBuffer::ScratchBuffer(ScratchBuffer&& other) noexcept
    : data(std::exchange(other.data, nullptr)), capacity(std::exchange(other.capacity, 0)) {}

auto WorkerPool::ScratchBuffer::operator=(ScratchBuffer&& other) noexcept -> ScratchBuffer& {
    if (this != &other) {
        if (data != nullptr) ::operator delete(data, std::align_val_t{CACHE_LINE_BYTES});
        data = std::exchange(other.data, nullptr);
        capacity = std::exchange(other.capacity, 0);
    }
    return *this;
}

auto WorkerPool::ScratchBuffer::ensure(std::size_t required_bytes) -> void {
    if (required_bytes <= capacity) return;
    const auto alloc_bytes = std::max(required_bytes, capacity * 2);
    auto* fresh = static_cast<std::byte*>(::operator new(alloc_bytes, std::align_val_t{CACHE_LINE_BYTES}));
    if (data != nullptr) ::operator delete(data, std::align_val_t{CACHE_LINE_BYTES});
    data = fresh;
    capacity = alloc_bytes;
}

WorkerPool::WorkerPool(int workers) : workers_(std::max(workers, 0)) {
    const auto total = total_threads();
    scratches_.resize(total);
    range_slots_ = std::make_unique<RangeSlot[]>(total);
    threads_.reserve(static_cast<std::size_t>(workers_));
    for (int i = 0; i < workers_; ++i) {
        const auto tid = static_cast<std::size_t>(i + 1);
        threads_.emplace_back([this, tid] { worker_main(tid); });
    }
}

WorkerPool::~WorkerPool() {
    stop_.store(true, std::memory_order_release);
    event_seq_.fetch_add(1, std::memory_order_seq_cst);
    event_seq_.notify_all();
    for (auto& thread : threads_) thread.join();
}

auto WorkerPool::record_exception(std::exception_ptr error) noexcept -> void {
    {
        std::scoped_lock lock(error_lock_);
        if (!team_error_) team_error_ = std::move(error);
    }
    team_barrier_.abort();
}

auto WorkerPool::execute_team_member(std::size_t tid) noexcept -> void {
    const ActiveTeamScope scope(this);
    const WorkerContext worker{
        .tid = tid,
        .thread_count = active_team_size_,
        .scratch = team_scratch_bytes_ > 0 ? scratches_[tid].data : nullptr,
        .scratch_size = team_scratch_bytes_,
    };
    try {
        team_fn_(team_ctx_, worker, team_barrier_);
    } catch (...) {
        record_exception(std::current_exception());
    }
}

auto WorkerPool::run_team(std::size_t max_threads, std::size_t scratch_bytes, void* context, TeamFn fn) -> void {
    if (max_threads == 0) return;
    const auto team_size = std::min(max_threads, total_threads());

    // Serial or re-entrant nested execution runs inline on the current thread without touching worker synchronization.
    if (team_size <= 1 || active_team_pool == this) {
        const InlineScratchScope inline_scratch(scratch_bytes);
        PhaseBarrier single_barrier(1);
        const WorkerContext worker{
            .tid = 0,
            .thread_count = 1,
            .scratch = inline_scratch.data(),
            .scratch_size = scratch_bytes,
        };
        fn(context, worker, single_barrier);
        return;
    }

    std::scoped_lock caller_guard(caller_lock_);
    while (async_active_.load(std::memory_order_acquire) != 0 || async_nonempty_.load(std::memory_order_acquire)) {
        if (!run_one_async()) cpu_relax();
    }

    if (scratch_bytes > 0) {
        for (std::size_t t = 0; t < team_size; ++t) scratches_[t].ensure(scratch_bytes);
    }

    active_team_size_ = team_size;
    team_fn_ = fn;
    team_ctx_ = context;
    team_scratch_bytes_ = scratch_bytes;
    team_error_ = nullptr;
    team_barrier_.reset(team_size);
    team_remaining_.store(team_size - 1, std::memory_order_relaxed);

    const auto token = ((++team_seq_) << 16) | static_cast<std::uint64_t>(team_size);
    team_id_.store(token, std::memory_order_release);
    event_seq_.fetch_add(1, std::memory_order_seq_cst);
    if (sleeping_.load(std::memory_order_seq_cst) > 0) {
        event_seq_.notify_all();
    }

    execute_team_member(0);

    if (team_remaining_.load(std::memory_order_acquire) != 0) {
        while (!spin_until_ready([&] { return team_remaining_.load(std::memory_order_acquire) == 0; })) {
            std::this_thread::yield();
        }
    }

    auto error = std::move(team_error_);
    if (error) std::rethrow_exception(error);
}

auto WorkerPool::run_ranges(std::size_t items, std::size_t min_grain, std::size_t scratch_bytes, void* context,
                            RangeFn fn) -> void {
    if (items == 0) return;
    min_grain = std::max<std::size_t>(1, min_grain);
    const auto max_by_grain = (items + min_grain - 1) / min_grain;
    const auto team_size = std::min(max_by_grain, total_threads());

    if (team_size <= 1 || active_team_pool == this) {
        const InlineScratchScope inline_scratch(scratch_bytes);
        const WorkerContext worker{
            .tid = 0,
            .thread_count = 1,
            .scratch = inline_scratch.data(),
            .scratch_size = scratch_bytes,
        };
        fn(context, worker, 0, items);
        return;
    }

    std::scoped_lock caller_guard(caller_lock_);
    while (async_active_.load(std::memory_order_acquire) != 0 || async_nonempty_.load(std::memory_order_acquire)) {
        if (!run_one_async()) cpu_relax();
    }

    if (scratch_bytes > 0) {
        for (std::size_t t = 0; t < team_size; ++t) scratches_[t].ensure(scratch_bytes);
    }

    // Partition `[0, items)` into cache-line-isolated per-worker home ranges aligned to `min_grain`.
    const auto total_units = (items + min_grain - 1) / min_grain;
    const auto base_units = total_units / team_size;
    const auto rem_units = total_units % team_size;
    std::size_t cursor = 0;
    for (std::size_t t = 0; t < team_size; ++t) {
        const auto units = base_units + (t < rem_units ? 1 : 0);
        const auto end = std::min(items, cursor + units * min_grain);
        const auto len = end - cursor;
        // Claim ~4 home sub-chunks locally to leave stealable work for faster P-cores when heterogeneous.
        const auto home_units = std::max<std::size_t>(1, (units + 3) / 4);
        range_slots_[t].next.store(cursor, std::memory_order_relaxed);
        range_slots_[t].end = end;
        range_slots_[t].home_grain = std::max(min_grain, std::min(len, home_units * min_grain));
        range_slots_[t].steal_grain = min_grain;
        cursor = end;
    }

    struct RangePayload {
        RangeSlot* slots;
        void* user_ctx;
        RangeFn user_fn;
    } payload{range_slots_.get(), context, fn};

    const TeamFn runner = [](void* raw, const WorkerContext& worker, PhaseBarrier& barrier) {
        const auto p = *static_cast<const RangePayload*>(raw);
        auto& home = p.slots[worker.tid];
        for (;;) {
            if (barrier.aborted()) return;
            const auto cur = home.next.fetch_add(home.home_grain, std::memory_order_relaxed);
            if (cur >= home.end) break;
            p.user_fn(p.user_ctx, worker, cur, std::min(home.end, cur + home.home_grain));
        }
        for (std::size_t offset = 1; offset < worker.thread_count; ++offset) {
            if (barrier.aborted()) return;
            auto& victim = p.slots[(worker.tid + offset) % worker.thread_count];
            if (victim.next.load(std::memory_order_relaxed) >= victim.end) continue;
            for (;;) {
                if (barrier.aborted()) return;
                const auto cur = victim.next.fetch_add(victim.steal_grain, std::memory_order_relaxed);
                if (cur >= victim.end) break;
                p.user_fn(p.user_ctx, worker, cur, std::min(victim.end, cur + victim.steal_grain));
            }
        }
    };

    active_team_size_ = team_size;
    team_fn_ = runner;
    team_ctx_ = &payload;
    team_scratch_bytes_ = scratch_bytes;
    team_error_ = nullptr;
    team_barrier_.reset(team_size);
    team_remaining_.store(team_size - 1, std::memory_order_relaxed);

    const auto token = ((++team_seq_) << 16) | static_cast<std::uint64_t>(team_size);
    team_id_.store(token, std::memory_order_release);
    event_seq_.fetch_add(1, std::memory_order_seq_cst);
    if (sleeping_.load(std::memory_order_seq_cst) > 0) {
        event_seq_.notify_all();
    }

    execute_team_member(0);

    if (team_remaining_.load(std::memory_order_acquire) != 0) {
        while (!spin_until_ready([&] { return team_remaining_.load(std::memory_order_acquire) == 0; })) {
            std::this_thread::yield();
        }
    }

    auto error = std::move(team_error_);
    if (error) std::rethrow_exception(error);
}

auto WorkerPool::run_tasks(std::size_t n, void* context, TaskFn fn) -> void {
    if (n == 0) return;
    struct Payload {
        void* ctx;
        TaskFn fn;
    } payload{context, fn};
    run_ranges(n, 1, 0, &payload, [](void* raw, const WorkerContext&, std::size_t begin, std::size_t end) {
        const auto p = *static_cast<const Payload*>(raw);
        for (std::size_t i = begin; i < end; ++i) p.fn(p.ctx, i);
    });
}

auto WorkerPool::parallel_for(std::size_t n, const std::function<void(std::size_t)>& body) -> void {
    if (n == 0) return;
    run_ranges(n, 1, 0, const_cast<std::function<void(std::size_t)>*>(&body),
               [](void* raw, const WorkerContext&, std::size_t begin, std::size_t end) {
                   const auto& fn = *static_cast<const std::function<void(std::size_t)>*>(raw);
                   for (std::size_t i = begin; i < end; ++i) fn(i);
               });
}

auto WorkerPool::schedule_async(void* task_context, AsyncFn task) -> void {
    if (workers_ <= 0) {
        const ActiveTeamScope scope(this);
        task(task_context);
        return;
    }
    async_active_.fetch_add(1, std::memory_order_relaxed);
    {
        std::scoped_lock lock(async_lock_);
        async_queue_.push_back({task_context, task});
        async_nonempty_.store(true, std::memory_order_release);
    }
    event_seq_.fetch_add(1, std::memory_order_seq_cst);
    if (sleeping_.load(std::memory_order_seq_cst) > 0) {
        event_seq_.notify_all();
    }
}

auto WorkerPool::run_one_async() -> bool {
    AsyncItem item{};
    {
        std::scoped_lock lock(async_lock_);
        if (async_queue_.empty()) {
            async_nonempty_.store(false, std::memory_order_relaxed);
            return false;
        }
        item = async_queue_.front();
        async_queue_.pop_front();
        if (async_queue_.empty()) async_nonempty_.store(false, std::memory_order_relaxed);
    }
    {
        const ActiveTeamScope scope(this);
        item.fn(item.context);
    }
    async_active_.fetch_sub(1, std::memory_order_release);
    return true;
}

auto WorkerPool::worker_main(std::size_t tid) -> void {
#if defined(__APPLE__)
    if (::pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) != 0)
        (void)::pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif
    std::uint64_t last_team_seq = 0;
    for (;;) {
        const auto seen = event_seq_.load(std::memory_order_acquire);
        if (stop_.load(std::memory_order_acquire)) break;

        const auto cur_token = team_id_.load(std::memory_order_acquire);
        const auto cur_seq = cur_token >> 16;
        if (cur_seq != last_team_seq) {
            last_team_seq = cur_seq;
            const auto tok_team_size = static_cast<std::size_t>(cur_token & 0xFFFFu);
            if (tid < tok_team_size) {
                execute_team_member(tid);
                team_remaining_.fetch_sub(1, std::memory_order_acq_rel);
                if (async_nonempty_.load(std::memory_order_acquire)) (void)run_one_async();
                continue;
            }
        }

        if (async_nonempty_.load(std::memory_order_acquire)) {
            if (run_one_async()) continue;
        }

        idle_worker(seen);
    }
}

auto WorkerPool::idle_worker(std::uint64_t seen) -> void {
    if (spin_until_ready([&] { return event_seq_.load(std::memory_order_acquire) != seen; })) return;
    sleeping_.fetch_add(1, std::memory_order_seq_cst);
    if (event_seq_.load(std::memory_order_seq_cst) == seen && !stop_.load(std::memory_order_acquire)) {
        event_seq_.wait(seen, std::memory_order_acquire);
    }
    sleeping_.fetch_sub(1, std::memory_order_relaxed);
}

} // namespace kidi::runtime::ynn
