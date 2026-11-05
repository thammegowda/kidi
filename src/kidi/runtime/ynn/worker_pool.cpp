#include "kidi/runtime/ynn/worker_pool.h"

#include <algorithm>
#include <cassert>
#include <utility>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace kidi::runtime::ynn {
namespace {

// Idle threads and the thread joining a loop poll in two phases: `pause` first, so a loop forked right after the
// previous one starts within a microsecond, then `yield`, so an oversubscribed machine still lets other runnable
// threads make progress. Past the budget a thread blocks.
constexpr int IDLE_PAUSES = 256;
constexpr int IDLE_YIELDS = 1000;
constexpr int JOIN_PAUSES = 2048;
constexpr int JOIN_YIELDS = 1000;
constexpr int LOCK_PAUSES = 128;

inline auto cpu_relax() noexcept -> void {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    _mm_pause();
#elif (defined(__aarch64__) || defined(__arm__)) && (defined(__GNUC__) || defined(__clang__))
    __asm__ __volatile__("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

/// Polls `ready` until it holds or the budget is spent. Returns whether it held.
template <typename Ready>
auto spin_until(Ready&& ready, int pauses, int yields) -> bool {
    for (int i = 0; i < pauses; ++i) {
        if (ready()) return true;
        cpu_relax();
    }
    for (int i = 0; i < yields; ++i) {
        if (ready()) return true;
        std::this_thread::yield();
    }
    return ready();
}

/// Loops this thread is executing, innermost last. A thread never joins a loop it is already inside of, which would
/// run the loop's iterations recursively on its own stack.
thread_local std::vector<const WorkerPool::Loop*> running_loops;

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

WorkerPool::WorkerPool(int workers) : workers_(workers) {
    threads_.reserve(static_cast<std::size_t>(std::max(workers, 0)));
    for (int i = 0; i < workers; ++i) threads_.emplace_back([this] { worker_main(); });
}

WorkerPool::~WorkerPool() {
    update([this] { stop_.store(true, std::memory_order_relaxed); }, Notify::EVERYONE);
    for (auto& thread : threads_) thread.join();
}

auto WorkerPool::enqueue(std::size_t n, task_body body, int max_workers) -> slinky::ref_count<task> {
    assert(n > 0);
    constexpr int MAX_SHARDS = 8;
    // A bounded worker count asks for an ordered loop, whose iterations must start in order from one shared counter.
    const bool ordered = max_workers < std::numeric_limits<int>::max();
    max_workers = static_cast<int>(std::min<std::size_t>(n, static_cast<std::size_t>(max_workers)));
    const int shards = ordered ? 1 : std::min(MAX_SHARDS, max_workers);
    auto loop = Loop::make(static_cast<std::size_t>(shards), n, std::move(body), max_workers);
    update([&] { queue_.push_back(loop); }, max_workers == 1 ? Notify::ONE : Notify::EVERYONE);
    return loop;
}

auto WorkerPool::wait_for(task* t) -> void {
    auto* loop = static_cast<Loop*>(t);
    if (run(*loop, -1)) loop_completed();
    if (loop->done()) return;
    // The remaining iterations are running on other threads: poll for their completion without the lock, then help
    // with other queued loops while waiting.
    if (spin_until([&] { return loop->done(); }, JOIN_PAUSES, JOIN_YIELDS)) return;
    wait_for([&] { return loop->done(); });
}

auto WorkerPool::wait_for(predicate_ref condition) -> void {
    // Count before the first check, so a loop completing from here on is announced to this thread. The count, the
    // completing thread's load of it, and `Loop::done()` are all sequentially consistent: either the completing
    // thread sees this registration or the condition sees the completion.
    waiting_helpers_.fetch_add(1);
    std::unique_lock lock(lock_);
    while (!condition()) {
        const auto seen = epoch_.load(std::memory_order_relaxed);
        int worker = -1;
        if (auto loop = dequeue(worker)) {
            lock.unlock();
            const bool completed = run(*loop, worker);
            loop = nullptr;
            if (completed) loop_completed();
            lock.lock();
        } else {
            idle(lock, seen, helper_waiters_);
        }
    }
    lock.unlock();
    waiting_helpers_.fetch_sub(1);
}

auto WorkerPool::atomic_call(slinky::function_ref<void()> t) -> void { update(t, Notify::EVERYONE); }

auto WorkerPool::worker_main() -> void {
    std::unique_lock lock(lock_);
    while (!stop_.load(std::memory_order_relaxed)) {
        const auto seen = epoch_.load(std::memory_order_relaxed);
        int worker = -1;
        if (auto loop = dequeue(worker)) {
            lock.unlock();
            const bool completed = run(*loop, worker);
            loop = nullptr;
            if (completed) loop_completed();
            lock.lock();
        } else {
            idle(lock, seen, worker_waiters_);
        }
    }
}

auto WorkerPool::dequeue(int& worker) -> slinky::ref_count<Loop> {
    for (auto i = queue_.begin(); i != queue_.end();) {
        slinky::ref_count<Loop>& loop = *i;
        if (loop->all_work_started()) {
            // No more threads can start working on this loop.
            i = queue_.erase(i);
        } else if (std::ranges::find(running_loops, &*loop) != running_loops.end()) {
            ++i;
        } else {
            worker = loop->allocate_worker();
            if (worker < 0) {
                i = queue_.erase(i);
            } else if (worker == 0) {
                // This is the last worker the loop admits.
                auto result = std::move(loop);
                queue_.erase(i);
                return result;
            } else {
                return loop;
            }
        }
    }
    return nullptr;
}

auto WorkerPool::run(Loop& loop, int worker) -> bool {
    running_loops.push_back(&loop);
    const bool completed = worker >= 0 ? loop.work(static_cast<std::size_t>(worker)) : loop.work();
    running_loops.pop_back();
    return completed;
}

auto WorkerPool::idle(std::unique_lock<SpinLock>& lock, std::uint64_t seen, Waiters& waiters) -> void {
    lock.unlock();
    const bool moved =
        spin_until([&] { return epoch_.load(std::memory_order_acquire) != seen; }, IDLE_PAUSES, IDLE_YIELDS);
    lock.lock();
    if (moved) return;
    ++waiters.sleeping;
    waiters.parked.wait(lock, [&] { return epoch_.load(std::memory_order_relaxed) != seen; });
    --waiters.sleeping;
}

auto WorkerPool::update(slinky::function_ref<void()> change, Notify notify) -> void {
    bool wake_workers = false;
    bool wake_helpers = false;
    {
        std::scoped_lock lock(lock_);
        change();
        epoch_.fetch_add(1, std::memory_order_release);
        wake_workers = notify != Notify::HELPERS && worker_waiters_.sleeping > 0;
        wake_helpers = helper_waiters_.sleeping > 0;
    }
    if (wake_workers) {
        if (notify == Notify::ONE) {
            worker_waiters_.parked.notify_one();
        } else {
            worker_waiters_.parked.notify_all();
        }
    }
    if (wake_helpers) {
        if (notify == Notify::ONE) {
            helper_waiters_.parked.notify_one();
        } else {
            helper_waiters_.parked.notify_all();
        }
    }
}

auto WorkerPool::loop_completed() -> void {
    // Only a thread inside `wait_for(predicate)` can be waiting for a completion, and it registers before it checks.
    if (waiting_helpers_.load() == 0) return;
    update([] {}, Notify::HELPERS);
}

} // namespace kidi::runtime::ynn
