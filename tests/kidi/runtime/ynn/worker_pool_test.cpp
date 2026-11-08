#include "kidi/runtime/ynn/worker_pool.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <thread>
#include <vector>

namespace {

using kidi::runtime::ynn::WorkerPool;

auto expect(bool condition, const char* what) -> bool {
    if (!condition) std::cerr << "FAILED: " << what << '\n';
    return condition;
}

constexpr auto sum_below(int n) -> int { return n * (n - 1) / 2; }

auto test_parallel_for() -> bool {
    for (const int workers : {0, 1, 3, 7}) {
        WorkerPool pool(workers);
        if (!expect(pool.thread_count() == workers, "thread_count reports the background workers")) return false;
        for (int n = 0; n < 100; ++n) {
            std::atomic<int> count = 0;
            std::atomic<int> sum = 0;
            pool.parallel_for(static_cast<std::size_t>(n), [&](std::size_t i) {
                ++count;
                sum += static_cast<int>(i);
            });
            if (!expect(count == n && sum == sum_below(n), "every iteration runs exactly once")) return false;
        }
    }
    return true;
}

auto test_nested() -> bool {
    WorkerPool pool(3);
    std::atomic<int> count = 0;
    std::atomic<int> sum = 0;
    pool.parallel_for(10, [&](std::size_t) {
        pool.parallel_for(8, [&](std::size_t j) {
            ++count;
            sum += static_cast<int>(j);
        });
    });
    return expect(count == 80 && sum == 10 * sum_below(8), "nested loops complete");
}

auto test_atomic_call() -> bool {
    WorkerPool pool(3);
    int sum = 0;
    pool.parallel_for(1000, [&](std::size_t i) { pool.atomic_call([&] { sum += static_cast<int>(i); }); });
    return expect(sum == sum_below(1000), "atomic calls are mutually exclusive");
}

auto test_barriers() -> bool {
    // Two threads hand control back and forth through predicates, as semaphores in a pipelined loop do.
    WorkerPool pool(3);
    bool first = false;
    bool second = false;
    bool third = false;
    std::thread other([&] {
        pool.atomic_call([&] { first = true; });
        pool.wait_for([&]() -> bool { return second; });
        pool.atomic_call([&] { third = true; });
    });
    pool.wait_for([&]() -> bool { return first; });
    pool.atomic_call([&] { second = true; });
    pool.wait_for([&]() -> bool { return third; });
    other.join();
    return true;
}

auto test_enqueue() -> bool {
    WorkerPool pool(3);
    for (const int max_workers : {1, 2, 1000000}) {
        std::atomic<int> sum = 0;
        auto loop = pool.enqueue(50, [&](std::size_t i) { sum += static_cast<int>(i); }, max_workers);
        pool.wait_for(&*loop);
        // A worker that registered late on the finished loop may still be leaving it, so only the work is checked.
        if (!expect(sum == sum_below(50), "enqueued loop completes")) return false;
    }
    return true;
}

auto test_helping() -> bool {
    // A thread waiting on a predicate runs queued loops, which is what lets a loop depend on a later one.
    WorkerPool pool(0);
    std::atomic<int> sum = 0;
    bool done = false;
    pool.enqueue(20, [&](std::size_t i) { sum += static_cast<int>(i); });
    pool.wait_for([&]() -> bool {
        if (sum == sum_below(20)) done = true;
        return done;
    });
    return expect(done, "waiting threads execute queued loops");
}

auto test_idle_then_wake() -> bool {
    // Let every worker give up spinning and block, then make sure the next loops still reach them.
    WorkerPool pool(4);
    for (int round = 0; round < 3; ++round) {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        std::vector<std::atomic<int>> hits(64);
        pool.parallel_for(hits.size(), [&](std::size_t i) { ++hits[i]; });
        for (const auto& hit : hits)
            if (!expect(hit == 1, "blocked workers wake up and iterations run once")) return false;
    }
    return true;
}

auto test_concurrent_callers() -> bool {
    // Several application threads share one pool, as concurrent sessions do.
    WorkerPool pool(3);
    std::atomic<int> total = 0;
    std::vector<std::thread> callers;
    for (int caller = 0; caller < 4; ++caller)
        callers.emplace_back([&] {
            for (int round = 0; round < 200; ++round) {
                std::atomic<int> sum = 0;
                pool.parallel_for(16, [&](std::size_t i) { sum += static_cast<int>(i); });
                if (sum == sum_below(16)) ++total;
            }
        });
    for (auto& caller : callers) caller.join();
    return expect(total == 800, "concurrent loops all complete");
}

auto test_lifecycle() -> bool {
    for (int i = 0; i < 50; ++i) {
        WorkerPool pool(i % 5);
        if (i % 2 == 0) pool.parallel_for(8, [](std::size_t) {});
    }
    return true;
}

} // namespace

auto main() -> int {
    return test_parallel_for() && test_nested() && test_atomic_call() && test_barriers() && test_enqueue() &&
                   test_helping() && test_idle_then_wake() && test_concurrent_callers() && test_lifecycle()
               ? 0
               : 1;
}
