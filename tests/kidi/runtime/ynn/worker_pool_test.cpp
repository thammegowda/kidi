#include "ynnpack/base/thread_pool.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace {

using ::ynn::PhaseBarrier;
using ::ynn::WorkerContext;
using ::ynn::WorkerPool;

constexpr auto sum_below(int n) -> int { return n * (n - 1) / 2; }

TEST(WorkerPoolTest, ParallelFor) {
    for (const int workers : {0, 1, 3, 7}) {
        WorkerPool pool(workers);
        ASSERT_EQ(pool.thread_count(), workers) << "thread_count reports the background workers";
        ASSERT_EQ(pool.total_threads(), static_cast<std::size_t>(workers + 1)) << "total_threads includes caller";
        for (int n = 0; n < 100; ++n) {
            std::atomic<int> count = 0;
            std::atomic<int> sum = 0;
            pool.parallel_for(static_cast<std::size_t>(n), [&](std::size_t i) {
                ++count;
                sum += static_cast<int>(i);
            });
            ASSERT_EQ(count, n) << "every iteration runs exactly once";
            ASSERT_EQ(sum, sum_below(n)) << "every iteration runs exactly once";
        }
    }
}

TEST(WorkerPoolTest, NestedLoopsCompleteInline) {
    WorkerPool pool(3);
    std::atomic<int> count = 0;
    std::atomic<int> sum = 0;
    pool.parallel_for(10, [&](std::size_t) {
        pool.parallel_for(8, [&](std::size_t j) {
            ++count;
            sum += static_cast<int>(j);
        });
    });
    EXPECT_EQ(count, 80) << "nested loops complete inline without deadlock";
    EXPECT_EQ(sum, 10 * sum_below(8)) << "nested loops complete inline without deadlock";
}

TEST(WorkerPoolTest, RunTeamAndPhaseBarrier) {
    WorkerPool pool(3);
    std::vector<int> phase1(4, 0);
    std::vector<int> phase2(4, 0);
    struct State {
        std::vector<int>* p1;
        std::vector<int>* p2;
    } state{&phase1, &phase2};
    pool.run_team(4, 256, &state, [](void* raw, const WorkerContext& worker, PhaseBarrier& barrier) {
        auto* s = static_cast<State*>(raw);
        (*s->p1)[worker.tid] = static_cast<int>(worker.tid + 10);
        if (!barrier.arrive_and_wait()) return;
        int total = 0;
        for (int v : *s->p1) total += v;
        (*s->p2)[worker.tid] = total + static_cast<int>(worker.tid);
        (void)barrier.arrive_and_wait();
    });
    constexpr int EXPECTED_P1_SUM = 10 + 11 + 12 + 13;
    for (int t = 0; t < 4; ++t) {
        ASSERT_EQ(phase2[t], EXPECTED_P1_SUM + t) << "PhaseBarrier synchronizes multi-phase team state";
    }

    // Stress alternating partial-team (2 of 4) and full-team (4 of 4) PhaseBarrier launches to verify non-participating
    // workers (tid >= team_size) never race with subsequent full-team dispatches.
    for (int iter = 0; iter < 5000; ++iter) {
        const std::size_t team = (iter % 2 == 0) ? 2 : 4;
        std::atomic<std::size_t> seen = 0;
        pool.run_team(team, 0, &seen, [](void* raw, const WorkerContext&, PhaseBarrier& barrier) {
            static_cast<std::atomic<std::size_t>*>(raw)->fetch_add(1, std::memory_order_relaxed);
            (void)barrier.arrive_and_wait();
        });
        ASSERT_EQ(seen.load(std::memory_order_relaxed), team) << "partial/full alternating team count exact";
    }
}

TEST(WorkerPoolTest, RunRanges) {
    WorkerPool pool(3);
    constexpr std::size_t N = 256;
    std::vector<std::atomic<int>> hits(N);
    for (auto& h : hits) h.store(0, std::memory_order_relaxed);
    std::atomic<bool> aligned_ok = true;
    struct Ctx {
        std::vector<std::atomic<int>>* hits;
        std::atomic<bool>* aligned_ok;
    } ctx{&hits, &aligned_ok};
    pool.run_ranges(N, 2, 1024, &ctx, [](void* raw, const WorkerContext& worker, std::size_t begin, std::size_t end) {
        auto* c = static_cast<Ctx*>(raw);
        if (worker.scratch == nullptr || (reinterpret_cast<std::uintptr_t>(worker.scratch) % 64 != 0) ||
            worker.scratch_size < 1024) {
            c->aligned_ok->store(false, std::memory_order_relaxed);
        }
        auto* buf = reinterpret_cast<std::size_t*>(worker.scratch);
        buf[0] = worker.tid;
        if (worker.tid == 1 && begin < 80) {
            // Simulate an E-core / OS-preempted straggler so fast workers steal its remaining home range.
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        if (buf[0] != worker.tid) c->aligned_ok->store(false, std::memory_order_relaxed);
        for (auto i = begin; i < end; ++i) c->hits->at(i).fetch_add(1, std::memory_order_relaxed);
    });
    ASSERT_TRUE(aligned_ok.load()) << "WorkerContext provides isolated 64B-aligned per-tid scratch";
    for (const auto& h : hits)
        ASSERT_EQ(h.load(std::memory_order_relaxed), 1) << "home-slot + stealing executes every item once";
}

TEST(WorkerPoolTest, NestedInlineScratch) {
    // Regression test (Finding #6): nested inline/serial calls requesting larger scratch_bytes must not alias or free
    // outer WorkerContext::scratch buffers.
    for (const int workers : {0, 3}) {
        WorkerPool pool(workers);
        std::atomic<bool> intact = true;
        struct State {
            WorkerPool* pool;
            std::atomic<bool>* intact;
        } state{&pool, &intact};
        pool.run_ranges(8, 1, 128, &state, [](void* raw, const WorkerContext& outer, std::size_t begin, std::size_t) {
            auto* s = static_cast<State*>(raw);
            if (outer.scratch == nullptr || outer.scratch_size < 128) {
                s->intact->store(false, std::memory_order_relaxed);
                return;
            }
            for (std::size_t i = 0; i < 128; ++i) outer.scratch[i] = static_cast<std::byte>((begin + i) & 0xFFu);

            // Trigger nested inline run_ranges and run_team with much larger scratch (64 KiB -> 128 KiB) to force
            // allocation/reallocation if TLS scratch were shared across nesting depths.
            s->pool->run_ranges(
                4, 1, 65536, s, [](void* inner_raw, const WorkerContext& mid, std::size_t, std::size_t) {
                    auto* is = static_cast<State*>(inner_raw);
                    if (mid.scratch == nullptr || mid.scratch_size < 65536) {
                        is->intact->store(false, std::memory_order_relaxed);
                        return;
                    }
                    for (std::size_t k = 0; k < 65536; ++k) mid.scratch[k] = std::byte{0xA5};
                    is->pool->run_team(2, 131072, is, [](void* leaf_raw, const WorkerContext& leaf, PhaseBarrier& b) {
                        auto* ls = static_cast<State*>(leaf_raw);
                        if (leaf.scratch == nullptr || leaf.scratch_size < 131072) {
                            ls->intact->store(false, std::memory_order_relaxed);
                            return;
                        }
                        for (std::size_t m = 0; m < 131072; ++m) leaf.scratch[m] = std::byte{0x5A};
                        (void)b.arrive_and_wait();
                    });
                    for (std::size_t k = 0; k < 65536; ++k) {
                        if (mid.scratch[k] != std::byte{0xA5}) {
                            is->intact->store(false, std::memory_order_relaxed);
                            break;
                        }
                    }
                });

            for (std::size_t i = 0; i < 128; ++i) {
                if (outer.scratch[i] != static_cast<std::byte>((begin + i) & 0xFFu)) {
                    s->intact->store(false, std::memory_order_relaxed);
                    break;
                }
            }
        });
        ASSERT_TRUE(intact.load()) << "nested inline scratch preserves outer buffers without UAF or aliasing";
    }
}

TEST(WorkerPoolTest, IdleThenWake) {
    // Let every worker give up spinning and block on atomic::wait, then make sure the next loops still reach them.
    WorkerPool pool(4);
    for (int round = 0; round < 3; ++round) {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        std::vector<std::atomic<int>> hits(64);
        pool.parallel_for(hits.size(), [&](std::size_t i) { ++hits[i]; });
        for (const auto& hit : hits)
            ASSERT_EQ(hit, 1) << "blocked workers wake up and iterations run once";
    }
}

TEST(WorkerPoolTest, ConcurrentCallers) {
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
    EXPECT_EQ(total, 800) << "concurrent loops all complete";
}

TEST(WorkerPoolTest, Lifecycle) {
    for (int i = 0; i < 50; ++i) {
        WorkerPool pool(i % 5);
        if (i % 2 == 0) pool.parallel_for(8, [](std::size_t) {});
    }
}

TEST(WorkerPoolTest, RunTasks) {
    WorkerPool pool(3);
    constexpr std::size_t N = 100003;
    std::vector<std::atomic<int>> hits(N);
    for (auto& h : hits) h.store(0, std::memory_order_relaxed);
    struct Context {
        std::vector<std::atomic<int>>* hits;
    } ctx{&hits};
    pool.run_tasks(N, &ctx, [](void* raw, std::size_t index) {
        static_cast<Context*>(raw)->hits->at(index).fetch_add(1, std::memory_order_relaxed);
    });
    for (const auto& h : hits)
        ASSERT_EQ(h.load(std::memory_order_relaxed), 1) << "large run_tasks executes each index exactly once";
}

TEST(WorkerPoolTest, ExceptionPropagation) {
    WorkerPool pool(3);
    bool caught = false;
    try {
        pool.parallel_for(32, [](std::size_t index) {
            if (index == 17) throw 42;
        });
    } catch (int value) {
        caught = (value == 42);
    }
    ASSERT_TRUE(caught) << "run_ranges propagates worker exception cleanly";

    bool barrier_caught = false;
    try {
        pool.run_team(4, 0, nullptr, [](void*, const WorkerContext& worker, PhaseBarrier& barrier) {
            if (worker.tid == 2) throw 99;
            (void)barrier.arrive_and_wait();
        });
    } catch (int value) {
        barrier_caught = (value == 99);
    }
    ASSERT_TRUE(barrier_caught) << "PhaseBarrier aborts cleanly when a team member throws";

    std::atomic<int> sum = 0;
    pool.parallel_for(32, [&](std::size_t index) { sum.fetch_add(static_cast<int>(index)); });
    EXPECT_EQ(sum, sum_below(32)) << "pool remains usable after exceptions";
}

} // namespace
