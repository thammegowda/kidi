#include "ynnpack/base/topology.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#if defined(__linux__) && !defined(__ANDROID__)
#include <sched.h>
#endif

namespace {

using ::ynn::NumaNode;

TEST(NumaTest, ParsesCpuLists) {
    using ::ynn::parse_cpu_list;
    EXPECT_EQ(parse_cpu_list("0-3,8,10-11\n"), (std::vector<int>{0, 1, 2, 3, 8, 10, 11})) << "ranges and singles";
    EXPECT_EQ(parse_cpu_list("5"), std::vector<int>{5}) << "single";
    EXPECT_TRUE(parse_cpu_list("").empty()) << "empty";
    EXPECT_TRUE(parse_cpu_list("3-1").empty()) << "descending range";
    EXPECT_TRUE(parse_cpu_list("a-b").empty()) << "letters";
    EXPECT_TRUE(parse_cpu_list("0-").empty()) << "open range";
    EXPECT_TRUE(parse_cpu_list("0,,2").empty()) << "empty item";
    EXPECT_TRUE(parse_cpu_list("0-70000").empty()) << "huge id";
}

TEST(NumaTest, ChoosesNode) {
    using ::ynn::choose_numa_node;
    const std::array<NumaNode, 2> two = {NumaNode{0, {0, 1, 2, 3}}, NumaNode{1, {4, 5, 6, 7}}};
    const std::array<NumaNode, 2> uneven = {NumaNode{0, {0, 1}}, NumaNode{1, {4, 5, 6, 7}}};
    const std::array<NumaNode, 1> one = {NumaNode{0, {0, 1, 2, 3}}};
    EXPECT_EQ(choose_numa_node(two, 5, 4), std::optional<std::size_t>(1)) << "prefers the current node";
    EXPECT_EQ(choose_numa_node(two, 1, 4), std::optional<std::size_t>(0)) << "prefers the current node on 0";
    EXPECT_EQ(choose_numa_node(two, -1, 2), std::optional<std::size_t>(0)) << "unknown cpu takes the first fit";
    EXPECT_EQ(choose_numa_node(uneven, 0, 4), std::optional<std::size_t>(1))
        << "current node too small, use the one that fits";
    EXPECT_FALSE(choose_numa_node(two, 0, 5)) << "threads exceed every node";
    EXPECT_FALSE(choose_numa_node(one, 0, 2)) << "single node needs no confinement";
    EXPECT_FALSE(choose_numa_node(two, 0, 0)) << "unknown thread count";
    EXPECT_FALSE(choose_numa_node({}, 0, 1)) << "no topology";
}

#if defined(__linux__) && !defined(__ANDROID__)
auto affinity() -> std::vector<int> {
    cpu_set_t set;
    CPU_ZERO(&set);
    std::vector<int> cpus;
    if (sched_getaffinity(0, sizeof(set), &set) != 0) return cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &set)) cpus.push_back(cpu);
    return cpus;
}

// Runs on a scratch thread so the affinity change does not leak into the rest of the process.
TEST(NumaTest, ConfinesToNode) {
    std::thread([] {
        const auto before = affinity();
        const auto node = ::ynn::confine_to_numa_node(1);
        const auto after = affinity();
        EXPECT_FALSE(after.empty()) << "affinity stays non-empty";
        EXPECT_TRUE(std::ranges::includes(before, after)) << "confinement only narrows the affinity";
        if (node) {
            EXPECT_LE(after.size(), before.size()) << "confined to a subset";
            std::vector<int> inherited;
            std::thread([&] { inherited = affinity(); }).join();
            EXPECT_EQ(inherited, after) << "threads created later inherit the confinement";
            EXPECT_TRUE(!::ynn::confine_to_numa_node(1) || affinity() == after) << "a second call stays on the same node";
        } else {
            EXPECT_EQ(after, before) << "no node chosen, no change";
        }
    }).join();
}
#else
TEST(NumaTest, ConfinementUnsupported) { EXPECT_FALSE(::ynn::confine_to_numa_node(1)); }
#endif

TEST(NumaTest, ChoosesPerformanceCpus) {
    using ::ynn::choose_performance_cpus;
    using ::ynn::count_performance_cpus;
    using ::ynn::CpuCapacity;
    const std::array<CpuCapacity, 8> hybrid = {
        CpuCapacity{0, 448},  CpuCapacity{1, 448},  CpuCapacity{2, 448},  CpuCapacity{3, 448},
        CpuCapacity{4, 1024}, CpuCapacity{5, 1024}, CpuCapacity{6, 1024}, CpuCapacity{7, 1024},
    };
    const std::array<CpuCapacity, 8> tri_cluster = {
        CpuCapacity{0, 320}, CpuCapacity{1, 320}, CpuCapacity{2, 320}, CpuCapacity{3, 320},
        CpuCapacity{4, 768}, CpuCapacity{5, 768}, CpuCapacity{6, 768}, CpuCapacity{7, 1024},
    };
    const std::array<CpuCapacity, 4> uniform = {
        CpuCapacity{0, 1024},
        CpuCapacity{1, 1024},
        CpuCapacity{2, 1024},
        CpuCapacity{3, 1024},
    };
    const std::vector<int> performance{4, 5, 6, 7};
    EXPECT_EQ(choose_performance_cpus(hybrid, 4), performance) << "picks P-cores when 4 fit";
    EXPECT_EQ(choose_performance_cpus(hybrid, 2), performance) << "keeps full P-core cluster";
    EXPECT_EQ(choose_performance_cpus(tri_cluster, 4), performance) << "tri-cluster picks Prime + Perf cores";
    EXPECT_TRUE(choose_performance_cpus(hybrid, 8).empty()) << "all cores requested needs no narrowing";
    EXPECT_TRUE(choose_performance_cpus(uniform, 2).empty()) << "uniform topology needs no P-core narrowing";
    EXPECT_TRUE(choose_performance_cpus(hybrid, 0).empty()) << "zero threads needs no narrowing";
    EXPECT_EQ(count_performance_cpus(hybrid), 4) << "2-tier auto count picks 4 P-cores";
    EXPECT_EQ(count_performance_cpus(tri_cluster), 4) << "3-tier auto count picks Prime + Perf cores";
    EXPECT_EQ(count_performance_cpus(uniform), 4) << "uniform auto count keeps all cores";
    EXPECT_EQ(count_performance_cpus({}), 0) << "empty topology returns 0";
}

} // namespace
