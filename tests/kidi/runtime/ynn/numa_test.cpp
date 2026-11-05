#include "kidi/runtime/ynn/numa.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

#if defined(__linux__) && !defined(__ANDROID__)
#include <sched.h>
#endif

namespace {

using kidi::runtime::ynn::NumaNode;

auto expect(bool condition, const char* what) -> bool {
    if (!condition) std::cerr << "FAILED: " << what << '\n';
    return condition;
}

auto test_parse() -> bool {
    using kidi::runtime::ynn::parse_cpu_list;
    return expect(parse_cpu_list("0-3,8,10-11\n") == std::vector<int>{0, 1, 2, 3, 8, 10, 11}, "ranges and singles") &&
           expect(parse_cpu_list("5") == std::vector<int>{5}, "single") &&
           expect(parse_cpu_list("").empty(), "empty") && expect(parse_cpu_list("3-1").empty(), "descending range") &&
           expect(parse_cpu_list("a-b").empty(), "letters") && expect(parse_cpu_list("0-").empty(), "open range") &&
           expect(parse_cpu_list("0,,2").empty(), "empty item") && expect(parse_cpu_list("0-70000").empty(), "huge id");
}

auto test_choose() -> bool {
    using kidi::runtime::ynn::choose_numa_node;
    const std::array<NumaNode, 2> two = {NumaNode{0, {0, 1, 2, 3}}, NumaNode{1, {4, 5, 6, 7}}};
    const std::array<NumaNode, 2> uneven = {NumaNode{0, {0, 1}}, NumaNode{1, {4, 5, 6, 7}}};
    const std::array<NumaNode, 1> one = {NumaNode{0, {0, 1, 2, 3}}};
    return expect(choose_numa_node(two, 5, 4) == std::optional<std::size_t>(1), "prefers the current node") &&
           expect(choose_numa_node(two, 1, 4) == std::optional<std::size_t>(0), "prefers the current node on 0") &&
           expect(choose_numa_node(two, -1, 2) == std::optional<std::size_t>(0), "unknown cpu takes the first fit") &&
           expect(choose_numa_node(uneven, 0, 4) == std::optional<std::size_t>(1),
                  "current node too small, use the one that fits") &&
           expect(!choose_numa_node(two, 0, 5), "threads exceed every node") &&
           expect(!choose_numa_node(one, 0, 2), "single node needs no confinement") &&
           expect(!choose_numa_node(two, 0, 0), "unknown thread count") &&
           expect(!choose_numa_node({}, 0, 1), "no topology");
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
auto test_confine() -> bool {
    bool ok = true;
    std::thread([&] {
        const auto before = affinity();
        setenv("KIDI_NUMA", "off", 1);
        ok = expect(!kidi::runtime::ynn::confine_to_numa_node(1), "KIDI_NUMA=off disables confinement") &&
             expect(affinity() == before, "KIDI_NUMA=off leaves the affinity alone");
        unsetenv("KIDI_NUMA");
        const auto node = kidi::runtime::ynn::confine_to_numa_node(1);
        const auto after = affinity();
        ok = expect(!after.empty(), "affinity stays non-empty") &&
             expect(std::ranges::includes(before, after), "confinement only narrows the affinity") && ok;
        if (node) {
            ok = expect(after.size() <= before.size(), "confined to a subset") && ok;
            std::vector<int> inherited;
            std::thread([&] { inherited = affinity(); }).join();
            ok = expect(inherited == after, "threads created later inherit the confinement") && ok;
            ok = expect(!kidi::runtime::ynn::confine_to_numa_node(1) || affinity() == after,
                        "a second call stays on the same node") &&
                 ok;
        } else {
            ok = expect(after == before, "no node chosen, no change") && ok;
        }
    }).join();
    return ok;
}
#else
auto test_confine() -> bool { return expect(!kidi::runtime::ynn::confine_to_numa_node(1), "unsupported platform"); }
#endif

} // namespace

auto main() -> int { return test_parse() && test_choose() && test_confine() ? 0 : 1; }
