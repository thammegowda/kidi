#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace kidi::runtime::ynn {

/// A NUMA node and the CPUs of it that the process may run on.
struct NumaNode {
    int id = 0;
    std::vector<int> cpus;
};

/// Parses a kernel CPU list such as "0-3,8,10-11". Malformed input yields an empty list.
auto parse_cpu_list(std::string_view text) -> std::vector<int>;

/// Picks the node that should host `threads` compute threads, as an index into `nodes`. Prefers the node of
/// `current_cpu`, then the node with the most CPUs. Returns nothing when the CPUs already belong to a single node or
/// when no node has `threads` CPUs, since then confining the threads would not keep them on one memory controller.
auto choose_numa_node(std::span<const NumaNode> nodes, int current_cpu, std::size_t threads)
    -> std::optional<std::size_t>;

/// A logical CPU and its relative compute capacity or maximum clock frequency.
struct CpuCapacity {
    int cpu = 0;
    std::uint64_t score = 0;
};

/// Returns the number of non-LITTLE / performance CPUs when `cpus` has heterogeneous tiers (excluding the lowest
/// capacity Efficiency/LITTLE tier on 2-tier P+E or 3-tier Prime+Perf+Little SoCs), or `cpus.size()` when uniform.
auto count_performance_cpus(std::span<const CpuCapacity> cpus) -> std::size_t;

/// Returns the default "auto" CPU thread count (`--threads 0`), preferring non-LITTLE performance cores on hybrid
/// Linux/Android systems unless `KIDI_NUMA=off` or `KIDI_BIG_CORES=off`.
auto default_thread_count() noexcept -> std::size_t;

/// Selects the highest-capacity subset of CPUs when `cpus` has heterogeneous tiers (e.g. ARM big.LITTLE or x86
/// P/E-cores) and the faster tier(s) contain at least `threads` CPUs. Returns empty when all CPUs have identical
/// capacity or when all CPUs are needed to satisfy `threads`.
auto choose_performance_cpus(std::span<const CpuCapacity> cpus, std::size_t threads) -> std::vector<int>;

/// Restricts the calling thread and future child threads on Linux/Android to high-capacity performance CPUs when the
/// allowed mask spans heterogeneous big.LITTLE / P+E tiers and the faster tiers fit `threads`. `KIDI_NUMA=off` or
/// `KIDI_BIG_CORES=off` disables it.
auto confine_to_performance_cpus(std::size_t threads) -> bool;

/// Restricts the calling thread, and the threads it creates afterwards, to the CPUs of one NUMA node.
///
/// Decoding streams every weight once per token, so it is bound by memory bandwidth, and a thread reading memory of
/// another socket gets a fraction of it. Pages belong to the node of the thread that first writes them, so call this
/// before loading a model and creating the thread pool. It only acts on Linux, when the process may run on several
/// nodes and `threads` threads fit within one of them. `KIDI_NUMA=off` disables it. Returns the node chosen, if any.
auto confine_to_numa_node(std::size_t threads) -> std::optional<int>;

} // namespace kidi::runtime::ynn
