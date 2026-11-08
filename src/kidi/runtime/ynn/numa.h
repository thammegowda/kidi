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

/// Restricts the calling thread, and the threads it creates afterwards, to the CPUs of one NUMA node.
///
/// Decoding streams every weight once per token, so it is bound by memory bandwidth, and a thread reading memory of
/// another socket gets a fraction of it. Pages belong to the node of the thread that first writes them, so call this
/// before loading a model and creating the thread pool. It only acts on Linux, when the process may run on several
/// nodes and `threads` threads fit within one of them. `KIDI_NUMA=off` disables it. Returns the node chosen, if any.
auto confine_to_numa_node(std::size_t threads) -> std::optional<int>;

} // namespace kidi::runtime::ynn
