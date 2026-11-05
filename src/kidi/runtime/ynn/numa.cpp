#include "kidi/runtime/ynn/numa.h"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>

#if defined(__linux__) && !defined(__ANDROID__)
#include <sched.h>
#endif

namespace kidi::runtime::ynn {
namespace {

auto parse_cpu_id(std::string_view text, int& value) -> bool {
    constexpr int MAX_CPU_ID = 65535;
    const auto [end, status] = std::from_chars(text.data(), text.data() + text.size(), value);
    return status == std::errc{} && end == text.data() + text.size() && value >= 0 && value <= MAX_CPU_ID;
}

} // namespace

auto parse_cpu_list(std::string_view text) -> std::vector<int> {
    while (!text.empty() && (text.back() == '\n' || text.back() == ' ')) text.remove_suffix(1);
    std::vector<int> cpus;
    while (!text.empty()) {
        const auto comma = text.find(',');
        const auto item = text.substr(0, comma);
        text = comma == std::string_view::npos ? std::string_view{} : text.substr(comma + 1);
        const auto dash = item.find('-');
        int first = 0;
        int last = 0;
        const bool valid = dash == std::string_view::npos
                               ? parse_cpu_id(item, first) && (last = first, true)
                               : parse_cpu_id(item.substr(0, dash), first) && parse_cpu_id(item.substr(dash + 1), last);
        if (!valid || last < first) return {};
        for (int cpu = first; cpu <= last; ++cpu) cpus.push_back(cpu);
    }
    return cpus;
}

auto choose_numa_node(std::span<const NumaNode> nodes, int current_cpu, std::size_t threads)
    -> std::optional<std::size_t> {
    if (threads == 0 || nodes.size() < 2) return std::nullopt;
    std::optional<std::size_t> largest;
    for (std::size_t index = 0; index < nodes.size(); ++index) {
        const auto& cpus = nodes[index].cpus;
        if (cpus.size() < threads) continue;
        if (std::ranges::find(cpus, current_cpu) != cpus.end()) return index;
        if (!largest || cpus.size() > nodes[*largest].cpus.size()) largest = index;
    }
    return largest;
}

auto confine_to_numa_node(std::size_t threads) -> std::optional<int> {
#if defined(__linux__) && !defined(__ANDROID__)
    if (const char* setting = std::getenv("KIDI_NUMA"); setting != nullptr && std::string_view(setting) == "off")
        return std::nullopt;
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return std::nullopt;

    const std::filesystem::path root = "/sys/devices/system/node";
    std::error_code error;
    std::vector<NumaNode> nodes;
    for (const auto& entry : std::filesystem::directory_iterator(root, error)) {
        const auto name = entry.path().filename().string();
        NumaNode node;
        if (!name.starts_with("node") || !parse_cpu_id(std::string_view(name).substr(4), node.id)) continue;
        std::ifstream file(entry.path() / "cpulist");
        const std::string list((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        for (const int cpu : parse_cpu_list(list))
            if (cpu < CPU_SETSIZE && CPU_ISSET(cpu, &allowed)) node.cpus.push_back(cpu);
        if (!node.cpus.empty()) nodes.push_back(std::move(node));
    }
    std::ranges::sort(nodes, {}, &NumaNode::id);

    const auto chosen = choose_numa_node(nodes, sched_getcpu(), threads);
    if (!chosen) return std::nullopt;
    cpu_set_t target;
    CPU_ZERO(&target);
    for (const int cpu : nodes[*chosen].cpus) CPU_SET(cpu, &target);
    if (sched_setaffinity(0, sizeof(target), &target) != 0) return std::nullopt;
    return nodes[*chosen].id;
#else
    (void)threads;
    return std::nullopt;
#endif
}

} // namespace kidi::runtime::ynn
