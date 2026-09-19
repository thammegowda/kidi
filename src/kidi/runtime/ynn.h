#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <ynnpack.h>

#include "kidi/core/error.h"

namespace kidi::runtime {

class YnnExecutable;
class YnnThreadPool;

void set_ynn_thread_count(std::size_t total_threads) noexcept;
[[nodiscard]] std::size_t ynn_thread_count() noexcept;
[[nodiscard]] std::uint64_t ynn_supported_arch_flags() noexcept;
[[nodiscard]] std::string ynn_supported_arch_names();

class YnnGraph {
public:
    YnnGraph(YnnGraph&& other) noexcept;
    YnnGraph& operator=(YnnGraph&& other) noexcept;
    ~YnnGraph();

    YnnGraph(const YnnGraph&) = delete;
    YnnGraph& operator=(const YnnGraph&) = delete;

    [[nodiscard]] static Result<YnnGraph> create(std::uint32_t external_value_count, std::uint32_t flags = 0);
    [[nodiscard]] ynn_subgraph_t get() const noexcept;
    [[nodiscard]] Result<YnnExecutable> compile() &&;

private:
    explicit YnnGraph(ynn_subgraph_t graph);
    [[nodiscard]] ynn_subgraph_t release() noexcept;

    ynn_subgraph_t graph_ = nullptr;
};

class YnnExecutable {
public:
    YnnExecutable(YnnExecutable&& other) noexcept;
    YnnExecutable& operator=(YnnExecutable&& other) noexcept;
    ~YnnExecutable();

    YnnExecutable(const YnnExecutable&) = delete;
    YnnExecutable& operator=(const YnnExecutable&) = delete;

    [[nodiscard]] Result<void> set_shape(std::uint32_t external_id, std::span<const std::size_t> dimensions);
    [[nodiscard]] Result<void> reshape();
    [[nodiscard]] Result<void> bind(std::uint32_t external_id, void* data);
    [[nodiscard]] Result<void> invoke();
    [[nodiscard]] Result<std::vector<std::size_t>> shape(std::uint32_t external_id) const;
    [[nodiscard]] Result<std::int32_t> concurrency() const;

private:
    friend class YnnGraph;
    YnnExecutable(ynn_subgraph_t graph, ynn_runtime_t runtime, std::shared_ptr<YnnThreadPool> thread_pool);

    ynn_subgraph_t graph_ = nullptr;
    ynn_runtime_t runtime_ = nullptr;
    std::shared_ptr<YnnThreadPool> thread_pool_;
    std::unordered_map<std::uint32_t, std::vector<std::size_t>> external_shapes_;
    bool reshape_needed_ = true;
};

[[nodiscard]] Result<void> check_ynn_status(ynn_status status, const char* operation);

} // namespace kidi::runtime