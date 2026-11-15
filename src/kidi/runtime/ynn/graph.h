#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <ynnpack.h>

#include "kidi/core/error.h"
#include "kidi/runtime/ynn/worker_pool.h"
#include "kidi/tensor/tensor.h"

namespace kidi::runtime::ynn {

class Executable;
class ThreadPool;

auto set_thread_count(std::size_t total_threads) noexcept -> void;
auto thread_count() noexcept -> std::size_t;
auto reserve_thread_pool(std::size_t total_threads) -> Result<void>;
auto parallel_for(std::size_t size, std::function<void(std::size_t)> body) -> Result<void>;
auto parallel_tasks(std::size_t tasks, void* context, void (*body)(void*, std::size_t)) -> Result<void>;
auto parallel_ranges(std::size_t items, std::size_t min_grain, void* context,
                     void (*body)(void*, std::size_t, std::size_t)) -> Result<void>;
auto parallel_ranges_with_worker(std::size_t items, std::size_t min_grain, std::size_t scratch_bytes, void* context,
                                 void (*body)(void*, const WorkerContext&, std::size_t, std::size_t)) -> Result<void>;
auto parallel_team(std::size_t max_threads, std::size_t scratch_bytes, void* context,
                   void (*body)(void*, const WorkerContext&, PhaseBarrier&)) -> Result<void>;

template <typename Body>
auto parallel_tasks(std::size_t tasks, Body&& body) -> Result<void> {
    using Callable = std::remove_reference_t<Body>;
    return parallel_tasks(tasks, const_cast<void*>(static_cast<const void*>(std::addressof(body))),
                          [](void* context, std::size_t task) { (*static_cast<Callable*>(context))(task); });
}

template <typename Body>
auto parallel_ranges(std::size_t items, std::size_t min_grain, Body&& body) -> Result<void> {
    using Callable = std::remove_reference_t<Body>;
    return parallel_ranges(
        items, min_grain, const_cast<void*>(static_cast<const void*>(std::addressof(body))),
        [](void* context, std::size_t begin, std::size_t end) { (*static_cast<Callable*>(context))(begin, end); });
}

template <typename Body>
auto parallel_ranges_with_worker(std::size_t items, std::size_t min_grain, std::size_t scratch_bytes, Body&& body)
    -> Result<void> {
    using Callable = std::remove_reference_t<Body>;
    return parallel_ranges_with_worker(items, min_grain, scratch_bytes,
                                       const_cast<void*>(static_cast<const void*>(std::addressof(body))),
                                       [](void* context, const WorkerContext& worker, std::size_t begin,
                                          std::size_t end) { (*static_cast<Callable*>(context))(worker, begin, end); });
}

template <typename Body>
auto parallel_team(std::size_t max_threads, std::size_t scratch_bytes, Body&& body) -> Result<void> {
    using Callable = std::remove_reference_t<Body>;
    return parallel_team(max_threads, scratch_bytes, const_cast<void*>(static_cast<const void*>(std::addressof(body))),
                         [](void* context, const WorkerContext& worker, PhaseBarrier& barrier) {
                             (*static_cast<Callable*>(context))(worker, barrier);
                         });
}
auto supported_arch_flags() noexcept -> std::uint64_t;
auto supported_arch_names() -> std::string;

class Graph {
public:
    Graph(Graph&& other) noexcept;
    auto operator=(Graph&& other) noexcept -> Graph&;
    ~Graph();

    Graph(const Graph&) = delete;
    auto operator=(const Graph&) -> Graph& = delete;

    static auto create(std::uint32_t external_value_count, std::uint32_t flags = 0) -> Result<Graph>;
    auto get() const noexcept -> ynn_subgraph_t;
    auto compile(std::size_t total_threads = 0) && -> Result<Executable>;

private:
    explicit Graph(ynn_subgraph_t graph);
    auto release() noexcept -> ynn_subgraph_t;

    ynn_subgraph_t graph_ = nullptr;
};

class Executable {
public:
    Executable(Executable&& other) noexcept;
    auto operator=(Executable&& other) noexcept -> Executable&;
    ~Executable();

    Executable(const Executable&) = delete;
    auto operator=(const Executable&) -> Executable& = delete;

    auto reshape() -> Result<void>;
    auto bind(std::uint32_t external_id, void* data) -> Result<void>;
    auto bind(std::uint32_t external_id, tensor::Tensor& tensor) -> Result<void>;
    auto bind(std::uint32_t external_id, const tensor::Tensor& tensor) -> Result<void>;
    auto invoke() -> Result<void>;
    auto shape(std::uint32_t external_id) const -> Result<std::vector<std::size_t>>;

private:
    friend class Graph;
    Executable(ynn_subgraph_t graph, ynn_runtime_t runtime, std::shared_ptr<ThreadPool> thread_pool);

    ynn_subgraph_t graph_ = nullptr;
    ynn_runtime_t runtime_ = nullptr;
    std::shared_ptr<ThreadPool> thread_pool_;
};

auto check_status(ynn_status status, const char* operation) -> Result<void>;

} // namespace kidi::runtime::ynn