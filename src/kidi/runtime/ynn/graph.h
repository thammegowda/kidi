#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <ynnpack.h>

#include "kidi/core/error.h"
#include "kidi/tensor/tensor.h"

namespace kidi::runtime::ynn {

class Executable;
class ThreadPool;

auto set_thread_count(std::size_t total_threads) noexcept -> void;
auto thread_count() noexcept -> std::size_t;
auto reserve_thread_pool(std::size_t total_threads) -> Result<void>;
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