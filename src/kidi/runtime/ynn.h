#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <vector>

#include <ynnpack.h>

#include "kidi/core/error.h"

namespace kidi::runtime {

class YnnExecutable;
class YnnThreadPool;

class YnnGraph {
public:
    YnnGraph(YnnGraph&& other) noexcept;
    YnnGraph& operator=(YnnGraph&& other) noexcept;
    ~YnnGraph();

    YnnGraph(const YnnGraph&) = delete;
    YnnGraph& operator=(const YnnGraph&) = delete;

    [[nodiscard]] static std::expected<YnnGraph, core::Error> create(std::uint32_t external_value_count);
    [[nodiscard]] ynn_subgraph_t get() const noexcept;
    [[nodiscard]] std::expected<YnnExecutable, core::Error> compile() &&;

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

    [[nodiscard]] std::expected<void, core::Error> set_shape(std::uint32_t external_id,
                                                             std::span<const std::size_t> dimensions);
    [[nodiscard]] std::expected<void, core::Error> reshape();
    [[nodiscard]] std::expected<void, core::Error> bind(std::uint32_t external_id, void* data);
    [[nodiscard]] std::expected<void, core::Error> invoke();
    [[nodiscard]] std::expected<std::vector<std::size_t>, core::Error> shape(std::uint32_t external_id) const;

private:
    friend class YnnGraph;
    YnnExecutable(ynn_subgraph_t graph, ynn_runtime_t runtime, std::shared_ptr<YnnThreadPool> thread_pool);

    ynn_subgraph_t graph_ = nullptr;
    ynn_runtime_t runtime_ = nullptr;
    std::shared_ptr<YnnThreadPool> thread_pool_;
};

[[nodiscard]] std::expected<void, core::Error> check_ynn_status(ynn_status status, const char* operation);

} // namespace kidi::runtime