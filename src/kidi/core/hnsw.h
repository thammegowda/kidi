#pragma once

#include "kidi/core/error.h"
#include "kidi/tensor/tensor.h"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace kidi::core {

struct HnswBuildOptions {
    std::uint32_t m = 32;
    std::size_t ef_construction = 200;
    std::uint64_t seed = 42;
    std::size_t candidates = 8, ef_search = 800;
};

struct HnswNeighbor {
    float distance;
    std::uint32_t id;
};

struct HnswWorkspace {
    std::vector<std::uint32_t> visited;
    std::uint32_t epoch = 0;
    std::vector<std::pair<float, std::uint32_t>> frontier, nearest;
    std::vector<HnswNeighbor> result;
};

using HnswProgress = std::function<void(std::size_t completed, std::size_t total)>;
/// The executor must complete all body calls before returning; graph updates remain ordered.
using HnswParallelFor = std::function<Result<void>(std::size_t, std::function<void(std::size_t)>)>;

class HnswIndex {
public:
    HnswIndex(HnswIndex&&) noexcept;
    auto operator=(HnswIndex&&) noexcept -> HnswIndex&;
    ~HnswIndex();

    HnswIndex(const HnswIndex&) = delete;
    auto operator=(const HnswIndex&) -> HnswIndex& = delete;

    static auto build(tensor::Tensor vectors, HnswBuildOptions options = {}, HnswProgress progress = {},
                      HnswParallelFor parallel_for = {}) -> Result<HnswIndex>;
    static auto load(const std::filesystem::path& path, tensor::Tensor vectors) -> Result<HnswIndex>;
    /// Imports an original 64-bit hnswlib FP32 export, checking every vector against the supplied table.
    static auto from_hnswlib(const std::filesystem::path& path, tensor::Tensor vectors, std::size_t candidates = 8,
                             std::size_t ef_search = 800) -> Result<HnswIndex>;
    /// Atomically publishes a complete graph-only file; existing paths are never overwritten.
    auto save(const std::filesystem::path& path) const -> Result<void>;
    /// The result borrows workspace storage until its next search; concurrent queries need separate workspaces.
    auto search(std::span<const float> query, HnswWorkspace& workspace, std::size_t candidates = 0,
                std::size_t ef = 0) const -> Result<std::span<const HnswNeighbor>>;

    auto size() const noexcept -> std::size_t;
    auto width() const noexcept -> std::size_t;
    auto candidates() const noexcept -> std::size_t;
    auto ef_search() const noexcept -> std::size_t;
    auto runtime_bytes() const noexcept -> std::size_t;
    auto vectors() const noexcept -> const tensor::Tensor&;

private:
    struct State;
    explicit HnswIndex(std::unique_ptr<State> state);
    auto query_distance(std::span<const float> query, std::uint32_t node) const -> float;
    auto node_distance(std::uint32_t left, std::uint32_t right) const -> float;

    std::unique_ptr<State> state_;
};

} // namespace kidi::core
