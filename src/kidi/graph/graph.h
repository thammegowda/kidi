#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "kidi/runtime/operator.h"
#include "kidi/tensor/tensor.h"

namespace kidi::graph {
using tensor::Tensor;

/// How a recorded operator writes its outputs.
enum class Mode { RESULT, IN_PLACE };

/// Where an operand's storage comes from: a step input, an earlier node output, or a fixed external tensor.
struct Source {
    enum class Kind { INPUT, NODE, EXTERNAL };
    Kind kind = Kind::EXTERNAL;
    std::size_t index = 0, output = 0;
    const void* storage = nullptr;
    std::int64_t offset = 0; // bytes from the source region start (from the storage start when EXTERNAL)
};

/// One recorded operator call with owned copies of its specification.
struct Node {
    runtime::Operation operation;
    tensor::DType dtype;
    float epsilon;
    bool dynamic_parameters, packed_prefill, vector_projection;
    std::vector<std::int64_t> attributes;
    std::array<std::vector<float>, 2> scales;
    std::array<std::vector<std::int32_t>, 2> zero_points;
    std::array<std::size_t, 2> block_sizes{};
    std::shared_ptr<runtime::Operator> prepared;
    Mode mode = Mode::RESULT;
    std::vector<Tensor> operands; // inputs, then outputs
    std::size_t input_count = 0;
    std::vector<Source> sources; // one per operand

    auto inputs() const -> std::span<const Tensor> { return {operands.data(), input_count}; }
    auto outputs() const -> std::span<const Tensor> { return std::span(operands).subspan(input_count); }
};

/// A captured fixed-shape step: pointer-stable buffers and prepared operators replayed in recorded order.
/// Step inputs may be rebound to other contiguous tensors with the same shape, dtype, and device.
class Graph {
public:
    /// Rebinds changed inputs, runs every node, and returns graph-owned outputs overwritten by the next replay.
    auto replay(std::span<const Tensor> inputs) -> std::span<const Tensor>;
    auto nodes() const noexcept -> std::span<const Node> { return nodes_; }
    auto inputs() const noexcept -> std::span<const Tensor> { return inputs_; }
    auto outputs() const noexcept -> std::span<const Tensor> { return outputs_; }

private:
    friend class Recorder;
    friend auto require_same_structure(const Graph& first, const Graph& second, std::string_view key) -> void;
    struct Use {
        std::size_t node, operand; // node == SIZE_MAX refers to graph output `operand`
    };
    auto bind(std::size_t slot, const Tensor& input) -> void;

    std::vector<Node> nodes_;
    std::vector<Tensor> inputs_, outputs_;
    std::vector<Source> output_sources_;
    std::vector<std::vector<Use>> uses_; // per input slot
};

/// Records the operator calls a step makes while it runs eagerly.
class Recorder {
public:
    explicit Recorder(std::span<const Tensor> inputs);
    auto record(const runtime::OperatorSpec& spec, std::shared_ptr<runtime::Operator> prepared, Mode mode,
                runtime::TensorInputs inputs, std::span<const Tensor> outputs) -> void;
    auto finish(std::span<const Tensor> outputs) && -> Graph;

private:
    auto locate(const Tensor& tensor) const -> Source;
    Graph graph_;
};

/// Throws ops::Failure describing the first difference between two captures of the same step.
auto require_same_structure(const Graph& first, const Graph& second, std::string_view key) -> void;

} // namespace kidi::graph
