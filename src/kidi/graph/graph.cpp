#include "kidi/graph/graph.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <string>
#include <utility>

#include "kidi/ops/context.h"

namespace kidi::graph {
namespace {
using ops::require;

constexpr auto OUTPUT_USE = std::numeric_limits<std::size_t>::max();

struct Region {
    const void* storage;
    std::int64_t begin, end;
};

auto region(const Tensor& tensor) -> Region {
    if (!tensor.defined() || !tensor.is_contiguous())
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "captured steps require defined contiguous tensors"});
    const auto begin = tensor.storage_offset() * static_cast<std::int64_t>(tensor::element_size(tensor.dtype()));
    return {tensor.storage_identity(), begin, begin + static_cast<std::int64_t>(tensor.nbytes())};
}

auto contains(const Region& outer, const Region& inner) -> bool {
    return outer.storage == inner.storage && outer.begin <= inner.begin && inner.end <= outer.end;
}

/// A contiguous view of `base` starting `offset` bytes in, with `like`'s shape.
auto view(const Tensor& base, std::int64_t offset, const Tensor& like) -> Tensor {
    const auto element = static_cast<std::int64_t>(tensor::element_size(base.dtype()));
    if (like.dtype() != base.dtype() || offset % element)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "captured step input view has an incompatible layout"});
    auto flat = require(base.reshape({static_cast<std::int64_t>(base.numel())}));
    auto window = require(flat.narrow(0, offset / element, static_cast<std::int64_t>(like.numel())));
    return require(window.reshape({like.shape().begin(), like.shape().end()}));
}

class InplaceScope {
public:
    InplaceScope() noexcept : previous_(std::exchange(ops::is_inplace, true)) {}
    ~InplaceScope() { ops::is_inplace = previous_; }
    InplaceScope(const InplaceScope&) = delete;
    auto operator=(const InplaceScope&) -> InplaceScope& = delete;

private:
    bool previous_;
};

auto same_source(const Source& first, const Source& second) -> bool {
    return first.kind == second.kind && first.index == second.index && first.output == second.output &&
           first.offset == second.offset && (first.kind != Source::Kind::EXTERNAL || first.storage == second.storage);
}

auto same_layout(const Tensor& first, const Tensor& second) -> bool {
    return first.dtype() == second.dtype() && first.device() == second.device() &&
           std::ranges::equal(first.shape(), second.shape());
}
} // namespace

auto Graph::bind(std::size_t slot, const Tensor& input) -> void {
    auto& current = inputs_[slot];
    if (input.storage_identity() == current.storage_identity() && input.storage_offset() == current.storage_offset() &&
        same_layout(input, current))
        return;
    if (!input.defined() || !input.is_contiguous() || !same_layout(input, current))
        throw ops::Failure(
            {ErrorCode::INVALID_ARGUMENT, "replayed step input " + std::to_string(slot) + " changed shape or dtype"});
    for (const auto& use : uses_[slot]) {
        auto& operand = use.node == OUTPUT_USE ? outputs_[use.operand] : nodes_[use.node].operands[use.operand];
        const auto& source =
            use.node == OUTPUT_USE ? output_sources_[use.operand] : nodes_[use.node].sources[use.operand];
        operand = view(input, source.offset, operand);
    }
    current = input;
}

auto Graph::replay(std::span<const Tensor> inputs) -> std::span<const Tensor> {
    if (inputs.size() != inputs_.size())
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "replayed step received a different number of inputs"});
    for (std::size_t slot = 0; slot < inputs.size(); ++slot) bind(slot, inputs[slot]);
    for (auto& node : nodes_) {
        const runtime::TensorInputs operands(node.inputs());
        auto outputs = std::span(node.operands).subspan(node.input_count);
        if (node.mode == Mode::IN_PLACE) {
            const InplaceScope mode;
            node.prepared->run_(operands, outputs.front());
        } else {
            node.prepared->run_into(operands, outputs);
        }
    }
    return outputs_;
}

Recorder::Recorder(std::span<const Tensor> inputs) {
    for (const auto& input : inputs) region(input);
    graph_.inputs_.assign(inputs.begin(), inputs.end());
    graph_.uses_.resize(inputs.size());
}

auto Recorder::locate(const Tensor& tensor) const -> Source {
    const auto target = region(tensor);
    for (std::size_t slot = 0; slot < graph_.inputs_.size(); ++slot) {
        const auto input = region(graph_.inputs_[slot]);
        if (contains(input, target)) return {Source::Kind::INPUT, slot, 0, nullptr, target.begin - input.begin};
    }
    for (std::size_t node = graph_.nodes_.size(); node-- > 0;) {
        const auto outputs = graph_.nodes_[node].outputs();
        for (std::size_t output = 0; output < outputs.size(); ++output) {
            const auto produced = region(outputs[output]);
            if (contains(produced, target))
                return {Source::Kind::NODE, node, output, nullptr, target.begin - produced.begin};
        }
    }
    return {Source::Kind::EXTERNAL, 0, 0, target.storage, target.begin};
}

auto Recorder::record(const runtime::OperatorSpec& spec, std::shared_ptr<runtime::Operator> prepared, Mode mode,
                      runtime::TensorInputs inputs, std::span<const Tensor> outputs) -> void {
    Node node{spec.operation,
              spec.dtype,
              spec.epsilon,
              spec.dynamic_parameters,
              spec.packed_prefill,
              spec.vector_projection,
              {spec.attributes.begin(), spec.attributes.end()}};
    for (std::size_t index = 0; index < spec.quantization.size(); ++index) {
        const auto& quantization = spec.quantization[index];
        node.scales[index].assign(quantization.scales.begin(), quantization.scales.end());
        node.zero_points[index].assign(quantization.zero_points.begin(), quantization.zero_points.end());
        node.block_sizes[index] = quantization.block_size;
    }
    node.prepared = std::move(prepared);
    node.mode = mode;
    node.input_count = inputs.size();
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        node.operands.push_back(inputs[index]);
        node.sources.push_back(locate(inputs[index]));
    }
    const auto index = graph_.nodes_.size();
    for (std::size_t output = 0; output < outputs.size(); ++output) {
        node.operands.push_back(outputs[output]);
        // In-place results alias their destination; fresh results are this node's own storage.
        node.sources.push_back(mode == Mode::IN_PLACE ? locate(outputs[output])
                                                      : Source{Source::Kind::NODE, index, output});
    }
    for (std::size_t operand = 0; operand < node.sources.size(); ++operand)
        if (node.sources[operand].kind == Source::Kind::INPUT)
            graph_.uses_[node.sources[operand].index].push_back({index, operand});
    graph_.nodes_.push_back(std::move(node));
}

auto Recorder::finish(std::span<const Tensor> outputs) && -> Graph {
    for (std::size_t index = 0; index < outputs.size(); ++index) {
        graph_.outputs_.push_back(outputs[index]);
        graph_.output_sources_.push_back(locate(outputs[index]));
        if (graph_.output_sources_.back().kind == Source::Kind::INPUT)
            graph_.uses_[graph_.output_sources_.back().index].push_back({OUTPUT_USE, index});
    }
    return std::move(graph_);
}

auto require_same_structure(const Graph& first, const Graph& second, std::string_view key) -> void {
    const auto fail = [&](std::size_t node, const std::string& reason) {
        std::string location = node < first.nodes_.size()
                                   ? "node " + std::to_string(node) + " (" +
                                         std::string(runtime::operation_name(first.nodes_[node].operation)) + ")"
                                   : "the step outputs";
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT,
                            "captured step '" + std::string(key) + "' changed between calls at " + location + ": " +
                                reason +
                                ". Values that change per call must be written into step inputs, not computed on "
                                "the host or baked into operator arguments"});
    };
    if (first.inputs_.size() != second.inputs_.size()) fail(0, "different input count");
    for (std::size_t slot = 0; slot < first.inputs_.size(); ++slot)
        if (!same_layout(first.inputs_[slot], second.inputs_[slot]))
            fail(0, "input " + std::to_string(slot) + " changed shape or dtype");
    if (first.nodes_.size() != second.nodes_.size())
        fail(std::min(first.nodes_.size(), second.nodes_.size()), "operator count changed from " +
                                                                      std::to_string(first.nodes_.size()) + " to " +
                                                                      std::to_string(second.nodes_.size()));
    for (std::size_t index = 0; index < first.nodes_.size(); ++index) {
        const auto& a = first.nodes_[index];
        const auto& b = second.nodes_[index];
        if (a.operation != b.operation) fail(index, "operator changed");
        if (a.prepared != b.prepared || a.mode != b.mode || a.dtype != b.dtype ||
            std::bit_cast<std::uint32_t>(a.epsilon) != std::bit_cast<std::uint32_t>(b.epsilon) ||
            a.attributes != b.attributes || a.scales != b.scales || a.zero_points != b.zero_points ||
            a.block_sizes != b.block_sizes || a.dynamic_parameters != b.dynamic_parameters)
            fail(index, "operator arguments or prepared shapes changed");
        if (a.input_count != b.input_count || a.operands.size() != b.operands.size())
            fail(index, "operand count changed");
        for (std::size_t operand = 0; operand < a.operands.size(); ++operand) {
            if (!same_layout(a.operands[operand], b.operands[operand]))
                fail(index, "operand " + std::to_string(operand) + " changed shape or dtype");
            if (!same_source(a.sources[operand], b.sources[operand]))
                fail(index,
                     "operand " + std::to_string(operand) +
                         (a.sources[operand].kind == Source::Kind::EXTERNAL ? " reads a tensor whose storage changed"
                                                                            : " reads from a different producer"));
        }
    }
    if (first.outputs_.size() != second.outputs_.size()) fail(first.nodes_.size(), "output count changed");
    for (std::size_t index = 0; index < first.outputs_.size(); ++index)
        if (!same_layout(first.outputs_[index], second.outputs_[index]) ||
            !same_source(first.output_sources_[index], second.output_sources_[index]))
            fail(first.nodes_.size(), "output " + std::to_string(index) + " changed");
}

} // namespace kidi::graph
