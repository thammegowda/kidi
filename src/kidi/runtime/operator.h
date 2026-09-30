#pragma once
#include <algorithm>
#include <array>
#include <memory>
#include <string_view>
#include <vector>
#include <initializer_list>
#include "kidi/tensor/tensor.h"
#include "kidi/tensor/arena.h"

namespace kidi::graph {
class Graph;
}

namespace kidi::runtime {
/// Borrowed operands, either a contiguous tensor span or a list of tensor addresses.
class TensorInputs {
public:
    TensorInputs(std::span<const tensor::Tensor> inputs) : contiguous_(inputs.data()), size_(inputs.size()) {}
    TensorInputs(std::initializer_list<const tensor::Tensor*> inputs)
        : indirect_(inputs.begin()), size_(inputs.size()) {}
    template <std::size_t Size>
    TensorInputs(const std::array<tensor::Tensor, Size>& inputs)
        : TensorInputs(std::span<const tensor::Tensor>(inputs)) {}
    TensorInputs(const std::vector<tensor::Tensor>& inputs) : TensorInputs(std::span<const tensor::Tensor>(inputs)) {}
    auto operator[](std::size_t index) const -> const tensor::Tensor& {
        return indirect_ ? *indirect_[index] : contiguous_[index];
    }
    auto front() const -> const tensor::Tensor& { return (*this)[0]; }
    auto back() const -> const tensor::Tensor& { return (*this)[size_ - 1]; }
    auto size() const noexcept -> std::size_t { return size_; }
    auto first(std::size_t count) const -> TensorInputs {
        auto result = *this;
        result.size_ = count;
        return result;
    }

private:
    const tensor::Tensor* contiguous_ = nullptr;
    const tensor::Tensor* const* indirect_ = nullptr;
    std::size_t size_;
};

enum class Operation {
    ADD,
    MULTIPLY,
    CAST,
    MATMUL,
    LINEAR,
    QUANTIZED_LINEAR,
    GELU,
    LAYER_NORM,
    SOFTMAX,
    LOG_SOFTMAX,
    TRANSPOSE,
    SLICE,
    GATHER,
    CONCAT,
    SCATTER,
    ATTENTION,
    RESIDUAL_NORM,
    RMS_NORM,
    TANH,
    ROTARY,
    PACKED_LINEAR,
    RMS_NORM_RESIDUAL,
    STATIC_ROUND,
    GREEDY_TOKEN,
    RMS_ROTARY,
    GELU_MULTIPLY,
    EMBEDDING,
    GATED_FEED_FORWARD
};
struct BlockwiseQuantizationSpec {
    std::span<const float> scales;
    std::span<const std::int32_t> zero_points;
    std::size_t block_size = 0;
};
struct OperatorSpec {
    Operation operation;
    std::span<const std::int64_t> attributes;
    tensor::DType dtype = tensor::DType::F32;
    float epsilon = 0;
    bool dynamic_parameters = false;
    bool packed_prefill = false;
    bool vector_projection = false;
    std::array<BlockwiseQuantizationSpec, 2> quantization;
};

struct AllocationStats {
    std::uint64_t count = 0;
    std::uint64_t bytes = 0;
};

class Operator {
public:
    virtual ~Operator() = default;
    virtual auto run(TensorInputs inputs) -> tensor::Tensor = 0;
    virtual auto run_(TensorInputs inputs, tensor::Tensor& destination) -> tensor::Tensor = 0;
    virtual auto run_pair(TensorInputs inputs) -> std::array<tensor::Tensor, 2> {
        return {tensor::Tensor{}, run(inputs)};
    }
    /// Writes into caller-owned outputs with the prepared shapes; `run_pair` order for paired results.
    /// Used to replay captured steps without allocating.
    virtual auto run_into(TensorInputs inputs, std::span<tensor::Tensor> outputs) -> void;
    virtual auto allocations() const -> AllocationStats { return {}; }
};
class OperatorBackend {
public:
    virtual ~OperatorBackend() = default;
    virtual auto prepare(const OperatorSpec&, TensorInputs) -> std::unique_ptr<Operator> = 0;
    virtual auto synchronize() -> void = 0;
    virtual auto release_cached_buffers() -> void {}
    /// True when every prepared operator implements `run_into`, so captured steps can replay.
    virtual auto supports_replay() const noexcept -> bool { return false; }
    virtual auto copy_slice_(tensor::Tensor& destination, const tensor::Tensor& source, std::size_t outer,
                             std::size_t source_bytes, std::size_t destination_bytes,
                             std::size_t offset_bytes) -> void = 0;
};
class OutputPool {
public:
    explicit OutputPool(tensor::Arena* arena = nullptr) : arena_(arena) { buffers_.reserve(32); }
    auto acquire(std::span<const std::int64_t> shape, tensor::DType dtype, tensor::Device device) -> tensor::Tensor;
    auto allocations() const -> AllocationStats { return allocations_; }

private:
    tensor::Arena* arena_;
    std::vector<tensor::Tensor> buffers_;
    AllocationStats allocations_;
};
auto cpu_operators() -> std::unique_ptr<OperatorBackend>;
auto metal_operators() -> std::unique_ptr<OperatorBackend>;
auto web_gpu_operators() -> std::unique_ptr<OperatorBackend>;
auto vulkan_operators() -> std::unique_ptr<OperatorBackend>;
auto operation_name(Operation operation) -> std::string_view;

/// A captured step compiled for an accelerator. Inputs and outputs have the captured layouts; outputs are the
/// captured step's own buffers.
class StepExecutable {
public:
    virtual ~StepExecutable() = default;
    virtual auto run(std::span<const tensor::Tensor> inputs, std::span<tensor::Tensor> outputs) -> void = 0;
};

/// Compiles captured steps of a CPU context for an accelerator such as an NPU. Steps it declines keep CPU replay.
class StepCompiler {
public:
    virtual ~StepCompiler() = default;
    virtual auto name() const -> std::string_view = 0;
    /// Returns nullptr to decline `graph`; throws ops::Failure when a supported step fails to compile.
    virtual auto compile(const graph::Graph& graph, std::string_view key) -> std::unique_ptr<StepExecutable> = 0;
    /// Key extent for a captured attention step: accelerators prefer few, coarse fixed shapes.
    virtual auto key_extent(std::size_t required, std::size_t capacity) const -> std::size_t {
        return std::min(capacity, (required + 127) / 128 * 128);
    }
    /// Whether captured steps may crop local-attention keys, which adds a shape per 128 positions.
    virtual auto crop_local_attention() const -> bool { return true; }
    /// Whether compile() may run on a background thread while CPU replay keeps serving the step. It must then read
    /// only the graph's structure and constant tensors, never buffers that replay writes, and be thread-safe with
    /// executables it produced earlier. A background failure leaves the step on CPU replay.
    virtual auto compiles_in_background() const -> bool { return false; }
    /// Per-step override: cached executables may load synchronously while cold compilation remains in the background.
    virtual auto compiles_in_background(const graph::Graph&, std::string_view) const -> bool {
        return compiles_in_background();
    }
    /// Whether models should capture prefill chunks as steps for compile(). Accelerators that offload prefill
    /// operators through prepare_operator() decline, so prefill runs eagerly without retaining step buffers.
    virtual auto captures_prefill() const -> bool { return true; }
    /// Preferred prompt chunk for this accelerator. Zero remains zero so caller validation still rejects it.
    virtual auto prefill_chunk_size(std::size_t requested) const -> std::size_t { return requested; }
    /// Offers one eager CPU operator (for example a many-row projection) to the accelerator. Returns nullptr to keep
    /// it on the CPU. The context caches the result like any prepared operator, and captured steps record it.
    virtual auto prepare_operator(const OperatorSpec&, TensorInputs) -> std::unique_ptr<Operator> { return nullptr; }
};

/// The Qualcomm NPU step compiler, or an error naming why it is unavailable (not built, libraries, or device).
auto npu_step_compiler() -> Result<std::shared_ptr<StepCompiler>>;
} // namespace kidi::runtime