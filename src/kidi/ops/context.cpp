#include "kidi/ops/context.h"
#include "kidi/graph/graph.h"
#include "kidi/graph/compiler.h"
#include "kidi/graph/session.h"
#include "kidi/runtime/operator.h"

#include "kidi/ops/quantization.h"
#include "kidi/ops/cpu_kernels.h"
#include <array>
#include <optional>
#include <bit>
#include <chrono>
#include <compare>
#include <map>
#include <list>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <atomic>
#include <limits>
#include <mutex>
#include <sstream>
#include <utility>
#include <functional>
#include <numeric>

namespace kidi::ops {
thread_local bool is_inplace = false;

using runtime::Operation;
using runtime::OperatorSpec;
using runtime::TensorInputs;
namespace {
thread_local bool decode_projections = false;
thread_local bool decode_attention = false;
thread_local std::shared_ptr<graph::Compiler> default_compiler;
thread_local bool eager_steps = false;

class InplaceScope {
public:
    explicit InplaceScope(bool enabled) noexcept : previous_(std::exchange(is_inplace, enabled)) {}
    ~InplaceScope() { is_inplace = previous_; }
    InplaceScope(const InplaceScope&) = delete;
    auto operator=(const InplaceScope&) -> InplaceScope& = delete;

private:
    bool previous_;
};

using Clock = std::chrono::steady_clock;
auto nanoseconds(Clock::time_point start) -> std::uint64_t {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}
using runtime::operation_name;
struct OperatorProfile {
    std::uint64_t calls = 0, prepare_ns = 0, dispatch_ns = 0, run_ns = 0, wait_ns = 0;
    std::uint64_t allocations = 0, allocated_bytes = 0, output_bytes = 0;
};

auto block_scaled_attributes(const BlockScaledWeight& weight, bool gated = false, float gate_cap = 0.F,
                             float up_cap = 0.F) -> std::array<std::int64_t, 7> {
    auto group = weight.group_size;
    if (!group) {
        switch (weight.format) {
            case BlockScaledFormat::NVFP4:
                group = 16;
                break;
            case BlockScaledFormat::MXFP8:
                group = 32;
                break;
            case BlockScaledFormat::INT4:
                group = 128;
                break;
            case BlockScaledFormat::VQ3D:
                group = 48;
                break;
            case BlockScaledFormat::VQ4D:
                group = 64;
                break;
            default:
                throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid block-scaled weight encoding"});
        }
    }
    const auto shift = group > 0 && std::has_single_bit(static_cast<std::uint32_t>(group))
                           ? std::countr_zero(static_cast<std::uint32_t>(group))
                           : -1;
    return {gated,
            std::bit_cast<std::int32_t>(gate_cap),
            std::bit_cast<std::int32_t>(up_cap),
            static_cast<std::int64_t>(weight.format),
            group,
            weight.global.defined(),
            shift};
}

// F32 values of a host tensor: borrowed when already contiguous F32, otherwise converted once.
class HostFloats {
public:
    HostFloats() = default;
    HostFloats(const HostFloats&) = delete;
    HostFloats(HostFloats&&) = default;
    auto operator=(const HostFloats&) -> HostFloats& = delete;
    auto operator=(HostFloats&&) -> HostFloats& = default;
    HostFloats(Context& context, const Tensor& input) {
        if (input.is_contiguous() && input.is_host_accessible() && input.dtype() == tensor::DType::F32) {
            view_ = require(input.data<float>());
            return;
        }
        if (input.is_contiguous() && input.is_host_accessible() && input.dtype() == tensor::DType::BF16) {
            const auto bytes = require(input.host_bytes());
            owned_.resize(input.numel());
            cpu::bf16_to_f32(reinterpret_cast<const std::uint16_t*>(bytes.data()),
                             static_cast<std::int64_t>(owned_.size()), owned_.data());
            view_ = owned_;
            return;
        }
        held_ = context.cast(input, tensor::DType::F32);
        view_ = require(std::as_const(held_).data<float>());
    }
    auto operator[](std::size_t index) const -> float { return view_[index]; }
    auto data() const -> const float* { return view_.data(); }
    auto size() const -> std::size_t { return view_.size(); }
    auto empty() const -> bool { return view_.empty(); }

private:
    std::vector<float> owned_;
    Tensor held_;
    std::span<const float> view_;
};

auto cpu_floats(Context& context, const Tensor& input) -> HostFloats { return {context, input}; }

auto cpu_tensor(Context& context, std::vector<std::int64_t> shape, std::span<const float> values, tensor::DType dtype)
    -> Tensor {
    if (dtype != tensor::DType::BF16) {
        auto output = require(Tensor::from_host(std::move(shape), values, context.device()));
        return dtype == tensor::DType::F32 ? output : context.cast(output, dtype);
    }
    auto output = require(Tensor::empty(std::move(shape), dtype, context.device()));
    const auto bytes = require(output.host_bytes());
    cpu::f32_to_bf16(values.data(), static_cast<std::int64_t>(values.size()),
                     reinterpret_cast<std::uint16_t*>(bytes.data()));
    return output;
}

// Runs `fill` on an F32 buffer of `shape` and returns it as `dtype`, writing F32 outputs in place.
template <typename Fill>
auto cpu_output(Context& context, std::vector<std::int64_t> shape, tensor::DType dtype, Fill&& fill) -> Tensor {
    if (dtype == tensor::DType::F32) {
        auto output = require(Tensor::empty(std::move(shape), dtype, context.device()));
        require(fill(require(output.data<float>()).data()));
        return output;
    }
    const auto count = std::accumulate(shape.begin(), shape.end(), std::int64_t{1}, std::multiplies<>{});
    std::vector<float> values(static_cast<std::size_t>(count));
    require(fill(values.data()));
    return cpu_tensor(context, std::move(shape), values, dtype);
}

auto validate_attention_quantization(const Tensor& key, const tensor::Device& device,
                                     const BlockwiseQuantization& key_quantization,
                                     const BlockwiseQuantization& value_quantization) -> void {
    const auto quantized = key.dtype() == tensor::DType::I8;
    if (quantized && !device.capabilities().attention.int8_cache_tokens)
        throw Failure({ErrorCode::UNSUPPORTED, "INT8 attention caches are unsupported by this backend"});
    const auto valid = [](const BlockwiseQuantization& metadata) {
        return metadata.block_size > 0 && !metadata.scales.empty() &&
               metadata.scales.size() == metadata.zero_points.size() &&
               std::ranges::all_of(metadata.scales, [](float value) { return std::isfinite(value) && value > 0; });
    };
    if (quantized != (valid(key_quantization) && valid(value_quantization)) ||
        (!quantized && (!key_quantization.scales.empty() || !value_quantization.scales.empty())))
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid attention cache quantization metadata"});
}
} // namespace
DecodeScope::DecodeScope(bool enabled) : DecodeScope(enabled, enabled) {}
DecodeScope::DecodeScope(bool attention, bool projections)
    : previous_attention_(std::exchange(decode_attention, attention)),
      previous_projections_(std::exchange(decode_projections, projections)) {}
DecodeScope::~DecodeScope() {
    decode_attention = previous_attention_;
    decode_projections = previous_projections_;
}
StepCompilerScope::StepCompilerScope(std::shared_ptr<graph::Compiler> compiler)
    : previous_(std::exchange(default_compiler, std::move(compiler))) {}
StepCompilerScope::~StepCompilerScope() { default_compiler = std::move(previous_); }
EagerScope::EagerScope() : previous_(std::exchange(eager_steps, true)) {}
EagerScope::~EagerScope() { eager_steps = previous_; }
struct Context::Impl {
    tensor::Device device;
    tensor::DType activation_dtype = tensor::DType::F32;
    bool packed_prefill = false;
    std::unique_ptr<runtime::OperatorBackend> backend;
    using DispatchKey = std::vector<std::int64_t>;
    std::list<const DispatchKey*> recent;
    struct Prepared {
        std::shared_ptr<runtime::Operator> operation;
        std::vector<Tensor> parameters;
        std::list<const DispatchKey*>::iterator recency;
    };
    std::map<DispatchKey, Prepared> operators;
    std::vector<std::int64_t> dispatch_key;
    std::size_t operator_capacity = 4096;
    struct PackedBinding {
        Tensor source;
        PackedWeight packed;
        std::int32_t bits = 0;
        std::int32_t group_size = 0;
        bool packed_prefill = false;
        bool released = false;
    };
    struct DequantizedBinding {
        Tensor weight, scales, value;
    };
    struct DequantizedKey {
        const std::byte* weight;
        const std::byte* scales;
        tensor::DType dtype;
        core::InferencePrecision precision;
        std::int32_t bits, group_size;
        std::size_t rows, packed_columns, scale_count;
        auto operator<=>(const DequantizedKey&) const = default;
    };
    std::map<const void*, PackedBinding> packed_weights;
    std::map<const void*, Tensor> packed_zeros;
    std::map<DequantizedKey, DequantizedBinding> dequantized_weights;
    tensor::DType parameter_dtype = tensor::DType::F32;
    core::InferencePrecision precision = core::InferencePrecision::CHECKPOINT;
    std::uint64_t preparation = 0;
    bool profiling = false;
    bool profile_requested = false;
    bool synchronize_operators = false;
    std::size_t skip_requests = 0, requests = 0;
    std::string phase = "unspecified";
    std::map<std::string, OperatorProfile> profiles;
    std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> synchronization;
    std::shared_ptr<graph::Compiler> compiler;
    graph::Session replays;
    ~Impl() {
        try {
            if (backend) backend->synchronize();
            for (const auto& [key, profile] : profiles)
                std::cerr << "kidi_op|device=" << tensor::to_string(device)
                          << "|mode=" << (synchronize_operators ? "sync" : "host") << '|' << key
                          << "|calls=" << profile.calls << "|prepare_ns=" << profile.prepare_ns
                          << "|dispatch_ns=" << profile.dispatch_ns << "|run_ns=" << profile.run_ns
                          << "|wait_ns=" << profile.wait_ns << "|allocations=" << profile.allocations
                          << "|allocated_bytes=" << profile.allocated_bytes << "|output_bytes=" << profile.output_bytes
                          << '\n';
            for (const auto& [label, counters] : synchronization)
                std::cerr << "kidi_op_sync|device=" << tensor::to_string(device) << "|phase=" << label
                          << "|calls=" << counters.first << "|wait_ns=" << counters.second << '\n';
        } catch (...) {
        }
    }
    auto run(OperatorSpec spec, TensorInputs inputs, Tensor* residual = nullptr, Tensor* destination = nullptr)
        -> Tensor {
        const InplaceScope mode(destination != nullptr);
        const auto& capabilities = device.capabilities();
        const auto& dispatch = device.policy().dispatch;
        spec.packed_prefill = packed_prefill;
        spec.vector_projection =
            (decode_projections && spec.operation == Operation::PACKED_LINEAR &&
             dispatch.projection == tensor::ProjectionDispatch::VECTOR_DECODE && inputs[0].dimensions() > 0 &&
             inputs[0].size(-1) && inputs[0].numel() / inputs[0].size(-1) >= 4) ||
            (decode_attention && spec.operation == Operation::ATTENTION && device.capabilities().compute.host());
        if (is_inplace && destination->is_host_accessible()) require(destination->host_bytes());
        const auto entered = profiling ? Clock::now() : Clock::time_point{};
        const auto preparation_before = preparation;
        auto& key = dispatch_key;
        key.clear();
        key.insert(key.end(), {static_cast<int>(spec.operation), static_cast<int>(spec.dtype),
                               std::bit_cast<std::int32_t>(spec.epsilon), is_inplace});
        key.push_back(spec.vector_projection);
        key.push_back(spec.attributes.size());
        key.insert(key.end(), spec.attributes.begin(), spec.attributes.end());
        for (const auto& quantization : spec.quantization) {
            key.push_back(quantization.block_size);
            key.push_back(quantization.scales.size());
            for (const auto scale : quantization.scales) key.push_back(std::bit_cast<std::int32_t>(scale));
            key.push_back(quantization.zero_points.size());
            key.insert(key.end(), quantization.zero_points.begin(), quantization.zero_points.end());
        }
        const bool constant_parameters =
            spec.operation == Operation::LINEAR || spec.operation == Operation::QUANTIZED_LINEAR ||
            spec.operation == Operation::LAYER_NORM || spec.operation == Operation::RESIDUAL_NORM ||
            spec.operation == Operation::RMS_NORM || spec.operation == Operation::PACKED_LINEAR ||
            spec.operation == Operation::EMBEDDING || spec.operation == Operation::EMBEDDING_SUM ||
            spec.operation == Operation::GATED_FEED_FORWARD || spec.operation == Operation::CONV1D ||
            spec.operation == Operation::SNAKE_CONV1D || spec.operation == Operation::SNAKE_QUANTIZED_LINEAR ||
            spec.operation == Operation::CONV_TRANSPOSE_FOLD || spec.operation == Operation::SNAKE;
        const std::size_t parameter_start = spec.operation == Operation::RESIDUAL_NORM ? 2 : 1;
        // Packed projections bake an interleaved copy of their weights, so each weight needs its own program.
        spec.dynamic_parameters =
            spec.operation == Operation::RMS_NORM ||
            (spec.operation == Operation::LINEAR && dispatch.parameters == tensor::ParameterBinding::DYNAMIC_LINEAR) ||
            (constant_parameters && spec.operation != Operation::PACKED_LINEAR &&
             spec.operation != Operation::GATED_FEED_FORWARD &&
             dispatch.parameters == tensor::ParameterBinding::DYNAMIC);
        for (std::size_t index = parameter_start; index < inputs.size(); ++index)
            spec.dynamic_parameters = spec.dynamic_parameters && inputs[index].device() == device;
        key.push_back(spec.dynamic_parameters);
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            const auto& input = inputs[index];
            if (!input.defined() || !input.is_contiguous())
                throw Failure({ErrorCode::INVALID_ARGUMENT, "eager operations require defined contiguous tensors"});
            if ((!constant_parameters || index < parameter_start) && input.device() != device)
                throw Failure({ErrorCode::INVALID_ARGUMENT, "eager operand device mismatch for " +
                                                                std::string(runtime::operation_name(spec.operation))});
            key.push_back(static_cast<int>(input.dtype()));
            key.push_back(input.dimensions());
            key.insert(key.end(), input.shape().begin(), input.shape().end());
            if (constant_parameters && !spec.dynamic_parameters && index >= parameter_start)
                key.insert(key.end(),
                           {reinterpret_cast<std::intptr_t>(input.storage_identity()), input.storage_offset()});
        }
        const auto invalid = [](bool condition, const char* message) {
            if (condition) throw Failure({ErrorCode::INVALID_ARGUMENT, message});
        };
        const auto& input = inputs.front();
        const auto floating = [&](tensor::DType dtype) {
            return dtype == tensor::DType::F32 ||
                   (activation_dtype == tensor::DType::BF16 && dtype == tensor::DType::BF16);
        };
        if (spec.operation == Operation::GREEDY_TOKEN)
            invalid(!floating(input.dtype()) || input.dimensions() == 0 || !input.numel() ||
                        input.size(-1) > INT32_MAX || input.numel() > UINT32_MAX,
                    "greedy selection expects nonempty FP32 rows within indexing bounds");
        if (spec.operation == Operation::BLOCK_SCALED_LINEAR) {
            invalid(
                spec.attributes.size() != 7 || inputs.size() < 3 || spec.attributes[3] < 0 || spec.attributes[3] > 2,
                "invalid block-scaled projection attributes");
            const bool fp8 = spec.attributes[3] == static_cast<int>(BlockScaledFormat::MXFP8);
            const auto gated = spec.attributes[0];
            const auto has_global = spec.attributes[5];
            invalid(
                gated < 0 || gated > 1 || has_global < 0 || has_global > 1 || inputs.size() != 3U + has_global + gated,
                "invalid block-scaled projection operands");
            invalid(input.dimensions() < 2 || !input.numel() ||
                        (input.dtype() != tensor::DType::F32 && input.dtype() != tensor::DType::BF16) ||
                        inputs[1].dimensions() != 2 || !inputs[1].numel() ||
                        inputs[1].dtype() != (fp8 ? tensor::DType::E4M3 : tensor::DType::U8),
                    "invalid block-scaled projection values");
            const auto width = input.size(-1);
            const auto block = spec.attributes[4];
            const auto scale_type = inputs[2].dtype();
            const bool byte_scale =
                spec.attributes[3] != static_cast<int>(BlockScaledFormat::INT4) && scale_type == tensor::DType::U8;
            invalid(block <= 0 || block > INT32_MAX || width % block || (!fp8 && block % 2) ||
                        inputs[1].size(1) != width / (fp8 ? 1 : 2) || inputs[2].dimensions() != 2 ||
                        inputs[2].size(0) != inputs[1].size(0) || inputs[2].size(1) != width / block ||
                        (!byte_scale && scale_type != tensor::DType::E4M3 && scale_type != tensor::DType::F32 &&
                         scale_type != tensor::DType::F16 && scale_type != tensor::DType::BF16) ||
                        (gated && inputs[1].size(0) % 2),
                    "invalid block-scaled projection geometry or scales");
            if (has_global)
                invalid(inputs[3].dtype() != tensor::DType::F32 || inputs[3].numel() != 1,
                        "block-scaled global scale must be a scalar FP32 tensor");
            if (gated) {
                const auto gate_cap = std::bit_cast<float>(static_cast<std::int32_t>(spec.attributes[1]));
                const auto up_cap = std::bit_cast<float>(static_cast<std::int32_t>(spec.attributes[2]));
                invalid(!std::isfinite(gate_cap) || gate_cap <= 0 || !std::isfinite(up_cap) || up_cap <= 0 ||
                            inputs.back().dtype() != tensor::DType::F32 ||
                            inputs.back().numel() != input.numel() / width,
                        "invalid block-scaled gate caps or routing weights");
            }
        }
        if (spec.operation == Operation::PACKED_LINEAR) {
            const auto bits = spec.attributes[0], group = spec.attributes[1];
            invalid((bits != 2 && bits != 4 && bits != 8) || group <= 0, "invalid packed linear precision or group");
            // A GELU-gate prologue halves the input width; an RMS-norm prologue keeps it.
            const auto prologue = spec.attributes.size() >= 4 ? spec.attributes[3] : 0;
            const bool gate = prologue == 2;
            invalid(prologue < 0 || prologue > 2 || (gate && input.dimensions() && input.size(-1) % 2) ||
                        (prologue == 1 && inputs.size() != 4) || (gate && inputs.size() != 3),
                    "invalid packed linear prologue");
            const auto width = input.dimensions() ? input.size(-1) / (gate ? 2 : 1) : 0;
            invalid(!floating(input.dtype()) || input.dimensions() < 2 || !width || width % group ||
                        group % (8 / bits) || inputs[1].dtype() != tensor::DType::U8 || inputs[1].dimensions() != 2 ||
                        !inputs[1].size(0) || inputs[1].size(1) != width / (8 / bits) ||
                        (inputs[2].dtype() != tensor::DType::F32 && inputs[2].dtype() != tensor::DType::F16) ||
                        inputs[2].dimensions() != 2 || inputs[2].size(0) != inputs[1].size(0) ||
                        inputs[2].size(1) != width / group,
                    "packed linear shape or dtype mismatch");
        }
        if (spec.operation == Operation::GATED_FEED_FORWARD) {
            invalid((spec.attributes.size() != 6 && spec.attributes.size() != 7) ||
                        (spec.attributes.size() == 7 && spec.attributes[6] != 1),
                    "invalid gated feed-forward attributes");
            const auto bits = spec.attributes[0], input_size = spec.attributes[1], intermediate = spec.attributes[2];
            invalid((bits != 2 && bits != 4 && bits != 8) || input_size <= 0 || intermediate <= 0 ||
                        input.dtype() != tensor::DType::F32 || input.dimensions() != 3 || input.size(-1) != input_size,
                    "invalid gated feed-forward input or attributes");
            invalid(inputs.size() != 5 || inputs[1].dtype() != tensor::DType::U8 || inputs[1].dimensions() != 2 ||
                        inputs[1].size(0) != 2 * intermediate || inputs[1].size(1) != input_size / (8 / bits) ||
                        inputs[2].dtype() != tensor::DType::F32 || inputs[2].dimensions() != 2 ||
                        inputs[2].size(0) != 2 * intermediate || inputs[2].size(1) != 1 ||
                        inputs[3].dtype() != tensor::DType::U8 || inputs[3].dimensions() != 2 ||
                        inputs[3].size(0) != input_size || inputs[3].size(1) != intermediate / (8 / bits) ||
                        inputs[4].dtype() != tensor::DType::F32 || inputs[4].dimensions() != 2 ||
                        inputs[4].size(0) != input_size || inputs[4].size(1) != 1,
                    "gated feed-forward parameter shape or dtype mismatch");
            invalid(input_size % (8 / bits) || intermediate % (8 / bits) || !std::isfinite(spec.epsilon) ||
                        spec.epsilon <= 0,
                    "invalid gated feed-forward precision or calibration");
            for (std::size_t index = 3; index < spec.attributes.size(); ++index) {
                const auto scale = std::bit_cast<float>(static_cast<std::int32_t>(spec.attributes[index]));
                invalid(!std::isfinite(scale) || scale <= 0, "invalid gated feed-forward calibration");
            }
        }
        if (spec.operation == Operation::ROTARY) {
            invalid(input.dimensions() != 4 || !floating(input.dtype()) || input.size(3) % 2,
                    "rotary expects FP32 [batch, sequence, heads, even width]");
            for (std::size_t index = 1; index < 3; ++index)
                invalid(inputs[index].dimensions() != 4 || inputs[index].dtype() != tensor::DType::F32 ||
                            inputs[index].size(0) != 1 || inputs[index].size(1) != input.size(1) ||
                            inputs[index].size(2) != 1 || inputs[index].size(3) != input.size(3) / 2,
                        "rotary frequency shape or dtype mismatch");
        }
        if (spec.operation == Operation::ATTENTION) {
            const auto& key = inputs[1];
            const auto& value = inputs[2];
            const auto heads = spec.attributes[0];
            const auto key_heads = spec.attributes.size() >= 2 ? spec.attributes[1] : heads;
            invalid(heads <= 0 || input.dimensions() != 3 || key.dimensions() != 3 || value.dimensions() != 3,
                    "attention expects rank-three tensors and positive head count");
            invalid(key_heads <= 0 || heads % key_heads, "invalid key/value head count");
            invalid(input.size(2) % heads != 0 || key.size(2) % key_heads ||
                        input.size(2) / heads != key.size(2) / key_heads ||
                        !std::ranges::equal(key.shape(), value.shape()) ||
                        (key.size(0) != 1 && key.size(0) != input.size(0)),
                    "attention shape mismatch");
            const auto causal_positions = spec.attributes.size() >= 7 && spec.attributes[5] == 1;
            for (std::size_t index = 0; index < inputs.size(); ++index) {
                const auto dtype = inputs[index].dtype();
                const auto cache_operand = index == 1 || index == 2;
                const auto position_operand = index == 3 && dtype == tensor::DType::I32 && causal_positions &&
                                              device.kind == tensor::DeviceKind::CUDA;
                invalid(!floating(dtype) && !position_operand &&
                            !(cache_operand && (dtype == tensor::DType::I8 ||
                                                capabilities.attention.accepts_cache(input.dtype(), dtype))),
                        "attention expects floating tensors with a backend-supported key/value cache dtype");
            }
            invalid(key.dtype() != value.dtype(), "attention key and value must share a dtype");
            if (key.dtype() == tensor::DType::I8) {
                invalid(!capabilities.attention.int8_cache_tokens, "INT8 attention requires a supported backend");
                const auto head_width = key.size(2) / key_heads;
                for (const auto& quantization : spec.quantization)
                    invalid(quantization.block_size == 0 || quantization.scales.empty() ||
                                quantization.scales.size() != quantization.zero_points.size() ||
                                quantization.scales.size() <
                                    (head_width + quantization.block_size - 1) / quantization.block_size,
                            "INT8 attention requires complete blockwise quantization metadata");
            }
            if (inputs.size() == 4) {
                const auto key_length =
                    spec.attributes.size() >= 4 ? static_cast<std::size_t>(spec.attributes[3]) : key.size(1);
                const std::array<std::size_t, 4> scores{input.size(0), static_cast<std::size_t>(heads), input.size(1),
                                                        key_length};
                const auto& mask = inputs[3];
                invalid(mask.dimensions() > scores.size(), "attention mask rank mismatch");
                for (std::size_t axis = 0; axis < mask.dimensions(); ++axis) {
                    auto extent = mask.size(axis);
                    invalid(extent != 1 && extent != scores[scores.size() - mask.dimensions() + axis],
                            "attention mask shape mismatch");
                }
            }
        }
        if (spec.operation == Operation::MATMUL || spec.operation == Operation::LINEAR ||
            spec.operation == Operation::QUANTIZED_LINEAR) {
            const auto& weight = inputs[1];
            invalid(input.dimensions() < 2 || weight.dimensions() < 2, "matrix operations require rank two or higher");
            const bool transpose = !spec.attributes.empty() && spec.attributes[0];
            invalid(input.size(-1) != weight.size(transpose ? -1 : -2), "matrix contraction dimensions differ");
            if (constant_parameters) {
                invalid(weight.dimensions() != 2, "linear weights require rank two");
                if (inputs.size() > 2) {
                    const auto& bias = inputs.back();
                    invalid(bias.dimensions() != 1 || bias.size(0) != weight.size(transpose ? 0 : 1) ||
                                bias.dtype() != tensor::DType::F32,
                            "linear parameter shape or dtype mismatch");
                }
            }
        }
        if (spec.operation == Operation::QUANTIZED_LINEAR)
            invalid(input.dtype() != tensor::DType::F32 || inputs[1].dtype() != tensor::DType::I8 ||
                        inputs[2].dtype() != tensor::DType::F32 || inputs[2].dimensions() != 2 ||
                        inputs[2].size(0) != inputs[1].size(!spec.attributes.empty() && spec.attributes[0] ? 0 : 1) ||
                        inputs[2].size(1) != 1,
                    "quantized linear parameter mismatch");
        if (spec.operation == Operation::LAYER_NORM || spec.operation == Operation::RESIDUAL_NORM ||
            spec.operation == Operation::RMS_NORM) {
            invalid(!input.dimensions() || !std::isfinite(spec.epsilon) || spec.epsilon <= 0,
                    "invalid normalization request");
            if (spec.operation == Operation::RMS_NORM)
                invalid(!floating(input.dtype()), "RMS normalization expects floating input");
            if (spec.operation == Operation::RESIDUAL_NORM)
                invalid(input.dtype() != tensor::DType::F32 || inputs[1].dtype() != input.dtype() ||
                            !std::ranges::equal(input.shape(), inputs[1].shape()),
                        "residual normalization shape or dtype mismatch");
            for (std::size_t index = parameter_start; index < inputs.size(); ++index)
                invalid(inputs[index].dimensions() != 1 || inputs[index].size(0) != input.size(-1) ||
                            inputs[index].dtype() != tensor::DType::F32,
                        "normalization parameter mismatch");
        }
        if (spec.operation == Operation::ADD || spec.operation == Operation::MULTIPLY) {
            const auto& other = inputs[1];
            const auto mixed_scalar =
                activation_dtype == tensor::DType::BF16 &&
                ((input.dtype() == tensor::DType::BF16 && other.dtype() == tensor::DType::F32 && other.numel() == 1) ||
                 (other.dtype() == tensor::DType::BF16 && input.dtype() == tensor::DType::F32 && input.numel() == 1));
            invalid(input.dtype() != other.dtype() && !mixed_scalar, "arithmetic operand dtype mismatch");
            const auto common = std::min(input.dimensions(), other.dimensions());
            for (std::size_t index = 1; index <= common; ++index) {
                const auto left = input.size(-static_cast<std::int64_t>(index)),
                           right = other.size(-static_cast<std::int64_t>(index));
                invalid(left != right && left != 1 && right != 1, "incompatible broadcast dimensions");
            }
            spec.dtype = mixed_scalar ? tensor::DType::BF16 : input.dtype();
        }
        if (spec.operation == Operation::TRANSPOSE) {
            invalid(spec.attributes.size() != input.dimensions(), "transpose rank mismatch");
            for (std::size_t index = 0; index < spec.attributes.size(); ++index) {
                const auto axis = spec.attributes[index];
                invalid(axis < 0 || static_cast<std::size_t>(axis) >= input.dimensions() ||
                            std::find(spec.attributes.begin(), spec.attributes.begin() + index, axis) !=
                                spec.attributes.begin() + index,
                        "invalid transpose permutation");
            }
        }
        if (spec.operation == Operation::SCATTER)
            invalid(input.dimensions() != 3 || inputs[1].dimensions() != 3 || input.size(0) != inputs[1].size(0) ||
                        input.size(2) != inputs[1].size(2) ||
                        !capabilities.attention.accepts_cache(inputs[1].dtype(), input.dtype()) ||
                        inputs[2].dtype() != tensor::DType::I32 || inputs[2].dimensions() != 1 ||
                        inputs[2].size(0) != inputs[1].size(1),
                    "cache scatter shape or dtype mismatch");
        auto found = operators.find(key);
        if (found == operators.end()) {
            auto start = std::chrono::steady_clock::now();
            auto prepared = compiler ? compiler->prepare_operator(spec, inputs) : nullptr;
            if (!prepared) prepared = backend->prepare(spec, inputs);
            if (operators.size() >= operator_capacity) {
                backend->synchronize();
                for (std::size_t count = 0; count < std::max(std::size_t{1}, operator_capacity / 4); ++count) {
                    operators.erase(*recent.back());
                    recent.pop_back();
                }
                backend->release_cached_buffers();
            }
            found = operators.try_emplace(key).first;
            found->second.operation = std::move(prepared);
            recent.push_front(&found->first);
            found->second.recency = recent.begin();
            if (constant_parameters && !spec.dynamic_parameters)
                for (std::size_t index = parameter_start; index < inputs.size(); ++index)
                    found->second.parameters.push_back(inputs[index]);
            preparation +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        }
        recent.splice(recent.begin(), recent, found->second.recency);
        const auto execute = [&]() -> Tensor {
            if (is_inplace) return found->second.operation->run_(inputs, *destination);
            if (!residual) return found->second.operation->run(inputs);
            auto outputs = found->second.operation->run_pair(inputs);
            *residual = std::move(outputs[0]);
            return std::move(outputs[1]);
        };
        const auto captured = [&](Tensor output) -> Tensor {
            auto* recorder = replays.recorder();
            if (!recorder) return output;
            if (is_inplace)
                recorder->record(spec, found->second.operation, graph::Mode::IN_PLACE, inputs, {destination, 1});
            else if (residual)
                recorder->record(spec, found->second.operation, graph::Mode::RESULT, inputs,
                                 std::array<Tensor, 2>{*residual, output});
            else
                recorder->record(spec, found->second.operation, graph::Mode::RESULT, inputs, {&output, 1});
            return output;
        };
        if (!profiling) return captured(execute());
        const auto dispatch_ns = nanoseconds(entered) - (preparation - preparation_before);
        const auto allocations_before = found->second.operation->allocations();
        auto started = Clock::now();
        auto output = execute();
        const auto run_ns = nanoseconds(started);
        std::uint64_t wait_ns = 0;
        if (synchronize_operators) {
            started = Clock::now();
            backend->synchronize();
            wait_ns = nanoseconds(started);
        }
        const auto allocations_after = found->second.operation->allocations();
        std::ostringstream label;
        label << "phase=" << phase << "|op=" << operation_name(spec.operation) << (is_inplace ? "_" : "") << "|shapes=";
        for (std::size_t operand = 0; operand < inputs.size(); ++operand) {
            if (operand) label << ';';
            label << '[';
            for (auto extent : inputs[operand].shape()) label << extent << ',';
            label << ']';
        }
        label << "|attrs=";
        for (auto attribute : spec.attributes) label << attribute << ',';
        auto& profile = profiles[label.str()];
        ++profile.calls;
        profile.prepare_ns += preparation - preparation_before;
        profile.dispatch_ns += dispatch_ns;
        profile.run_ns += run_ns;
        profile.wait_ns += wait_ns;
        profile.allocations += allocations_after.count - allocations_before.count;
        profile.allocated_bytes += allocations_after.bytes - allocations_before.bytes;
        profile.output_bytes += output.nbytes() + (residual ? residual->nbytes() : 0);
        return captured(std::move(output));
    }
};
namespace {
std::mutex profile_mutex;
ProfileOptions configured_profile;
std::atomic<std::size_t> configured_metal_prefill_cache{std::numeric_limits<std::size_t>::max()};
} // namespace

auto set_profile_options(ProfileOptions options) -> void {
    const std::lock_guard lock(profile_mutex);
    configured_profile = options;
}
auto profile_options() -> ProfileOptions {
    const std::lock_guard lock(profile_mutex);
    return configured_profile;
}
auto parse_profile_mode(std::string_view name) -> std::optional<ProfileMode> {
    if (name == "off") return ProfileMode::OFF;
    if (name == "host") return ProfileMode::HOST;
    if (name == "sync") return ProfileMode::SYNC;
    if (name == "graph") return ProfileMode::GRAPH;
    if (name == "events") return ProfileMode::EVENTS;
    return std::nullopt;
}
auto set_metal_prefill_cache_bytes(std::size_t bytes) -> void { configured_metal_prefill_cache = bytes; }
auto metal_prefill_cache_bytes() -> std::size_t { return configured_metal_prefill_cache; }

Context::Context(tensor::Device device, bool packed_prefill, std::optional<bool> replay_enabled,
                 tensor::DType parameter_dtype)
    : impl_(std::make_unique<Impl>()) {
    impl_->device = device;
    impl_->packed_prefill = packed_prefill;
    impl_->dispatch_key.reserve(128);
    impl_->operator_capacity = device.policy().dispatch.cache_capacity;
    graph::SessionOptions options;
    options.capacity = device.policy().replay.cache_capacity;
    const auto profile = profile_options();
    impl_->profiling = profile.mode == ProfileMode::HOST || profile.mode == ProfileMode::SYNC;
    impl_->profile_requested = impl_->profiling;
    impl_->skip_requests = profile.skip_requests;
    if (impl_->skip_requests) impl_->profiling = false;
    impl_->synchronize_operators = profile.mode == ProfileMode::SYNC;
    if (device == tensor::Device::cpu()) impl_->backend = runtime::cpu_operators();
#if defined(KIDI_HAS_CUDA)
    else if (device.kind == tensor::DeviceKind::CUDA)
        impl_->backend = runtime::cuda_operators(device);
#endif
#if defined(KIDI_HAS_WEBGPU)
    else if (device == tensor::Device::web_gpu())
        impl_->backend = runtime::web_gpu_operators();
#endif
#if defined(KIDI_HAS_VULKAN)
    else if (device == tensor::Device::vulkan())
        impl_->backend = runtime::vulkan_operators();
#endif
#if defined(KIDI_HAS_METAL)
    else if (device == tensor::Device::apple_gpu())
        impl_->backend = runtime::metal_operators();
#endif
    else
        throw Failure({ErrorCode::UNSUPPORTED, "no eager backend for requested device"});
    impl_->parameter_dtype = parameter_dtype;
    impl_->activation_dtype = impl_->backend->activation_dtype(parameter_dtype);
    if (device.capabilities().compute.host()) impl_->compiler = default_compiler;
    const auto accelerator_replay = impl_->compiler && impl_->compiler->captures_prefill();
    options.enabled =
        !eager_steps && impl_->backend->supports_replay() && (replay_enabled.value_or(true) || accelerator_replay);
    impl_->replays = graph::Session(impl_->compiler ? impl_->compiler : impl_->backend->graph_compiler(), options);
}
Context::~Context() = default;
Context::Context(Context&&) noexcept = default;
auto Context::operator=(Context&&) noexcept -> Context& = default;
auto Context::device() const noexcept -> tensor::Device { return impl_->device; }
auto Context::activation_dtype() const noexcept -> tensor::DType { return impl_->activation_dtype; }
auto Context::stage_inputs(std::span<const Tensor> inputs) -> std::vector<Tensor> {
    return impl_->backend->stage_inputs(inputs, impl_->device);
}
auto Context::synchronize() -> void {
    if (!impl_->profiling) return impl_->backend->synchronize();
    const auto started = Clock::now();
    impl_->backend->synchronize();
    const auto duration = nanoseconds(started);
    auto& counters = impl_->synchronization[impl_->phase];
    ++counters.first;
    counters.second += duration;
}
auto Context::profile_phase(std::string_view phase) -> void {
    if (!impl_->profile_requested) return;
    if (phase == "encoder" || phase == "gemma4_request") impl_->profiling = ++impl_->requests > impl_->skip_requests;
    impl_->phase = phase;
}
auto Context::preparation_ns() const noexcept -> std::uint64_t { return impl_->preparation; }
auto Context::configure_cache(const std::filesystem::path& directory, const std::filesystem::path& source) -> void {
    impl_->backend->configure_cache(directory, source);
}
auto Context::set_precision(core::InferencePrecision precision) noexcept -> void {
    if (impl_->precision == precision) return;
    impl_->replays.clear();
    impl_->recent.clear();
    impl_->operators.clear();
    impl_->replays.clear_outputs();
    impl_->dequantized_weights.clear();
    impl_->precision = precision;
    const auto effective_dtype =
        (precision == core::InferencePrecision::BF16 || precision == core::InferencePrecision::Q8) &&
                !device().capabilities().compute.host()
            ? tensor::DType::BF16
            : impl_->parameter_dtype;
    impl_->activation_dtype = impl_->backend->activation_dtype(effective_dtype);
}
auto Context::precision() const noexcept -> core::InferencePrecision { return impl_->precision; }
auto Context::replay_enabled() const noexcept -> bool { return impl_->replays.enabled(); }
auto Context::set_replay_capacity(std::size_t capacity) -> void { impl_->replays.set_capacity(capacity); }
auto Context::clear_replays() -> void { impl_->replays.clear(); }
auto Context::release_workspaces() -> void {
    if (impl_->replays.recorder())
        throw Failure({ErrorCode::INVALID_ARGUMENT, "cannot release workspaces inside a captured step"});
    synchronize();
    impl_->replays.clear();
    impl_->replays.clear_outputs();
    impl_->operators.clear();
    impl_->recent.clear();
    impl_->backend->release_cached_buffers();
}
auto Context::trim_workspaces() -> void {
    if (impl_->replays.recorder())
        throw Failure({ErrorCode::INVALID_ARGUMENT, "cannot trim workspaces inside a captured step"});
    synchronize();
    impl_->replays.clear_outputs();
    for (auto& [key, prepared] : impl_->operators) prepared.operation->release_workspaces();
    impl_->backend->release_cached_buffers();
}
auto Context::clear_replays(std::string_view prefix) -> void { impl_->replays.clear(prefix); }
auto Context::accelerator() const noexcept -> std::string_view {
    return impl_->compiler ? impl_->compiler->name() : std::string_view{};
}
auto Context::execution() const -> std::string {
    if (const auto name = accelerator(); !name.empty()) return "cpu+" + std::string(name);
    if (device().kind == tensor::DeviceKind::CUDA) return tensor::to_string(device());
    static constexpr tensor::EnumMap<tensor::DeviceKind, std::string_view,
                                     static_cast<std::size_t>(tensor::DeviceKind::COUNT)>
        NAMES{std::array{
            std::pair{tensor::DeviceKind::CPU, std::string_view{"ynnpack"}},
            std::pair{tensor::DeviceKind::Q_NPU, std::string_view{"qnn"}},
            std::pair{tensor::DeviceKind::CUDA, std::string_view{"cuda"}},
            std::pair{tensor::DeviceKind::A_GPU, std::string_view{"mps+ynnpack"}},
            std::pair{tensor::DeviceKind::WEB_GPU, std::string_view{"webgpu"}},
            std::pair{tensor::DeviceKind::VULKAN, std::string_view{"vulkan+ynnpack"}},
        }};
    return std::string(NAMES[device().kind]);
}
auto Context::step_compiler() const noexcept -> std::shared_ptr<graph::Compiler> { return impl_->compiler; }
auto Context::step_extent(std::size_t required, std::size_t capacity) const -> std::size_t {
    return impl_->compiler ? impl_->compiler->key_extent(required, capacity)
                           : std::min(capacity, (required + 127) / 128 * 128);
}
auto Context::crop_local_attention() const noexcept -> bool {
    return !impl_->compiler || impl_->compiler->crop_local_attention();
}
auto Context::prefill_chunk_size(std::size_t requested) const -> std::size_t {
    return impl_->compiler ? impl_->compiler->prefill_chunk_size(requested) : requested;
}
auto Context::replay(std::string_view key, std::span<const Tensor> inputs, const Step& step, bool validate_structure)
    -> std::span<const Tensor> {
    auto& impl = *impl_;
    const auto started = impl.profiling ? Clock::now() : Clock::time_point{};
    const auto result = impl.replays.replay(key, inputs, step, validate_structure, [&] {
        if (impl.device.policy().replay.region_layers) trim_workspaces();
    });
    if (impl.profiling && result.mode != graph::ReplayMode::EAGER) {
        static constexpr std::array MODES{"eager", "capture", "replay", "accelerated"};
        const auto mode = MODES[static_cast<std::size_t>(result.mode)];
        auto& profile = impl.profiles["phase=" + impl.phase + "|op=" + mode + "|step=" + std::string(key)];
        ++profile.calls;
        profile.run_ns += nanoseconds(started);
    }
    return result.outputs;
}
auto Context::add(const Tensor& left, const Tensor& right) -> Tensor {
    return impl_->run({Operation::ADD}, {&left, &right});
}
auto Context::greedy_token(const Tensor& logits, TokenOutput output) -> Tensor {
    const std::array<std::int64_t, 1> attributes{static_cast<std::int64_t>(output)};
    return impl_->run({Operation::GREEDY_TOKEN, attributes, tensor::DType::I32}, {&logits});
}
auto Context::embedding(const Tensor& indices, const Tensor& weight, const Tensor& scales, std::int32_t width,
                        std::int32_t bits, float multiplier, bool unsigned_packed) -> Tensor {
    if (indices.dtype() != tensor::DType::I32 || indices.dimensions() != 1 || weight.dimensions() != 2 || width <= 0 ||
        (bits != 0 && bits != 2 && bits != 4 && bits != 8) || !std::isfinite(multiplier))
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid embedding operands"});
    const std::array<std::int64_t, 4> attributes{width, bits, bits ? static_cast<std::int64_t>(scales.size(1)) : 1,
                                                 unsigned_packed ? 1 : 0};
    const OperatorSpec spec{Operation::EMBEDDING, attributes, impl_->activation_dtype, multiplier};
    if (bits) {
        if (const auto found = impl_->packed_zeros.find(scales.storage_identity());
            found != impl_->packed_zeros.end() && found->second.defined())
            return impl_->run(spec, {&indices, &weight, &scales, &found->second});
        return impl_->run(spec, {&indices, &weight, &scales});
    }
    return impl_->run(spec, {&indices, &weight});
}
auto Context::add_(Tensor& left, const Tensor& right) -> Tensor& {
    impl_->run({Operation::ADD}, {&left, &right}, nullptr, &left);
    return left;
}
auto Context::embedding_sum(const Tensor& tokens, const Tensor& weight, const Tensor& scales, std::int32_t width,
                            std::int32_t bits, float multiplier) -> Tensor {
    if (!device().capabilities().compute.embedding_sum || tokens.dtype() != tensor::DType::I32 ||
        tokens.dimensions() != 2 || !tokens.size(0) || !tokens.size(1) || weight.dimensions() != 2 ||
        weight.size(0) % tokens.size(0) || (bits != 0 && bits != 2 && bits != 4 && bits != 8))
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid device embedding sum"});
    const std::array<std::int64_t, 4> attributes{width, bits, bits ? static_cast<std::int64_t>(scales.size(1)) : 1,
                                                 static_cast<std::int64_t>(weight.size(0) / tokens.size(0))};
    if (bits)
        return impl_->run({Operation::EMBEDDING_SUM, attributes, tensor::DType::F32, multiplier},
                          {&tokens, &weight, &scales});
    return impl_->run({Operation::EMBEDDING_SUM, attributes, tensor::DType::F32, multiplier}, {&tokens, &weight});
}
auto Context::diffusion_candidates(const Tensor& logits, std::size_t target, std::int32_t codebooks,
                                   std::int32_t vocabulary, std::int32_t mask, float guidance) -> Tensor {
    if (!device().capabilities().sampling.diffusion || logits.dtype() != tensor::DType::F32 ||
        logits.dimensions() != 3 || logits.size(0) != 1 || !target || codebooks <= 0 || vocabulary <= 1 || mask < 0 ||
        mask >= vocabulary || logits.size(2) != static_cast<std::size_t>(codebooks) * vocabulary ||
        (logits.size(1) != target && logits.size(1) != 2 * target) || !std::isfinite(guidance))
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid device diffusion logits"});
    const std::array<std::int64_t, 4> attributes{vocabulary, mask, codebooks, static_cast<std::int64_t>(target)};
    return impl_->run({Operation::DIFFUSION_CANDIDATES, attributes, tensor::DType::F32, guidance}, {&logits});
}
auto Context::diffusion_update(const Tensor& tokens, const Tensor& candidates, const Tensor& noise, std::size_t count,
                               std::int32_t mask, float penalty, float temperature) -> Tensor {
    if (!device().capabilities().sampling.diffusion || tokens.dtype() != tensor::DType::I32 ||
        tokens.dimensions() != 2 || !tokens.numel() || candidates.dtype() != tensor::DType::F32 ||
        candidates.numel() != tokens.numel() * 2 || noise.dtype() != tensor::DType::F32 || !noise.numel() ||
        count > tokens.numel() || !std::isfinite(penalty) || !std::isfinite(temperature) || temperature < 0)
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid device diffusion update"});
    const std::array<std::int64_t, 4> attributes{static_cast<std::int64_t>(count), mask,
                                                 std::bit_cast<std::int32_t>(penalty),
                                                 std::bit_cast<std::int32_t>(temperature)};
    return impl_->run({Operation::DIFFUSION_UPDATE, attributes, tensor::DType::I32}, {&tokens, &candidates, &noise});
}
auto Context::conv1d(const Tensor& input, const Tensor& weight, const Tensor& scales, const Tensor& bias,
                     std::int32_t kernel, std::int32_t stride, std::int32_t padding, std::int32_t dilation) -> Tensor {
    if (!device().capabilities().compute.native_audio() || input.dtype() != tensor::DType::F32 ||
        input.dimensions() != 3 || input.size(0) != 1 || kernel <= 0 || stride <= 0 || padding < 0 || dilation <= 0 ||
        weight.dimensions() != 2 || weight.size(1) != input.size(2) * kernel || bias.dtype() != tensor::DType::F32 ||
        bias.numel() != weight.size(0))
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid device convolution"});
    const std::array<std::int64_t, 4> attributes{kernel, stride, padding, dilation};
    if (scales.defined()) return impl_->run({Operation::CONV1D, attributes}, {&input, &weight, &scales, &bias});
    return impl_->run({Operation::CONV1D, attributes}, {&input, &weight, &bias});
}
auto Context::snake_conv1d(const Tensor& input, const Tensor& alpha, const Tensor& weight, const Tensor& scales,
                           const Tensor& bias, std::int32_t kernel, std::int32_t stride, std::int32_t padding,
                           std::int32_t dilation) -> Tensor {
    if (!device().capabilities().compute.native_audio() || input.dtype() != tensor::DType::F32 ||
        input.dimensions() != 3 || input.size(0) != 1 || alpha.dtype() != tensor::DType::F32 ||
        alpha.dimensions() != 1 || alpha.numel() != input.size(2) || kernel <= 0 || stride <= 0 || padding < 0 ||
        dilation <= 0 || weight.dimensions() != 2 || weight.size(1) != input.size(2) * kernel ||
        bias.dtype() != tensor::DType::F32 || bias.numel() != weight.size(0))
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid device Snake convolution"});
    const std::array<std::int64_t, 4> attributes{kernel, stride, padding, dilation};
    if (scales.defined())
        return impl_->run({Operation::SNAKE_CONV1D, attributes}, {&input, &alpha, &weight, &scales, &bias});
    return impl_->run({Operation::SNAKE_CONV1D, attributes}, {&input, &alpha, &weight, &bias});
}
auto Context::snake_quantized_linear(const Tensor& input, const Tensor& alpha, const Tensor& weight,
                                     const Tensor& scale, const Tensor& bias, bool transpose_weight) -> Tensor {
    if (!device().capabilities().compute.native_audio() || input.dtype() != tensor::DType::F32 ||
        input.dimensions() != 3 || input.size(0) != 1 || alpha.dtype() != tensor::DType::F32 ||
        alpha.dimensions() != 1 || alpha.numel() != input.size(2))
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid device Snake projection"});
    const std::array<std::int64_t, 1> attributes{transpose_weight};
    return impl_->run({Operation::SNAKE_QUANTIZED_LINEAR, attributes}, {&input, &alpha, &weight, &scale, &bias});
}
auto Context::conv_transpose_fold(const Tensor& projected, const Tensor& bias, std::int32_t kernel, std::int32_t stride,
                                  std::int32_t padding, std::int32_t output_padding, std::int32_t dilation) -> Tensor {
    if (!device().capabilities().compute.native_audio() || projected.dtype() != tensor::DType::F32 ||
        projected.dimensions() != 3 || projected.size(0) != 1 || kernel <= 0 || stride <= 0 || padding < 0 ||
        output_padding < 0 || output_padding >= stride || dilation <= 0 || projected.size(2) % kernel ||
        bias.dtype() != tensor::DType::F32 || bias.numel() != projected.size(2) / kernel)
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid transposed convolution folding"});
    const std::array<std::int64_t, 5> attributes{kernel, stride, padding, output_padding, dilation};
    return impl_->run({Operation::CONV_TRANSPOSE_FOLD, attributes}, {&projected, &bias});
}
auto Context::snake(const Tensor& input, const Tensor& alpha) -> Tensor {
    if (!device().capabilities().compute.native_audio() || input.dtype() != tensor::DType::F32 ||
        input.dimensions() != 3 || alpha.dtype() != tensor::DType::F32 || alpha.dimensions() != 1 ||
        alpha.numel() != input.size(2))
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid device Snake activation"});
    return impl_->run({Operation::SNAKE}, {&input, &alpha});
}
auto Context::multiply(const Tensor& left, const Tensor& right) -> Tensor {
    return impl_->run({Operation::MULTIPLY}, {&left, &right});
}
auto Context::multiply_(Tensor& left, const Tensor& right) -> Tensor& {
    impl_->run({Operation::MULTIPLY}, {&left, &right}, nullptr, &left);
    return left;
}
auto Context::cast(const Tensor& input, tensor::DType dtype, float scale) -> Tensor {
    if (!input.defined() || input.device() != device())
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid cast operand"});
    if (scale != 0) {
        if (!std::isfinite(scale) || scale < 0 || dtype != tensor::DType::I8 || input.dtype() != tensor::DType::F32)
            throw Failure({ErrorCode::INVALID_ARGUMENT, "a scaled cast rounds FP32 into INT8 with a positive scale"});
        if (!device().capabilities().compute.supports(tensor::TransformerSupport::QUANTIZED))
            throw Failure({ErrorCode::UNSUPPORTED, "scaled INT8 casts are unsupported by this backend"});
    }
    if (input.dtype() == dtype) return input;
    return impl_->run({Operation::CAST, {}, dtype, scale}, {&input});
}
auto Context::matmul(const Tensor& left, const Tensor& right, bool transpose_right) -> Tensor {
    const std::array<std::int64_t, 1> attributes{transpose_right};
    const auto dtype = impl_->activation_dtype == tensor::DType::BF16 && left.dtype() == tensor::DType::BF16 &&
                               right.dtype() == tensor::DType::BF16
                           ? tensor::DType::BF16
                           : tensor::DType::F32;
    return impl_->run({Operation::MATMUL, attributes, dtype}, {&left, &right});
}
auto Context::scaled_dot_product_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                           std::int32_t heads, const Tensor& mask) -> Tensor {
    const std::array<std::int64_t, 1> attributes{heads};
    if (mask.defined())
        return impl_->run({Operation::ATTENTION, attributes, query.dtype()}, {&query, &key, &value, &mask});
    return impl_->run({Operation::ATTENTION, attributes, query.dtype()}, {&query, &key, &value});
}
auto Context::linear(const Tensor& input, const Tensor& weight, const Tensor& bias, bool transpose_weight) -> Tensor {
    if (transpose_weight && !bias.defined() && !impl_->packed_weights.empty()) {
        const auto found = impl_->packed_weights.find(weight.storage_identity());
        const bool single_row = input.dimensions() > 0 && input.numel() == input.size(-1);
        if (found != impl_->packed_weights.end() && found->second.source.dtype() == weight.dtype() &&
            std::ranges::equal(found->second.source.shape(), weight.shape()) &&
            (found->second.packed_prefill || decode_attention || decode_projections || single_row)) {
            auto& binding = found->second;
            if (!binding.packed.values.defined()) {
                const auto started = Clock::now();
                const auto host_source = binding.source.is_host_accessible()
                                             ? binding.source
                                             : require(binding.source.to(tensor::Device::cpu()));
                auto packed = require(pack_weight(host_source, binding.bits, binding.group_size));
                binding.source.release_pages(false);
                packed.values = require(packed.values.to(device()));
                packed.scales = require(packed.scales.to(device()));
                binding.packed = std::move(packed);
                impl_->preparation += nanoseconds(started);
            }
            auto output = packed_linear(input, binding.packed.values, binding.packed.scales, binding.packed.bits,
                                        binding.packed.group_size, 0.F, 0.F, binding.packed.zeros);
            if (!binding.released && device().kind == tensor::DeviceKind::CPU &&
                impl_->precision == core::InferencePrecision::CHECKPOINT && single_row &&
                binding.packed.group_size < input.size(-1) &&
                binding.packed.values.nbytes() >= (std::size_t{1} << 20) &&
                binding.packed.values.size(0) % (8 / binding.packed.bits) == 0) {
                binding.packed.values.release_pages(true);
                binding.released = true;
            }
            return output;
        }
    }
    const auto activation = weight.dtype() == tensor::DType::BF16 ? cast(input, tensor::DType::BF16) : input;
    const std::array<std::int64_t, 1> attributes{transpose_weight};
    const auto dtype = impl_->activation_dtype == tensor::DType::BF16 && weight.dtype() == tensor::DType::BF16
                           ? tensor::DType::BF16
                           : tensor::DType::F32;
    if (!bias.defined()) return impl_->run({Operation::LINEAR, attributes, dtype}, {&activation, &weight});
    return impl_->run({Operation::LINEAR, attributes, dtype}, {&activation, &weight, &bias});
}
auto Context::prepare_linear_weights(const Tensor& weight, std::int32_t bits, std::int32_t group_size,
                                     bool packed_prefill) -> void {
    if (!weight.defined() || weight.dimensions() != 2 || !weight.is_contiguous() ||
        (weight.dtype() != tensor::DType::F16 && weight.dtype() != tensor::DType::BF16 &&
         weight.dtype() != tensor::DType::F32) ||
        (bits != 2 && bits != 4 && bits != 8) || group_size <= 0 || group_size % (8 / bits) ||
        weight.size(1) % group_size || !weight.size(0) || !weight.size(1))
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid packed weight dimensions, precision or group size"});
    const auto address = weight.storage_identity();
    if (const auto found = impl_->packed_weights.find(address); found != impl_->packed_weights.end()) {
        if (found->second.bits != bits || found->second.group_size != group_size ||
            found->second.packed_prefill != packed_prefill ||
            !std::ranges::equal(found->second.source.shape(), weight.shape()))
            throw Failure({ErrorCode::INVALID_ARGUMENT, "weight already packed with a different layout"});
        return;
    }
    if (device().kind == tensor::DeviceKind::CPU && weight.nbytes() >= (std::size_t{1} << 20) &&
        weight.nbytes() < (std::size_t{256} << 20)) {
        impl_->packed_weights.emplace(address,
                                      Impl::PackedBinding{weight, {}, bits, group_size, packed_prefill, false});
        return;
    }
    const auto host_weight = weight.is_host_accessible() ? weight : require(weight.to(tensor::Device::cpu()));
    auto packed = require(pack_weight(host_weight, bits, group_size));
    if (weight.is_host_accessible()) weight.release_pages(false);
    packed.values = require(packed.values.to(device()));
    packed.scales = require(packed.scales.to(device()));
    impl_->packed_weights.emplace(
        address, Impl::PackedBinding{weight, std::move(packed), bits, group_size, packed_prefill, false});
}
auto Context::bind_packed_zero(const Tensor& scales, Tensor zeros) -> void {
    if (scales.defined() && zeros.defined()) impl_->packed_zeros[scales.storage_identity()] = std::move(zeros);
}
auto Context::quantized_linear(const Tensor& input, const Tensor& weight, const Tensor& scale, const Tensor& bias,
                               bool transpose_weight) -> Tensor {
    const std::array<std::int64_t, 1> attributes{transpose_weight};
    return impl_->run({Operation::QUANTIZED_LINEAR, attributes}, {&input, &weight, &scale, &bias});
}
auto Context::apply(const Tensor& input, const ProjectionPrologue& prologue) -> Tensor {
    switch (prologue.kind) {
        case ProjectionPrologue::Kind::NONE:
            return input;
        case ProjectionPrologue::Kind::RMS_NORM:
            if (!prologue.scale) throw Failure({ErrorCode::INVALID_ARGUMENT, "RMS-norm prologue requires a scale"});
            return rms_norm(input, *prologue.scale, prologue.epsilon);
        case ProjectionPrologue::Kind::GELU_GATE:
            return gelu_gate(input);
    }
    throw Failure({ErrorCode::INVALID_ARGUMENT, "unknown projection prologue"});
}
auto Context::fuses(const Tensor& input, const ProjectionPrologue& prologue) const -> bool {
    // Fuse only multi-row CUDA W8A8 prefill; decode graphs keep their grouped single-row projections.
    constexpr std::int64_t max_width = 10240, min_rows = 32;
    if (prologue.kind == ProjectionPrologue::Kind::NONE || device().kind != tensor::DeviceKind::CUDA ||
        impl_->precision != core::InferencePrecision::CHECKPOINT || impl_->activation_dtype != tensor::DType::F32 ||
        !input.defined() || input.dtype() != tensor::DType::F32 || input.dimensions() < 2 || !input.size(-1))
        return false;
    const bool gate = prologue.kind == ProjectionPrologue::Kind::GELU_GATE;
    const auto width = input.size(-1) / (gate ? 2 : 1);
    return (!gate || width % 4 == 0) && width <= max_width && input.numel() / input.size(-1) >= min_rows &&
           (gate || (prologue.scale && prologue.scale->dtype() == tensor::DType::F32 &&
                     prologue.scale->numel() == static_cast<std::size_t>(width)));
}
auto Context::packed_linear(const Tensor& input, const Tensor& weight, const Tensor& scales, std::int32_t bits,
                            std::int32_t group_size, float input_scale, float output_scale, const Tensor& zeros,
                            const ProjectionPrologue& prologue) -> Tensor {
    if (!std::isfinite(input_scale) || input_scale < 0 || !std::isfinite(output_scale) || output_scale < 0)
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid calibrated projection scales"});
    const Tensor* bound_zero = zeros.defined() ? &zeros : nullptr;
    if (!bound_zero) {
        if (const auto found = impl_->packed_zeros.find(scales.storage_identity());
            found != impl_->packed_zeros.end() && found->second.defined())
            bound_zero = &found->second;
    }
    if (prologue.kind != ProjectionPrologue::Kind::NONE) {
        const bool gate = prologue.kind == ProjectionPrologue::Kind::GELU_GATE;
        const auto width = input.size(-1) / (gate ? 2 : 1);
        if (!fuses(input, prologue) || bits != 8 || bound_zero || group_size != width)
            return packed_linear(apply(input, prologue), weight, scales, bits, group_size, input_scale, output_scale,
                                 zeros);
        // Attribute layout: {bits, group, output_scale, prologue kind (1 = RMS norm, 2 = GELU gate), epsilon}.
        const std::array<std::int64_t, 5> attributes{bits, group_size, std::bit_cast<std::int32_t>(output_scale),
                                                     gate ? 2 : 1, std::bit_cast<std::int32_t>(prologue.epsilon)};
        if (gate)
            return impl_->run({Operation::PACKED_LINEAR, attributes, tensor::DType::F32, input_scale},
                              {&input, &weight, &scales});
        return impl_->run({Operation::PACKED_LINEAR, attributes, tensor::DType::F32, input_scale},
                          {&input, &weight, &scales, prologue.scale});
    }
    if (device().kind == tensor::DeviceKind::CUDA && impl_->precision == core::InferencePrecision::CHECKPOINT &&
        bits == 4 && group_size > 0 && input.dtype() == tensor::DType::BF16 && input_scale == 0.F &&
        output_scale == 0.F && (!bound_zero || !bound_zero->defined()) && weight.dtype() == tensor::DType::U8 &&
        weight.device() == device() && scales.device() == device() &&
        (scales.dtype() == tensor::DType::F32 || scales.dtype() == tensor::DType::F16))
        return block_scaled_linear(input, {BlockScaledFormat::INT4, weight, scales, {}, group_size});
    if (device().kind == tensor::DeviceKind::CUDA && impl_->precision == core::InferencePrecision::Q8 && bits == 8 &&
        (!bound_zero || !bound_zero->defined())) {
        const auto activation = input.dtype() == tensor::DType::BF16 ? input : cast(input, tensor::DType::BF16);
        const std::array<std::int64_t, 3> attributes{bits, group_size, 0};
        return impl_->run({Operation::PACKED_LINEAR, attributes, tensor::DType::BF16, 0.F},
                          {&activation, &weight, &scales});
    }
    if (impl_->precision == core::InferencePrecision::BF16 || impl_->precision == core::InferencePrecision::QAT_FP32) {
        const auto valid_bits = bits == 2 || bits == 4 || bits == 8;
        const auto values_per_byte = valid_bits ? 8 / bits : 0;
        const auto width = weight.dimensions() == 2 ? weight.size(1) * values_per_byte : 0;
        const auto groups = group_size > 0 ? width / group_size : 0;
        if (!device().capabilities().compute.host() || !valid_bits || input.dimensions() == 0 || group_size <= 0 ||
            width != input.size(-1) || width % group_size || weight.dimensions() != 2 ||
            (weight.dtype() != tensor::DType::U8 && weight.dtype() != tensor::DType::I8) ||
            scales.dtype() != tensor::DType::F32 || scales.numel() != weight.size(0) * groups)
            throw Failure({ErrorCode::UNSUPPORTED,
                           "selected precision requires CPU packed weights with compatible group scales"});
        const bool qat_fp32 = impl_->precision == core::InferencePrecision::QAT_FP32;
        const auto weight_dtype = qat_fp32 ? tensor::DType::F32 : tensor::DType::BF16;
        const auto bytes = require(weight.host_bytes());
        const auto scale_bytes = require(scales.host_bytes());
        const auto key = Impl::DequantizedKey{
            .weight = bytes.data(),
            .scales = scale_bytes.data(),
            .dtype = weight_dtype,
            .precision = impl_->precision,
            .bits = bits,
            .group_size = group_size,
            .rows = weight.size(0),
            .packed_columns = weight.size(1),
            .scale_count = scales.numel(),
        };
        auto found = impl_->dequantized_weights.find(key);
        if (found == impl_->dequantized_weights.end()) {
            const auto scale_values = require(scales.data<float>());
            if (!std::ranges::all_of(scale_values, [](float scale) { return std::isfinite(scale) && scale > 0.F; }))
                throw Failure({ErrorCode::INVALID_ARGUMENT, "packed weight scales must be finite and positive"});
            const auto* values = reinterpret_cast<const std::uint8_t*>(bytes.data());
            std::vector<float> dequantized(weight.size(0) * width);
            const auto mask = (1 << bits) - 1;
            const auto sign = 1 << (bits - 1);
            for (std::size_t row = 0; row < weight.size(0); ++row)
                for (std::size_t column = 0; column < width; ++column) {
                    const auto packed_column = column / values_per_byte;
                    auto value =
                        (values[row * weight.size(1) + packed_column] >> ((column % values_per_byte) * bits)) & mask;
                    if (value & sign) value -= 1 << bits;
                    dequantized[row * width + column] =
                        static_cast<float>(value) * scale_values[row * groups + column / group_size];
                }
            auto unpacked =
                require(Tensor::from_host({static_cast<std::int64_t>(weight.size(0)), static_cast<std::int64_t>(width)},
                                          std::span<const float>(dequantized)));
            if (weight_dtype == tensor::DType::BF16) unpacked = cast(unpacked, tensor::DType::BF16);
            found =
                impl_->dequantized_weights.emplace(key, Impl::DequantizedBinding{weight, scales, std::move(unpacked)})
                    .first;
        }
        auto output = linear(qat_fp32 ? static_round(input, input_scale) : input, found->second.value, {}, true);
        if (output.dtype() != tensor::DType::F32) output = cast(output, tensor::DType::F32);
        return qat_fp32 ? static_round(output, output_scale) : output;
    }
    if (impl_->precision != core::InferencePrecision::CHECKPOINT)
        throw Failure({ErrorCode::UNSUPPORTED, std::string(core::to_string(impl_->precision)) +
                                                   " is not implemented for packed projections on this device"});
    const std::array<std::int64_t, 3> attributes{bits, group_size, std::bit_cast<std::int32_t>(output_scale)};
    const auto output_dtype = device().kind == tensor::DeviceKind::CUDA && bits == 8 &&
                                      impl_->activation_dtype == tensor::DType::BF16 &&
                                      input.dtype() == tensor::DType::BF16 && input_scale == 0.F &&
                                      output_scale == 0.F && (!bound_zero || !bound_zero->defined())
                                  ? tensor::DType::BF16
                                  : tensor::DType::F32;
    if (bound_zero && bound_zero->defined())
        return impl_->run({Operation::PACKED_LINEAR, attributes, output_dtype, input_scale},
                          {&input, &weight, &scales, bound_zero});
    return impl_->run({Operation::PACKED_LINEAR, attributes, output_dtype, input_scale}, {&input, &weight, &scales});
}
auto Context::gelu(const Tensor& input, bool approximate) -> Tensor {
    const std::array<std::int64_t, 1> attributes{approximate};
    return impl_->run({Operation::GELU, attributes, input.dtype()}, {&input});
}
auto Context::gelu_multiply(const Tensor& gate, const Tensor& value) -> Tensor {
    if (!gate.defined() || !value.defined() ||
        (gate.dtype() != tensor::DType::F32 && gate.dtype() != impl_->activation_dtype) ||
        value.dtype() != gate.dtype() || !std::ranges::equal(gate.shape(), value.shape()))
        throw Failure({ErrorCode::INVALID_ARGUMENT, "GELU multiplication requires matching floating tensors"});
    return impl_->run({Operation::GELU_MULTIPLY, {}, gate.dtype()}, {&gate, &value});
}
auto Context::silu_multiply(const Tensor& gate, const Tensor& value) -> Tensor {
    if (!gate.defined() || !value.defined() ||
        (gate.dtype() != tensor::DType::F32 && gate.dtype() != impl_->activation_dtype) ||
        value.dtype() != gate.dtype() || !std::ranges::equal(gate.shape(), value.shape()))
        throw Failure({ErrorCode::INVALID_ARGUMENT, "SiLU multiplication requires matching floating tensors"});
    return impl_->run({Operation::SILU_MULTIPLY, {}, gate.dtype()}, {&gate, &value});
}
auto Context::gelu_gate(const Tensor& gate_up) -> Tensor {
    if (!gate_up.defined() || (gate_up.dtype() != tensor::DType::F32 && gate_up.dtype() != impl_->activation_dtype) ||
        !gate_up.dimensions() || gate_up.size(-1) % 2)
        throw Failure({ErrorCode::INVALID_ARGUMENT, "GELU gate requires floating [..., 2 * intermediate] input"});
    return impl_->run({Operation::GELU_GATE, {}, gate_up.dtype()}, {&gate_up});
}
auto Context::silu_gate(const Tensor& gate_up, float gate_cap, float up_cap, const Tensor& row_weights) -> Tensor {
    const auto cpu_bf16 = device().kind == tensor::DeviceKind::CPU && gate_up.defined() &&
                          gate_up.dtype() == tensor::DType::BF16 && (gate_cap || up_cap || row_weights.defined());
    if (device().kind == tensor::DeviceKind::CPU && gate_up.defined() && gate_up.dtype() == tensor::DType::BF16 &&
        !cpu_bf16)
        return cast(silu_gate(cast(gate_up, tensor::DType::F32), gate_cap, up_cap, row_weights), tensor::DType::BF16);
    if (!gate_up.defined() ||
        (gate_up.dtype() != tensor::DType::F32 && gate_up.dtype() != impl_->activation_dtype && !cpu_bf16) ||
        !gate_up.dimensions() || gate_up.size(-1) % 2)
        throw Failure({ErrorCode::INVALID_ARGUMENT, "SiLU gate requires floating [..., 2 * intermediate] input"});
    if (!gate_cap && !up_cap && !row_weights.defined())
        return impl_->run({Operation::SILU_GATE, {}, gate_up.dtype()}, {&gate_up});
    if (!std::isfinite(gate_cap) || !std::isfinite(up_cap) || !(gate_cap > 0) || !(up_cap > 0))
        throw Failure({ErrorCode::INVALID_ARGUMENT, "softcapped SiLU gate requires positive caps"});
    if (device().kind == tensor::DeviceKind::CPU) {
        if (gate_up.dimensions() != 2 || (row_weights.defined() && (row_weights.numel() != gate_up.size(0) ||
                                                                    row_weights.dtype() != tensor::DType::F32)))
            throw Failure({ErrorCode::INVALID_ARGUMENT, "softcapped SiLU gate requires [rows, 2 * width] input"});
        const auto values = cpu_floats(*this, gate_up);
        const auto weights = row_weights.defined() ? cpu_floats(*this, row_weights) : HostFloats{};
        const auto rows = static_cast<std::int64_t>(gate_up.size(0));
        const auto width = static_cast<std::int64_t>(gate_up.size(1) / 2);
        return cpu_output(*this, {rows, width}, gate_up.dtype(), [&](float* output) {
            return cpu::silu_gate(values.data(), rows, width, gate_cap, up_cap,
                                  weights.empty() ? nullptr : weights.data(), output);
        });
    }
    if (device().kind != tensor::DeviceKind::CUDA && device().kind != tensor::DeviceKind::A_GPU)
        throw Failure({ErrorCode::UNSUPPORTED, "softcapped SiLU gate requires CPU, CUDA or Metal"});
    if (gate_up.dimensions() != 2 ||
        (row_weights.defined() && (row_weights.dimensions() != 1 || row_weights.size(0) != gate_up.size(0) ||
                                   row_weights.dtype() != tensor::DType::F32 || row_weights.device() != device())))
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid softcapped SiLU gate row weights"});
    const std::array<std::int64_t, 2> attributes{std::bit_cast<std::int32_t>(gate_cap),
                                                 std::bit_cast<std::int32_t>(up_cap)};
    if (row_weights.defined())
        return impl_->run({Operation::SILU_GATE, attributes, gate_up.dtype()}, {&gate_up, &row_weights});
    return impl_->run({Operation::SILU_GATE, attributes, gate_up.dtype()}, {&gate_up});
}
auto Context::gated_feed_forward(const Tensor& input, const Tensor& gate_up_weight, const Tensor& gate_up_scales,
                                 const Tensor& down_weight, const Tensor& down_scales, std::int32_t bits,
                                 std::int32_t input_size, std::int32_t intermediate_size, float gate_up_input_scale,
                                 float gate_up_output_scale, float down_input_scale, float down_output_scale, bool silu)
    -> Tensor {
    if (impl_->precision != core::InferencePrecision::CHECKPOINT ||
        !device().capabilities().compute.supports_native_calibrated_packed(bits)) {
        const auto gate_up = packed_linear(input, gate_up_weight, gate_up_scales, bits, input_size, gate_up_input_scale,
                                           gate_up_output_scale);
        const auto gate = slice(gate_up, -1, 0, intermediate_size);
        const auto up = slice(gate_up, -1, intermediate_size, intermediate_size);
        const auto hidden = silu ? silu_multiply(gate, up) : gelu_multiply(gate, up);
        return packed_linear(hidden, down_weight, down_scales, bits, intermediate_size, down_input_scale,
                             down_output_scale);
    }

    const std::array<std::int64_t, 7> attributes{bits,
                                                 input_size,
                                                 intermediate_size,
                                                 std::bit_cast<std::int32_t>(gate_up_output_scale),
                                                 std::bit_cast<std::int32_t>(down_input_scale),
                                                 std::bit_cast<std::int32_t>(down_output_scale),
                                                 silu};
    const auto count = silu ? attributes.size() : attributes.size() - 1;
    return impl_->run(
        {Operation::GATED_FEED_FORWARD, std::span(attributes).first(count), tensor::DType::F32, gate_up_input_scale},
        {&input, &gate_up_weight, &gate_up_scales, &down_weight, &down_scales});
}
auto Context::dequantize_experts(const Tensor& values, const Tensor& scales, const Tensor& global, bool interleaved)
    -> Tensor {
    if (impl_->device.kind != tensor::DeviceKind::CUDA)
        throw Failure({ErrorCode::UNSUPPORTED, "expert dequantization requires CUDA"});
    const std::array<std::int64_t, 1> attributes{interleaved};
    if (global.defined())
        return impl_->run({Operation::DEQUANTIZE_EXPERTS, attributes, tensor::DType::BF16},
                          {&values, &scales, &global});
    return impl_->run({Operation::DEQUANTIZE_EXPERTS, attributes, tensor::DType::BF16}, {&values, &scales});
}

auto Context::block_scaled_linear(const Tensor& input, const BlockScaledWeight& weight) -> Tensor {
    const auto attributes = block_scaled_attributes(weight);
    const runtime::OperatorSpec spec{Operation::BLOCK_SCALED_LINEAR, attributes, input.dtype()};
    return weight.global.defined() ? impl_->run(spec, {&input, &weight.values, &weight.scales, &weight.global})
                                   : impl_->run(spec, {&input, &weight.values, &weight.scales});
}

auto Context::block_scaled_silu(const Tensor& input, const BlockScaledWeight& weight, const Tensor& row_weights,
                                float gate_cap, float up_cap) -> Tensor {
    const auto attributes = block_scaled_attributes(weight, true, gate_cap, up_cap);
    const runtime::OperatorSpec spec{Operation::BLOCK_SCALED_LINEAR, attributes, input.dtype()};
    return weight.global.defined()
               ? impl_->run(spec, {&input, &weight.values, &weight.scales, &weight.global, &row_weights})
               : impl_->run(spec, {&input, &weight.values, &weight.scales, &row_weights});
}

auto Context::route_experts(const Tensor& logits, const Tensor& bias, std::int32_t top, float scale) -> Tensor {
    if (device().kind != tensor::DeviceKind::CUDA && device().kind != tensor::DeviceKind::A_GPU)
        throw Failure({ErrorCode::UNSUPPORTED, "expert routing requires CUDA or Metal"});
    const std::array<std::int64_t, 2> attributes{top, std::bit_cast<std::int32_t>(scale)};
    return impl_->run({Operation::ROUTE_EXPERTS, attributes, tensor::DType::F32}, {&logits, &bias});
}

namespace {
auto check_expert_backend(tensor::Device device, BlockScaledFormat format) -> void {
    const bool vq = format == BlockScaledFormat::VQ3D || format == BlockScaledFormat::VQ4D;
    if (device.kind == tensor::DeviceKind::CUDA ||
        (vq && (device.kind == tensor::DeviceKind::CPU || device.kind == tensor::DeviceKind::A_GPU)))
        return;
    throw Failure({ErrorCode::UNSUPPORTED, "unsupported routed expert format for this backend"});
}
} // namespace

auto Context::expert_silu(const Tensor& input, const BlockScaledWeight& weight, const Tensor& routes, float gate_cap,
                          float up_cap) -> Tensor {
    check_expert_backend(device(), weight.format);
    const auto attributes = block_scaled_attributes(weight, true, gate_cap, up_cap);
    const runtime::OperatorSpec spec{Operation::EXPERT_LINEAR, attributes, input.dtype()};
    return weight.global.defined() ? impl_->run(spec, {&input, &weight.values, &weight.scales, &weight.global, &routes})
                                   : impl_->run(spec, {&input, &weight.values, &weight.scales, &routes});
}

auto Context::expert_linear(const Tensor& input, const BlockScaledWeight& weight, const Tensor& routes,
                            std::optional<tensor::DType> output_dtype) -> Tensor {
    check_expert_backend(device(), weight.format);
    const auto attributes = block_scaled_attributes(weight);
    const auto dtype = output_dtype.value_or(tensor::DType::F32);
    const runtime::OperatorSpec spec{Operation::EXPERT_LINEAR, attributes, dtype};
    return weight.global.defined() ? impl_->run(spec, {&input, &weight.values, &weight.scales, &weight.global, &routes})
                                   : impl_->run(spec, {&input, &weight.values, &weight.scales, &routes});
}

auto Context::pack_experts(const Tensor& values, const Tensor& scales, const Tensor& global, bool interleaved)
    -> PackedWeight {
    if (device().kind == tensor::DeviceKind::CPU)
        return require(pack_expert_weights(values, scales, global, interleaved));
    if (device().kind != tensor::DeviceKind::CUDA)
        throw Failure({ErrorCode::UNSUPPORTED, "expert packing requires CPU or CUDA"});
    const std::array<std::int64_t, 1> attributes{interleaved};
    const auto row_scales =
        global.defined()
            ? impl_->run({Operation::DEQUANTIZE_EXPERTS, attributes, tensor::DType::F32}, {&values, &scales, &global})
            : impl_->run({Operation::DEQUANTIZE_EXPERTS, attributes, tensor::DType::F32}, {&values, &scales});
    const auto host_scales = require(row_scales.to(tensor::Device::cpu()));
    if (!std::ranges::all_of(require(host_scales.data<float>()),
                             [](float scale) { return std::isfinite(scale) && scale > 0.F; }))
        throw Failure({ErrorCode::INVALID_ARGUMENT, "non-finite expert weight or quantization scale underflow"});
    auto packed = global.defined() ? impl_->run({Operation::DEQUANTIZE_EXPERTS, attributes, tensor::DType::U8},
                                                {&values, &scales, &global, &row_scales})
                                   : impl_->run({Operation::DEQUANTIZE_EXPERTS, attributes, tensor::DType::U8},
                                                {&values, &scales, &row_scales});
    if (packed.size(2) > INT32_MAX)
        throw Failure({ErrorCode::INVALID_ARGUMENT, "expert width exceeds packed projection bounds"});
    const auto width = static_cast<std::int32_t>(packed.size(2));
    return {std::move(packed), row_scales, {}, 8, width};
}

auto Context::gelu_(Tensor& input, bool approximate) -> Tensor& {
    const std::array<std::int64_t, 1> attributes{approximate};
    impl_->run({Operation::GELU, attributes}, {&input}, nullptr, &input);
    return input;
}
auto Context::residual_mix(std::span<const Tensor> sources, const Tensor& queries, float epsilon) -> Tensor {
    if (sources.empty() || sources.size() > 7 || !std::isfinite(epsilon) || epsilon <= 0)
        throw Failure({ErrorCode::INVALID_ARGUMENT, "residual mix requires one to seven sources and positive epsilon"});
    if (device().kind == tensor::DeviceKind::CPU) {
        if (queries.dimensions() != 2 || !queries.size(0) || queries.size(1) <= 0)
            throw Failure({ErrorCode::INVALID_ARGUMENT, "residual mix queries must have shape [queries, width]"});
        const auto width = static_cast<std::int64_t>(queries.size(1));
        const auto query_values = cpu_floats(*this, queries);
        std::vector<HostFloats> values;
        values.reserve(sources.size());
        for (const auto& source : sources) {
            if (source.dimensions() != 2 || source.size(1) != width || !source.size(0))
                throw Failure({ErrorCode::INVALID_ARGUMENT, "residual mix sources must have shape [tokens, width]"});
            values.push_back(cpu_floats(*this, source));
        }
        const auto tokens = static_cast<std::int64_t>(sources.front().size(0));
        if (std::ranges::any_of(sources, [tokens](const Tensor& source) { return source.size(0) != tokens; }))
            throw Failure({ErrorCode::INVALID_ARGUMENT, "residual mix source lengths must match"});
        const auto query_count = static_cast<std::int64_t>(queries.size(0));
        std::vector<const float*> pointers;
        for (const auto& value : values) pointers.push_back(value.data());
        return cpu_output(*this, {query_count, tokens, width}, tensor::DType::F32, [&](float* output) {
            return cpu::residual_mix(pointers, query_values.data(), query_count, tokens, width, epsilon, output);
        });
    }
    if (device().kind != tensor::DeviceKind::CUDA && device().kind != tensor::DeviceKind::A_GPU)
        throw Failure({ErrorCode::UNSUPPORTED, "residual mix requires CPU, CUDA or Metal"});
    std::vector<Tensor> inputs(sources.begin(), sources.end());
    inputs.push_back(queries);
    return impl_->run({Operation::RESIDUAL_MIX, {}, tensor::DType::F32, epsilon}, inputs);
}
auto Context::short_convolution(const Tensor& padded, const Tensor& weight, const Tensor& taps, float scale) -> Tensor {
    if (weight.dimensions() != 2 || !weight.size(0) || !weight.size(1) || !std::isfinite(scale))
        throw Failure({ErrorCode::INVALID_ARGUMENT, "short convolution requires nonempty [width, channels] taps"});
    if (device().kind == tensor::DeviceKind::CPU) {
        const auto width = static_cast<std::int64_t>(weight.size(0));
        const auto channels = static_cast<std::int64_t>(weight.size(1));
        if (padded.dimensions() != 2 || padded.size(1) != channels || padded.size(0) < width)
            throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid short convolution input shape"});
        const auto rows = static_cast<std::int64_t>(padded.size(0)) - width + 1;
        const auto input_values = cpu_floats(*this, padded);
        const auto weight_values = cpu_floats(*this, weight);
        const auto tap_values = taps.defined() ? cpu_floats(*this, taps) : HostFloats{};
        std::int64_t groups = 1;
        if (taps.defined()) {
            if (taps.dimensions() != 2 || taps.size(0) != rows || !taps.size(1) || taps.size(1) % width ||
                channels % (taps.size(1) / width))
                throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid short convolution dynamic taps"});
            groups = taps.size(1) / width;
        }
        std::vector<const float*> sources(static_cast<std::size_t>(rows * width));
        for (std::int64_t token = 0; token < rows; ++token)
            for (std::int64_t lag = 0; lag < width; ++lag)
                sources[static_cast<std::size_t>(token * width + lag)] =
                    input_values.data() + (token + width - 1 - lag) * channels;
        return cpu_output(*this, {rows, channels}, padded.dtype(), [&](float* output) {
            return cpu::short_convolution(sources.data(), input_values.data() + (width - 1) * channels,
                                          weight_values.data(), taps.defined() ? tap_values.data() : nullptr, nullptr,
                                          {rows, channels, width, groups, scale}, output);
        });
    }
    if (device().kind != tensor::DeviceKind::CUDA && device().kind != tensor::DeviceKind::A_GPU)
        throw Failure({ErrorCode::UNSUPPORTED, "short convolution requires CPU, CUDA or Metal"});
    const std::array<std::int64_t, 1> attributes{static_cast<std::int64_t>(weight.size(0))};
    if (taps.defined())
        return impl_->run({Operation::SHORT_CONVOLUTION, attributes, padded.dtype(), scale}, {&padded, &weight, &taps});
    return impl_->run({Operation::SHORT_CONVOLUTION, attributes, padded.dtype(), scale}, {&padded, &weight});
}
auto Context::short_convolution(const Tensor& input, const Tensor& state, const Tensor& weight, const Tensor& taps,
                                const Tensor& slot, float scale, const Tensor& residual,
                                std::optional<tensor::DType> output_dtype, bool update_state) -> Tensor {
    if (weight.dimensions() != 2 || !weight.size(0) || !weight.size(1) || !std::isfinite(scale))
        throw Failure({ErrorCode::INVALID_ARGUMENT, "short convolution requires nonempty [width, channels] taps"});
    const auto width = static_cast<std::int64_t>(weight.size(0));
    const auto channels = static_cast<std::int64_t>(weight.size(1));
    if (input.dimensions() != 2 || input.size(1) != channels || !input.size(0) || state.dimensions() != 2 ||
        state.size(1) != channels || state.size(0) < width - 1 || state.dtype() != input.dtype() ||
        slot.dtype() != tensor::DType::I32 || slot.numel() != 1)
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid ring short convolution input, state or slot"});
    const auto rows = static_cast<std::int64_t>(input.size(0));
    const auto ring = static_cast<std::int64_t>(state.size(0));
    if (update_state && (rows != 1 || ring < width))
        throw Failure(
            {ErrorCode::INVALID_ARGUMENT, "ring short convolution in-place update requires 1 token and ring >= width"});
    if (taps.defined() && (taps.dimensions() != 2 || taps.size(0) != rows || !taps.size(1) || taps.size(1) % width ||
                           channels % (taps.size(1) / width)))
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid short convolution dynamic taps"});
    if (residual.defined() && (residual.dimensions() != 2 || residual.size(0) != rows || residual.size(1) != channels ||
                               (residual.dtype() != tensor::DType::BF16 && residual.dtype() != tensor::DType::F32)))
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid short convolution residual tensor"});
    const auto out_dtype = output_dtype.value_or(input.dtype());
    if (device().kind == tensor::DeviceKind::CPU) {
        const auto input_values = cpu_floats(*this, input);
        const auto state_values = cpu_floats(*this, state);
        const auto weight_values = cpu_floats(*this, weight);
        const auto tap_values = taps.defined() ? cpu_floats(*this, taps) : HostFloats{};
        const auto res_values = residual.defined() ? cpu_floats(*this, residual) : HostFloats{};
        const auto slot_value = static_cast<std::int64_t>(require(slot.data<std::int32_t>())[0]);
        if (slot_value < 0 || slot_value >= ring)
            throw Failure({ErrorCode::INVALID_ARGUMENT, "short convolution slot outside the ring"});
        const auto groups = taps.defined() ? static_cast<std::int64_t>(taps.size(1)) / width : 1;
        std::vector<const float*> sources(static_cast<std::size_t>(rows * width));
        for (std::int64_t token = 0; token < rows; ++token)
            for (std::int64_t lag = 0; lag < width; ++lag) {
                const auto prior = token - lag;
                sources[static_cast<std::size_t>(token * width + lag)] =
                    prior >= 0 ? input_values.data() + prior * channels
                               : state_values.data() + (((slot_value + prior) % ring + ring) % ring) * channels;
            }
        auto result = cpu_output(*this, {rows, channels}, out_dtype, [&](float* output) {
            return cpu::short_convolution(
                sources.data(), input_values.data(), weight_values.data(), taps.defined() ? tap_values.data() : nullptr,
                residual.defined() ? res_values.data() : nullptr, {rows, channels, width, groups, scale}, output);
        });
        if (update_state) {
            auto ring_state = state;
            const auto target = require(ring_state.host_bytes());
            const auto source = require(input.host_bytes());
            std::memmove(target.data() + slot_value * source.size(), source.data(), source.size());
        }
        return result;
    }
    if (device().kind != tensor::DeviceKind::CUDA && device().kind != tensor::DeviceKind::A_GPU)
        throw Failure({ErrorCode::UNSUPPORTED, "short convolution requires CPU, CUDA or Metal"});
    if (residual.defined() || update_state) {
        const std::int64_t flags = (taps.defined() ? 1 : 0) | (residual.defined() ? 2 : 0) | (update_state ? 4 : 0);
        const std::array<std::int64_t, 3> attributes{width, ring, flags};
        if (taps.defined() && residual.defined())
            return impl_->run({Operation::SHORT_CONVOLUTION, attributes, out_dtype, scale},
                              {&input, &state, &weight, &slot, &taps, &residual});
        if (taps.defined())
            return impl_->run({Operation::SHORT_CONVOLUTION, attributes, out_dtype, scale},
                              {&input, &state, &weight, &slot, &taps});
        if (residual.defined())
            return impl_->run({Operation::SHORT_CONVOLUTION, attributes, out_dtype, scale},
                              {&input, &state, &weight, &slot, &residual});
        return impl_->run({Operation::SHORT_CONVOLUTION, attributes, out_dtype, scale},
                          {&input, &state, &weight, &slot});
    }
    const std::array<std::int64_t, 2> attributes{width, ring};
    if (taps.defined())
        return impl_->run({Operation::SHORT_CONVOLUTION, attributes, out_dtype, scale},
                          {&input, &state, &weight, &slot, &taps});
    return impl_->run({Operation::SHORT_CONVOLUTION, attributes, out_dtype, scale}, {&input, &state, &weight, &slot});
}
auto Context::l2_clip(const Tensor& input, const Tensor& scale, float threshold) -> Tensor {
    if (!std::isfinite(threshold) || threshold <= 0)
        throw Failure({ErrorCode::INVALID_ARGUMENT, "L2 clip requires a positive threshold"});
    if (device().kind == tensor::DeviceKind::CPU) {
        if (!input.dimensions() || input.size(-1) <= 0)
            throw Failure({ErrorCode::INVALID_ARGUMENT, "L2 clip requires a nonempty final dimension"});
        const auto width = static_cast<std::int64_t>(input.size(-1));
        const auto rows = static_cast<std::int64_t>(input.numel()) / width;
        const auto values = cpu_floats(*this, input);
        const auto scales = scale.defined() ? cpu_floats(*this, scale) : HostFloats{};
        if (scale.defined() && (scale.dimensions() != 2 || input.dimensions() < 3 || scale.size(0) != input.size(-2) ||
                                scale.size(1) != input.size(-1)))
            throw Failure({ErrorCode::INVALID_ARGUMENT, "L2 clip scale must match the input's final two dimensions"});
        return cpu_output(*this, std::vector(input.shape().begin(), input.shape().end()), input.dtype(),
                          [&](float* output) {
                              return cpu::l2_clip(values.data(), scales.empty() ? nullptr : scales.data(),
                                                  scale.defined() ? static_cast<std::int64_t>(scale.size(0)) : 1, rows,
                                                  width, threshold, output);
                          });
    }
    if (device().kind != tensor::DeviceKind::CUDA && device().kind != tensor::DeviceKind::A_GPU)
        throw Failure({ErrorCode::UNSUPPORTED, "L2 clip requires CPU, CUDA or Metal"});
    if (scale.defined()) return impl_->run({Operation::L2_CLIP, {}, input.dtype(), threshold}, {&input, &scale});
    return impl_->run({Operation::L2_CLIP, {}, input.dtype(), threshold}, {&input});
}
auto Context::relative_attention(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& relative,
                                 const Tensor& sink, const Tensor& gate, RelativeAttention shape,
                                 const Tensor& positions, const Tensor& key_scales, const Tensor& value_scales)
    -> Tensor {
    if (positions.defined() &&
        (positions.dtype() != tensor::DType::I32 || positions.dimensions() != 1 || positions.size(0) != query.size(0)))
        throw Failure({ErrorCode::INVALID_ARGUMENT, "relative attention positions must be I32 [tokens]"});
    if (!std::isfinite(shape.log_alpha) || !std::isfinite(shape.log_floor) || shape.log_floor < 0)
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid relative attention log scaling"});
    if (device().kind == tensor::DeviceKind::CPU) {
        if (query.dimensions() != 2 || key.dimensions() != 2 || !std::ranges::equal(value.shape(), key.shape()) ||
            relative.dimensions() != 3 || sink.dimensions() != 2 || !std::ranges::equal(gate.shape(), query.shape()) ||
            shape.heads <= 0 || shape.key_heads <= 0 || shape.heads % shape.key_heads || shape.start < 0 ||
            shape.window < 0 || shape.extent < 0 || query.size(1) % shape.heads || relative.size(0) != query.size(0) ||
            relative.size(1) != shape.heads || relative.size(2) != shape.extent || sink.size(0) != query.size(0) ||
            sink.size(1) != shape.heads || key.size(1) != query.size(1) / shape.heads * shape.key_heads ||
            shape.start + query.size(0) > key.size(0))
            throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid relative attention shapes"});
        const auto queries = cpu_floats(*this, query), offsets = cpu_floats(*this, relative);
        const auto sinks = cpu_floats(*this, sink), gates = cpu_floats(*this, gate);
        const auto tokens = static_cast<std::int64_t>(query.size(0));
        std::vector<std::int32_t> given(static_cast<std::size_t>(tokens));
        if (positions.defined()) {
            const auto host = require(positions.to(tensor::Device::cpu()));
            const auto data = require(host.data<std::int32_t>());
            given.assign(data.begin(), data.end());
        } else {
            std::iota(given.begin(), given.end(), static_cast<std::int32_t>(shape.start));
        }
        if (std::ranges::any_of(given, [&](std::int32_t position) {
                return position < 0 || position >= static_cast<std::int64_t>(key.size(0));
            }))
            throw Failure({ErrorCode::INVALID_ARGUMENT, "relative attention position outside the cache"});
        const auto rows = [&](const Tensor& cache, const Tensor& scales) {
            if (!cache.is_contiguous() || !cache.is_host_accessible() ||
                (cache.dtype() != tensor::DType::F32 && cache.dtype() != tensor::DType::BF16 &&
                 cache.dtype() != tensor::DType::I8))
                throw Failure(
                    {ErrorCode::INVALID_ARGUMENT, "CPU relative attention reads contiguous F32, BF16 or I8 caches"});
            if (cache.dtype() == tensor::DType::I8) {
                if (!scales.defined() || scales.dtype() != tensor::DType::F32 || !scales.is_contiguous() ||
                    !scales.is_host_accessible() || scales.dimensions() != 2 || scales.size(0) != cache.size(0) ||
                    scales.size(1) == 0 || scales.size(1) % shape.key_heads || cache.size(1) % scales.size(1))
                    throw Failure({ErrorCode::INVALID_ARGUMENT, "INT8 attention needs F32 per-group cache scales"});
                return cpu::Rows{require(cache.host_bytes()).data(), cpu::Rows::Storage::I8,
                                 static_cast<std::int64_t>(cache.size(1)), require(scales.data<float>()).data(),
                                 static_cast<std::int64_t>(cache.size(1) / scales.size(1))};
            }
            if (scales.defined())
                throw Failure({ErrorCode::INVALID_ARGUMENT, "attention scales require an INT8 cache"});
            return cpu::Rows{require(cache.host_bytes()).data(),
                             cache.dtype() == tensor::DType::F32 ? cpu::Rows::Storage::F32 : cpu::Rows::Storage::BF16,
                             static_cast<std::int64_t>(cache.size(1))};
        };
        std::vector<float> output(static_cast<std::size_t>(query.numel()));
        require(cpu::relative_attention(
            queries.data(), rows(key, key_scales), rows(value, value_scales), offsets.data(), sinks.data(),
            gates.data(), given,
            {tokens, shape.heads, shape.key_heads, static_cast<std::int64_t>(query.size(1)) / shape.heads, shape.extent,
             shape.window, shape.log_alpha, shape.log_floor},
            output.data()));
        return cpu_tensor(*this, std::vector(query.shape().begin(), query.shape().end()), output, query.dtype());
    }
    if ((key_scales.defined() || value_scales.defined()) && device().kind != tensor::DeviceKind::A_GPU)
        throw Failure({ErrorCode::UNSUPPORTED, "scaled relative attention caches require CPU or Metal"});
    if (device().kind != tensor::DeviceKind::CUDA && device().kind != tensor::DeviceKind::A_GPU)
        throw Failure({ErrorCode::UNSUPPORTED, "relative attention requires CPU, CUDA or Metal"});
    const bool ring_mode = shape.ring_mode && shape.window > 0 && device().kind == tensor::DeviceKind::A_GPU;
    const std::array<std::int64_t, 8> attributes{shape.heads,
                                                 shape.key_heads,
                                                 shape.start,
                                                 shape.window,
                                                 shape.extent,
                                                 std::bit_cast<std::int32_t>(shape.log_alpha),
                                                 std::bit_cast<std::int32_t>(shape.log_floor),
                                                 ring_mode ? 1 : 0};
    const auto attr_span =
        ring_mode ? std::span<const std::int64_t>(attributes) : std::span<const std::int64_t>(attributes).first(7);
    if (key_scales.defined() || value_scales.defined()) {
        if (!positions.defined() || !key_scales.defined() || !value_scales.defined())
            throw Failure({ErrorCode::INVALID_ARGUMENT,
                           "Metal INT8 relative attention requires positions, key_scales, and value_scales"});
        return impl_->run({Operation::RELATIVE_ATTENTION, attr_span, query.dtype()},
                          {&query, &key, &value, &relative, &sink, &gate, &positions, &key_scales, &value_scales});
    }
    if (positions.defined())
        return impl_->run({Operation::RELATIVE_ATTENTION, attr_span, query.dtype()},
                          {&query, &key, &value, &relative, &sink, &gate, &positions});
    return impl_->run({Operation::RELATIVE_ATTENTION, attr_span, query.dtype()},
                      {&query, &key, &value, &relative, &sink, &gate});
}
auto Context::quantize_cache_(Tensor& cache, Tensor& scales, const Tensor& input, const Tensor& positions) -> void {
    if (device().kind == tensor::DeviceKind::A_GPU) {
        if (input.dimensions() != 3 || cache.dimensions() != 3 || cache.size(0) != 1 || input.size(1) == 0 ||
            input.size(2) == 0 || cache.size(2) != input.size(1) * input.size(2) ||
            cache.dtype() != tensor::DType::I8 || scales.dimensions() != 2 || scales.size(0) != cache.size(1) ||
            scales.size(1) == 0 || cache.size(2) != scales.size(1) * 32 || scales.dtype() != tensor::DType::F32 ||
            positions.dtype() != tensor::DType::I32 || positions.dimensions() != 1 ||
            positions.size(0) != input.size(0))
            throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid Metal INT8 cache update"});
        auto flat = require(
            input.reshape({static_cast<std::int64_t>(input.size(0)), static_cast<std::int64_t>(cache.size(2))}));
        impl_->run({Operation::QUANTIZE_CACHE, {}, tensor::DType::F32}, {&flat, &positions, &cache, &scales});
        return;
    }
    if (device().kind != tensor::DeviceKind::CPU || input.dimensions() != 3 || cache.dimensions() != 3 ||
        cache.size(0) != 1 || input.size(1) == 0 || input.size(2) == 0 ||
        cache.size(2) != input.size(1) * input.size(2) || cache.dtype() != tensor::DType::I8 ||
        !cache.is_contiguous() || !cache.is_host_accessible() || scales.dimensions() != 2 ||
        scales.size(0) != cache.size(1) || scales.size(1) == 0 || scales.size(1) % input.size(1) ||
        cache.size(2) % scales.size(1) || scales.dtype() != tensor::DType::F32 || !scales.is_contiguous() ||
        !scales.is_host_accessible() || positions.dtype() != tensor::DType::I32 || positions.dimensions() != 1 ||
        positions.size(0) != input.size(0) || !positions.is_contiguous() || !positions.is_host_accessible())
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid CPU INT8 cache update"});
    const auto indices = require(positions.data<std::int32_t>());
    if (std::ranges::any_of(indices,
                            [&](auto index) { return index < 0 || static_cast<std::size_t>(index) >= cache.size(1); }))
        throw Failure({ErrorCode::INVALID_ARGUMENT, "INT8 cache position outside capacity"});
    const auto values = cpu_floats(*this, input);
    auto codes = require(cache.data<std::int8_t>());
    auto factors = require(scales.data<float>());
    const auto heads = static_cast<std::int64_t>(input.size(1));
    const auto width = static_cast<std::int64_t>(input.size(2));
    const auto groups = static_cast<std::int64_t>(scales.size(1));
    const auto group_width = heads * width / groups;
    // Serial writes also support repeated positions deterministically (the last update wins).
    for (std::size_t t = 0; t < indices.size(); ++t) {
        const auto position = static_cast<std::int64_t>(indices[t]);
        cpu::quantize_rows(values.data() + t * heads * width, 1, heads * width, group_width,
                           codes.data() + position * heads * width, factors.data() + position * groups);
    }
}
auto Context::tanh(const Tensor& input) -> Tensor { return impl_->run({Operation::TANH, {}, input.dtype()}, {&input}); }
auto Context::static_round(const Tensor& input, float scale) -> Tensor {
    if (!input.defined() || (input.dtype() != tensor::DType::F32 && input.dtype() != impl_->activation_dtype) ||
        input.device() != device() || !std::isfinite(scale) || scale < 0)
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid static activation rounding"});
    if (scale == 0) return input;
    return impl_->run({Operation::STATIC_ROUND, {}, input.dtype(), scale}, {&input});
}
auto Context::rotary(const Tensor& input, const Tensor& cosine, const Tensor& sine) -> Tensor {
    return impl_->run({Operation::ROTARY, {}, input.dtype()}, {&input, &cosine, &sine});
}
auto Context::grouped_query_attention(const Tensor& query, const Tensor& key, const Tensor& value, std::int32_t heads,
                                      std::int32_t key_value_heads, const Tensor& mask, float scale,
                                      std::int64_t key_start, const BlockwiseQuantization& key_quantization,
                                      const BlockwiseQuantization& value_quantization) -> Tensor {
    if (!std::isfinite(scale) || scale <= 0 || !mask.defined() || !mask.dimensions() || key.dimensions() != 3 ||
        key_start < 0 || static_cast<std::size_t>(key_start) > key.size(1) || !mask.size(-1) ||
        mask.size(-1) > key.size(1) - key_start)
        throw Failure({ErrorCode::INVALID_ARGUMENT, "grouped attention requires a mask and positive scale"});
    validate_attention_quantization(key, device(), key_quantization, value_quantization);
    const auto key_length = !key_start && mask.size(-1) == 1 ? key.size(1) : mask.size(-1);
    const auto parity_math = impl_->precision == core::InferencePrecision::QAT_FP32;
    const std::array<std::int64_t, 5> attributes{heads, key_value_heads, key_start,
                                                 static_cast<std::int64_t>(key_length), parity_math};
    const auto attribute_count = parity_math ? attributes.size() : attributes.size() - 1;
    OperatorSpec spec{Operation::ATTENTION, std::span(attributes).first(attribute_count), query.dtype(), scale};
    spec.quantization = {{{key_quantization.scales, key_quantization.zero_points, key_quantization.block_size},
                          {value_quantization.scales, value_quantization.zero_points, value_quantization.block_size}}};
    return impl_->run(spec, {&query, &key, &value, &mask});
}
auto Context::grouped_query_attention_causal(const Tensor& query, const Tensor& key, const Tensor& value,
                                             std::int32_t heads, std::int32_t key_value_heads,
                                             const Tensor& query_positions, std::int64_t window, float scale,
                                             std::int64_t key_start, const BlockwiseQuantization& key_quantization,
                                             const BlockwiseQuantization& value_quantization) -> Tensor {
    if (device().kind != tensor::DeviceKind::CUDA || !std::isfinite(scale) || scale <= 0 || window < 0 ||
        key.dimensions() != 3 || key_start < 0 || static_cast<std::size_t>(key_start) >= key.size(1) ||
        query_positions.dtype() != tensor::DType::I32 || query_positions.device() != device() ||
        query_positions.dimensions() != 4 || query_positions.size(0) != 1 || query_positions.size(1) != 1 ||
        query_positions.size(2) != query.size(1) || query_positions.size(3) != 1)
        throw Failure({ErrorCode::INVALID_ARGUMENT, "causal attention requires CUDA query positions and a positive scale"});
    validate_attention_quantization(key, device(), key_quantization, value_quantization);
    const auto parity_math = impl_->precision == core::InferencePrecision::QAT_FP32;
    const std::array<std::int64_t, 7> attributes{
        heads, key_value_heads, key_start, static_cast<std::int64_t>(key.size(1)) - key_start, parity_math, 1, window};
    OperatorSpec spec{Operation::ATTENTION, attributes, query.dtype(), scale};
    spec.quantization = {{{key_quantization.scales, key_quantization.zero_points, key_quantization.block_size},
                          {value_quantization.scales, value_quantization.zero_points, value_quantization.block_size}}};
    return impl_->run(spec, {&query, &key, &value, &query_positions});
}
auto Context::blend_prologue(const Tensor& destination, const Tensor& message, float epsilon) -> Tensor {
    if (destination.dimensions() != 2 || !destination.size(0) || !destination.size(1) ||
        !std::ranges::equal(destination.shape(), message.shape()) || !std::isfinite(epsilon) || epsilon <= 0)
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid blend prologue operands"});
    const auto rows = static_cast<std::int64_t>(destination.size(0));
    const auto width = static_cast<std::int64_t>(destination.size(1));
    if (device().kind == tensor::DeviceKind::CPU) {
        const auto d = cpu_floats(*this, destination);
        const auto m = cpu_floats(*this, message);
        std::vector<float> out(static_cast<std::size_t>(rows * 2 * width));
        require(cpu::parallel_rows(rows, 2 * width, [&](std::int64_t first, std::int64_t last) {
            for (auto t = first; t < last; ++t) {
                float sd = 0, sm = 0;
                for (std::int64_t c = 0; c < width; ++c) {
                    const float dv = d[static_cast<std::size_t>(t * width + c)];
                    const float mv = m[static_cast<std::size_t>(t * width + c)];
                    sd += dv * dv;
                    sm += mv * mv;
                }
                const float id = 1.F / std::sqrt(sd / static_cast<float>(width) + epsilon);
                const float im = 1.F / std::sqrt(sm / static_cast<float>(width) + epsilon);
                for (std::int64_t c = 0; c < width; ++c) {
                    out[static_cast<std::size_t>(t * 2 * width + c)] = d[static_cast<std::size_t>(t * width + c)] * id;
                    out[static_cast<std::size_t>(t * 2 * width + width + c)] =
                        m[static_cast<std::size_t>(t * width + c)] * im;
                }
            }
        }));
        return cpu_tensor(*this, {rows, 2 * width}, out, tensor::DType::BF16);
    }
    if (device().kind != tensor::DeviceKind::CUDA && device().kind != tensor::DeviceKind::A_GPU)
        throw Failure({ErrorCode::UNSUPPORTED, "blend prologue requires CPU, CUDA or Metal"});
    const auto dtype = device().kind == tensor::DeviceKind::A_GPU ? tensor::DType::F32 : tensor::DType::BF16;
    return impl_->run({Operation::BLEND_PROLOGUE, {}, dtype, epsilon}, {&destination, &message});
}
auto Context::blend_epilogue(const Tensor& destination, const Tensor& message, const Tensor& gain_forget,
                             const Tensor& norm, float epsilon) -> Tensor {
    if (destination.dimensions() != 2 || !destination.size(0) || !destination.size(1) ||
        !std::ranges::equal(destination.shape(), message.shape()) || gain_forget.dimensions() != 2 ||
        gain_forget.size(0) != destination.size(0) || gain_forget.size(1) != 2 * destination.size(1) ||
        norm.dimensions() != 1 || norm.size(0) != destination.size(1) || norm.dtype() != tensor::DType::F32 ||
        !std::isfinite(epsilon) || epsilon <= 0)
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid blend epilogue operands"});
    const auto rows = static_cast<std::int64_t>(destination.size(0));
    const auto width = static_cast<std::int64_t>(destination.size(1));
    if (device().kind == tensor::DeviceKind::CPU) {
        const auto d = cpu_floats(*this, destination);
        const auto m = cpu_floats(*this, message);
        const auto gf = cpu_floats(*this, gain_forget);
        const auto n = cpu_floats(*this, norm);
        return cpu_output(*this, {rows, width}, tensor::DType::BF16, [&](float* output) {
            return cpu::blend_epilogue(d.data(), m.data(), gf.data(), n.data(), rows, width, epsilon, output);
        });
    }
    if (device().kind != tensor::DeviceKind::CUDA && device().kind != tensor::DeviceKind::A_GPU)
        throw Failure({ErrorCode::UNSUPPORTED, "blend epilogue requires CPU, CUDA or Metal"});
    const auto dtype = device().kind == tensor::DeviceKind::A_GPU ? tensor::DType::F32 : tensor::DType::BF16;
    return impl_->run({Operation::BLEND_EPILOGUE, {}, dtype, epsilon}, {&destination, &message, &gain_forget, &norm});
}
auto Context::rms_norm(const Tensor& input, const Tensor& scale, float epsilon,
                       std::optional<tensor::DType> output_dtype) -> Tensor {
    Tensor implicit_scale;
    if (!scale.defined()) {
        if (!input.dimensions() || !input.size(-1))
            throw Failure({ErrorCode::INVALID_ARGUMENT, "unweighted RMS norm requires a nonempty final dimension"});
        const std::vector<float> ones(input.size(-1), 1.F);
        implicit_scale = require(
            Tensor::from_host({static_cast<std::int64_t>(input.size(-1))}, std::span<const float>(ones), device()));
    }
    const auto& weight = scale.defined() ? scale : implicit_scale;
    const auto dtype = output_dtype.value_or(input.dtype());
    if (impl_->precision == core::InferencePrecision::QAT_FP32) {
        const std::array<std::int64_t, 1> attributes{1};
        return impl_->run({Operation::RMS_NORM, attributes, tensor::DType::F32, epsilon}, {&input, &weight});
    }
    if (device().kind == tensor::DeviceKind::CPU &&
        (input.dtype() == tensor::DType::BF16 || (input.dtype() == tensor::DType::F32 && dtype == tensor::DType::BF16)))
        return cast(rms_norm(cast(input, tensor::DType::F32), weight, epsilon, tensor::DType::F32), dtype);
    return impl_->run({Operation::RMS_NORM, {}, dtype, epsilon}, {&input, &weight});
}
auto Context::rms_rotary(const Tensor& input, const Tensor& scale, const Tensor& cosine, const Tensor& sine,
                         float epsilon) -> Tensor {
    if (input.dimensions() != 4 || !input.numel() ||
        (input.dtype() != tensor::DType::F32 && input.dtype() != impl_->activation_dtype) || input.size(3) % 2 ||
        scale.dimensions() != 1 || scale.size(0) != input.size(3) || scale.dtype() != tensor::DType::F32 ||
        !std::isfinite(epsilon) || epsilon <= 0)
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid RMS rotary operands"});
    for (const auto* angle : {&cosine, &sine})
        if (angle->dimensions() != 4 || angle->size(0) != 1 || angle->size(1) != input.size(1) || angle->size(2) != 1 ||
            angle->size(3) != input.size(3) / 2 || angle->dtype() != tensor::DType::F32)
            throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid RMS rotary angles"});
    if (impl_->compiler && !impl_->compiler->supports_step_operation(Operation::RMS_ROTARY))
        return rotary(rms_norm(input, scale, epsilon), cosine, sine);
    return impl_->run({Operation::RMS_ROTARY, {}, input.dtype(), epsilon}, {&input, &scale, &cosine, &sine});
}
auto Context::rms_rotary_slice(const Tensor& input, const Tensor& scale, const Tensor& cosine, const Tensor& sine,
                               float epsilon, std::int64_t heads, std::int64_t width, std::int64_t channel_offset,
                               std::int64_t sequence_offset, std::int64_t sequence_length) -> Tensor {
    if (input.dimensions() != 3 || input.dtype() != tensor::DType::F32 || heads <= 0 || width <= 0 || width % 2 ||
        channel_offset < 0 || sequence_offset < 0 || sequence_length <= 0 ||
        channel_offset + heads * width > input.size(2) || sequence_offset + sequence_length > input.size(1) ||
        scale.dimensions() != 1 || scale.size(0) != width || scale.dtype() != tensor::DType::F32 ||
        !std::isfinite(epsilon) || epsilon <= 0)
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid sliced RMS rotary operands"});
    for (const auto* angle : {&cosine, &sine})
        if (angle->dimensions() != 4 || angle->size(0) != 1 || angle->size(1) != sequence_length ||
            angle->size(2) != 1 || angle->size(3) != width / 2 || angle->dtype() != tensor::DType::F32)
            throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid sliced RMS rotary angles"});
    const std::array attributes{heads, width, channel_offset, sequence_offset, sequence_length};
    return impl_->run({Operation::RMS_ROTARY, attributes, tensor::DType::F32, epsilon},
                      {&input, &scale, &cosine, &sine});
}
auto Context::rms_axial_rotary(const Tensor& input, const Tensor& scale, const Tensor& cosine, const Tensor& sine,
                               float epsilon) -> Tensor {
    if (input.dimensions() != 4 || !input.numel() || input.dtype() != tensor::DType::F32 || input.size(3) % 4 ||
        scale.dimensions() != 1 || scale.size(0) != input.size(3) || scale.dtype() != tensor::DType::F32 ||
        !std::isfinite(epsilon) || epsilon <= 0)
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid axial RMS rotary operands"});
    for (const auto* angle : {&cosine, &sine})
        if (angle->dimensions() != 4 || angle->size(0) != 2 || angle->size(1) != input.size(1) || angle->size(2) != 1 ||
            angle->size(3) != input.size(3) / 4 || angle->dtype() != tensor::DType::F32)
            throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid axial RMS rotary angles"});
    if (!device().capabilities().attention.fused()) {
        const auto normalized = rms_norm(input, scale, epsilon);
        std::array<Tensor, 2> axes;
        for (int axis = 0; axis < 2; ++axis)
            axes[axis] = rotary(slice(normalized, 3, axis * input.size(3) / 2, input.size(3) / 2),
                                slice(cosine, 0, axis, 1), slice(sine, 0, axis, 1));
        return concat(axes, 3);
    }
    return impl_->run({Operation::RMS_ROTARY, {}, tensor::DType::F32, epsilon}, {&input, &scale, &cosine, &sine});
}
auto Context::rms_norm_residual(const Tensor& input, const Tensor& scale, const Tensor& residual, float epsilon,
                                const Tensor& output_scale) -> Tensor {
    if (!input.defined() || !input.dimensions() ||
        (input.dtype() != tensor::DType::F32 && input.dtype() != impl_->activation_dtype) || !scale.defined() ||
        scale.dimensions() != 1 || scale.size(0) != input.size(-1) || scale.dtype() != tensor::DType::F32 ||
        !residual.defined() || residual.dtype() != input.dtype() ||
        !std::ranges::equal(input.shape(), residual.shape()) || !std::isfinite(epsilon) || epsilon <= 0 ||
        (output_scale.defined() && (output_scale.dtype() != tensor::DType::F32 || output_scale.numel() != 1)))
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid RMS residual normalization operands"});
    if (impl_->precision == core::InferencePrecision::QAT_FP32) {
        const std::array<std::int64_t, 1> attributes{1};
        const OperatorSpec spec{Operation::RMS_NORM_RESIDUAL, attributes, tensor::DType::F32, epsilon};
        if (output_scale.defined()) return impl_->run(spec, {&input, &scale, &residual, &output_scale});
        return impl_->run(spec, {&input, &scale, &residual});
    }
    const OperatorSpec spec{Operation::RMS_NORM_RESIDUAL, {}, input.dtype(), epsilon};
    if (output_scale.defined()) return impl_->run(spec, {&input, &scale, &residual, &output_scale});
    return impl_->run(spec, {&input, &scale, &residual});
}
auto Context::layer_norm(const Tensor& input, const Tensor& scale, const Tensor& bias, float epsilon) -> Tensor {
    return impl_->run({Operation::LAYER_NORM, {}, tensor::DType::F32, epsilon}, {&input, &scale, &bias});
}
auto Context::layer_norm_(Tensor& input, const Tensor& scale, const Tensor& bias, float epsilon) -> Tensor& {
    impl_->run({Operation::LAYER_NORM, {}, tensor::DType::F32, epsilon}, {&input, &scale, &bias}, nullptr, &input);
    return input;
}
auto Context::residual_layer_norm(const Tensor& input, const Tensor& residual, const Tensor& scale, const Tensor& bias,
                                  float epsilon) -> std::array<Tensor, 2> {
    Tensor sum;
    auto normalized = impl_->run({Operation::RESIDUAL_NORM, {}, tensor::DType::F32, epsilon},
                                 {&input, &residual, &scale, &bias}, &sum);
    return {std::move(sum), std::move(normalized)};
}
auto Context::softmax(const Tensor& input, bool logarithmic) -> Tensor {
    return impl_->run({logarithmic ? Operation::LOG_SOFTMAX : Operation::SOFTMAX}, {&input});
}
auto Context::softmax_(Tensor& input, bool logarithmic) -> Tensor& {
    impl_->run({logarithmic ? Operation::LOG_SOFTMAX : Operation::SOFTMAX}, {&input}, nullptr, &input);
    return input;
}
auto Context::transpose(const Tensor& input, std::span<const std::int64_t> axes) -> Tensor {
    return impl_->run({Operation::TRANSPOSE, axes, input.dtype()}, {&input});
}
auto Context::slice(const Tensor& input, std::int64_t axis, std::int64_t start, std::int64_t length) -> Tensor {
    if (input.device() != device()) throw Failure({ErrorCode::INVALID_ARGUMENT, "slice operand device mismatch"});
    auto view = require(input.narrow(axis, start, length));
    if (view.is_contiguous() && (device().capabilities().memory.offset_views || view.storage_offset() == 0))
        return view;
    const std::array attributes{axis, start, length};
    return impl_->run({Operation::SLICE, attributes, input.dtype()}, {&input});
}
auto Context::gather(const Tensor& input, const Tensor& indices, std::int64_t axis) -> Tensor {
    const std::array attributes{axis};
    return impl_->run({Operation::GATHER, attributes, input.dtype()}, {&input, &indices});
}
auto Context::concat(std::span<const Tensor> inputs, std::int64_t axis) -> Tensor {
    if (inputs.empty()) throw Failure({ErrorCode::INVALID_ARGUMENT, "empty concatenation"});
    const std::array attributes{axis};
    return impl_->run({Operation::CONCAT, attributes, inputs.front().dtype()}, inputs);
}
auto Context::reshape(const Tensor& input, std::vector<std::int64_t> shape) -> Tensor {
    return require(input.reshape(std::move(shape)));
}
auto Context::scatter(const Tensor& input, const Tensor& updates, const Tensor& indices) -> Tensor {
    return impl_->run({Operation::SCATTER, {}, input.dtype()}, {&input, &updates, &indices});
}
auto Context::scatter_(Tensor& input, const Tensor& updates, const Tensor& indices) -> Tensor& {
    impl_->run({Operation::SCATTER, {}, input.dtype()}, {&input, &updates, &indices}, nullptr, &input);
    return input;
}
auto Context::scatter_unique_(Tensor& input, const Tensor& updates, const Tensor& indices) -> Tensor& {
    if (impl_->device.kind == tensor::DeviceKind::CPU || impl_->device.kind == tensor::DeviceKind::A_GPU)
        return scatter_(input, updates, indices);
    if (impl_->device.kind != tensor::DeviceKind::CUDA)
        throw Failure({ErrorCode::INVALID_ARGUMENT, "unique-index scatter requires CPU or CUDA"});
    const std::array<std::int64_t, 1> attributes{1};
    impl_->run({Operation::SCATTER, attributes, input.dtype()}, {&input, &updates, &indices}, nullptr, &input);
    return input;
}
auto Context::copy_slice_(Tensor& destination, const Tensor& source, std::int64_t axis, std::int64_t start) -> Tensor& {
    if (impl_->replays.recorder())
        throw Failure({ErrorCode::INVALID_ARGUMENT,
                       "copy_slice_ bakes its offset into a captured step; use scatter_ with an index input"});
    const InplaceScope mode(true);
    const auto rank = static_cast<std::int64_t>(destination.dimensions());
    if (axis < 0) axis += rank;
    if (!destination.defined() || !source.defined() || source.dimensions() != destination.dimensions() ||
        !destination.is_contiguous() || !source.is_contiguous() || destination.dtype() != source.dtype() ||
        destination.device() != device() || source.device() != device() || axis < 0 || axis >= rank || start < 0)
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid contiguous slice copy operands"});
    if (static_cast<std::size_t>(start) > destination.size(axis) ||
        source.size(axis) > destination.size(axis) - static_cast<std::size_t>(start))
        throw Failure({ErrorCode::INVALID_ARGUMENT, "slice copy outside destination"});
    for (std::int64_t dimension = 0; dimension < rank; ++dimension)
        if (dimension != axis && destination.size(dimension) != source.size(dimension))
            throw Failure({ErrorCode::INVALID_ARGUMENT, "slice copy shape mismatch"});
    if (destination.is_host_accessible()) require(destination.host_bytes());
    if (!source.numel()) return destination;
    std::size_t outer = 1, inner = tensor::element_size(source.dtype());
    for (std::int64_t dimension = 0; dimension < axis; ++dimension) outer *= source.size(dimension);
    for (std::int64_t dimension = axis + 1; dimension < rank; ++dimension) inner *= source.size(dimension);
    const auto entered = impl_->profiling ? Clock::now() : Clock::time_point{};
    impl_->backend->copy_slice_(destination, source, outer, source.size(axis) * inner, destination.size(axis) * inner,
                                static_cast<std::size_t>(start) * inner);
    if (impl_->profiling) {
        const auto run = nanoseconds(entered);
        const auto waiting = Clock::now();
        if (impl_->synchronize_operators) impl_->backend->synchronize();
        const auto wait = impl_->synchronize_operators ? nanoseconds(waiting) : 0;
        std::ostringstream label;
        label << "phase=" << impl_->phase << "|op=copy_slice_|shapes=[";
        for (auto extent : source.shape()) label << extent << ',';
        label << "]|axis=" << axis;
        auto& profile = impl_->profiles[label.str()];
        ++profile.calls;
        profile.run_ns += run;
        profile.wait_ns += wait;
        profile.output_bytes += source.nbytes();
    }
    return destination;
}
} // namespace kidi::ops