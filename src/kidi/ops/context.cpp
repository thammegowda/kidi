#include "kidi/ops/context.h"
#include "kidi/runtime/operator.h"
#include "kidi/ops/quantization.h"
#include <array>
#include <bit>
#include <chrono>
#include <map>
#include <list>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <utility>

namespace kidi::ops {
thread_local bool is_inplace = false;

using runtime::Operation;
using runtime::OperatorSpec;
using runtime::TensorInputs;
namespace {
thread_local bool decode_projections = false;
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
auto operation_name(Operation operation) -> std::string_view {
    constexpr std::array names{"add",           "multiply",
                               "cast",          "matmul",
                               "linear",        "quantized_linear",
                               "gelu",          "layer_norm",
                               "softmax",       "log_softmax",
                               "transpose",     "slice",
                               "gather",        "concat",
                               "scatter",       "attention",
                               "residual_norm", "rms_norm",
                               "tanh",          "rotary",
                               "packed_linear", "rms_norm_residual",
                               "static_round",  "greedy_token",
                               "rms_rotary",    "gelu_multiply"};
    return names.at(static_cast<std::size_t>(operation));
}
struct OperatorProfile {
    std::uint64_t calls = 0, prepare_ns = 0, dispatch_ns = 0, run_ns = 0, wait_ns = 0;
    std::uint64_t allocations = 0, allocated_bytes = 0, output_bytes = 0;
};
} // namespace
DecodeScope::DecodeScope(bool enabled) : previous_(std::exchange(decode_projections, enabled)) {}
DecodeScope::~DecodeScope() { decode_projections = previous_; }
struct Context::Impl {
    tensor::Device device;
    bool packed_prefill = false;
    std::unique_ptr<runtime::OperatorBackend> backend;
    using DispatchKey = std::vector<std::int64_t>;
    std::list<const DispatchKey*> recent;
    struct Prepared {
        std::unique_ptr<runtime::Operator> operation;
        std::vector<Tensor> parameters;
        std::list<const DispatchKey*>::iterator recency;
    };
    std::map<DispatchKey, Prepared> operators;
    std::vector<std::int64_t> dispatch_key;
    std::size_t operator_capacity = 4096;
    struct PackedBinding {
        Tensor source;
        PackedWeight packed;
        bool packed_prefill;
    };
    std::map<const std::byte*, PackedBinding> packed_weights;
    std::uint64_t preparation = 0;
    bool profiling = false;
    bool profile_requested = false;
    bool synchronize_operators = false;
    std::size_t skip_requests = 0, requests = 0;
    std::string phase = "unspecified";
    std::map<std::string, OperatorProfile> profiles;
    std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> synchronization;
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
        spec.packed_prefill = packed_prefill;
        spec.vector_projection = decode_projections && spec.operation == Operation::PACKED_LINEAR &&
                                 device == tensor::Device::apple_gpu() && inputs[0].dimensions() > 0 &&
                                 inputs[0].size(-1) && inputs[0].numel() / inputs[0].size(-1) >= 4;
        if (is_inplace) require(destination->host_bytes());
        const auto entered = profiling ? Clock::now() : Clock::time_point{};
        const auto preparation_before = preparation;
        auto& key = dispatch_key;
        key.clear();
        key.insert(key.end(), {static_cast<int>(spec.operation), static_cast<int>(spec.dtype),
                               std::bit_cast<std::int32_t>(spec.epsilon), is_inplace});
        key.push_back(spec.vector_projection);
        key.push_back(spec.attributes.size());
        key.insert(key.end(), spec.attributes.begin(), spec.attributes.end());
        const bool constant_parameters =
            spec.operation == Operation::LINEAR || spec.operation == Operation::QUANTIZED_LINEAR ||
            spec.operation == Operation::LAYER_NORM || spec.operation == Operation::RESIDUAL_NORM ||
            spec.operation == Operation::RMS_NORM || spec.operation == Operation::PACKED_LINEAR;
        const std::size_t parameter_start = spec.operation == Operation::RESIDUAL_NORM ? 2 : 1;
        spec.dynamic_parameters = spec.operation == Operation::RMS_NORM ||
                                  (spec.operation == Operation::LINEAR && device == tensor::Device::apple_gpu());
        for (std::size_t index = parameter_start; index < inputs.size(); ++index)
            spec.dynamic_parameters = spec.dynamic_parameters && inputs[index].device() == device;
        key.push_back(spec.dynamic_parameters);
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            const auto& input = inputs[index];
            if (!input.defined() || !input.is_contiguous())
                throw Failure({ErrorCode::INVALID_ARGUMENT, "eager operations require defined contiguous tensors"});
            if ((!constant_parameters || index < parameter_start) && input.device() != device)
                throw Failure({ErrorCode::INVALID_ARGUMENT, "eager operand device mismatch"});
            key.push_back(static_cast<int>(input.dtype()));
            key.push_back(input.dimensions());
            key.insert(key.end(), input.shape().begin(), input.shape().end());
            if (constant_parameters && !spec.dynamic_parameters && index >= parameter_start)
                key.push_back(reinterpret_cast<std::intptr_t>(require(input.host_bytes()).data()));
        }
        const auto invalid = [](bool condition, const char* message) {
            if (condition) throw Failure({ErrorCode::INVALID_ARGUMENT, message});
        };
        const auto& input = inputs.front();
        if (spec.operation == Operation::GREEDY_TOKEN)
            invalid(input.dtype() != tensor::DType::F32 || input.dimensions() == 0 || !input.numel() ||
                        input.size(-1) > INT32_MAX || input.numel() > UINT32_MAX,
                    "greedy selection expects nonempty FP32 rows within indexing bounds");
        if (spec.operation == Operation::PACKED_LINEAR) {
            const auto bits = spec.attributes[0], group = spec.attributes[1];
            invalid((bits != 2 && bits != 4 && bits != 8) || group <= 0, "invalid packed linear precision or group");
            invalid(input.dtype() != tensor::DType::F32 || input.dimensions() < 2 || !input.size(-1) ||
                        input.size(-1) % group || group % (8 / bits) || inputs[1].dtype() != tensor::DType::U8 ||
                        inputs[1].dimensions() != 2 || !inputs[1].size(0) ||
                        inputs[1].size(1) != input.size(-1) / (8 / bits) || inputs[2].dtype() != tensor::DType::F32 ||
                        inputs[2].dimensions() != 2 || inputs[2].size(0) != inputs[1].size(0) ||
                        inputs[2].size(1) != input.size(-1) / group,
                    "packed linear shape or dtype mismatch");
        }
        if (spec.operation == Operation::ROTARY) {
            invalid(input.dimensions() != 4 || input.dtype() != tensor::DType::F32 || input.size(3) % 2,
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
            for (std::size_t index = 0; index < inputs.size(); ++index)
                invalid(inputs[index].dtype() != tensor::DType::F32, "attention expects FP32 tensors");
            if (inputs.size() == 4) {
                const auto key_length =
                    spec.attributes.size() == 4 ? static_cast<std::size_t>(spec.attributes[3]) : key.size(1);
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
            const bool transpose = spec.operation != Operation::QUANTIZED_LINEAR && spec.attributes[0];
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
                        inputs[2].size(0) != inputs[1].size(1) || inputs[2].size(1) != 1,
                    "quantized linear parameter mismatch");
        if (spec.operation == Operation::LAYER_NORM || spec.operation == Operation::RESIDUAL_NORM ||
            spec.operation == Operation::RMS_NORM) {
            invalid(!input.dimensions() || !std::isfinite(spec.epsilon) || spec.epsilon <= 0,
                    "invalid normalization request");
            if (spec.operation == Operation::RMS_NORM)
                invalid(input.dtype() != tensor::DType::F32, "RMS normalization expects FP32 input");
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
            invalid(input.dtype() != other.dtype(), "arithmetic operand dtype mismatch");
            const auto common = std::min(input.dimensions(), other.dimensions());
            for (std::size_t index = 1; index <= common; ++index) {
                const auto left = input.size(-static_cast<std::int64_t>(index)),
                           right = other.size(-static_cast<std::int64_t>(index));
                invalid(left != right && left != 1 && right != 1, "incompatible broadcast dimensions");
            }
            spec.dtype = input.dtype();
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
                        input.size(2) != inputs[1].size(2) || inputs[1].dtype() != input.dtype() ||
                        inputs[2].dtype() != tensor::DType::I32 || inputs[2].dimensions() != 1 ||
                        inputs[2].size(0) != inputs[1].size(1),
                    "cache scatter shape or dtype mismatch");
        auto found = operators.find(key);
        if (found == operators.end()) {
            auto start = std::chrono::steady_clock::now();
            auto prepared = backend->prepare(spec, inputs);
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
        if (!profiling) return execute();
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
        return output;
    }
};
Context::Context(tensor::Device device, bool packed_prefill) : impl_(std::make_unique<Impl>()) {
    impl_->device = device;
    impl_->packed_prefill = packed_prefill;
    impl_->dispatch_key.reserve(128);
    const auto profile = std::getenv("KIDI_PROFILE_OPS");
    impl_->profiling = profile && (std::string_view(profile) == "host" || std::string_view(profile) == "sync");
    impl_->profile_requested = impl_->profiling;
    if (const auto skip = std::getenv("KIDI_PROFILE_OPS_SKIP")) impl_->skip_requests = std::strtoull(skip, nullptr, 10);
    if (impl_->skip_requests) impl_->profiling = false;
    impl_->synchronize_operators = profile && std::string_view(profile) == "sync";
    if (device == tensor::Device::cpu()) impl_->backend = runtime::cpu_operators();
#if defined(KIDI_HAS_METAL)
    else if (device == tensor::Device::apple_gpu())
        impl_->backend = runtime::metal_operators();
#endif
    else
        throw Failure({ErrorCode::UNSUPPORTED, "no eager backend for requested device"});
}
Context::~Context() = default;
Context::Context(Context&&) noexcept = default;
auto Context::operator=(Context&&) noexcept -> Context& = default;
auto Context::device() const noexcept -> tensor::Device { return impl_->device; }
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
auto Context::add(const Tensor& left, const Tensor& right) -> Tensor {
    return impl_->run({Operation::ADD}, {&left, &right});
}
auto Context::greedy_token(const Tensor& logits) -> Tensor {
    return impl_->run({Operation::GREEDY_TOKEN, {}, tensor::DType::I32}, {&logits});
}
auto Context::add_(Tensor& left, const Tensor& right) -> Tensor& {
    impl_->run({Operation::ADD}, {&left, &right}, nullptr, &left);
    return left;
}
auto Context::multiply(const Tensor& left, const Tensor& right) -> Tensor {
    return impl_->run({Operation::MULTIPLY}, {&left, &right});
}
auto Context::multiply_(Tensor& left, const Tensor& right) -> Tensor& {
    impl_->run({Operation::MULTIPLY}, {&left, &right}, nullptr, &left);
    return left;
}
auto Context::cast(const Tensor& input, tensor::DType dtype) -> Tensor {
    if (!input.defined() || input.device() != device())
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid cast operand"});
    if (input.dtype() == dtype) return input;
    return impl_->run({Operation::CAST, {}, dtype}, {&input});
}
auto Context::matmul(const Tensor& left, const Tensor& right, bool transpose_right) -> Tensor {
    const std::array<std::int64_t, 1> attributes{transpose_right};
    return impl_->run({Operation::MATMUL, attributes}, {&left, &right});
}
auto Context::scaled_dot_product_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                           std::int32_t heads, const Tensor& mask) -> Tensor {
    const std::array<std::int64_t, 1> attributes{heads};
    if (mask.defined()) return impl_->run({Operation::ATTENTION, attributes}, {&query, &key, &value, &mask});
    return impl_->run({Operation::ATTENTION, attributes}, {&query, &key, &value});
}
auto Context::linear(const Tensor& input, const Tensor& weight, const Tensor& bias, bool transpose_weight) -> Tensor {
    if (transpose_weight && !bias.defined() && !impl_->packed_weights.empty()) {
        const auto found = impl_->packed_weights.find(require(weight.host_bytes()).data());
        if (found != impl_->packed_weights.end() && found->second.source.dtype() == weight.dtype() &&
            std::ranges::equal(found->second.source.shape(), weight.shape()) &&
            (found->second.packed_prefill || (input.dimensions() > 0 && input.numel() == input.size(-1)))) {
            const auto& packed = found->second.packed;
            return packed_linear(input, packed.values, packed.scales, packed.bits, packed.group_size);
        }
    }
    const auto activation = weight.dtype() == tensor::DType::BF16 ? cast(input, tensor::DType::BF16) : input;
    const std::array<std::int64_t, 1> attributes{transpose_weight};
    if (!bias.defined()) return impl_->run({Operation::LINEAR, attributes}, {&activation, &weight});
    return impl_->run({Operation::LINEAR, attributes}, {&activation, &weight, &bias});
}
auto Context::prepare_linear_weights(const Tensor& weight, std::int32_t bits, std::int32_t group_size,
                                     bool packed_prefill) -> void {
    const auto address = require(weight.host_bytes()).data();
    if (const auto found = impl_->packed_weights.find(address); found != impl_->packed_weights.end()) {
        if (found->second.packed.bits != bits || found->second.packed.group_size != group_size ||
            found->second.packed_prefill != packed_prefill ||
            !std::ranges::equal(found->second.source.shape(), weight.shape()))
            throw Failure({ErrorCode::INVALID_ARGUMENT, "weight already packed with a different layout"});
        return;
    }
    auto packed = require(pack_weight(weight, bits, group_size));
    packed.values = require(packed.values.to(device()));
    packed.scales = require(packed.scales.to(device()));
    impl_->packed_weights.emplace(address, Impl::PackedBinding{weight, std::move(packed), packed_prefill});
}
auto Context::quantized_linear(const Tensor& input, const Tensor& weight, const Tensor& scale, const Tensor& bias)
    -> Tensor {
    return impl_->run({Operation::QUANTIZED_LINEAR}, {&input, &weight, &scale, &bias});
}
auto Context::packed_linear(const Tensor& input, const Tensor& weight, const Tensor& scales, std::int32_t bits,
                            std::int32_t group_size, float input_scale, float output_scale) -> Tensor {
    if (!std::isfinite(input_scale) || input_scale < 0 || !std::isfinite(output_scale) || output_scale < 0)
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid calibrated projection scales"});
    const std::array<std::int64_t, 3> attributes{bits, group_size, std::bit_cast<std::int32_t>(output_scale)};
    return impl_->run({Operation::PACKED_LINEAR, attributes, tensor::DType::F32, input_scale},
                      {&input, &weight, &scales});
}
auto Context::gelu(const Tensor& input, bool approximate) -> Tensor {
    const std::array<std::int64_t, 1> attributes{approximate};
    return impl_->run({Operation::GELU, attributes}, {&input});
}
auto Context::gelu_multiply(const Tensor& gate, const Tensor& value) -> Tensor {
    if (!gate.defined() || !value.defined() || gate.dtype() != tensor::DType::F32 ||
        value.dtype() != tensor::DType::F32 || !std::ranges::equal(gate.shape(), value.shape()))
        throw Failure({ErrorCode::INVALID_ARGUMENT, "GELU multiplication requires matching FP32 tensors"});
    if (device() != tensor::Device::apple_gpu()) return multiply(gelu(gate, true), value);
    return impl_->run({Operation::GELU_MULTIPLY}, {&gate, &value});
}
auto Context::gelu_(Tensor& input, bool approximate) -> Tensor& {
    const std::array<std::int64_t, 1> attributes{approximate};
    impl_->run({Operation::GELU, attributes}, {&input}, nullptr, &input);
    return input;
}
auto Context::tanh(const Tensor& input) -> Tensor { return impl_->run({Operation::TANH}, {&input}); }
auto Context::static_round(const Tensor& input, float scale) -> Tensor {
    if (!input.defined() || input.dtype() != tensor::DType::F32 || input.device() != device() ||
        !std::isfinite(scale) || scale < 0)
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid static activation rounding"});
    if (scale == 0) return input;
    return impl_->run({Operation::STATIC_ROUND, {}, tensor::DType::F32, scale}, {&input});
}
auto Context::rotary(const Tensor& input, const Tensor& cosine, const Tensor& sine) -> Tensor {
    return impl_->run({Operation::ROTARY}, {&input, &cosine, &sine});
}
auto Context::grouped_query_attention(const Tensor& query, const Tensor& key, const Tensor& value, std::int32_t heads,
                                      std::int32_t key_value_heads, const Tensor& mask, float scale,
                                      std::int64_t key_start) -> Tensor {
    if (!std::isfinite(scale) || scale <= 0 || !mask.defined() || !mask.dimensions() || key.dimensions() != 3 ||
        key_start < 0 || static_cast<std::size_t>(key_start) > key.size(1) || !mask.size(-1) ||
        mask.size(-1) > key.size(1) - key_start)
        throw Failure({ErrorCode::INVALID_ARGUMENT, "grouped attention requires a mask and positive scale"});
    const auto key_length = !key_start && mask.size(-1) == 1 ? key.size(1) : mask.size(-1);
    const std::array<std::int64_t, 4> attributes{heads, key_value_heads, key_start,
                                                 static_cast<std::int64_t>(key_length)};
    return impl_->run({Operation::ATTENTION, attributes, tensor::DType::F32, scale}, {&query, &key, &value, &mask});
}
auto Context::rms_norm(const Tensor& input, const Tensor& scale, float epsilon) -> Tensor {
    return impl_->run({Operation::RMS_NORM, {}, tensor::DType::F32, epsilon}, {&input, &scale});
}
auto Context::rms_rotary(const Tensor& input, const Tensor& scale, const Tensor& cosine, const Tensor& sine,
                         float epsilon) -> Tensor {
    if (input.dimensions() != 4 || !input.numel() || input.dtype() != tensor::DType::F32 || input.size(3) % 2 ||
        scale.dimensions() != 1 || scale.size(0) != input.size(3) || scale.dtype() != tensor::DType::F32 ||
        !std::isfinite(epsilon) || epsilon <= 0)
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid RMS rotary operands"});
    for (const auto* angle : {&cosine, &sine})
        if (angle->dimensions() != 4 || angle->size(0) != 1 || angle->size(1) != input.size(1) || angle->size(2) != 1 ||
            angle->size(3) != input.size(3) / 2 || angle->dtype() != tensor::DType::F32)
            throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid RMS rotary angles"});
    if (device() != tensor::Device::apple_gpu()) return rotary(rms_norm(input, scale, epsilon), cosine, sine);
    return impl_->run({Operation::RMS_ROTARY, {}, tensor::DType::F32, epsilon}, {&input, &scale, &cosine, &sine});
}
auto Context::rms_norm_residual(const Tensor& input, const Tensor& scale, const Tensor& residual, float epsilon,
                                const Tensor& output_scale) -> Tensor {
    if (!input.defined() || !input.dimensions() || input.dtype() != tensor::DType::F32 || !scale.defined() ||
        scale.dimensions() != 1 || scale.size(0) != input.size(-1) || scale.dtype() != tensor::DType::F32 ||
        !residual.defined() || residual.dtype() != tensor::DType::F32 ||
        !std::ranges::equal(input.shape(), residual.shape()) || !std::isfinite(epsilon) || epsilon <= 0 ||
        (output_scale.defined() && (output_scale.dtype() != tensor::DType::F32 || output_scale.numel() != 1)))
        throw Failure({ErrorCode::INVALID_ARGUMENT, "invalid RMS residual normalization operands"});
    const OperatorSpec spec{Operation::RMS_NORM_RESIDUAL, {}, tensor::DType::F32, epsilon};
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
    if (view.is_contiguous() && (device() == tensor::Device::cpu() || view.storage_offset() == 0)) return view;
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
auto Context::copy_slice_(Tensor& destination, const Tensor& source, std::int64_t axis, std::int64_t start) -> Tensor& {
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
    require(destination.host_bytes());
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