#include "kidi/runtime/operator.h"
#include "kidi/graph/compiler.h"
#include "kidi/graph/graph.h"
#include "kidi/ops/context.h"
#include "kidi/tensor/cuda.h"
#include "ynnpack/include/ynnpack_cuda.h"
#include <cuda_runtime_api.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <tuple>

namespace kidi::runtime {
namespace {
using ops::require;
using tensor::Device;
using tensor::Tensor;
static_assert(static_cast<int>(Operation::GELU_GATE) == static_cast<int>(ynn::cuda::Op::GELU_GATE));
static_assert(static_cast<int>(tensor::DType::E5M2) == static_cast<int>(ynn::cuda::Type::E5M2));

auto view(const Tensor& tensor, bool bind = true) -> ynn::cuda::Tensor {
    if (tensor.dimensions() > 8)
        throw ops::Failure({ErrorCode::UNSUPPORTED, "CUDA tensors support at most eight dimensions"});
    ynn::cuda::Tensor result;
    result.type = static_cast<ynn::cuda::Type>(tensor.dtype());
    result.rank = static_cast<int>(tensor.dimensions());
    result.device = tensor.device().index;
    std::copy(tensor.shape().begin(), tensor.shape().end(), result.shape.begin());
    if (bind) result.data = require(tensor::cuda_buffer(tensor)).data;
    return result;
}
class CudaOperator final : public Operator {
public:
    CudaOperator(ynn::cuda::Descriptor descriptor, std::vector<Tensor> constants, Device device)
        : program_(descriptor, device.index),
          constants_(std::move(constants)),
          selected_(
              descriptor.operation == ynn::cuda::Op::GREEDY_TOKEN &&
              (!descriptor.attribute_count || descriptor.attributes[0] == static_cast<int>(ops::TokenOutput::HOST))),
          scatter_(descriptor.operation == ynn::cuda::Op::SCATTER),
          device_(device) {
        const auto& output = program_.output();
        shape_.assign(output.shape.begin(), output.shape.begin() + output.rank);
        dtype_ = static_cast<tensor::DType>(output.type);
        if (descriptor.operation == ynn::cuda::Op::RESIDUAL_NORM) {
            auto sum = descriptor;
            sum.operation = ynn::cuda::Op::ADD;
            sum.input_count = 2;
            residual_ = std::make_unique<ynn::cuda::Program>(sum, device.index);
        }
    }
    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(shape_, dtype_, device_);
        run_into(inputs, {&output, 1});
        return selected_ ? require(output.to(Device::cpu())) : output;
    }
    auto run_(TensorInputs inputs, Tensor& destination) -> Tensor override {
        if (destination.device() != device_ || destination.dtype() != dtype_ ||
            !std::ranges::equal(destination.shape(), shape_))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "CUDA in-place output shape or dtype changed"});
        if (scatter_ && inputs.size() == 3 && inputs[0].storage_identity() == destination.storage_identity() &&
            inputs[0].storage_offset() == destination.storage_offset() &&
            inputs[1].storage_identity() != destination.storage_identity() &&
            inputs[2].storage_identity() != destination.storage_identity()) {
            run_into(inputs, {&destination, 1});
            return destination;
        }
        auto result = run(inputs);
        if (destination.device() != device_ || result.dtype() != destination.dtype() ||
            !std::ranges::equal(result.shape(), destination.shape()))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "CUDA in-place output shape or dtype changed"});
        ynn::cuda::copy_device(require(tensor::cuda_buffer(destination)).data,
                               require(tensor::cuda_buffer(result)).data, result.nbytes(), device_.index);
        return destination;
    }
    auto run_pair(TensorInputs inputs) -> std::array<Tensor, 2> override {
        if (!residual_) return {Tensor{}, run(inputs)};
        auto sum = residual_pool_.acquire(shape_, dtype_, device_);
        auto normalized = pool_.acquire(shape_, dtype_, device_);
        std::array outputs{sum, normalized};
        run_into(inputs, outputs);
        return outputs;
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override {
        if (selected_ && outputs.size() == 1 && outputs[0].device() == Device::cpu()) {
            auto temporary = pool_.acquire(shape_, dtype_, device_);
            run_into(inputs, {&temporary, 1});
            require(outputs[0].copy_from_host(require(temporary.copy_to_host())));
            return;
        }
        if (inputs.size() != constants_.size())
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "CUDA operand count changed after preparation"});
        if ((outputs.size() != 1 && !(residual_ && outputs.size() == 2)) ||
            std::ranges::any_of(outputs, [&](const auto& output) { return output.device() != device_; }))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "CUDA output belongs to a different device"});
        for (const auto& output : outputs)
            if (output.dtype() != dtype_ || !std::ranges::equal(output.shape(), shape_))
                throw ops::Failure(
                    {ErrorCode::INVALID_ARGUMENT, "CUDA output shape or dtype changed after preparation"});
        std::array<ynn::cuda::Tensor, 8> bindings{};
        for (std::size_t index = 0; index < inputs.size(); ++index)
            bindings[index] = view(constants_[index].defined() ? constants_[index] : inputs[index]);
        try {
            if (outputs.size() == 2) residual_->run(bindings.data(), 2, require(tensor::cuda_buffer(outputs[0])).data);
            program_.run(bindings.data(), static_cast<int>(inputs.size()),
                         require(tensor::cuda_buffer(outputs.back())).data);
        } catch (const std::exception& error) {
            throw ops::Failure({ErrorCode::RUNTIME, error.what()});
        }
    }
    auto allocations() const -> AllocationStats override {
        const auto primary = pool_.allocations(), residual = residual_pool_.allocations();
        return {primary.count + residual.count, primary.bytes + residual.bytes};
    }
    auto release_workspaces() -> void override {
        pool_.clear();
        residual_pool_.clear();
    }

private:
    ynn::cuda::Program program_;
    std::vector<Tensor> constants_;
    std::vector<std::int64_t> shape_;
    tensor::DType dtype_;
    bool selected_, scatter_;
    Device device_;
    OutputPool pool_;
    OutputPool residual_pool_;
    std::unique_ptr<ynn::cuda::Program> residual_;
};

auto check_cuda(cudaError_t status) -> void {
    if (status != cudaSuccess) throw ops::Failure({ErrorCode::RUNTIME, cudaGetErrorString(status)});
}

class CudaDeviceScope {
public:
    explicit CudaDeviceScope(Device device) {
        check_cuda(cudaGetDevice(&previous_));
        if (device.index != previous_) check_cuda(cudaSetDevice(device.index));
    }
    ~CudaDeviceScope() { cudaSetDevice(previous_); }

private:
    int previous_ = 0;
};

struct CudaUpload;

struct CudaGraphResources {
    int device = 0;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaEvent_t before = nullptr, after = nullptr;
    void* selection = nullptr;
    std::vector<std::shared_ptr<CudaUpload>> bindings;
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> profiles;
    std::array<cudaEvent_t, 4> phases{};
    ~CudaGraphResources() {
        int previous = 0;
        if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device) != cudaSuccess) return;
        if (stream) cudaStreamSynchronize(stream);
        if (selection) cudaFreeHost(selection);
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        for (const auto& [begin, end] : profiles) {
            if (begin) cudaEventDestroy(begin);
            if (end) cudaEventDestroy(end);
        }
        for (const auto phase : phases)
            if (phase) cudaEventDestroy(phase);
        if (before) cudaEventDestroy(before);
        if (after) cudaEventDestroy(after);
        if (stream) cudaStreamDestroy(stream);
        cudaSetDevice(previous);
    }
};

struct CudaGraphInput {
    Tensor value;
    bool direct = false;
};

struct CudaUpload {
    int device = 0;
    void* data = nullptr;
    std::size_t capacity = 0;
    cudaEvent_t ready = nullptr;
    ~CudaUpload() {
        int previous = 0;
        if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device) != cudaSuccess) return;
        if (ready) cudaEventSynchronize(ready);
        if (data) cudaFreeHost(data);
        if (ready) cudaEventDestroy(ready);
        cudaSetDevice(previous);
    }
};

class CudaExecutable final : public graph::Executable {
public:
    CudaExecutable(const graph::Graph& source, Device device, std::vector<std::shared_ptr<CudaGraphInput>> bindings,
                   std::string key)
        : device_(device),
          graph_(source),
          bindings_(std::move(bindings)),
          key_(std::move(key)),
          resources_(std::make_unique<CudaGraphResources>()) {
        const CudaDeviceScope selected(device_);
        resources_->device = device_.index;
        check_cuda(cudaStreamCreateWithFlags(&resources_->stream, cudaStreamNonBlocking));
        check_cuda(cudaEventCreateWithFlags(&resources_->before, cudaEventDisableTiming));
        check_cuda(cudaEventCreateWithFlags(&resources_->after, cudaEventDisableTiming));
        std::vector<Tensor> staged;
        for (const auto& binding : bindings_) staged.push_back(binding->value);
        graph_.rebind(staged);
        for (std::size_t index = 0; index < bindings_.size(); ++index)
            if (bindings_[index]->direct) direct_slots_.push_back(index);
        if (!direct_slots_.empty()) {
            binding_table_ =
                require(Tensor::empty({static_cast<std::int64_t>(direct_slots_.size())}, tensor::DType::U64, device_));
            auto* table = static_cast<void**>(require(tensor::cuda_buffer(binding_table_)).data);
            latest_bindings_.resize(direct_slots_.size());
            for (std::size_t index = 0; index < direct_slots_.size(); ++index) {
                const auto& input = bindings_[direct_slots_[index]]->value;
                dynamic_bindings_.push_back({require(tensor::cuda_buffer(input)).data, input.nbytes(), table + index});
            }
        }
        prepare_projections();
        if (projections_.empty()) graph_.plan_intermediates();
        if (const auto* profile = std::getenv("KIDI_PROFILE_OPS"); profile && std::string_view(profile) == "graph")
            for (auto& phase : resources_->phases) check_cuda(cudaEventCreate(&phase));
        if (const auto* profile = std::getenv("KIDI_PROFILE_OPS"); profile && std::string_view(profile) == "events") {
            resources_->profiles.resize(graph_.nodes().size());
            for (auto& [begin, end] : resources_->profiles) {
                check_cuda(cudaEventCreate(&begin));
                check_cuda(cudaEventCreate(&end));
            }
        }
        for (const auto& node : graph_.nodes())
            if (node.mode == graph::Mode::IN_PLACE) {
                const auto slot = node.sources[node.input_count].index;
                if (!bindings_[slot]->direct && std::ranges::find(mutated_, slot) == mutated_.end())
                    mutated_.push_back(slot);
            }
        for (const auto& output : graph_.outputs()) {
            if (output.device() == Device::cpu()) {
                selection_ =
                    require(Tensor::empty({output.shape().begin(), output.shape().end()}, output.dtype(), device_));
                exports_.push_back(output);
            } else {
                exports_.push_back(
                    require(Tensor::empty({output.shape().begin(), output.shape().end()}, output.dtype(), device_)));
            }
        }
        if (selection_.defined())
            check_cuda(cudaHostAlloc(&resources_->selection, selection_.nbytes(), cudaHostAllocDefault));
        check_cuda(cudaEventRecord(resources_->before, nullptr));
        check_cuda(cudaStreamWaitEvent(resources_->stream, resources_->before, 0));
        const ynn::cuda::StreamScope stream(resources_->stream);
        copy_inputs(source.inputs());
        execute_nodes();
        check_cuda(cudaStreamSynchronize(resources_->stream));
        check_cuda(cudaStreamBeginCapture(resources_->stream, cudaStreamCaptureModeThreadLocal));
        try {
            execute_nodes();
        } catch (...) {
            cudaGraph_t abandoned = nullptr;
            cudaStreamEndCapture(resources_->stream, &abandoned);
            if (abandoned) cudaGraphDestroy(abandoned);
            throw;
        }
        check_cuda(cudaStreamEndCapture(resources_->stream, &resources_->graph));
        check_cuda(cudaGraphInstantiateWithFlags(&resources_->executable, resources_->graph, 0));
    }

    auto projection_groups() const noexcept -> std::size_t { return projections_.size(); }

    auto run(std::span<const Tensor> inputs, std::span<Tensor> outputs) -> void override {
        const auto started = std::chrono::steady_clock::now();
        if (outputs.size() != exports_.size())
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "CUDA graph output count changed"});
        const CudaDeviceScope selected(device_);
        check_cuda(cudaEventRecord(resources_->before, nullptr));
        check_cuda(cudaStreamWaitEvent(resources_->stream, resources_->before, 0));
        const ynn::cuda::StreamScope stream(resources_->stream);
        if (resources_->phases[0]) check_cuda(cudaEventRecord(resources_->phases[0], resources_->stream));
        copy_inputs(inputs);
        if (resources_->phases[1]) check_cuda(cudaEventRecord(resources_->phases[1], resources_->stream));
        check_cuda(cudaGraphLaunch(resources_->executable, resources_->stream));
        if (resources_->phases[2]) check_cuda(cudaEventRecord(resources_->phases[2], resources_->stream));
        destinations_.resize(mutated_.size());
        sources_.resize(mutated_.size());
        sizes_.resize(mutated_.size());
        for (std::size_t index = 0; index < mutated_.size(); ++index) {
            const auto slot = mutated_[index];
            destinations_[index] = require(tensor::cuda_buffer(inputs[slot])).data;
            sources_[index] = require(tensor::cuda_buffer(bindings_[slot]->value)).data;
            sizes_[index] = inputs[slot].nbytes();
        }
        ynn::cuda::copy_device_batch(destinations_.data(), sources_.data(), sizes_.data(), sizes_.size(),
                                     device_.index);
        if (selection_.defined())
            check_cuda(cudaMemcpyAsync(resources_->selection, require(tensor::cuda_buffer(selection_)).data,
                                       selection_.nbytes(), cudaMemcpyDeviceToHost, resources_->stream));
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            auto& output = exports_[index];
            if (output.device() != Device::cpu()) {
                ynn::cuda::copy_device(require(tensor::cuda_buffer(output)).data,
                                       require(tensor::cuda_buffer(graph_.outputs()[index])).data, output.nbytes(),
                                       device_.index);
            }
            outputs[index] = output;
        }
        if (resources_->phases[3]) check_cuda(cudaEventRecord(resources_->phases[3], resources_->stream));
        check_cuda(cudaEventRecord(resources_->after, resources_->stream));
        if (selection_.defined()) {
            check_cuda(cudaEventSynchronize(resources_->after));
            for (auto& output : outputs)
                if (output.device() == Device::cpu()) {
                    auto destination = require(output.host_bytes());
                    std::memcpy(destination.data(), resources_->selection, destination.size());
                }
        } else {
            check_cuda(cudaStreamWaitEvent(nullptr, resources_->after, 0));
        }
        if (resources_->phases[0]) {
            check_cuda(cudaStreamSynchronize(resources_->stream));
            std::array<float, 3> milliseconds{};
            for (std::size_t index = 0; index < milliseconds.size(); ++index)
                check_cuda(cudaEventElapsedTime(&milliseconds[index], resources_->phases[index],
                                                resources_->phases[index + 1]));
            const auto wall =
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count();
            std::cerr << "kidi_cuda_replay|key=" << key_ << "|input_us=" << milliseconds[0] * 1000
                      << "|compute_us=" << milliseconds[1] * 1000 << "|output_us=" << milliseconds[2] * 1000
                      << "|wall_us=" << wall << '\n';
        }
        if (!resources_->profiles.empty() && !profiled_) {
            check_cuda(cudaStreamSynchronize(resources_->stream));
            for (std::size_t index = 0; index < graph_.nodes().size(); ++index) {
                float milliseconds = 0;
                check_cuda(cudaEventElapsedTime(&milliseconds, resources_->profiles[index].first,
                                                resources_->profiles[index].second));
                const auto& node = graph_.nodes()[index];
                std::cerr << "kidi_cuda_node|key=" << key_ << "|op=" << operation_name(node.operation) << "|shapes=";
                for (const auto& input : node.inputs()) {
                    std::cerr << '[';
                    for (const auto extent : input.shape()) std::cerr << extent << ',';
                    std::cerr << ']';
                }
                std::cerr << "|us=" << milliseconds * 1000 << '\n';
            }
            profiled_ = true;
        }
    }

private:
    struct ProjectionGroup {
        std::vector<std::size_t> nodes;
        Tensor weight, output;
        std::unique_ptr<ynn::cuda::Program> program;
    };
    auto prepare_projections() -> void {
        auto nodes = graph_.nodes();
        projection_nodes_.assign(nodes.size(), SIZE_MAX);
        const auto eligible = [&](const graph::Node& node) {
            return node.operation == Operation::LINEAR && node.input_count == 2 && node.attributes.size() == 1 &&
                   node.attributes[0] == 1 && node.inputs()[0].dtype() == tensor::DType::BF16 &&
                   node.inputs()[1].dtype() == tensor::DType::BF16 && node.inputs()[1].device() == device_ &&
                   node.inputs()[1].dimensions() == 2 && node.inputs()[0].numel() == node.inputs()[0].size(-1) &&
                   node.outputs()[0].dtype() == tensor::DType::BF16 &&
                   node.sources[0].kind == graph::Source::Kind::NODE &&
                   node.sources[1].kind == graph::Source::Kind::EXTERNAL &&
                   std::ranges::none_of(graph_.outputs(), [&](const auto& output) {
                       return output.storage_identity() == node.outputs()[0].storage_identity();
                   });
        };
        for (std::size_t first = 0; first < nodes.size(); ++first) {
            if (projection_nodes_[first] != SIZE_MAX || !eligible(nodes[first])) continue;
            ProjectionGroup group;
            const auto& source = nodes[first].sources[0];
            std::int64_t columns = 0;
            for (std::size_t index = first; index < nodes.size(); ++index) {
                if (projection_nodes_[index] != SIZE_MAX || !eligible(nodes[index])) continue;
                const auto& other = nodes[index].sources[0];
                if (source.kind != other.kind || source.index != other.index || source.output != other.output ||
                    source.offset != other.offset ||
                    !std::ranges::equal(nodes[first].inputs()[0].shape(), nodes[index].inputs()[0].shape()))
                    continue;
                group.nodes.push_back(index);
                columns += nodes[index].inputs()[1].size(0);
            }
            if (group.nodes.size() < 2) continue;
            const auto width = static_cast<std::int64_t>(nodes[first].inputs()[0].size(-1));
            group.weight = require(Tensor::empty({columns, width}, tensor::DType::BF16, device_));
            auto shape = std::vector<std::int64_t>(nodes[first].outputs()[0].shape().begin(),
                                                   nodes[first].outputs()[0].shape().end());
            shape.back() = columns;
            group.output = require(Tensor::empty(shape, tensor::DType::BF16, device_));
            std::int64_t start = 0;
            for (const auto index : group.nodes) {
                const auto original = nodes[index].outputs()[0];
                const auto length = static_cast<std::int64_t>(original.size(-1));
                auto* packed =
                    static_cast<std::byte*>(require(tensor::cuda_buffer(group.weight)).data) + start * width * 2;
                ynn::cuda::copy_device(packed, require(tensor::cuda_buffer(nodes[index].inputs()[1])).data,
                                       nodes[index].inputs()[1].nbytes(), device_.index);
                const auto flat = require(group.output.reshape({columns}));
                for (auto& node : nodes)
                    for (auto& operand : node.operands) {
                        if (operand.storage_identity() != original.storage_identity()) continue;
                        const auto offset = operand.storage_offset() - original.storage_offset();
                        if (offset < 0 || offset + operand.numel() > original.numel())
                            throw ops::Failure(
                                {ErrorCode::INVALID_ARGUMENT, "CUDA projection output alias exceeds its source"});
                        operand =
                            require(require(flat.narrow(0, start + offset, static_cast<std::int64_t>(operand.numel())))
                                        .reshape({operand.shape().begin(), operand.shape().end()}));
                    }
                projection_nodes_[index] = projections_.size();
                start += length;
            }
            ynn::cuda::Descriptor descriptor;
            descriptor.operation = ynn::cuda::Op::LINEAR;
            descriptor.input_count = 2;
            descriptor.inputs[0] = view(nodes[first].inputs()[0]);
            descriptor.inputs[1] = view(group.weight);
            descriptor.output.type = ynn::cuda::Type::BF16;
            descriptor.attribute_count = 1;
            descriptor.attributes[0] = 1;
            group.program = std::make_unique<ynn::cuda::Program>(descriptor, device_.index);
            projections_.push_back(std::move(group));
        }
    }
    auto copy_inputs(std::span<const Tensor> inputs) -> void {
        bool changed = false;
        for (std::size_t index = 0; index < direct_slots_.size(); ++index) {
            const auto pointer = require(tensor::cuda_buffer(inputs[direct_slots_[index]])).data;
            if (pointer != latest_bindings_[index]) changed = true;
        }
        if (changed) {
            CudaUpload* upload = nullptr;
            for (const auto& candidate : resources_->bindings) {
                const auto status = cudaEventQuery(candidate->ready);
                if (status == cudaErrorNotReady) continue;
                check_cuda(status);
                upload = candidate.get();
                break;
            }
            if (!upload) {
                if (resources_->bindings.size() == 64)
                    throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "CUDA binding update queue is full"});
                auto created = std::make_shared<CudaUpload>();
                created->device = device_.index;
                created->capacity = direct_slots_.size() * sizeof(void*);
                check_cuda(cudaHostAlloc(&created->data, created->capacity, cudaHostAllocDefault));
                check_cuda(cudaEventCreateWithFlags(&created->ready, cudaEventDisableTiming));
                upload = created.get();
                resources_->bindings.push_back(std::move(created));
            }
            for (std::size_t index = 0; index < direct_slots_.size(); ++index)
                latest_bindings_[index] = require(tensor::cuda_buffer(inputs[direct_slots_[index]])).data;
            std::memcpy(upload->data, latest_bindings_.data(), latest_bindings_.size() * sizeof(void*));
            check_cuda(cudaMemcpyAsync(require(tensor::cuda_buffer(binding_table_)).data, upload->data,
                                       latest_bindings_.size() * sizeof(void*), cudaMemcpyHostToDevice,
                                       resources_->stream));
            check_cuda(cudaEventRecord(upload->ready, resources_->stream));
        }
        destinations_.resize(inputs.size());
        sources_.resize(inputs.size());
        sizes_.resize(inputs.size());
        std::size_t copies = 0;
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            if (bindings_[index]->direct) continue;
            destinations_[copies] = require(tensor::cuda_buffer(bindings_[index]->value)).data;
            sources_[copies] = require(tensor::cuda_buffer(inputs[index])).data;
            sizes_[copies++] = inputs[index].nbytes();
        }
        ynn::cuda::copy_device_batch(destinations_.data(), sources_.data(), sizes_.data(), copies, device_.index);
    }
    auto execute_nodes() -> void {
        const ynn::cuda::BindingScope bindings(dynamic_bindings_.data(), dynamic_bindings_.size());
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        if (!resources_->profiles.empty()) check_cuda(cudaStreamIsCapturing(resources_->stream, &status));
        const auto flags = status == cudaStreamCaptureStatusActive ? cudaEventRecordExternal : cudaEventRecordDefault;
        for (std::size_t index = 0; index < graph_.nodes().size(); ++index) {
            const auto& node = graph_.nodes()[index];
            if (!resources_->profiles.empty())
                check_cuda(cudaEventRecordWithFlags(resources_->profiles[index].first, resources_->stream, flags));
            std::vector<Tensor> outputs(node.outputs().begin(), node.outputs().end());
            if (node.operation == Operation::GREEDY_TOKEN && selection_.defined()) outputs[0] = selection_;
            if (projection_nodes_[index] == SIZE_MAX) {
                node.prepared->run_into(node.inputs(), outputs);
            } else {
                auto& group = projections_[projection_nodes_[index]];
                if (group.nodes.front() == index) {
                    const std::array bindings{view(node.inputs()[0]), view(group.weight)};
                    group.program->run(bindings.data(), 2, require(tensor::cuda_buffer(group.output)).data);
                }
            }
            if (!resources_->profiles.empty())
                check_cuda(cudaEventRecordWithFlags(resources_->profiles[index].second, resources_->stream, flags));
        }
    }

    Device device_;
    graph::Graph graph_;
    std::vector<std::shared_ptr<CudaGraphInput>> bindings_;
    std::vector<std::size_t> mutated_;
    std::vector<std::size_t> direct_slots_;
    std::vector<void*> latest_bindings_;
    std::vector<ynn::cuda::TensorBinding> dynamic_bindings_;
    std::vector<Tensor> exports_;
    std::vector<void*> destinations_;
    std::vector<const void*> sources_;
    std::vector<std::size_t> sizes_;
    std::vector<ProjectionGroup> projections_;
    std::vector<std::size_t> projection_nodes_;
    Tensor selection_;
    Tensor binding_table_;
    std::string key_;
    bool profiled_ = false;
    std::unique_ptr<CudaGraphResources> resources_;
};

class CudaCompiler final : public graph::Compiler {
public:
    explicit CudaCompiler(Device device) : device_(device) {}
    auto name() const -> std::string_view override { return "cuda-graph"; }
    auto captures_prefill() const -> bool override { return false; }
    auto compile(const graph::Graph& source, std::string_view key) -> std::unique_ptr<graph::Executable> override {
        if (source.nodes().empty() || key.starts_with("gemma4_prefill:") ||
            std::ranges::any_of(source.inputs(), [&](const auto& input) { return input.device() != device_; }))
            return nullptr;
        for (std::size_t index = 0; index < source.nodes().size(); ++index) {
            const auto& node = source.nodes()[index];
            if (node.mode == graph::Mode::IN_PLACE &&
                (node.operation != Operation::SCATTER ||
                 node.sources[node.input_count].kind != graph::Source::Kind::INPUT))
                return nullptr;
            if (node.operation == Operation::GREEDY_TOKEN && index + 1 != source.nodes().size()) return nullptr;
        }
        for (const auto& output : source.outputs())
            if (output.device() != device_ &&
                (output.device() != Device::cpu() || source.nodes().back().operation != Operation::GREEDY_TOKEN ||
                 output.storage_identity() != source.nodes().back().outputs()[0].storage_identity()))
                return nullptr;
        std::erase_if(inputs_, [](const auto& entry) { return entry.second.expired(); });
        std::vector<std::shared_ptr<CudaGraphInput>> bindings;
        for (std::size_t index = 0; index < source.inputs().size(); ++index) {
            const auto& input = source.inputs()[index];
            bool direct = false, supported = true;
            for (const auto& node : source.nodes()) {
                if (node.mode == graph::Mode::IN_PLACE && node.sources[node.input_count].index == index) direct = true;
                for (std::size_t operand = 0; operand < node.input_count; ++operand) {
                    const auto& origin = node.sources[operand];
                    if (origin.kind != graph::Source::Kind::INPUT || origin.index != index) continue;
                    if ((node.operation != Operation::SCATTER && node.operation != Operation::ATTENTION) ||
                        (node.operation == Operation::ATTENTION && node.inputs()[0].size(1) != 1))
                        supported = false;
                }
            }
            const bool exported = std::ranges::any_of(source.outputs(), [&](const auto& output) {
                return output.storage_identity() == input.storage_identity();
            });
            if (direct && supported && !exported) {
                auto binding = std::make_shared<CudaGraphInput>();
                binding->value = input;
                binding->direct = true;
                bindings.push_back(std::move(binding));
                continue;
            }
            auto layout =
                std::tuple{index, input.dtype(), std::vector<std::int64_t>(input.shape().begin(), input.shape().end())};
            auto& cached = inputs_[layout];
            auto binding = cached.lock();
            if (!binding) {
                binding = std::make_shared<CudaGraphInput>();
                binding->value =
                    require(Tensor::empty({input.shape().begin(), input.shape().end()}, input.dtype(), device_));
                cached = binding;
            }
            bindings.push_back(std::move(binding));
        }
        auto executable = std::make_unique<CudaExecutable>(source, device_, std::move(bindings), std::string(key));
        const auto timestamp =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count();
        std::cerr << "kidi_cuda_graph|key=" << key << "|nodes=" << source.nodes().size()
                  << "|projection_groups=" << executable->projection_groups()
                  << "|capture=completed|steady_ns=" << timestamp << '\n';
        return executable;
    }

private:
    Device device_;
    using Layout = std::tuple<std::size_t, tensor::DType, std::vector<std::int64_t>>;
    std::map<Layout, std::weak_ptr<CudaGraphInput>> inputs_;
};

class CudaBackend final : public OperatorBackend {
public:
    explicit CudaBackend(Device device) : device_(device), compiler_(std::make_shared<CudaCompiler>(device)) {}
    auto supports_replay() const noexcept -> bool override { return true; }
    auto graph_compiler() -> std::shared_ptr<graph::Compiler> override { return compiler_; }
    auto stage_inputs(std::span<const Tensor> inputs, Device device) -> std::vector<Tensor> override {
        if (device != device_)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "CUDA staging destination belongs to a different device"});
        const CudaDeviceScope selected(device_);
        std::vector<Tensor> staged;
        std::vector<std::size_t> offsets(inputs.size());
        staged.reserve(inputs.size());
        std::size_t bytes = 0;
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            const auto& input = inputs[index];
            if (!input.defined() || !input.is_contiguous())
                throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "CUDA staging requires defined contiguous tensors"});
            if (input.device() != Device::cpu()) {
                staged.push_back(require(input.to(device_)));
                continue;
            }
            offsets[index] = bytes;
            bytes += (input.nbytes() + 15) / 16 * 16;
            staged.push_back(staging_.acquire(input.shape(), input.dtype(), device_));
        }
        if (!bytes) return staged;
        CudaUpload* upload = nullptr;
        for (const auto& candidate : uploads_) {
            const auto status = cudaEventQuery(candidate->ready);
            if (status == cudaErrorNotReady) continue;
            check_cuda(status);
            if (!upload || candidate->capacity > upload->capacity) upload = candidate.get();
            if (candidate->capacity >= bytes) {
                upload = candidate.get();
                break;
            }
        }
        if (!upload) {
            if (uploads_.size() == 64)
                throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "CUDA input staging queue is full"});
            auto created = std::make_unique<CudaUpload>();
            created->device = device_.index;
            check_cuda(cudaEventCreateWithFlags(&created->ready, cudaEventDisableTiming));
            upload = created.get();
            uploads_.push_back(std::move(created));
        }
        if (bytes > upload->capacity) {
            if (upload->data) {
                check_cuda(cudaFreeHost(upload->data));
                upload->data = nullptr;
                upload->capacity = 0;
            }
            check_cuda(cudaHostAlloc(&upload->data, bytes, cudaHostAllocDefault));
            upload->capacity = bytes;
        }
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            if (inputs[index].device() != Device::cpu() || !inputs[index].nbytes()) continue;
            const auto source = require(inputs[index].host_bytes());
            auto* pinned = static_cast<std::byte*>(upload->data) + offsets[index];
            std::memcpy(pinned, source.data(), source.size());
            check_cuda(cudaMemcpyAsync(require(tensor::cuda_buffer(staged[index])).data, pinned, source.size(),
                                       cudaMemcpyHostToDevice, nullptr));
        }
        check_cuda(cudaEventRecord(upload->ready, nullptr));
        return staged;
    }
    auto release_cached_buffers() -> void override {
        staging_.clear();
        std::erase_if(constants_, [](const auto& entry) { return entry.second.second.owns_unique_storage(); });
    }
    auto activation_dtype(tensor::DType parameters) const noexcept -> tensor::DType override {
        return parameters == tensor::DType::BF16 ? tensor::DType::BF16 : tensor::DType::F32;
    }
    auto prepare(const OperatorSpec& spec, TensorInputs inputs) -> std::unique_ptr<Operator> override {
        if (inputs.size() > 8 || spec.attributes.size() > 32)
            throw ops::Failure({ErrorCode::UNSUPPORTED, "CUDA operand or attribute count exceeded"});
        ynn::cuda::Descriptor descriptor;
        descriptor.operation = static_cast<ynn::cuda::Op>(spec.operation);
        descriptor.output.type = static_cast<ynn::cuda::Type>(spec.dtype);
        descriptor.input_count = static_cast<int>(inputs.size());
        descriptor.attribute_count = static_cast<int>(spec.attributes.size());
        descriptor.epsilon = spec.epsilon;
        descriptor.immutable_weights = !spec.dynamic_parameters;
        descriptor.projection_cache = projection_cache_;
        std::copy(spec.attributes.begin(), spec.attributes.end(), descriptor.attributes.begin());
        std::vector<Tensor> constants(inputs.size());
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            descriptor.inputs[index] = view(inputs[index], false);
            if (inputs[index].device() == device_) continue;
            if (inputs[index].device().kind != tensor::DeviceKind::CPU)
                throw ops::Failure({ErrorCode::INVALID_ARGUMENT,
                                    "CUDA operands belong to different devices; transfer parameters explicitly"});
            const auto key =
                std::tuple{inputs[index].storage_identity(), inputs[index].storage_offset(), inputs[index].dtype(),
                           std::vector<std::int64_t>(inputs[index].shape().begin(), inputs[index].shape().end())};
            auto found = constants_.find(key);
            if (found == constants_.end())
                found = constants_.emplace(key, std::pair{inputs[index], require(inputs[index].to(device_))}).first;
            constants[index] = found->second.second;
        }
        try {
            return std::make_unique<CudaOperator>(descriptor, std::move(constants), device_);
        } catch (const std::exception& error) {
            throw ops::Failure({ErrorCode::UNSUPPORTED, error.what()});
        }
    }
    auto synchronize() -> void override {
        const CudaDeviceScope selected(device_);
        check_cuda(cudaStreamSynchronize(nullptr));
    }
    auto copy_slice_(Tensor& destination, const Tensor& source, std::size_t outer, std::size_t source_bytes,
                     std::size_t destination_bytes, std::size_t offset) -> void override {
        if (destination.device() != device_ || source.device() != device_)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "CUDA slice operands belong to different devices"});
        auto* target = static_cast<std::byte*>(require(tensor::cuda_buffer(destination)).data);
        const auto* input = static_cast<const std::byte*>(require(tensor::cuda_buffer(source)).data);
        if (outer * source_bytes > source.nbytes() || outer * destination_bytes > destination.nbytes() ||
            offset > destination_bytes || source_bytes > destination_bytes - offset)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "CUDA slice copy is out of bounds"});
        for (std::size_t index = 0; index < outer; ++index)
            ynn::cuda::copy_device(target + index * destination_bytes + offset, input + index * source_bytes,
                                   source_bytes, device_.index);
    }

private:
    Device device_;
    std::shared_ptr<graph::Compiler> compiler_;
    std::shared_ptr<ynn::cuda::ProjectionCache> projection_cache_ = ynn::cuda::make_projection_cache();
    using ConstantKey = std::tuple<const void*, std::int64_t, tensor::DType, std::vector<std::int64_t>>;
    std::map<ConstantKey, std::pair<Tensor, Tensor>> constants_;
    OutputPool staging_;
    std::vector<std::unique_ptr<CudaUpload>> uploads_;
};
} // namespace
auto cuda_operators(tensor::Device device) -> std::unique_ptr<OperatorBackend> {
    return std::make_unique<CudaBackend>(device);
}
} // namespace kidi::runtime