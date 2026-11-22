#include "cuda/runtime.h"
#include "cuda/attention.h"
#include "cuda/elementwise.h"
#include "cuda/launch.h"
#include "cuda/projection.h"
#if defined(KIDI_CUDA_USE_CUTLASS_INT8)
#include "cuda/dense_gemm.h"
#include "cuda/int8_gemm.h"
#endif

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

namespace kidi::cuda {
namespace {
std::mutex allocations_mutex;
struct Allocation {
    std::size_t bytes;
    int device;
};
std::unordered_map<void*, Allocation> allocations;
std::atomic<std::size_t> live_bytes{0};
std::size_t peak_bytes = 0;
std::unordered_map<int, std::size_t> device_bytes, device_peaks;
thread_local cudaStream_t execution_stream = nullptr;
thread_local const TensorBinding* execution_bindings = nullptr;
thread_local std::size_t execution_binding_count = 0;

View view(const Tensor& tensor) {
    View result{};
    result.data = tensor.data;
    result.type = static_cast<int>(tensor.type);
    result.rank = tensor.rank;
    std::copy(tensor.shape.begin(), tensor.shape.end(), result.shape);
    return result;
}

void bind(View& tensor) {
    const auto address = reinterpret_cast<std::uintptr_t>(tensor.data);
    for (std::size_t index = 0; index < execution_binding_count; ++index) {
        const auto& candidate = execution_bindings[index];
        const auto begin = reinterpret_cast<std::uintptr_t>(candidate.data);
        if (address >= begin && address - begin < candidate.bytes) {
            tensor.binding = candidate.slot;
            tensor.binding_offset = address - begin;
            return;
        }
    }
}

struct PreparedProjection {
    std::unique_ptr<void, decltype(&release)> expanded{nullptr, release};
    std::unique_ptr<void, decltype(&release)> sums{nullptr, release};
    cudaEvent_t ready{};
    int device;
    PreparedProjection(Launch launch, int bits, int selected_device) : device(selected_device) {
        const auto width = launch.inputs[0].shape[launch.inputs[0].rank - 1];
        const auto columns = launch.output.shape[launch.output.rank - 1];
        sums.reset(allocate(columns * sizeof(int), device));
        if (bits != 8) expanded.reset(allocate(columns * width, device));
        check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        prepare(launch, bits);
    }
    ~PreparedProjection() {
        int previous = 0;
        if (ready && cudaGetDevice(&previous) == cudaSuccess && cudaSetDevice(device) == cudaSuccess) {
            cudaEventDestroy(ready);
            cudaSetDevice(previous);
        }
    }
    void prepare(Launch launch, int bits) {
        prepare_projection_weights(launch, bits, expanded.get(), sums.get(), execution_stream);
        check(cudaGetLastError());
        check(cudaEventRecord(ready, execution_stream));
    }
};

int minimum_inputs(const Descriptor& descriptor) {
    const auto operation = descriptor.operation;
    switch (operation) {
        case Op::ADD:
        case Op::MULTIPLY:
        case Op::MATMUL:
        case Op::LINEAR:
        case Op::GATHER:
        case Op::RMS_NORM:
        case Op::GELU_MULTIPLY:
        case Op::SILU_MULTIPLY:
        case Op::SNAKE:
        case Op::CONV_TRANSPOSE_FOLD:
            return 2;
        case Op::LAYER_NORM:
        case Op::RMS_NORM_RESIDUAL:
        case Op::ROTARY:
        case Op::PACKED_LINEAR:
        case Op::ATTENTION:
        case Op::SCATTER:
        case Op::DIFFUSION_UPDATE:
            return 3;
        case Op::SCATTER_KV:
            return 5;
        case Op::RESIDUAL_NORM:
        case Op::RMS_ROTARY:
        case Op::QUANTIZED_LINEAR:
            return 4;
        case Op::EMBEDDING:
        case Op::EMBEDDING_SUM:
            // Quantized tables carry a scale operand.
            return descriptor.attribute_count >= 2 && descriptor.attributes[1] ? 3 : 2;
        case Op::CONV1D:
        case Op::SNAKE_CONV1D:
        case Op::SNAKE_QUANTIZED_LINEAR: {
            const int weight = operation == Op::CONV1D ? 1 : 2;
            if (descriptor.input_count <= weight) return weight + 1;
            return weight + (descriptor.inputs[weight].type == Type::I8 ? 3 : 2);
        }
        default:
            return 1;
    }
}

void validate_attention(const Descriptor& descriptor) {
    const auto& query = descriptor.inputs[0];
    const auto& key = descriptor.inputs[1];
    const auto& value = descriptor.inputs[2];
    const auto count = descriptor.attribute_count;
    const std::int64_t heads = count >= 1 ? descriptor.attributes[0] : 0;
    const std::int64_t key_heads = count >= 2 ? descriptor.attributes[1] : heads;
    const std::int64_t offset = count >= 3 ? descriptor.attributes[2] : 0;
    if (descriptor.input_count > 4 || query.rank != 3 || key.rank != 3 || value.rank != 3 || heads <= 0 ||
        key_heads <= 0 || heads % key_heads || query.shape[2] % heads ||
        key.shape[2] != key_heads * (query.shape[2] / heads) ||
        !std::equal(key.shape.begin(), key.shape.begin() + 3, value.shape.begin()) ||
        (key.shape[0] != 1 && key.shape[0] != query.shape[0]))
        throw std::runtime_error("CUDA: invalid attention geometry");
    const std::int64_t keys = count >= 4 ? descriptor.attributes[3] : key.shape[1];
    if (offset < 0 || keys < 0 || offset > key.shape[1] || keys > key.shape[1] - offset)
        throw std::runtime_error("CUDA: attention keys exceed the cache");
}

void infer(Descriptor& descriptor) {
    if (descriptor.input_count < 1 || descriptor.input_count > 8) throw std::runtime_error("CUDA: invalid input count");
    if (descriptor.attribute_count < 0 || descriptor.attribute_count > 32)
        throw std::runtime_error("CUDA: invalid attribute count");
    const auto required = minimum_inputs(descriptor);
    if (descriptor.input_count < required)
        throw std::runtime_error("CUDA: operation " + std::to_string(static_cast<int>(descriptor.operation)) +
                                 " requires at least " + std::to_string(required) + " inputs");
    for (int index = 0; index < descriptor.input_count; ++index) {
        descriptor.inputs[index].elements();
        switch (descriptor.inputs[index].type) {
            case Type::BOOL:
            case Type::U8:
            case Type::I8:
            case Type::I32:
            case Type::F16:
            case Type::BF16:
            case Type::F32:
                break;
            default:
                throw std::runtime_error("CUDA: unsupported input dtype");
        }
    }
    auto& output = descriptor.output;
    const auto requested_type = output.type;
    const auto& input = descriptor.inputs[0];
    output.rank = input.rank;
    output.shape = input.shape;
    if (descriptor.operation != Op::CAST) output.type = input.type;
    switch (descriptor.operation) {
        case Op::ADD:
        case Op::MULTIPLY: {
            const auto& other = descriptor.inputs[1];
            output.rank = std::max(input.rank, other.rank);
            for (int axis = 0; axis < output.rank; ++axis) {
                const int left_axis = axis - output.rank + input.rank, right_axis = axis - output.rank + other.rank;
                const auto left = left_axis < 0 ? 1 : input.shape[left_axis];
                const auto right = right_axis < 0 ? 1 : other.shape[right_axis];
                if (left != right && left != 1 && right != 1) throw std::runtime_error("CUDA: invalid broadcast");
                output.shape[axis] = left == 1 ? right : left;
            }
            if (requested_type == Type::BF16) output.type = Type::BF16;
            break;
        }
        case Op::CAST:
        case Op::GELU:
        case Op::TANH:
        case Op::STATIC_ROUND:
        case Op::RMS_NORM:
        case Op::RMS_NORM_RESIDUAL:
        case Op::LAYER_NORM:
        case Op::RESIDUAL_NORM:
        case Op::SOFTMAX:
        case Op::LOG_SOFTMAX:
        case Op::ROTARY:
        case Op::GELU_MULTIPLY:
        case Op::SILU_MULTIPLY:
        case Op::SNAKE:
            break;
        case Op::ATTENTION:
            validate_attention(descriptor);
            break;
        case Op::SCATTER:
            if (descriptor.input_count != 3 || input.rank != 3 || descriptor.inputs[1].rank != 3 ||
                descriptor.inputs[2].rank != 1 || descriptor.inputs[2].type != Type::I32 ||
                descriptor.attribute_count > 1 ||
                (descriptor.attribute_count == 1 && descriptor.attributes[0] != 1) ||
                (descriptor.inputs[1].type != input.type &&
                 !(input.type == Type::BF16 && descriptor.inputs[1].type == Type::F32)) ||
                descriptor.inputs[1].shape[0] != input.shape[0] ||
                descriptor.inputs[1].shape[2] != input.shape[2] ||
                descriptor.inputs[1].shape[1] != descriptor.inputs[2].shape[0])
                throw std::runtime_error("CUDA: invalid scatter geometry");
            break;
        case Op::SCATTER_KV: {
            const auto& value_cache = descriptor.inputs[1];
            const auto& key_updates = descriptor.inputs[2];
            const auto& value_updates = descriptor.inputs[3];
            const auto& indices = descriptor.inputs[4];
            const auto cache_accepts = [](Type cache, Type updates) {
                return cache == updates || (cache == Type::BF16 && updates == Type::F32);
            };
            const bool caches_match =
                value_cache.rank == input.rank &&
                std::equal(value_cache.shape.begin(), value_cache.shape.begin() + value_cache.rank, input.shape.begin());
            const bool updates_match =
                key_updates.rank == 3 && value_updates.rank == key_updates.rank &&
                std::equal(value_updates.shape.begin(), value_updates.shape.begin() + value_updates.rank,
                           key_updates.shape.begin()) &&
                key_updates.shape[0] == input.shape[0] && key_updates.shape[2] == input.shape[2] &&
                key_updates.shape[1] == indices.shape[0];
            if (descriptor.input_count != 5 || !caches_match || input.rank != 3 || key_updates.rank != 3 ||
                indices.rank != 1 || indices.type != Type::I32 || descriptor.attribute_count != 1 ||
                descriptor.attributes[0] != 1 || !updates_match || !cache_accepts(input.type, key_updates.type) ||
                !cache_accepts(value_cache.type, value_updates.type))
                throw std::runtime_error("CUDA: invalid key/value scatter geometry");
            break;
        }
        case Op::GELU_GATE:
        case Op::SILU_GATE:
            output.shape[input.rank - 1] /= 2;
            break;
        case Op::RMS_ROTARY:
            if (descriptor.attribute_count == 5) {
                output.rank = 4;
                output.shape = {input.shape[0], descriptor.attributes[4], descriptor.attributes[0],
                                descriptor.attributes[1]};
            }
            break;
        case Op::PACKED_LINEAR:
        case Op::QUANTIZED_LINEAR:
            output.shape[input.rank - 1] = descriptor.inputs[1].shape[0];
            if (descriptor.operation == Op::QUANTIZED_LINEAR && !descriptor.attributes[0])
                output.shape[input.rank - 1] = descriptor.inputs[1].shape[1];
            output.type = descriptor.operation == Op::PACKED_LINEAR && requested_type == Type::BF16 &&
                                  input.type == Type::BF16 && descriptor.input_count == 3 &&
                                  descriptor.attribute_count >= 2 && descriptor.attributes[0] == 8 &&
                                  descriptor.epsilon == 0
                              ? Type::BF16
                              : Type::F32;
            break;
        case Op::SNAKE_QUANTIZED_LINEAR:
            output.shape[input.rank - 1] = descriptor.inputs[2].shape[descriptor.attributes[0] ? 0 : 1];
            output.type = Type::F32;
            break;
        case Op::CONV1D:
        case Op::SNAKE_CONV1D: {
            const auto kernel = descriptor.attributes[0], stride = descriptor.attributes[1],
                       padding = descriptor.attributes[2], dilation = descriptor.attributes[3];
            output.shape[1] = (input.shape[1] + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
            output.shape[2] = descriptor.inputs[descriptor.operation == Op::SNAKE_CONV1D ? 2 : 1].shape[0];
            output.type = Type::F32;
            break;
        }
        case Op::CONV_TRANSPOSE_FOLD:
            output.shape[1] = (input.shape[1] - 1) * descriptor.attributes[1] - 2 * descriptor.attributes[2] +
                              descriptor.attributes[4] * (descriptor.attributes[0] - 1) + descriptor.attributes[3] + 1;
            output.shape[2] = input.shape[2] / descriptor.attributes[0];
            output.type = Type::F32;
            break;
        case Op::TRANSPOSE:
            for (int axis = 0; axis < input.rank; ++axis) output.shape[axis] = input.shape[descriptor.attributes[axis]];
            break;
        case Op::SLICE:
        case Op::GATHER: {
            const int axis = static_cast<int>(descriptor.attributes[0] < 0 ? descriptor.attributes[0] + input.rank
                                                                           : descriptor.attributes[0]);
            if (axis < 0 || axis >= input.rank) throw std::runtime_error("CUDA: invalid layout axis");
            output.shape[axis] =
                descriptor.operation == Op::SLICE ? descriptor.attributes[2] : descriptor.inputs[1].elements();
            break;
        }
        case Op::CONCAT: {
            const int axis = static_cast<int>(descriptor.attributes[0] < 0 ? descriptor.attributes[0] + input.rank
                                                                           : descriptor.attributes[0]);
            output.shape[axis] = 0;
            for (int index = 0; index < descriptor.input_count; ++index)
                output.shape[axis] += descriptor.inputs[index].shape[axis];
            break;
        }
        case Op::EMBEDDING:
            output.rank = 3;
            output.shape = {1, static_cast<std::int64_t>(input.elements()), descriptor.attributes[0]};
            output.type = requested_type == Type::BF16 ? Type::BF16 : Type::F32;
            break;
        case Op::GREEDY_TOKEN:
            output.rank = input.rank - 1;
            output.type = Type::I32;
            break;
        case Op::EMBEDDING_SUM:
            output.rank = 3;
            output.shape = {1, input.shape[1], descriptor.attributes[0]};
            output.type = Type::F32;
            break;
        case Op::DIFFUSION_CANDIDATES:
            output.rank = 3;
            output.shape = {descriptor.attributes[2], descriptor.attributes[3], 2};
            output.type = Type::F32;
            break;
        case Op::DIFFUSION_UPDATE:
            output.type = Type::I32;
            break;
        case Op::MATMUL:
        case Op::LINEAR: {
            const auto& right = descriptor.inputs[1];
            if (input.rank < 2 || right.rank < 2) throw std::runtime_error("CUDA: matrix rank must be at least two");
            const bool transpose = descriptor.attribute_count && descriptor.attributes[0];
            output.rank = std::max(input.rank, right.rank);
            for (int axis = 0; axis < output.rank - 2; ++axis) {
                const int left_axis = axis - output.rank + input.rank, right_axis = axis - output.rank + right.rank;
                const auto left = left_axis < 0 ? 1 : input.shape[left_axis];
                const auto other = right_axis < 0 ? 1 : right.shape[right_axis];
                if (left != other && left != 1 && other != 1)
                    throw std::runtime_error("CUDA: invalid matrix batch broadcast");
                output.shape[axis] = left == 1 ? other : left;
            }
            output.shape[output.rank - 2] = input.shape[input.rank - 2];
            output.shape[output.rank - 1] = right.shape[right.rank - (transpose ? 2 : 1)];
            if (input.shape[input.rank - 1] != right.shape[right.rank - (transpose ? 1 : 2)])
                throw std::runtime_error("CUDA: matrix contraction mismatch");
            output.type = requested_type == Type::BF16 ? Type::BF16 : Type::F32;
            break;
        }
        default:
            throw std::runtime_error("CUDA: unsupported operation " +
                                     std::to_string(static_cast<int>(descriptor.operation)));
    }
    switch (output.type) {
        case Type::BOOL:
        case Type::U8:
        case Type::I8:
        case Type::I32:
        case Type::F16:
        case Type::BF16:
        case Type::F32:
            break;
        default:
            throw std::runtime_error("CUDA: unsupported output dtype");
    }
}
} // namespace

StreamScope::StreamScope(void* stream) noexcept : previous_(execution_stream) {
    execution_stream = static_cast<cudaStream_t>(stream);
}
StreamScope::~StreamScope() { execution_stream = static_cast<cudaStream_t>(previous_); }

BindingScope::BindingScope(const TensorBinding* bindings, std::size_t count) noexcept
    : previous_(execution_bindings), previous_count_(execution_binding_count) {
    execution_bindings = bindings;
    execution_binding_count = count;
}
BindingScope::~BindingScope() {
    execution_bindings = previous_;
    execution_binding_count = previous_count_;
}

class ProjectionCache {
public:
    std::shared_ptr<PreparedProjection> get(Launch launch, int bits, int device) {
        const auto key = std::tuple{device, reinterpret_cast<std::uintptr_t>(launch.inputs[1].data),
                                    launch.inputs[0].shape[launch.inputs[0].rank - 1],
                                    launch.output.shape[launch.output.rank - 1], bits};
        std::lock_guard<std::mutex> lock(mutex_);
        auto& cached = weights_[key];
        if (auto result = cached.lock()) return result;
        auto result = std::make_shared<PreparedProjection>(launch, bits, device);
        cached = result;
        return result;
    }
    std::shared_ptr<void> workspace(int device, std::size_t bytes) {
        if (!bytes) return nullptr;
        const auto key = std::pair{device, execution_stream};
        std::lock_guard<std::mutex> lock(mutex_);
        auto& entry = workspaces_[key];
        if (entry.first && entry.second >= bytes) return entry.first;
        auto buffer = std::shared_ptr<void>(allocate(bytes, device), release);
        entry = {buffer, bytes};
        return buffer;
    }

private:
    std::mutex mutex_;
    std::map<std::tuple<int, std::uintptr_t, std::int64_t, std::int64_t, int>, std::weak_ptr<PreparedProjection>>
        weights_;
    std::map<std::pair<int, cudaStream_t>, std::pair<std::shared_ptr<void>, std::size_t>> workspaces_;
};
std::shared_ptr<ProjectionCache> make_projection_cache() { return std::make_shared<ProjectionCache>(); }

std::size_t Tensor::elements() const {
    if (rank < 0 || rank > 8) throw std::runtime_error("CUDA: unsupported tensor rank");
    std::size_t count = 1;
    for (int axis = 0; axis < rank; ++axis) {
        if (shape[axis] < 0) throw std::runtime_error("CUDA: negative tensor extent");
        if (shape[axis] && count > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(shape[axis]))
            throw std::runtime_error("CUDA: tensor size overflow");
        count *= static_cast<std::size_t>(shape[axis]);
    }
    return count;
}
int device_count() noexcept {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess ? count : 0;
}
int current_device() {
    int device = 0;
    check(cudaGetDevice(&device));
    return device;
}
bool available(int device) noexcept { return device >= 0 && device < device_count(); }
std::string device_name(int device) {
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device));
    return properties.name;
}
void* allocate(std::size_t bytes, int device) {
    DeviceScope scope(device);
    void* result = nullptr;
    check(cudaMalloc(&result, std::max<std::size_t>(bytes, 1)));
    std::lock_guard<std::mutex> lock(allocations_mutex);
    allocations.emplace(result, Allocation{bytes, device});
    live_bytes += bytes;
    device_bytes[device] += bytes;
    device_peaks[device] = std::max(device_peaks[device], device_bytes[device]);
    peak_bytes = std::max(peak_bytes, live_bytes.load());
    return result;
}
void release(void* pointer) noexcept {
    if (!pointer) return;
    std::lock_guard<std::mutex> lock(allocations_mutex);
    const auto found = allocations.find(pointer);
    if (found == allocations.end()) return;
    int previous = 0;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(found->second.device) != cudaSuccess) return;
    const auto status = cudaFree(pointer);
    cudaSetDevice(previous);
    if (status == cudaSuccess) {
        live_bytes -= found->second.bytes;
        device_bytes[found->second.device] -= found->second.bytes;
        allocations.erase(found);
    }
}
void copy_to_device(void* destination, const void* source, std::size_t bytes, int device) {
    DeviceScope scope(device);
    if (bytes) {
        check(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, execution_stream));
        check(cudaStreamSynchronize(execution_stream));
    }
}
void copy_to_host(void* destination, const void* source, std::size_t bytes, int device) {
    DeviceScope scope(device);
    if (bytes) {
        check(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToHost, execution_stream));
        check(cudaStreamSynchronize(execution_stream));
    }
}
void copy_device(void* destination, const void* source, std::size_t bytes, int device) {
    DeviceScope scope(device);
    if (bytes) check(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, execution_stream));
}
void copy_device_batch(void* const* destinations, const void* const* sources, const std::size_t* bytes,
                       std::size_t count, int device) {
    if (!count) return;
    DeviceScope scope(device);
    if (!execution_stream || std::any_of(bytes, bytes + count, [](std::size_t size) { return !size; })) {
        for (std::size_t index = 0; index < count; ++index)
            if (bytes[index])
                check(cudaMemcpyAsync(destinations[index], sources[index], bytes[index], cudaMemcpyDeviceToDevice,
                                      execution_stream));
        return;
    }
    cudaMemcpyAttributes attributes{};
    attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    std::size_t first = 0;
    check(cudaMemcpyBatchAsync(destinations, sources, bytes, count, &attributes, &first, 1, execution_stream));
}
void clear(void* pointer, std::size_t bytes, int device) {
    DeviceScope scope(device);
    if (bytes) check(cudaMemsetAsync(pointer, 0, bytes, execution_stream));
}
void synchronize(int device) {
    DeviceScope scope(device);
    check(cudaDeviceSynchronize());
}
void start_profiler() { check(cudaProfilerStart()); }
void stop_profiler() { check(cudaProfilerStop()); }
std::size_t allocated_bytes(int device) noexcept {
    if (device < 0) return live_bytes;
    std::lock_guard<std::mutex> lock(allocations_mutex);
    std::size_t bytes = 0;
    for (const auto& allocation : allocations)
        if (allocation.second.device == device) bytes += allocation.second.bytes;
    return bytes;
}
std::size_t peak_allocated_bytes(int device) noexcept {
    std::lock_guard<std::mutex> lock(allocations_mutex);
    if (device < 0) return peak_bytes;
    const auto found = device_peaks.find(device);
    return found == device_peaks.end() ? 0 : found->second;
}

const DenseDevice& dense_device(int device) {
    static thread_local std::map<int, DenseDevice> devices;
    auto found = devices.find(device);
    if (found == devices.end()) {
        DenseDevice properties{};
        check(cudaDeviceGetAttribute(&properties.major, cudaDevAttrComputeCapabilityMajor, device));
        check(cudaDeviceGetAttribute(&properties.multiprocessors, cudaDevAttrMultiProcessorCount, device));
        found = devices.emplace(device, properties).first;
    }
    return found->second;
}

struct Program::Impl {
    Descriptor descriptor;
    int device;
    std::unique_ptr<void, decltype(&release)> scratch{nullptr, release};
    std::unique_ptr<void, decltype(&release)> scales{nullptr, release};
    std::unique_ptr<void, decltype(&release)> zeros{nullptr, release};
    std::unique_ptr<void, decltype(&release)> columns{nullptr, release};
    std::unique_ptr<void, decltype(&release)> accumulators{nullptr, release};
    std::unique_ptr<void, decltype(&release)> attention_partials{nullptr, release};
    std::unique_ptr<void, decltype(&release)> token_candidates{nullptr, release};
    std::shared_ptr<PreparedProjection> prepared;
    std::shared_ptr<void> shared_workspace;
    const void* prepared_weight = nullptr;
    AttentionLauncher attention_launcher = nullptr;
    bool tensor_projection = false;
    bool vector_projection = false;
    bool w8a16_projection = false;
    bool fused_w8a8 = false;
    int dense_slices = 1;
    explicit Impl(Descriptor value, int selected_device) : descriptor(value), device(selected_device) {
        DeviceScope scope(device);
        infer(descriptor);
        descriptor.output.device = device;
        const auto operation = descriptor.operation;
        const auto& input = descriptor.inputs[0];
        const bool convolution = operation == Op::CONV1D || operation == Op::SNAKE_CONV1D;
        const bool snake_linear = operation == Op::SNAKE_QUANTIZED_LINEAR;
        const int weight_index = operation == Op::SNAKE_CONV1D || snake_linear ? 2 : 1;
        const auto last_width = input.rank ? input.shape[input.rank - 1] : 0;
        const auto rows = convolution  ? descriptor.output.shape[1]
                          : last_width ? static_cast<std::int64_t>(input.elements() / last_width)
                                       : 0;
        const auto width = convolution ? input.shape[2] * descriptor.attributes[0] : last_width;
#if defined(KIDI_CUDA_USE_CUTLASS_INT8)
        const bool bfloat = input.type == Type::BF16;
        if (!descriptor.reference_projection && (operation == Op::LINEAR || operation == Op::MATMUL) &&
            descriptor.input_count == 2 && (bfloat || input.type == Type::F32) &&
            descriptor.inputs[1].type == input.type && descriptor.output.type == input.type &&
            descriptor.inputs[1].rank == 2 && descriptor.attribute_count && descriptor.attributes[0] && width > 0 &&
            width <= INT32_MAX && width % (bfloat ? 8 : 4) == 0 && rows <= 256) {
            const auto output_columns = descriptor.output.shape[descriptor.output.rank - 1];
            if (output_columns > 0 && output_columns <= 4096 && output_columns % 8 == 0) {
                const auto& properties = dense_device(device);
                if (properties.major >= 8) {
                    dense_slices = dense_gemm_slices(static_cast<int>(rows), static_cast<int>(output_columns),
                                                     static_cast<int>(width), properties.multiprocessors, bfloat);
                    if (dense_slices > 1) {
                        const auto bytes = descriptor.output.elements() * dense_slices * sizeof(float);
                        if (descriptor.projection_cache)
                            shared_workspace = descriptor.projection_cache->workspace(device, bytes);
                        else
                            scratch.reset(allocate(bytes, device));
                    }
                }
            }
        }
#endif
        if (convolution || snake_linear) columns.reset(allocate(rows * width * sizeof(float), device));
        if (operation == Op::PACKED_LINEAR && descriptor.output.type == Type::BF16) {
            w8a16_projection = true;
#if defined(KIDI_CUDA_USE_CUTLASS_INT8)
            const auto output_columns = descriptor.output.shape[descriptor.output.rank - 1];
            const auto group_size = descriptor.attributes[1];
            if (!descriptor.reference_projection && rows > 4 && rows <= 256 && width > 0 && width <= INT32_MAX &&
                width % 64 == 0 && group_size > 0 && group_size % 64 == 0 && width % group_size == 0 &&
                output_columns > 0 && output_columns <= INT32_MAX && output_columns % 8 == 0) {
                const auto& properties = dense_device(device);
                if (properties.major >= 8) {
                    dense_slices = w8a16_gemm_slices(static_cast<int>(rows), static_cast<int>(output_columns),
                                                     static_cast<int>(width), static_cast<int>(group_size),
                                                     properties.multiprocessors);
                    if (dense_slices > 1) {
                        const auto bytes = descriptor.output.elements() * dense_slices * sizeof(float);
                        if (descriptor.projection_cache)
                            shared_workspace = descriptor.projection_cache->workspace(device, bytes);
                        else
                            scratch.reset(allocate(bytes, device));
                    }
                }
            }
#endif
        } else if (operation == Op::PACKED_LINEAR || operation == Op::QUANTIZED_LINEAR ||
                   ((convolution || snake_linear) && descriptor.inputs[weight_index].type == Type::I8)) {
            scratch.reset(allocate(rows * width, device));
            scales.reset(allocate(rows * sizeof(float), device));
            zeros.reset(allocate(rows * sizeof(int), device));
            const auto output_columns = descriptor.output.shape[descriptor.output.rank - 1];
            const bool transposed =
                convolution || snake_linear || (operation == Op::QUANTIZED_LINEAR && descriptor.attributes[0]);
            const bool supported_weights =
                operation == Op::QUANTIZED_LINEAR || convolution || snake_linear ||
                (operation == Op::PACKED_LINEAR &&
                 (descriptor.attributes[0] == 2 || descriptor.attributes[0] == 4 || descriptor.attributes[0] == 8) &&
                 descriptor.attributes[1] == width);
            cudaDeviceProp properties{};
            check(cudaGetDeviceProperties(&properties, device));
            const auto bits = operation == Op::PACKED_LINEAR ? descriptor.attributes[0] : 8;
            vector_projection = !descriptor.reference_projection && properties.major >= 7 &&
                                (transposed || operation == Op::PACKED_LINEAR) &&
                                (bits == 2 || bits == 4 || bits == 8) &&
                                (operation != Op::PACKED_LINEAR || descriptor.attributes[1] == width) && rows > 0 &&
                                rows <= 4 && width > 0 && width <= 65536 && width % (32 / bits) == 0 &&
                                output_columns > 0 && output_columns <= INT32_MAX;
#if defined(KIDI_CUDA_USE_CUTLASS_INT8)
            constexpr int MIN_MAJOR = 8, WIDTH_ALIGNMENT = 16;
            tensor_projection = !descriptor.reference_projection && properties.major >= MIN_MAJOR &&
                                (transposed || operation == Op::PACKED_LINEAR) && supported_weights && width > 0 &&
                                width <= 65536 && width % WIDTH_ALIGNMENT == 0 && output_columns > 0 &&
                                output_columns <= INT32_MAX && output_columns % 4 == 0 && rows > 0 && rows <= INT32_MAX;
            const int scale_index = weight_index + 1;
            const bool f32_scales = descriptor.inputs[scale_index].type == Type::F32 &&
                                    (operation != Op::QUANTIZED_LINEAR || descriptor.input_count <= weight_index + 2 ||
                                     descriptor.inputs[weight_index + 2].type == Type::F32);
            const bool f32_or_bf16_out = descriptor.output.type == Type::F32 || descriptor.output.type == Type::BF16;
            fused_w8a8 = tensor_projection && !vector_projection && rows > 4 && width % 64 == 0 &&
                         output_columns % 8 == 0 && f32_scales && f32_or_bf16_out;
            if (fused_w8a8) {
                dense_slices = w8a8_gemm_slices(static_cast<int>(rows), static_cast<int>(output_columns),
                                                static_cast<int>(width), dense_device(device).multiprocessors);
                if (dense_slices > 1) {
                    const auto bytes = descriptor.output.elements() * dense_slices * sizeof(int);
                    if (descriptor.projection_cache)
                        shared_workspace = descriptor.projection_cache->workspace(device, bytes);
                    else
                        accumulators.reset(allocate(bytes, device));
                }
            } else if (tensor_projection && !vector_projection) {
                accumulators.reset(allocate(descriptor.output.elements() * sizeof(int), device));
            }
#endif
        } else if (operation == Op::DIFFUSION_UPDATE) {
            scratch.reset(allocate(input.elements() * sizeof(float), device));
        } else if (operation == Op::GREEDY_TOKEN && input.shape[input.rank - 1] > 4096) {
            token_candidates.reset(allocate(
                descriptor.output.elements() * ((input.shape[input.rank - 1] + 2047) / 2048) * token_candidate_bytes(),
                device));
        } else if (operation == Op::ATTENTION) {
            if ((descriptor.inputs[1].type != Type::F32 && descriptor.inputs[1].type != Type::BF16) ||
                descriptor.inputs[2].type != descriptor.inputs[1].type)
                throw std::runtime_error("CUDA: quantized KV attention not yet supported");
            const auto keys =
                descriptor.attribute_count >= 4 ? descriptor.attributes[3] : descriptor.inputs[1].shape[1];
            const auto head_width = input.shape[2] / descriptor.attributes[0];
            if (!descriptor.reference_attention && (input.type == Type::BF16 || input.type == Type::F32) &&
                (descriptor.inputs[1].type == input.type ||
                 (input.type == Type::F32 && descriptor.inputs[1].type == Type::BF16)) &&
                descriptor.output.type == input.type &&
                input.shape[1] == 1 && (head_width == 256 || head_width == 512))
                attention_partials.reset(allocate(
                    input.shape[0] * descriptor.attributes[0] *
                        ((keys + DECODE_ATTENTION_PARTITION_SIZE - 1) / DECODE_ATTENTION_PARTITION_SIZE) *
                        (head_width + 2) * sizeof(float),
                    device));
            if (!attention_partials) {
                attention_launcher = select_parallel_attention(descriptor, keys, head_width, device);
                if (!attention_launcher)
                    scratch.reset(allocate(
                        input.shape[0] * descriptor.attributes[0] * input.shape[1] * keys * sizeof(float), device));
            }
        }
    }
    void floating(Launch launch) {
        floating_projection(launch, device, descriptor.reference_projection, shared_workspace.get(), scratch.get(),
                            dense_slices, execution_stream);
    }
    void project(Launch launch) {
#if defined(KIDI_CUDA_USE_CUTLASS_INT8)
        const auto output_columns = launch.output.shape[launch.output.rank - 1];
        const auto width = launch.inputs[0].shape[launch.inputs[0].rank - 1];
        const bool need_prepared =
            tensor_projection &&
            !(vector_projection && reinterpret_cast<std::uintptr_t>(launch.inputs[1].data) % 4 == 0) &&
            reinterpret_cast<std::uintptr_t>(launch.inputs[1].data) % 16 == 0;
        if (need_prepared) {
            const auto bits = launch.operation == static_cast<int>(Op::PACKED_LINEAR) ? launch.attributes[0] : 8;
            if (!descriptor.immutable_weights || prepared_weight != launch.inputs[1].data) {
                if (descriptor.immutable_weights && descriptor.projection_cache)
                    prepared = descriptor.projection_cache->get(launch, bits, device);
                else if (!prepared || prepared_weight != launch.inputs[1].data)
                    prepared = std::make_shared<PreparedProjection>(launch, bits, device);
                else
                    prepared->prepare(launch, bits);
                check(cudaStreamWaitEvent(execution_stream, prepared->ready));
                prepared_weight = launch.inputs[1].data;
            }
        }
        (void)output_columns;
        (void)width;
#endif
        quantized_projection(launch, device, descriptor.reference_projection, vector_projection, tensor_projection,
                             fused_w8a8, scratch.get(), scales.get(), zeros.get(),
                             prepared ? (prepared->expanded ? prepared->expanded.get() : nullptr) : nullptr,
                             prepared ? prepared->sums.get() : nullptr, accumulators.get(), shared_workspace.get(),
                             dense_slices, execution_stream);
    }
    void project_w8a16(Launch launch) {
        kidi::cuda::w8a16_projection(launch, device, descriptor.reference_projection, shared_workspace.get(),
                                     scratch.get(), dense_slices, execution_stream);
    }
};
Program::Program(Descriptor descriptor, int device) : impl_(std::make_unique<Impl>(descriptor, device)) {}
Program::~Program() = default;
const Tensor& Program::output() const { return impl_->descriptor.output; }
void Program::run(const Tensor* inputs, int count, void* output) {
    DeviceScope scope(impl_->device);
    const auto& descriptor = impl_->descriptor;
    if (count != descriptor.input_count) throw std::runtime_error("CUDA: input count changed");
    Launch launch{};
    launch.operation = static_cast<int>(descriptor.operation);
    launch.count = count;
    launch.attribute_count = descriptor.attribute_count;
    launch.epsilon = descriptor.epsilon;
    std::copy(descriptor.attributes.begin(), descriptor.attributes.end(), launch.attributes);
    for (int index = 0; index < count; ++index) {
        if (inputs[index].device != impl_->device)
            throw std::runtime_error("CUDA: operands belong to different devices");
        const auto& expected = descriptor.inputs[index];
        if (inputs[index].rank != expected.rank || inputs[index].type != expected.type ||
            !std::equal(inputs[index].shape.begin(), inputs[index].shape.begin() + inputs[index].rank,
                        expected.shape.begin()))
            throw std::runtime_error("CUDA: operand shape or dtype changed after preparation");
        if (!inputs[index].data && inputs[index].elements()) throw std::runtime_error("CUDA: null input pointer");
        launch.inputs[index] = view(inputs[index]);
        bind(launch.inputs[index]);
    }
    launch.output = view(descriptor.output);
    launch.output.data = output;
    bind(launch.output);
    launch.elements = descriptor.output.elements();
    if (!launch.elements) return;
    if (!output) throw std::runtime_error("CUDA: null output pointer");
    const auto operation = descriptor.operation;
    const bool relocated =
        launch.output.binding ||
        std::any_of(launch.inputs, launch.inputs + count, [](const View& tensor) { return tensor.binding != nullptr; });
    if (relocated && operation != Op::SCATTER && operation != Op::SCATTER_KV &&
        !(operation == Op::ATTENTION && descriptor.inputs[0].shape[1] == 1))
        throw std::runtime_error("CUDA: relocated operands are only supported by scatter and single-query attention");
    if (operation == Op::CONV1D || operation == Op::SNAKE_CONV1D || operation == Op::SNAKE_QUANTIZED_LINEAR) {
        const bool convolution = operation != Op::SNAKE_QUANTIZED_LINEAR;
        const bool snake = operation != Op::CONV1D;
        const auto rows = convolution ? launch.output.shape[1]
                                      : static_cast<std::int64_t>(descriptor.inputs[0].elements() /
                                                                  launch.inputs[0].shape[launch.inputs[0].rank - 1]);
        const auto width = convolution ? launch.inputs[0].shape[2] * launch.attributes[0]
                                       : launch.inputs[0].shape[launch.inputs[0].rank - 1];
        if (convolution) {
            unfold_convolution(launch, static_cast<float*>(impl_->columns.get()), rows * width, execution_stream);
        } else {
            Launch activated = launch;
            activated.operation = static_cast<int>(Op::SNAKE);
            activated.elements = descriptor.inputs[0].elements();
            activated.output = launch.inputs[0];
            activated.output.data = impl_->columns.get();
            pointwise(activated, execution_stream);
        }
        const int weight = snake ? 2 : 1;
        const bool quantized = descriptor.inputs[weight].type == Type::I8;
        Launch projected = launch;
        projected.inputs[0] = {impl_->columns.get(), static_cast<int>(Type::F32), 3, {1, rows, width}};
        projected.inputs[1] = launch.inputs[weight];
        projected.count = quantized ? 4 : 3;
        projected.inputs[2] = launch.inputs[weight + 1];
        if (quantized) projected.inputs[3] = launch.inputs[weight + 2];
        projected.attribute_count = 1;
        projected.attributes[0] = 1;
        projected.operation = static_cast<int>(quantized ? Op::QUANTIZED_LINEAR : Op::LINEAR);
        if (quantized) {
            quantize_rows(projected, rows, impl_->scratch.get(), impl_->scales.get(), impl_->zeros.get(),
                          execution_stream);
            impl_->project(projected);
        } else {
            execute_matrix(projected, impl_->device, true, nullptr, 1, execution_stream);
        }
    } else if (operation == Op::CONV_TRANSPOSE_FOLD) {
        fold_convolution(launch, execution_stream);
    } else if (operation == Op::MATMUL || operation == Op::LINEAR) {
        impl_->floating(launch);
    } else if (operation == Op::PACKED_LINEAR || operation == Op::QUANTIZED_LINEAR) {
        if (impl_->w8a16_projection) {
            impl_->project_w8a16(launch);
        } else {
            const auto rows =
                descriptor.inputs[0].elements() / descriptor.inputs[0].shape[descriptor.inputs[0].rank - 1];
            quantize_rows(launch, rows, impl_->scratch.get(), impl_->scales.get(), impl_->zeros.get(),
                          execution_stream);
            impl_->project(launch);
        }
    } else if (operation == Op::RMS_NORM || operation == Op::RMS_NORM_RESIDUAL || operation == Op::RMS_ROTARY ||
               operation == Op::LAYER_NORM || operation == Op::RESIDUAL_NORM || operation == Op::SOFTMAX ||
               operation == Op::LOG_SOFTMAX) {
        normalize(launch, operation, execution_stream);
    } else if (operation == Op::SCATTER && output == inputs[0].data) {
        const auto bytes = [](Type type) -> std::size_t {
            return type == Type::F32 || type == Type::I32 ? 4 : type == Type::F16 || type == Type::BF16 ? 2 : 1;
        };
        const auto begin = reinterpret_cast<std::uintptr_t>(output);
        const auto end = begin + descriptor.output.elements() * bytes(descriptor.output.type);
        for (int index = 1; index < 3; ++index) {
            const auto source = reinterpret_cast<std::uintptr_t>(inputs[index].data);
            if (source < end && begin < source + inputs[index].elements() * bytes(inputs[index].type))
                throw std::runtime_error("CUDA: in-place scatter requires nonoverlapping updates and indices");
        }
        launch.elements = inputs[1].elements();
        scatter_updates(launch, execution_stream);
    } else if (operation == Op::SCATTER_KV && output == inputs[0].data) {
        const auto bytes = [](Type type) -> std::size_t {
            return type == Type::F32 || type == Type::I32 ? 4 : type == Type::F16 || type == Type::BF16 ? 2 : 1;
        };
        const auto overlaps = [&](int left, int right) {
            const auto left_begin = reinterpret_cast<std::uintptr_t>(inputs[left].data);
            const auto right_begin = reinterpret_cast<std::uintptr_t>(inputs[right].data);
            const auto left_end = left_begin + inputs[left].elements() * bytes(inputs[left].type);
            const auto right_end = right_begin + inputs[right].elements() * bytes(inputs[right].type);
            return left_begin < right_end && right_begin < left_end;
        };
        bool source_overlaps_cache = false;
        for (int source = 2; source < 5; ++source)
            source_overlaps_cache = source_overlaps_cache || overlaps(0, source) || overlaps(1, source);
        if (overlaps(0, 1) || output != inputs[0].data || source_overlaps_cache)
            throw std::runtime_error("CUDA: in-place key/value scatter requires nonoverlapping operands");
        launch.elements = inputs[2].elements();
        scatter_key_value_updates(launch, execution_stream);
    } else if (operation == Op::SLICE) {
        const auto& source = inputs[0];
        const auto axis =
            descriptor.attributes[0] < 0 ? descriptor.attributes[0] + source.rank : descriptor.attributes[0];
        std::size_t inner = 1, outer = 1;
        for (int dimension = 0; dimension < axis; ++dimension) outer *= source.shape[dimension];
        for (int dimension = static_cast<int>(axis) + 1; dimension < source.rank; ++dimension)
            inner *= source.shape[dimension];
        const auto type = source.type;
        const std::size_t bytes = type == Type::F64 || type == Type::I64 || type == Type::U64   ? 8
                                  : type == Type::F32 || type == Type::I32 || type == Type::U32 ? 4
                                  : type == Type::F16 || type == Type::BF16 || type == Type::I16 || type == Type::U16
                                      ? 2
                                      : 1;
        const auto pitch = source.shape[axis] * inner * bytes;
        const auto width = descriptor.output.shape[axis] * inner * bytes;
        const auto* source_bytes =
            static_cast<const unsigned char*>(source.data) + descriptor.attributes[1] * inner * bytes;
        check(cudaMemcpy2DAsync(output, width, source_bytes, pitch, width, outer, cudaMemcpyDeviceToDevice,
                                execution_stream));
    } else if (operation == Op::TRANSPOSE || operation == Op::GATHER || operation == Op::CONCAT ||
               operation == Op::SCATTER) {
        layout(launch, execution_stream);
    } else if (operation == Op::GELU_GATE || operation == Op::SILU_GATE) {
        gated_activation(launch, execution_stream);
    } else if (operation == Op::EMBEDDING) {
        embedding(launch, execution_stream);
    } else if (operation == Op::GREEDY_TOKEN) {
        greedy_token(launch, impl_->token_candidates.get(), execution_stream);
    } else if (operation == Op::EMBEDDING_SUM) {
        embedding_sum(launch, execution_stream);
    } else if (operation == Op::DIFFUSION_CANDIDATES) {
        diffusion_candidates(launch, execution_stream);
    } else if (operation == Op::DIFFUSION_UPDATE) {
        diffusion_update(launch, static_cast<float*>(impl_->scratch.get()), execution_stream);
    } else if (operation == Op::ATTENTION) {
        const auto heads = descriptor.attributes[0],
                   key_heads = descriptor.attribute_count >= 2 ? descriptor.attributes[1] : heads;
        const auto keys = descriptor.attribute_count >= 4 ? descriptor.attributes[3] : descriptor.inputs[1].shape[1];
        const auto width = descriptor.inputs[0].shape[2] / heads;
        if (impl_->attention_partials) {
            decode_attention(launch, static_cast<float*>(impl_->attention_partials.get()),
                             descriptor.inputs[0].shape[0], heads, key_heads, keys, width,
                             static_cast<int>(descriptor.inputs[0].type), execution_stream);
            check(cudaGetLastError());
            return;
        }
        if (impl_->attention_launcher) {
            impl_->attention_launcher(launch, heads, key_heads, keys, execution_stream);
            return;
        }
        fallback_attention(launch, static_cast<float*>(impl_->scratch.get()), descriptor.inputs[0].shape[0], heads,
                           key_heads, descriptor.inputs[0].shape[1], keys, execution_stream);
    } else {
        pointwise(launch, execution_stream);
    }
    check(cudaGetLastError());
}
} // namespace kidi::cuda