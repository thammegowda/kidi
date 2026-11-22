#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace kidi::cuda {

enum class Type { BOOL, U8, I8, U16, I16, U32, I32, U64, I64, F16, BF16, F32, F64, E4M3, E5M2 };
enum class Op {
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
    SILU_MULTIPLY,
    SILU_GATE,
    EMBEDDING,
    GATED_FEED_FORWARD,
    DIFFUSION_CANDIDATES,
    DIFFUSION_UPDATE,
    EMBEDDING_SUM,
    CONV1D,
    SNAKE_CONV1D,
    SNAKE_QUANTIZED_LINEAR,
    CONV_TRANSPOSE_FOLD,
    SNAKE,
    GELU_GATE,
    SCATTER_KV
};

// Optional work a PACKED_LINEAR applies to its input inside the activation quantization (attributes[3]).
enum class ProjectionPrologue { NONE, RMS_NORM, GELU_GATE };

struct Tensor {
    void* data = nullptr;
    Type type = Type::F32;
    int rank = 0;
    std::array<std::int64_t, 8> shape{};
    int device = 0;
    std::size_t elements() const;
};

class ProjectionCache;
std::shared_ptr<ProjectionCache> make_projection_cache();

struct Descriptor {
    Op operation = Op::ADD;
    std::array<Tensor, 8> inputs{};
    int input_count = 0;
    Tensor output;
    std::array<std::int64_t, 32> attributes{};
    int attribute_count = 0;
    float epsilon = 0;
    bool reference_projection = false;
    bool reference_attention = false;
    bool immutable_weights = false;
    std::shared_ptr<ProjectionCache> projection_cache;
};

int device_count() noexcept;
int current_device();
bool available(int device = 0) noexcept;
std::string device_name(int device = 0);
void* allocate(std::size_t bytes, int device = 0);
void release(void* pointer) noexcept;
void copy_to_device(void* destination, const void* source, std::size_t bytes, int device = 0);
void copy_to_host(void* destination, const void* source, std::size_t bytes, int device = 0);
void copy_device(void* destination, const void* source, std::size_t bytes, int device = 0);
void copy_device_batch(void* const* destinations, const void* const* sources, const std::size_t* bytes,
                       std::size_t count, int device = 0);
void clear(void* pointer, std::size_t bytes, int device = 0);
void synchronize(int device = 0);
void start_profiler();
void stop_profiler();
std::size_t allocated_bytes(int device = -1) noexcept;
std::size_t peak_allocated_bytes(int device = -1) noexcept;

// A bound operand is read through its device-side slot; only scatter and single-query attention accept one.
struct TensorBinding {
    const void* data;
    std::size_t bytes;
    void* const* slot;
};

class BindingScope {
public:
    BindingScope(const TensorBinding* bindings, std::size_t count) noexcept;
    ~BindingScope();
    BindingScope(const BindingScope&) = delete;
    BindingScope& operator=(const BindingScope&) = delete;

private:
    const TensorBinding* previous_;
    std::size_t previous_count_;
};

class StreamScope {
public:
    explicit StreamScope(void* stream) noexcept;
    ~StreamScope();
    StreamScope(const StreamScope&) = delete;
    StreamScope& operator=(const StreamScope&) = delete;

private:
    void* previous_;
};

class Program {
public:
    explicit Program(Descriptor descriptor, int device = 0);
    ~Program();
    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;
    const Tensor& output() const;
    void run(const Tensor* inputs, int count, void* output);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace kidi::cuda