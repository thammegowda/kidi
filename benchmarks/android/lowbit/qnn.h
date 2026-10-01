#pragma once

#if defined(KIDI_BENCH_QNN)

#include <HTP/QnnHtpDevice.h>
#include <QnnInterface.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace kidi::bench::qnn {

auto check(Qnn_ErrorHandle_t status, const std::string& operation) -> void;
auto read_file(const std::string& path) -> std::vector<std::uint8_t>;
auto write_file(const std::string& path, std::span<const std::uint8_t> bytes) -> void;
auto library_loaded(const char* name) -> bool;

// One HTP backend, device, and context loaded through dlopen, with a sustained-performance power vote.
class Session {
public:
    explicit Session(const std::string& library);
    ~Session();
    Session(const Session&) = delete;
    auto operator=(const Session&) -> Session& = delete;

    auto create_context() -> void;
    auto load_context(std::span<const std::uint8_t> binary) -> void;
    auto save_context() -> std::vector<std::uint8_t>;
    auto create_graph(const std::string& name) -> Qnn_GraphHandle_t;
    auto retrieve_graph(const std::string& name) -> Qnn_GraphHandle_t;
    auto register_memory(int fd, std::span<const std::uint32_t> dims, Qnn_DataType_t data_type) -> Qnn_MemHandle_t;

    QNN_INTERFACE_VER_TYPE api{};
    Qnn_ContextHandle_t context{};
    std::string api_version;
    std::string build_id;
    bool burst = false;

private:
    auto request_burst() -> void;

    void* library_{};
    Qnn_LogHandle_t log_{};
    Qnn_BackendHandle_t backend_{};
    Qnn_DeviceHandle_t device_{};
    const QnnHtpDevice_PerfInfrastructure_t* perf_{};
    std::uint32_t power_{};
};

auto make_tensor(const char* name, Qnn_TensorType_t type, Qnn_DataType_t data_type,
                 std::vector<std::uint32_t>& dims) -> Qnn_Tensor_t;
auto per_tensor(float scale, std::int32_t offset) -> Qnn_QuantizeParams_t;

// Static signed weights with per-channel scales along `axis`. Sub-byte weights use either the documented 8-bit
// container (bitwidth-annotated) or tightly packed QNN sub-byte types. QNN copies static data at registration.
struct Weights {
    std::vector<std::uint8_t> data;
    std::vector<float> scales;
    std::vector<Qnn_ScaleOffset_t> scale_offsets;
    Qnn_DataType_t data_type = QNN_DATATYPE_SFIXED_POINT_8;
    Qnn_QuantizeParams_t params{};
    std::string encoding;
};
auto make_weights(std::span<const std::int8_t> values, std::span<const float> scales, std::int32_t axis, int bits,
                  bool packed) -> Weights;

// FastRPC shared memory: one allocation visible to the CPU and HTP without per-execution copies.
class SharedMemory {
public:
    struct Block {
        void* data{};
        int fd = -1;
        std::size_t size = 0;
    };
    SharedMemory();
    ~SharedMemory();
    SharedMemory(const SharedMemory&) = delete;
    auto operator=(const SharedMemory&) -> SharedMemory& = delete;
    auto allocate(std::size_t size) -> Block;

private:
    void* library_{};
    void* (*alloc_)(int, std::uint32_t, int) = nullptr;
    void (*free_)(void*) = nullptr;
    int (*to_fd_)(void*) = nullptr;
    std::vector<void*> blocks_;
};

} // namespace kidi::bench::qnn

#endif
