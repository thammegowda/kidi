#if defined(KIDI_BENCH_QNN)

#include "qnn.h"

#include <HTP/QnnHtpCommon.h>
#include <HTP/QnnHtpGraph.h>
#include <HTP/QnnHtpPerfInfrastructure.h>

#include <dlfcn.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace kidi::bench::qnn {
namespace {

using GetProviders = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, std::uint32_t*);

auto log_callback(const char* format, QnnLog_Level_t level, std::uint64_t, va_list arguments) -> void {
    std::fprintf(stderr, "[qnn:%d] ", static_cast<int>(level));
    std::vfprintf(stderr, format, arguments);
    std::fputc('\n', stderr);
}

} // namespace

auto check(Qnn_ErrorHandle_t status, const std::string& operation) -> void {
    if (status != QNN_SUCCESS)
        throw std::runtime_error(operation + ": QNN error " + std::to_string(QNN_GET_ERROR_CODE(status)));
}

auto read_file(const std::string& path) -> std::vector<std::uint8_t> {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot read " + path);
    return {std::istreambuf_iterator<char>(stream), {}};
}

auto write_file(const std::string& path, std::span<const std::uint8_t> bytes) -> void {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!stream) throw std::runtime_error("cannot write " + path);
}

auto library_loaded(const char* name) -> bool {
    void* handle = dlopen(name, RTLD_NOW | RTLD_NOLOAD);
    if (handle) dlclose(handle);
    return handle != nullptr;
}

Session::Session(const std::string& library) {
    library_ = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library_) throw std::runtime_error("load " + library + ": " + dlerror());
    const auto get = reinterpret_cast<GetProviders>(dlsym(library_, "QnnInterface_getProviders"));
    if (!get) throw std::runtime_error(library + " does not export QnnInterface_getProviders");
    const QnnInterface_t** providers = nullptr;
    std::uint32_t count = 0;
    check(get(&providers, &count), "get QNN providers");
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto& version = providers[index]->apiVersion.coreApiVersion;
        if (providers[index]->backendId == QNN_BACKEND_ID_HTP && version.major == QNN_API_VERSION_MAJOR) {
            api = providers[index]->QNN_INTERFACE_VER_NAME;
            api_version = std::to_string(version.major) + "." + std::to_string(version.minor) + "." +
                          std::to_string(version.patch);
            break;
        }
    }
    if (api_version.empty()) throw std::runtime_error(library + " has no compatible QNN HTP provider");
    if (const char* level = std::getenv("KIDI_QNN_LOG"))
        check(api.logCreate(log_callback, static_cast<QnnLog_Level_t>(std::atoi(level)), &log_), "create log");
    check(api.backendCreate(log_, nullptr, &backend_), "create HTP backend");
    const char* build = nullptr;
    if (api.backendGetBuildId && api.backendGetBuildId(&build) == QNN_SUCCESS && build) build_id = build;
    check(api.deviceCreate(log_, nullptr, &device_), "create HTP device");
    request_burst();
}

// The library stays loaded: unloading HTP runtime libraries during process exit races FastRPC teardown.
Session::~Session() {
    if (context) api.contextFree(context, nullptr);
    if (perf_ && power_) perf_->destroyPowerConfigId(power_);
    if (device_) api.deviceFree(device_);
    if (backend_) api.backendFree(backend_);
    if (log_) api.logFree(log_);
}

// Sustained-performance power vote: DCVS off, maximum voltage corners, and low-latency RPC polling.
auto Session::request_burst() -> void {
    QnnDevice_Infrastructure_t infrastructure = nullptr;
    if (!api.deviceGetInfrastructure || api.deviceGetInfrastructure(&infrastructure) != QNN_SUCCESS || !infrastructure)
        return;
    auto* htp = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(infrastructure);
    if (htp->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) return;
    perf_ = &htp->perfInfra;
    check(perf_->createPowerConfigId(0, 0, &power_), "create HTP power config");
    QnnHtpPerfInfrastructure_PowerConfig_t dcvs = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    auto& vote = dcvs.dcvsV3Config;
    vote.contextId = power_;
    vote.setDcvsEnable = 1;
    vote.dcvsEnable = 0;
    vote.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    vote.setSleepLatency = 1;
    vote.sleepLatency = 40;
    vote.setSleepDisable = 1;
    vote.sleepDisable = 1;
    vote.setBusParams = 1;
    vote.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    vote.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    vote.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    vote.setCoreParams = 1;
    vote.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    vote.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    vote.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    QnnHtpPerfInfrastructure_PowerConfig_t latency = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    latency.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    latency.rpcControlLatencyConfig = 100;
    QnnHtpPerfInfrastructure_PowerConfig_t polling = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    polling.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    polling.rpcPollingTimeConfig = 9999;
    const QnnHtpPerfInfrastructure_PowerConfig_t* configs[] = {&dcvs, &latency, &polling, nullptr};
    check(perf_->setPowerConfig(power_, configs), "set HTP performance mode");
    burst = true;
}

auto Session::create_context() -> void {
    check(api.contextCreate(backend_, device_, nullptr, &context), "create context");
}

auto Session::load_context(std::span<const std::uint8_t> binary) -> void {
    check(api.contextCreateFromBinary(backend_, device_, nullptr, binary.data(), binary.size(), &context, nullptr),
          "load context binary");
}

auto Session::save_context() -> std::vector<std::uint8_t> {
    Qnn_ContextBinarySize_t size = 0;
    check(api.contextGetBinarySize(context, &size), "get context binary size");
    std::vector<std::uint8_t> binary(size);
    Qnn_ContextBinarySize_t written = 0;
    check(api.contextGetBinary(context, binary.data(), size, &written), "get context binary");
    binary.resize(written);
    return binary;
}

auto Session::create_graph(const std::string& name) -> Qnn_GraphHandle_t {
    QnnHtpGraph_CustomConfig_t optimization = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    optimization.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
    optimization.optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
    optimization.optimizationOption.floatValue = 3;
    QnnGraph_Config_t config = QNN_GRAPH_CONFIG_INIT;
    config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    config.customConfig = &optimization;
    const QnnGraph_Config_t* configs[] = {&config, nullptr};
    Qnn_GraphHandle_t graph{};
    check(api.graphCreate(context, name.c_str(), configs, &graph), "create graph " + name);
    return graph;
}

auto Session::retrieve_graph(const std::string& name) -> Qnn_GraphHandle_t {
    Qnn_GraphHandle_t graph{};
    check(api.graphRetrieve(context, name.c_str(), &graph), "retrieve graph " + name);
    return graph;
}

auto Session::register_memory(int fd, std::span<const std::uint32_t> dims, Qnn_DataType_t data_type)
    -> Qnn_MemHandle_t {
    Qnn_MemDescriptor_t descriptor = QNN_MEM_DESCRIPTOR_INIT;
    descriptor.memShape = {static_cast<std::uint32_t>(dims.size()), const_cast<std::uint32_t*>(dims.data()), nullptr};
    descriptor.dataType = data_type;
    descriptor.memType = QNN_MEM_TYPE_ION;
    descriptor.ionInfo.fd = fd;
    Qnn_MemHandle_t handle{};
    check(api.memRegister(context, &descriptor, 1, &handle), "register shared memory");
    return handle;
}

auto make_tensor(const char* name, Qnn_TensorType_t type, Qnn_DataType_t data_type, std::vector<std::uint32_t>& dims)
    -> Qnn_Tensor_t {
    Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
    tensor.version = QNN_TENSOR_VERSION_1;
    tensor.v1.name = name;
    tensor.v1.type = type;
    tensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    tensor.v1.dataType = data_type;
    tensor.v1.rank = static_cast<std::uint32_t>(dims.size());
    tensor.v1.dimensions = dims.data();
    tensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
    return tensor;
}

auto per_tensor(float scale, std::int32_t offset) -> Qnn_QuantizeParams_t {
    Qnn_QuantizeParams_t params = QNN_QUANTIZE_PARAMS_INIT;
    params.encodingDefinition = QNN_DEFINITION_DEFINED;
    params.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    params.scaleOffsetEncoding = {scale, offset};
    return params;
}

auto make_weights(std::span<const std::int8_t> values, std::span<const float> scales, std::int32_t axis, int bits,
                  bool packed) -> Weights {
    Weights weights;
    weights.scales.assign(scales.begin(), scales.end());
    const auto channels = static_cast<std::uint32_t>(scales.size());
    weights.params = QNN_QUANTIZE_PARAMS_INIT;
    weights.params.encodingDefinition = QNN_DEFINITION_DEFINED;
    const bool sub_byte_packed = packed && bits < 8;
    if (sub_byte_packed) {
        weights.data_type = bits == 4 ? QNN_DATATYPE_SFIXED_POINT_4 : QNN_DATATYPE_SFIXED_POINT_2;
        const auto per_byte = static_cast<std::size_t>(8 / bits);
        const auto mask = static_cast<std::uint8_t>((1U << bits) - 1);
        weights.data.assign(values.size() / per_byte, 0);
        for (std::size_t index = 0; index < values.size(); ++index)
            weights.data[index / per_byte] |= static_cast<std::uint8_t>(
                (static_cast<std::uint8_t>(values[index]) & mask) << ((index % per_byte) * bits));
        weights.encoding = "sfixed" + std::to_string(bits) + "-packed";
    } else {
        weights.data.assign(reinterpret_cast<const std::uint8_t*>(values.data()),
                            reinterpret_cast<const std::uint8_t*>(values.data()) + values.size());
        weights.encoding = bits == 8 ? "sfixed8" : "sfixed8-container-bw" + std::to_string(bits);
    }
    if (bits == 8 || sub_byte_packed) {
        weights.scale_offsets.resize(channels);
        for (std::uint32_t channel = 0; channel < channels; ++channel)
            weights.scale_offsets[channel] = {weights.scales[channel], 0};
        weights.params.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
        weights.params.axisScaleOffsetEncoding = {axis, channels, weights.scale_offsets.data()};
    } else {
        weights.params.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET;
        weights.params.bwAxisScaleOffsetEncoding = {static_cast<std::uint32_t>(bits), axis, channels,
                                                    weights.scales.data(), nullptr};
    }
    return weights;
}

SharedMemory::SharedMemory() {
    library_ = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    if (!library_) throw std::runtime_error(std::string("load libcdsprpc.so: ") + dlerror());
    alloc_ = reinterpret_cast<void* (*)(int, std::uint32_t, int)>(dlsym(library_, "rpcmem_alloc"));
    free_ = reinterpret_cast<void (*)(void*)>(dlsym(library_, "rpcmem_free"));
    to_fd_ = reinterpret_cast<int (*)(void*)>(dlsym(library_, "rpcmem_to_fd"));
    if (!alloc_ || !free_ || !to_fd_) throw std::runtime_error("libcdsprpc.so lacks the rpcmem API");
}

SharedMemory::~SharedMemory() {
    for (auto* block : blocks_) free_(block);
}

auto SharedMemory::allocate(std::size_t size) -> Block {
    constexpr int HEAP_ID_SYSTEM = 25;
    constexpr std::uint32_t DEFAULT_FLAGS = 1;
    void* data = alloc_(HEAP_ID_SYSTEM, DEFAULT_FLAGS, static_cast<int>(size));
    if (!data) throw std::runtime_error("rpcmem_alloc failed for " + std::to_string(size) + " bytes");
    blocks_.push_back(data);
    const int fd = to_fd_(data);
    if (fd < 0) throw std::runtime_error("rpcmem_to_fd failed");
    std::memset(data, 0, size);
    return {data, fd, size};
}

} // namespace kidi::bench::qnn

#endif
