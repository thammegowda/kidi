#include "kidi/runtime/operator.h"

#include <QnnInterface.h>
#include <QnnOpDef.h>
#include <QnnOpPackage.h>
#include <HTP/QnnHtpCommon.h>
#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpGraph.h>
#include <HTP/QnnHtpPerfInfrastructure.h>
#include <HTP/QnnHtpProfile.h>
#include <QnnProfile.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <thread>
#include <future>
#include <functional>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include <dlfcn.h>
#if defined(__ANDROID__)
#include <android/log.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "kidi/graph/graph.h"
#include "kidi/ops/context.h"

namespace kidi::runtime {
namespace {

/// Kidi's HTP op package: fused GELU*up, fused RMSNorm, and opt-in structured attention.
constexpr auto CUSTOM_PACKAGE = "KidiOps";
using graph::Graph;
using graph::Node;
using graph::Source;
using ops::require;
using tensor::DType;
using tensor::Tensor;

using GetProviders = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, std::uint32_t*);

constexpr std::uint64_t FNV_OFFSET = 14695981039346656037ULL;
constexpr std::uint64_t FNV_PRIME = 1099511628211ULL;
constexpr std::string_view CACHE_VERSION = "kidi-qnn-v4";

auto diagnostic_log(const std::string& message) -> void {
    std::cerr << message << '\n';
#if defined(__ANDROID__)
    __android_log_write(ANDROID_LOG_INFO, "KidiQNN", message.c_str());
#endif
}

struct Hash {
    std::uint64_t value = FNV_OFFSET;

    auto bytes(std::span<const std::byte> data) -> void {
        for (const auto byte : data) {
            value ^= static_cast<std::uint8_t>(byte);
            value *= FNV_PRIME;
        }
    }
    auto text(std::string_view text_value) -> void {
        bytes(std::as_bytes(std::span(text_value.data(), text_value.size())));
    }
    template <typename Value>
    auto pod(const Value& pod_value) -> void {
        bytes(std::as_bytes(std::span(&pod_value, std::size_t{1})));
    }
};

auto hex(std::uint64_t value) -> std::string {
    constexpr char DIGITS[] = "0123456789abcdef";
    std::string result(16, '0');
    for (std::size_t index = 0; index < result.size(); ++index)
        result[result.size() - 1 - index] = DIGITS[(value >> (index * 4)) & 0xf];
    return result;
}

auto getenv_string(const char* name) -> std::string {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string{};
}

auto join_path(const std::string& directory, const char* name) -> std::string {
    if (directory.empty()) return name;
    return (std::filesystem::path(directory) / name).string();
}

auto cache_directory() -> std::filesystem::path {
    if (const auto env = getenv_string("KIDI_QNN_CACHE_DIR"); !env.empty()) return env;
    return std::filesystem::path(".kidi-qnn-cache");
}

auto prepend_adsp_paths(const std::string& directory) -> void {
#if defined(__ANDROID__)
    constexpr char SEPARATOR = ';';
#else
    constexpr char SEPARATOR = ':';
#endif
    std::vector<std::string> additions;
    if (const auto skel = getenv_string("KIDI_QNN_SKEL_DIR"); !skel.empty()) additions.push_back(skel);
    if (!directory.empty()) {
        additions.push_back(directory);
        additions.push_back((std::filesystem::path(directory).parent_path() / "hexagon-v79" / "unsigned").string());
    }
#if defined(__ANDROID__)
    additions.insert(additions.end(), {"/vendor/lib/rfsa/adsp", "/vendor/dsp/cdsp", "/vendor/dsp"});
#endif
    const auto current = getenv_string("ADSP_LIBRARY_PATH");
    std::string combined;
    const auto contains = [](std::string_view paths, std::string_view path) {
        for (std::size_t start = 0; start <= paths.size();) {
            const auto end = paths.find(SEPARATOR, start);
            if (paths.substr(start, end - start) == path) return true;
            if (end == std::string_view::npos) break;
            start = end + 1;
        }
        return false;
    };
    for (const auto& path : additions) {
        if (path.empty() || contains(current, path) || contains(combined, path)) continue;
        if (!combined.empty()) combined += SEPARATOR;
        combined += path;
    }
    if (combined.empty()) return;
    if (!current.empty()) {
        combined += SEPARATOR;
        combined += current;
    }
    setenv("ADSP_LIBRARY_PATH", combined.c_str(), 1);
}

std::atomic<std::uint64_t> graph_memory_failures = 0;

auto log_callback(const char* format, QnnLog_Level_t level, std::uint64_t, va_list arguments) -> void {
    std::array<char, 2048> message{};
    va_list copy;
    va_copy(copy, arguments);
    std::vsnprintf(message.data(), message.size(), format, copy);
    va_end(copy);
    const std::string_view text(message.data());
    if (text.contains("Failed to map weights buffer") || text.contains("Could not allocate persistent weights buffer") ||
        text.contains("Failed to initialize graph memory"))
        graph_memory_failures.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr, "[kidi-qnn:%d] %s\n", static_cast<int>(level), message.data());
}

auto qnn_error(Qnn_ErrorHandle_t status) -> std::uint32_t { return QNN_GET_ERROR_CODE(status); }

auto check(Qnn_ErrorHandle_t status, const std::string& operation) -> void {
    if (status != QNN_SUCCESS)
        throw ops::Failure({ErrorCode::RUNTIME, operation + ": QNN error " + std::to_string(qnn_error(status))});
}

auto try_dlopen(const std::string& path, int flags = RTLD_NOW | RTLD_LOCAL) -> void* {
    return dlopen(path.c_str(), flags);
}

auto library_loaded(const std::string& path) -> bool {
    if (void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_NOLOAD)) {
        dlclose(handle);
        return true;
    }
    return false;
}

auto read_file(const std::filesystem::path& path) -> std::vector<std::uint8_t> {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw ops::Failure({ErrorCode::RUNTIME, "cannot read " + path.string()});
    return {std::istreambuf_iterator<char>(stream), {}};
}

auto write_file(const std::filesystem::path& path, std::span<const std::uint8_t> bytes) -> void {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!stream) throw ops::Failure({ErrorCode::RUNTIME, "cannot write " + path.string()});
}

auto write_text(const std::filesystem::path& path, std::string_view text) -> void {
    std::ofstream stream(path, std::ios::trunc);
    stream << text;
    if (!stream) throw ops::Failure({ErrorCode::RUNTIME, "cannot write " + path.string()});
}

class QnnRuntime final {
public:
    QnnRuntime() {
        library_dir_ = getenv_string("KIDI_QNN_LIBRARY_DIR");
        prepend_adsp_paths(library_dir_);
        const auto load = [&](const char* name, int flags = RTLD_NOW | RTLD_LOCAL) {
            auto* result = try_dlopen(name, flags);
            return result ? result : try_dlopen(join_path(library_dir_, name), flags);
        };
        system_library_ = load("libQnnSystem.so", RTLD_NOW | RTLD_GLOBAL);
        prepare_library_ = load("libQnnHtpPrepare.so");
        library_ = load("libQnnHtp.so");
        if (!library_) throw ops::Failure({ErrorCode::UNSUPPORTED, std::string("load libQnnHtp.so: ") + dlerror()});
        stub_library_ = load("libQnnHtpV79Stub.so", RTLD_NOW | RTLD_GLOBAL);
        const auto get = reinterpret_cast<GetProviders>(dlsym(library_, "QnnInterface_getProviders"));
        if (!get)
            throw ops::Failure({ErrorCode::UNSUPPORTED, "libQnnHtp.so does not export QnnInterface_getProviders"});
        const QnnInterface_t** providers = nullptr;
        std::uint32_t provider_count = 0;
        check(get(&providers, &provider_count), "get QNN providers");
        for (std::uint32_t index = 0; index < provider_count; ++index) {
            const auto& version = providers[index]->apiVersion.coreApiVersion;
            if (providers[index]->backendId == QNN_BACKEND_ID_HTP && version.major == QNN_API_VERSION_MAJOR) {
                api_ = providers[index]->QNN_INTERFACE_VER_NAME;
                api_version_ = std::to_string(version.major) + "." + std::to_string(version.minor) + "." +
                               std::to_string(version.patch);
                break;
            }
        }
        if (api_version_.empty())
            throw ops::Failure({ErrorCode::UNSUPPORTED, "libQnnHtp.so has no compatible HTP provider"});
        if (const char* dump = std::getenv("KIDI_QNN_DUMP"); dump && *dump && api_.propertyHasCapability) {
            const auto status = api_.propertyHasCapability(QNN_PROPERTY_TENSOR_SUPPORT_DYNAMIC_DIMENSIONS);
            diagnostic_log("kidi_qnn_capability|dynamic_dimensions=" +
                           std::string(status == QNN_PROPERTY_SUPPORTED ? "supported" : "unsupported") +
                           "|status=" + std::to_string(qnn_error(status)));
        }
        const auto log_level = [] {
            if (const char* level = std::getenv("KIDI_QNN_LOG"))
                return static_cast<QnnLog_Level_t>(std::atoi(level));
            return QNN_LOG_LEVEL_ERROR;
        }();
        check(api_.logCreate(log_callback, log_level, &log_), "create QNN log");
        check(api_.backendCreate(log_, nullptr, &backend_), "create QNN HTP backend");
        // The host (prepare) and DSP (execute) builds of the package have different file names so both fit in an APK.
        if (const auto package = getenv_string("KIDI_QNN_OP_PACKAGE"); !package.empty()) {
            auto htp_package = getenv_string("KIDI_QNN_OP_PACKAGE_HTP");
            if (htp_package.empty()) htp_package = package;
            // A missing or incompatible package only disables the fused lowering; dense QNN graphs still work.
            const auto host_status =
                api_.backendRegisterOpPackage(backend_, package.c_str(), "KidiOpsInterfaceProvider", "CPU");
            const auto htp_status =
                host_status == QNN_SUCCESS
                    ? api_.backendRegisterOpPackage(backend_, htp_package.c_str(), "KidiOpsInterfaceProvider", "HTP")
                    : host_status;
            custom_ops_ = host_status == QNN_SUCCESS && htp_status == QNN_SUCCESS;
            diagnostic_log("kidi_qnn_op_package|name=KidiOps|host=" + package + "|htp=" + htp_package +
                           "|host_status=" + std::to_string(qnn_error(host_status)) +
                           "|htp_status=" + std::to_string(qnn_error(htp_status)) +
                           "|status=" + (custom_ops_ ? "registered" : "unavailable"));
            if (const char* dump = std::getenv("KIDI_QNN_DUMP"); dump && *dump) {
                auto* library = dlopen(package.c_str(), RTLD_NOW | RTLD_LOCAL);
                using Provider = Qnn_ErrorHandle_t (*)(QnnOpPackage_Interface_t*);
                auto provider =
                    library ? reinterpret_cast<Provider>(dlsym(library, "KidiOpsInterfaceProvider")) : nullptr;
                QnnOpPackage_Interface_t package_interface{};
                const QnnOpPackage_Info_t* info = nullptr;
                const auto provider_status =
                    provider ? provider(&package_interface) : QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
                const auto info_status = provider_status == QNN_SUCCESS && package_interface.v1_4.getInfo
                                             ? package_interface.v1_4.getInfo(&info)
                                             : QNN_OP_PACKAGE_ERROR_GENERAL;
                diagnostic_log("kidi_qnn_op_package|provider_status=" + std::to_string(qnn_error(provider_status)) +
                               "|info_status=" + std::to_string(qnn_error(info_status)) + "|reported_package=" +
                               (info && info->packageName ? info->packageName : "") + "|reported_op=" +
                               (info && info->numOperations && info->operationNames ? info->operationNames[0] : ""));
                if (library) dlclose(library);
            }
            if (api_.backendGetSupportedOperations) {
                std::uint32_t count = 0;
                const QnnBackend_OperationName_t* operations = nullptr;
                if (api_.backendGetSupportedOperations(backend_, &count, &operations) == QNN_SUCCESS)
                    for (std::uint32_t index = 0; index < count; ++index) {
                        const auto& operation = operations[index];
                        if (operation.packageName && std::string_view(operation.packageName).contains("Kidi"))
                            diagnostic_log(
                                "kidi_qnn_op_package|reported_package=" + std::string(operation.packageName) +
                                "|op=" + (operation.name ? operation.name : "") +
                                "|target=" + (operation.target ? operation.target : ""));
                    }
            }
        }
        const char* build = nullptr;
        if (api_.backendGetBuildId && api_.backendGetBuildId(&build) == QNN_SUCCESS && build) build_id_ = build;
        if (!stub_library_) {
            const char* error = dlerror();
            throw ops::Failure({ErrorCode::UNSUPPORTED,
                                std::string("load libQnnHtpV79Stub.so: ") + (error ? error : "unknown error")});
        }
        check(api_.deviceCreate(log_, nullptr, &device_), "create QNN HTP device");
        request_burst();
    }

    ~QnnRuntime() {
        if (perf_ && power_) perf_->destroyPowerConfigId(power_);
        if (device_) api_.deviceFree(device_);
        if (backend_) api_.backendFree(backend_);
        if (log_) api_.logFree(log_);
    }

    QnnRuntime(const QnnRuntime&) = delete;
    auto operator=(const QnnRuntime&) -> QnnRuntime& = delete;

    auto api() const noexcept -> const QNN_INTERFACE_VER_TYPE& { return api_; }
    auto backend() const noexcept -> Qnn_BackendHandle_t { return backend_; }
    auto device() const noexcept -> Qnn_DeviceHandle_t { return device_; }
    auto build_id() const noexcept -> std::string_view { return build_id_; }
    auto api_version() const noexcept -> std::string_view { return api_version_; }
    auto prepare_available() const noexcept -> bool {
        return prepare_library_ || library_loaded(join_path(library_dir_, "libQnnHtpPrepare.so"));
    }
    auto burst() const noexcept -> bool { return burst_; }
    /// Custom attention measured slower than QNN's HMX attention, so it stays opt-in.
    auto structured_attention() const noexcept -> bool {
        return custom_ops_ && getenv_string("KIDI_QNN_CUSTOM_ATTENTION") == "1";
    }
    /// Fused GELU*up and RMSNorm restore the CPU backend's fusions on the NPU.
    auto fused_ops() const noexcept -> bool { return custom_ops_ && getenv_string("KIDI_QNN_FUSED_OPS") != "0"; }

private:
    auto request_burst() -> void {
        QnnDevice_Infrastructure_t infrastructure = nullptr;
        if (!api_.deviceGetInfrastructure || api_.deviceGetInfrastructure(&infrastructure) != QNN_SUCCESS ||
            !infrastructure)
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
        burst_ = true;
    }

    std::string library_dir_;
    void* system_library_{};
    void* library_{};
    void* stub_library_{};
    void* prepare_library_{};
    QNN_INTERFACE_VER_TYPE api_{};
    Qnn_LogHandle_t log_{};
    Qnn_BackendHandle_t backend_{};
    Qnn_DeviceHandle_t device_{};
    const QnnHtpDevice_PerfInfrastructure_t* perf_{};
    std::uint32_t power_{};
    std::string api_version_;
    std::string build_id_;
    bool burst_ = false;
    bool custom_ops_ = false;
};

class QnnContext final {
public:
    explicit QnnContext(std::shared_ptr<QnnRuntime> runtime) : runtime_(std::move(runtime)) {}
    ~QnnContext() {
        if (context_) runtime_->api().contextFree(context_, nullptr);
    }
    QnnContext(const QnnContext&) = delete;
    auto operator=(const QnnContext&) -> QnnContext& = delete;

    auto create() -> void {
        check(runtime_->api().contextCreate(runtime_->backend(), runtime_->device(), nullptr, &context_),
              "create QNN context");
    }
    auto load(std::span<const std::uint8_t> binary) -> void {
        check(runtime_->api().contextCreateFromBinary(runtime_->backend(), runtime_->device(), nullptr, binary.data(),
                                                      binary.size(), &context_, nullptr),
              "load QNN context binary");
    }
    auto save() -> std::vector<std::uint8_t> {
        Qnn_ContextBinarySize_t size = 0;
        check(runtime_->api().contextGetBinarySize(context_, &size), "get QNN context binary size");
        std::vector<std::uint8_t> binary(size);
        Qnn_ContextBinarySize_t written = 0;
        check(runtime_->api().contextGetBinary(context_, binary.data(), size, &written), "get QNN context binary");
        binary.resize(written);
        return binary;
    }
    auto create_graph(const std::string& name, bool float16 = false) -> Qnn_GraphHandle_t {
        QnnHtpGraph_CustomConfig_t optimization = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
        optimization.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
        optimization.optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
        optimization.optimizationOption.floatValue = 3;
        QnnGraph_Config_t config = QNN_GRAPH_CONFIG_INIT;
        config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        config.customConfig = &optimization;
        // Mixed graphs keep quantized projections and run float tensors in FP16.
        QnnHtpGraph_CustomConfig_t precision = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
        precision.option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
        precision.precision = QNN_PRECISION_FLOAT16;
        QnnGraph_Config_t precision_config = QNN_GRAPH_CONFIG_INIT;
        precision_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        precision_config.customConfig = &precision;
        const QnnGraph_Config_t* configs[] = {&config, float16 ? &precision_config : nullptr, nullptr};
        Qnn_GraphHandle_t graph{};
        check(runtime_->api().graphCreate(context_, name.c_str(), configs, &graph), "create QNN graph " + name);
        return graph;
    }
    auto retrieve_graph(const std::string& name) -> Qnn_GraphHandle_t {
        Qnn_GraphHandle_t graph{};
        check(runtime_->api().graphRetrieve(context_, name.c_str(), &graph), "retrieve QNN graph " + name);
        return graph;
    }
    auto execute(Qnn_GraphHandle_t graph, Qnn_Tensor_t* inputs, std::uint32_t input_count, Qnn_Tensor_t* outputs,
                 std::uint32_t output_count, const char* label = "") -> void {
        const auto& api = runtime_->api();
        // KIDI_QNN_PROFILE=1 reports HTP cycles per op type for every graph execution.
        static const bool profiling = !getenv_string("KIDI_QNN_PROFILE").empty();
        Qnn_ProfileHandle_t profile{};
        if (profiling)
            check(api.profileCreate(runtime_->backend(), QNN_PROFILE_LEVEL_DETAILED, &profile), "create QNN profile");
        check(api.graphExecute(graph, inputs, input_count, outputs, output_count, profile, nullptr),
              "execute QNN graph");
        if (profile) {
            report_profile(profile, label);
            api.profileFree(profile);
        }
    }
    auto runtime() const noexcept -> const QnnRuntime& { return *runtime_; }
    auto handle() const noexcept -> Qnn_ContextHandle_t { return context_; }
    /// Registers an rpcmem buffer (by file descriptor) for tensors of `dims` and `type`.
    auto register_memory(int fd, std::vector<std::uint32_t>& dims, Qnn_DataType_t type) -> Qnn_MemHandle_t {
        Qnn_MemDescriptor_t descriptor = QNN_MEM_DESCRIPTOR_INIT;
        descriptor.memShape = {static_cast<std::uint32_t>(dims.size()), dims.data(), nullptr};
        descriptor.dataType = type;
        descriptor.memType = QNN_MEM_TYPE_ION;
        descriptor.ionInfo.fd = fd;
        Qnn_MemHandle_t result{};
        check(runtime_->api().memRegister(context_, &descriptor, 1, &result), "register QNN shared memory");
        return result;
    }

private:
    auto report_profile(Qnn_ProfileHandle_t profile, const char* label) const -> void {
        const auto& api = runtime_->api();
        const QnnProfile_EventId_t* events = nullptr;
        std::uint32_t event_count = 0;
        if (api.profileGetEvents(profile, &events, &event_count) != QNN_SUCCESS) return;
        for (std::uint32_t index = 0; index < event_count; ++index) {
            QnnProfile_EventData_t data = QNN_PROFILE_EVENT_DATA_INIT;
            if (api.profileGetEventData(events[index], &data) != QNN_SUCCESS) continue;
            if (data.type != QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_TIME_CYCLE &&
                data.type != QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_TIME_MICROSEC)
                continue;
            const bool cycles = data.type == QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_TIME_CYCLE;
            std::map<std::string, std::pair<std::uint64_t, std::uint32_t>> by_type;
            std::vector<std::pair<std::uint64_t, std::string>> nodes;
            const QnnProfile_EventId_t* children = nullptr;
            std::uint32_t child_count = 0;
            if (api.profileGetSubEvents(events[index], &children, &child_count) == QNN_SUCCESS)
                for (std::uint32_t child = 0; child < child_count; ++child) {
                    QnnProfile_EventData_t node = QNN_PROFILE_EVENT_DATA_INIT;
                    if (api.profileGetEventData(children[child], &node) != QNN_SUCCESS || !node.identifier) continue;
                    std::string name = node.identifier;
                    name = name.substr(0, name.find(':'));
                    const auto slash = name.rfind('/');
                    const auto separator = name.find('_', slash == std::string::npos ? 0 : slash);
                    auto& entry = by_type[separator == std::string::npos ? name : name.substr(separator + 1)];
                    entry.first += node.value;
                    ++entry.second;
                    nodes.emplace_back(node.value, std::move(name));
                }
            std::ostringstream message;
            message << "kidi_qnn_profile|graph=" << label << "|unit=" << (cycles ? "cycles" : "us")
                    << "|total=" << data.value << "|nodes=" << nodes.size();
            std::vector<std::pair<std::uint64_t, std::string>> types;
            for (const auto& [type, entry] : by_type)
                types.emplace_back(entry.first,
                                   type + ":" + std::to_string(entry.first) + "/" + std::to_string(entry.second));
            std::sort(types.rbegin(), types.rend());
            for (std::size_t item = 0; item < std::min<std::size_t>(types.size(), 16); ++item)
                message << "|" << types[item].second;
            diagnostic_log(message.str());
            if (!cycles) continue;
            if (const auto path = getenv_string("KIDI_QNN_PROFILE_CSV"); !path.empty()) {
                std::ofstream csv(path, std::ios::app);
                for (const auto& [value, name] : nodes) csv << label << ',' << name << ',' << value << '\n';
            }
            std::sort(nodes.rbegin(), nodes.rend());
            std::ostringstream top;
            top << "kidi_qnn_profile_top|graph=" << label;
            for (std::size_t item = 0; item < std::min<std::size_t>(nodes.size(), 12); ++item)
                top << "|" << nodes[item].second << ":" << nodes[item].first;
            diagnostic_log(top.str());
        }
    }

    std::shared_ptr<QnnRuntime> runtime_;
    Qnn_ContextHandle_t context_{};
};

auto qnn_dims(const Tensor& tensor) -> std::vector<std::uint32_t> {
    std::vector<std::uint32_t> result;
    result.reserve(tensor.dimensions());
    for (const auto extent : tensor.shape()) result.push_back(static_cast<std::uint32_t>(extent));
    return result;
}

auto make_tensor(const char* name, Qnn_TensorType_t type, Qnn_DataType_t data_type,
                 std::vector<std::uint32_t>& dims) -> Qnn_Tensor_t {
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

struct Weights {
    std::vector<std::uint8_t> data;
    std::vector<float> scales;
    std::vector<Qnn_ScaleOffset_t> scale_offsets;
    Qnn_DataType_t data_type = QNN_DATATYPE_SFIXED_POINT_8;
    Qnn_QuantizeParams_t params{};
};

auto unpack_packed_weight(const Tensor& weight, int bits, std::size_t logical_columns) -> std::vector<std::int8_t> {
    const auto bytes = require(weight.host_bytes());
    const auto* data = reinterpret_cast<const std::uint8_t*>(bytes.data());
    const auto rows = weight.size(0);
    const auto per_byte = static_cast<std::size_t>(8 / bits);
    std::vector<std::int8_t> result(rows * logical_columns);
    const int sign = 1 << (bits - 1);
    const int mask = (1 << bits) - 1;
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < logical_columns; ++column) {
            const auto raw = (data[row * weight.size(1) + column / per_byte] >> ((column % per_byte) * bits)) & mask;
            result[row * logical_columns + column] = static_cast<std::int8_t>((raw ^ sign) - sign);
        }
    return result;
}

auto make_weights(std::span<const std::int8_t> values, std::span<const float> scales, int bits) -> Weights {
    Weights weights;
    weights.data.assign(reinterpret_cast<const std::uint8_t*>(values.data()),
                        reinterpret_cast<const std::uint8_t*>(values.data()) + values.size());
    weights.scales.assign(scales.begin(), scales.end());
    weights.params = QNN_QUANTIZE_PARAMS_INIT;
    weights.params.encodingDefinition = QNN_DEFINITION_DEFINED;
    if (bits == 8) {
        weights.scale_offsets.resize(weights.scales.size());
        for (std::size_t index = 0; index < weights.scales.size(); ++index)
            weights.scale_offsets[index] = {weights.scales[index], 0};
        weights.params.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
        weights.params.axisScaleOffsetEncoding = {0, static_cast<std::uint32_t>(weights.scale_offsets.size()),
                                                  weights.scale_offsets.data()};
    } else {
        weights.params.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET;
        weights.params.bwAxisScaleOffsetEncoding = {static_cast<std::uint32_t>(bits), 0,
                                                    static_cast<std::uint32_t>(weights.scales.size()),
                                                    weights.scales.data(), nullptr};
    }
    return weights;
}

auto scalar_bits(std::int64_t value) -> float { return std::bit_cast<float>(static_cast<std::int32_t>(value)); }

auto tensor_scalar(const Tensor& tensor) -> float {
    if (!tensor.defined() || tensor.numel() != 1 || tensor.dtype() != DType::F32)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "expected scalar FP32 tensor"});
    return require(tensor.data<float>())[0];
}

auto source_matches_node(const Source& source, std::size_t node, std::size_t output = 0) -> bool {
    return source.kind == Source::Kind::NODE && source.index == node && source.output == output;
}

auto source_matches_input(const Source& source, std::size_t input) -> bool {
    return source.kind == Source::Kind::INPUT && source.index == input;
}

auto storage_begin(const Tensor& tensor) -> const std::byte* {
    const auto bytes = require(tensor.host_bytes());
    return bytes.data() - tensor.storage_offset() * static_cast<std::int64_t>(tensor::element_size(tensor.dtype()));
}

auto input_view(const Tensor& base, std::int64_t offset, const Tensor& like) -> Tensor {
    const auto element = static_cast<std::int64_t>(tensor::element_size(base.dtype()));
    if (like.dtype() != base.dtype() || offset % element)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "replayed QNN input view has an incompatible layout"});
    auto flat = require(base.reshape({static_cast<std::int64_t>(base.numel())}));
    auto window = require(flat.narrow(0, offset / element, static_cast<std::int64_t>(like.numel())));
    return require(window.reshape({like.shape().begin(), like.shape().end()}));
}

struct PartitionPlan {
    enum class Kind { FFN, GATED_FFN, PACKED_LINEAR };
    Kind kind = Kind::PACKED_LINEAR;
    std::size_t start = 0;
    std::size_t end = 0;
    int bits = 0;
    std::size_t input_width = 0;
    std::size_t intermediate = 0;
    std::size_t output_width = 0;
    float input_scale = 0;
    float gate_output_scale = 0;
    float down_input_scale = 0;
    float output_scale = 0;
};

auto is_calibrated_packed_linear(const Node& node) -> bool {
    return node.operation == Operation::PACKED_LINEAR && node.attributes.size() == 3 && node.epsilon > 0 &&
           scalar_bits(node.attributes[2]) > 0 && node.inputs().size() == 3 && node.inputs()[0].dtype() == DType::F32 &&
           node.inputs()[1].dtype() == DType::U8 && node.inputs()[2].dtype() == DType::F32 &&
           node.inputs()[2].dimensions() == 2 && node.inputs()[2].size(1) == 1 &&
           static_cast<std::size_t>(node.attributes[1]) == node.inputs()[0].size(-1);
}

auto find_ffn(const std::span<const Node> nodes, std::size_t index) -> std::optional<PartitionPlan> {
    if (index + 2 >= nodes.size()) return std::nullopt;
    const auto& gate = nodes[index];
    const auto& activation = nodes[index + 1];
    const auto& down = nodes[index + 2];
    if (!is_calibrated_packed_linear(gate) || activation.operation != Operation::GELU_MULTIPLY ||
        !is_calibrated_packed_linear(down))
        return std::nullopt;
    if (gate.outputs().size() != 1 || activation.outputs().size() != 1 || down.outputs().size() != 1 ||
        activation.inputs().size() != 2 || down.inputs().size() != 3)
        return std::nullopt;
    if (!source_matches_node(activation.sources[0], index) || !source_matches_node(activation.sources[1], index) ||
        !source_matches_node(down.sources[0], index + 1))
        return std::nullopt;
    const auto output_width = down.outputs()[0].size(-1);
    const auto input_width = gate.inputs()[0].size(-1);
    const auto projected_width = gate.outputs()[0].size(-1);
    if (projected_width % 2 || activation.inputs()[0].size(-1) != projected_width / 2 ||
        activation.inputs()[1].size(-1) != projected_width / 2 || down.inputs()[0].size(-1) != projected_width / 2 ||
        down.inputs()[1].size(0) != output_width || gate.attributes[0] != down.attributes[0])
        return std::nullopt;
    const auto second_offset = static_cast<std::int64_t>(projected_width / 2 * sizeof(float));
    if (activation.sources[0].offset != 0 || activation.sources[1].offset != second_offset ||
        down.sources[0].offset != 0)
        return std::nullopt;
    return PartitionPlan{PartitionPlan::Kind::FFN,
                         index,
                         index + 3,
                         static_cast<int>(gate.attributes[0]),
                         input_width,
                         projected_width / 2,
                         output_width,
                         gate.epsilon,
                         scalar_bits(gate.attributes[2]),
                         down.epsilon,
                         scalar_bits(down.attributes[2])};
}

auto find_sliced_ffn(const std::span<const Node> nodes, std::size_t index) -> std::optional<PartitionPlan> {
    if (index + 4 >= nodes.size()) return std::nullopt;
    const auto& gate = nodes[index];
    const auto& gate_slice = nodes[index + 1];
    const auto& up_slice = nodes[index + 2];
    const auto& activation = nodes[index + 3];
    const auto& down = nodes[index + 4];
    if (!is_calibrated_packed_linear(gate) || gate_slice.operation != Operation::SLICE ||
        up_slice.operation != Operation::SLICE || activation.operation != Operation::GELU_MULTIPLY ||
        !is_calibrated_packed_linear(down))
        return std::nullopt;
    if (gate.outputs().size() != 1 || gate_slice.outputs().size() != 1 || up_slice.outputs().size() != 1 ||
        activation.outputs().size() != 1 || down.outputs().size() != 1 || gate_slice.inputs().size() != 1 ||
        up_slice.inputs().size() != 1 || activation.inputs().size() != 2 || down.inputs().size() != 3)
        return std::nullopt;
    if (!source_matches_node(gate_slice.sources[0], index) || !source_matches_node(up_slice.sources[0], index) ||
        !source_matches_node(activation.sources[0], index + 1) ||
        !source_matches_node(activation.sources[1], index + 2) || !source_matches_node(down.sources[0], index + 3))
        return std::nullopt;
    if (gate_slice.attributes.size() != 3 || up_slice.attributes.size() != 3 || gate_slice.attributes[0] != -1 ||
        up_slice.attributes[0] != -1 || gate_slice.attributes[1] != 0)
        return std::nullopt;
    const auto input_width = gate.inputs()[0].size(-1);
    const auto projected_width = gate.outputs()[0].size(-1);
    if (projected_width % 2) return std::nullopt;
    const auto intermediate = projected_width / 2;
    const auto output_width = down.outputs()[0].size(-1);
    if (gate_slice.attributes[2] != intermediate || up_slice.attributes[1] != intermediate ||
        up_slice.attributes[2] != intermediate || gate_slice.outputs()[0].size(-1) != intermediate ||
        up_slice.outputs()[0].size(-1) != intermediate || activation.outputs()[0].size(-1) != intermediate ||
        down.inputs()[0].size(-1) != intermediate || down.inputs()[1].size(0) != output_width ||
        gate.attributes[0] != down.attributes[0])
        return std::nullopt;
    return PartitionPlan{PartitionPlan::Kind::FFN,
                         index,
                         index + 5,
                         static_cast<int>(gate.attributes[0]),
                         input_width,
                         intermediate,
                         output_width,
                         gate.epsilon,
                         scalar_bits(gate.attributes[2]),
                         down.epsilon,
                         scalar_bits(down.attributes[2])};
}

auto find_gated_ffn(const std::span<const Node> nodes, std::size_t index) -> std::optional<PartitionPlan> {
    const auto& node = nodes[index];
    if (node.operation != Operation::GATED_FEED_FORWARD || node.inputs().size() != 5 || node.outputs().size() != 1 ||
        node.attributes.size() != 6 || node.epsilon <= 0)
        return std::nullopt;
    const auto bits = static_cast<int>(node.attributes[0]);
    const auto input_width = static_cast<std::size_t>(node.attributes[1]);
    const auto intermediate = static_cast<std::size_t>(node.attributes[2]);
    const auto output_width = static_cast<std::size_t>(node.outputs()[0].size(-1));
    if ((bits != 2 && bits != 4 && bits != 8) || !input_width || !intermediate || !output_width ||
        node.inputs()[0].dtype() != DType::F32 || node.inputs()[1].dtype() != DType::U8 ||
        node.inputs()[2].dtype() != DType::F32 || node.inputs()[3].dtype() != DType::U8 ||
        node.inputs()[4].dtype() != DType::F32 || node.inputs()[1].size(0) != 2 * intermediate ||
        node.inputs()[2].size(0) != 2 * intermediate || node.inputs()[3].size(0) != output_width ||
        node.inputs()[4].size(0) != output_width)
        return std::nullopt;
    return PartitionPlan{PartitionPlan::Kind::GATED_FFN,
                         index,
                         index + 1,
                         bits,
                         input_width,
                         intermediate,
                         output_width,
                         node.epsilon,
                         scalar_bits(node.attributes[3]),
                         scalar_bits(node.attributes[4]),
                         scalar_bits(node.attributes[5])};
}

auto minimum_single_projection_bytes() -> std::size_t {
    if (const auto env = getenv_string("KIDI_QNN_MIN_PROJECTION_BYTES"); !env.empty()) {
        std::size_t value = 0;
        const auto result = std::from_chars(env.data(), env.data() + env.size(), value);
        if (result.ec == std::errc{}) return value;
    }
    return 8 * 1024 * 1024;
}

auto find_single_projection(const std::span<const Node> nodes, std::size_t index) -> std::optional<PartitionPlan> {
    const auto& node = nodes[index];
    if (!is_calibrated_packed_linear(node) || node.outputs().size() != 1) return std::nullopt;
    const auto input_width = node.inputs()[0].size(-1);
    const auto output_width = node.outputs()[0].size(-1);
    const auto weight_bytes = node.inputs()[1].nbytes();
    if (weight_bytes < minimum_single_projection_bytes()) return std::nullopt;
    return PartitionPlan{PartitionPlan::Kind::PACKED_LINEAR,
                         index,
                         index + 1,
                         static_cast<int>(node.attributes[0]),
                         input_width,
                         0,
                         output_width,
                         node.epsilon,
                         0,
                         0,
                         scalar_bits(node.attributes[2])};
}

auto plan_partitions(const Graph& graph) -> std::vector<PartitionPlan> {
    const auto nodes = graph.nodes();
    std::vector<PartitionPlan> plans;
    for (std::size_t index = 0; index < nodes.size();) {
        if (auto ffn = find_sliced_ffn(nodes, index); ffn || (ffn = find_ffn(nodes, index))) {
            plans.push_back(*ffn);
            index = ffn->end;
            continue;
        }
        if (auto gated = find_gated_ffn(nodes, index)) {
            plans.push_back(*gated);
            index = gated->end;
            continue;
        }
        if (auto projection = find_single_projection(nodes, index)) plans.push_back(*projection);
        ++index;
    }
    return plans;
}

class GraphBuilder final {
public:
    GraphBuilder(QnnContext& context, Qnn_GraphHandle_t graph, const std::string& prefix)
        : context_(context), graph_(graph), prefix_(prefix) {}

    auto tensor(const std::string& suffix, Qnn_TensorType_t type, Qnn_DataType_t dtype, std::vector<std::uint32_t> dims,
                float scale = 0) -> Qnn_Tensor_t {
        dims_.push_back(std::move(dims));
        auto name = prefix_ + suffix;
        names_.push_back(std::move(name));
        auto result = make_tensor(names_.back().c_str(), type, dtype, dims_.back());
        if (scale > 0) result.v1.quantizeParams = per_tensor(scale, -128);
        check(context_.runtime().api().tensorCreateGraphTensor(graph_, &result),
              "register QNN tensor " + names_.back());
        return result;
    }
    auto tensor(const std::string& suffix, Qnn_TensorType_t type, Qnn_DataType_t dtype, std::vector<std::uint32_t> dims,
                Qnn_QuantizeParams_t params) -> Qnn_Tensor_t {
        dims_.push_back(std::move(dims));
        names_.push_back(prefix_ + suffix);
        auto result = make_tensor(names_.back().c_str(), type, dtype, dims_.back());
        result.v1.quantizeParams = params;
        check(context_.runtime().api().tensorCreateGraphTensor(graph_, &result),
              "register QNN tensor " + names_.back());
        return result;
    }

    auto static_weight(const std::string& suffix, const Tensor& weight_tensor, const Tensor& scales_tensor, int bits,
                       std::size_t logical_columns) -> Qnn_Tensor_t {
        const auto unpacked = unpack_packed_weight(weight_tensor, bits, logical_columns);
        const auto scales = require(scales_tensor.data<float>());
        weights_.push_back(make_weights(unpacked, scales, bits));
        auto& weights = weights_.back();
        return static_weight(suffix, weights, weight_tensor.size(0), logical_columns);
    }

    auto static_weight(const std::string& suffix, Weights& weights, std::size_t rows,
                       std::size_t columns) -> Qnn_Tensor_t {
        dims_.push_back({static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(columns)});
        names_.push_back(prefix_ + suffix);
        auto result = make_tensor(names_.back().c_str(), QNN_TENSOR_TYPE_STATIC, weights.data_type, dims_.back());
        result.v1.quantizeParams = weights.params;
        result.v1.clientBuf = {weights.data.data(), static_cast<std::uint32_t>(weights.data.size())};
        check(context_.runtime().api().tensorCreateGraphTensor(graph_, &result),
              "register QNN weights " + names_.back());
        return result;
    }

    auto add_node(const std::string& suffix, const char* type, std::span<Qnn_Tensor_t> inputs,
                  std::span<Qnn_Tensor_t> outputs) -> void {
        names_.push_back(prefix_ + suffix);
        Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
        op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = names_.back().c_str();
        op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
        op.v1.typeName = type;
        op.v1.numOfInputs = static_cast<std::uint32_t>(inputs.size());
        op.v1.inputTensors = inputs.data();
        op.v1.numOfOutputs = static_cast<std::uint32_t>(outputs.size());
        op.v1.outputTensors = outputs.data();
        check(context_.runtime().api().graphAddNode(graph_, op), "add QNN node " + names_.back());
    }

private:
    QnnContext& context_;
    Qnn_GraphHandle_t graph_;
    std::string prefix_;
    std::deque<std::string> names_;
    std::deque<std::vector<std::uint32_t>> dims_;
    std::deque<Weights> weights_;
};

struct BuiltPartition {
    PartitionPlan plan;
    std::string name;
    std::uint32_t input_id = 0;
    std::uint32_t output_id = 0;
};

auto build_projection_graph(QnnContext& context, const std::string& name, const Node& node,
                            const PartitionPlan& plan) -> BuiltPartition {
    const auto graph = context.create_graph(name);
    GraphBuilder builder(context, graph, name + "/");
    const auto rows = node.inputs()[0].numel() / plan.input_width;
    auto input = builder.tensor("input", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_UFIXED_POINT_8,
                                {static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(plan.input_width)},
                                plan.input_scale);
    auto weight = builder.static_weight("weight", node.inputs()[1], node.inputs()[2], plan.bits, plan.input_width);
    auto output = builder.tensor("output", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_UFIXED_POINT_8,
                                 {static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(plan.output_width)},
                                 plan.output_scale);
    std::array inputs{input, weight};
    std::array outputs{output};
    builder.add_node("fc", QNN_OP_FULLY_CONNECTED, inputs, outputs);
    check(context.runtime().api().graphFinalize(graph, nullptr, nullptr), "finalize QNN graph " + name);
    return {plan, name, input.v1.id, output.v1.id};
}

auto build_ffn_graph(QnnContext& context, const std::string& name, const std::span<const Node> nodes,
                     const PartitionPlan& plan) -> BuiltPartition {
    const auto graph = context.create_graph(name);
    GraphBuilder builder(context, graph, name + "/");
    const auto& gate_node = nodes[plan.start];
    const auto& down_node = nodes[plan.end - 1];
    const auto rows = gate_node.inputs()[0].numel() / plan.input_width;
    auto input = builder.tensor("input", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_UFIXED_POINT_8,
                                {static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(plan.input_width)},
                                plan.input_scale);
    const auto gate_values = unpack_packed_weight(gate_node.inputs()[1], plan.bits, plan.input_width);
    const auto gate_scales = require(gate_node.inputs()[2].data<float>());
    const auto first_values = std::span(gate_values).first(plan.intermediate * plan.input_width);
    const auto second_values =
        std::span(gate_values).subspan(plan.intermediate * plan.input_width, plan.intermediate * plan.input_width);
    auto gate_weights = make_weights(first_values, gate_scales.first(plan.intermediate), plan.bits);
    auto up_weights = make_weights(second_values, gate_scales.subspan(plan.intermediate, plan.intermediate), plan.bits);
    auto gate_weight = builder.static_weight("gate_weight", gate_weights, plan.intermediate, plan.input_width);
    auto up_weight = builder.static_weight("up_weight", up_weights, plan.intermediate, plan.input_width);
    auto gate = builder.tensor("gate", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_UFIXED_POINT_8,
                               {static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(plan.intermediate)},
                               plan.gate_output_scale);
    auto up = builder.tensor("up", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_UFIXED_POINT_8,
                             {static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(plan.intermediate)},
                             plan.gate_output_scale);
    std::array gate_inputs{input, gate_weight};
    std::array gate_outputs{gate};
    builder.add_node("gate_fc", QNN_OP_FULLY_CONNECTED, gate_inputs, gate_outputs);
    std::array up_inputs{input, up_weight};
    std::array up_outputs{up};
    builder.add_node("up_fc", QNN_OP_FULLY_CONNECTED, up_inputs, up_outputs);
    auto gelu = builder.tensor("gelu", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_UFIXED_POINT_8,
                               {static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(plan.intermediate)},
                               plan.gate_output_scale);
    std::array gelu_inputs{gate};
    std::array gelu_outputs{gelu};
    builder.add_node("gelu", QNN_OP_GELU, gelu_inputs, gelu_outputs);
    auto hidden = builder.tensor("hidden", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_UFIXED_POINT_8,
                                 {static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(plan.intermediate)},
                                 plan.down_input_scale);
    std::array multiply_inputs{gelu, up};
    std::array multiply_outputs{hidden};
    builder.add_node("multiply", QNN_OP_ELEMENT_WISE_MULTIPLY, multiply_inputs, multiply_outputs);
    auto down_weight = builder.static_weight("down_weight", down_node.inputs()[1], down_node.inputs()[2], plan.bits,
                                             plan.intermediate);
    auto output = builder.tensor("output", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_UFIXED_POINT_8,
                                 {static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(plan.output_width)},
                                 plan.output_scale);
    std::array down_inputs{hidden, down_weight};
    std::array down_outputs{output};
    builder.add_node("down_fc", QNN_OP_FULLY_CONNECTED, down_inputs, down_outputs);
    check(context.runtime().api().graphFinalize(graph, nullptr, nullptr), "finalize QNN graph " + name);
    return {plan, name, input.v1.id, output.v1.id};
}

/// How a fused feed-forward graph computes gelu(gate) * up between its projections: "u8" requantizes the GELU onto
/// the gate grid, "u16" keeps it at 16 bits, and "fp16" computes both in FP16 before one INT8 quantization at the
/// down projection's input scale, as the CPU kernel does.
auto build_gated_ffn_graph(QnnContext& context, const std::string& name, const Node& node, const PartitionPlan& plan,
                           std::string_view variant = "u8") -> BuiltPartition {
    const bool float16 = variant == "fp16";
    const auto graph = context.create_graph(name, float16);
    GraphBuilder builder(context, graph, name + "/");
    const auto rows = node.inputs()[0].numel() / plan.input_width;
    auto input = builder.tensor("input", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_UFIXED_POINT_8,
                                {static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(plan.input_width)},
                                plan.input_scale);
    const auto gate_values = unpack_packed_weight(node.inputs()[1], plan.bits, plan.input_width);
    const auto gate_scales = require(node.inputs()[2].data<float>());
    const auto first_values = std::span(gate_values).first(plan.intermediate * plan.input_width);
    const auto second_values =
        std::span(gate_values).subspan(plan.intermediate * plan.input_width, plan.intermediate * plan.input_width);
    auto gate_weights = make_weights(first_values, gate_scales.first(plan.intermediate), plan.bits);
    auto up_weights = make_weights(second_values, gate_scales.subspan(plan.intermediate, plan.intermediate), plan.bits);
    auto gate_weight = builder.static_weight("gate_weight", gate_weights, plan.intermediate, plan.input_width);
    auto up_weight = builder.static_weight("up_weight", up_weights, plan.intermediate, plan.input_width);
    auto gate = builder.tensor("gate", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_UFIXED_POINT_8,
                               {static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(plan.intermediate)},
                               plan.gate_output_scale);
    auto up = builder.tensor("up", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_UFIXED_POINT_8,
                             {static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(plan.intermediate)},
                             plan.gate_output_scale);
    std::array gate_inputs{input, gate_weight};
    std::array gate_outputs{gate};
    builder.add_node("gate_fc", QNN_OP_FULLY_CONNECTED, gate_inputs, gate_outputs);
    std::array up_inputs{input, up_weight};
    std::array up_outputs{up};
    builder.add_node("up_fc", QNN_OP_FULLY_CONNECTED, up_inputs, up_outputs);
    const std::vector<std::uint32_t> wide{static_cast<std::uint32_t>(rows),
                                          static_cast<std::uint32_t>(plan.intermediate)};
    auto hidden =
        builder.tensor("hidden", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_UFIXED_POINT_8, wide, plan.down_input_scale);
    if (float16) {
        auto gate_half = builder.tensor("gate_half", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, wide);
        auto up_half = builder.tensor("up_half", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, wide);
        auto gelu = builder.tensor("gelu", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, wide);
        auto product = builder.tensor("product", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, wide);
        std::array gate_in{gate}, gate_out{gate_half}, up_in{up}, up_out{up_half};
        builder.add_node("gate_dequantize", QNN_OP_DEQUANTIZE, gate_in, gate_out);
        builder.add_node("up_dequantize", QNN_OP_DEQUANTIZE, up_in, up_out);
        std::array gelu_in{gate_half}, gelu_out{gelu};
        builder.add_node("gelu", QNN_OP_GELU, gelu_in, gelu_out);
        std::array multiply_in{gelu, up_half};
        std::array multiply_out{product};
        builder.add_node("multiply", QNN_OP_ELEMENT_WISE_MULTIPLY, multiply_in, multiply_out);
        std::array quantize_in{product}, quantize_out{hidden};
        builder.add_node("quantize", QNN_OP_QUANTIZE, quantize_in, quantize_out);
    } else {
        // GELU of the INT8 gate grid lies in [-0.17, 127 * scale]; 16 bits keep it well below one output quantum.
        const auto low = -0.2F, high = 127.F * plan.gate_output_scale;
        const auto wide_scale = variant == "u16" ? (high - low) / 65535.F : plan.gate_output_scale;
        auto gelu = builder.tensor(
            "gelu", QNN_TENSOR_TYPE_NATIVE,
            variant == "u16" ? QNN_DATATYPE_UFIXED_POINT_16 : QNN_DATATYPE_UFIXED_POINT_8, wide,
            variant == "u16" ? per_tensor(wide_scale, static_cast<std::int32_t>(std::lround(low / wide_scale)))
                             : per_tensor(plan.gate_output_scale, -128));
        std::array gelu_inputs{gate};
        std::array gelu_outputs{gelu};
        builder.add_node("gelu", QNN_OP_GELU, gelu_inputs, gelu_outputs);
        std::array multiply_inputs{gelu, up};
        std::array multiply_outputs{hidden};
        builder.add_node("multiply", QNN_OP_ELEMENT_WISE_MULTIPLY, multiply_inputs, multiply_outputs);
    }
    auto down_weight =
        builder.static_weight("down_weight", node.inputs()[3], node.inputs()[4], plan.bits, plan.intermediate);
    auto output = builder.tensor("output", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_UFIXED_POINT_8,
                                 {static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(plan.output_width)},
                                 plan.output_scale);
    std::array down_inputs{hidden, down_weight};
    std::array down_outputs{output};
    builder.add_node("down_fc", QNN_OP_FULLY_CONNECTED, down_inputs, down_outputs);
    check(context.runtime().api().graphFinalize(graph, nullptr, nullptr), "finalize QNN graph " + name);
    return {plan, name, input.v1.id, output.v1.id};
}

auto hash_tensor(Hash& hash, const Tensor& tensor) -> void {
    hash.pod(static_cast<int>(tensor.dtype()));
    hash.pod(tensor.dimensions());
    for (const auto extent : tensor.shape()) hash.pod(extent);
    const auto bytes = require(tensor.host_bytes());
    hash.bytes(bytes);
}

auto fingerprint(const Graph& graph, std::span<const PartitionPlan> plans, std::string_view key,
                 std::string_view build_id) -> std::string {
    Hash hash;
    hash.text(CACHE_VERSION);
    hash.text(key);
    hash.text(build_id);
    for (const auto& input : graph.inputs()) {
        hash.pod(static_cast<int>(input.dtype()));
        hash.pod(input.dimensions());
        for (const auto extent : input.shape()) hash.pod(extent);
    }
    for (const auto& plan : plans) {
        hash.pod(static_cast<int>(plan.kind));
        hash.pod(plan.start);
        hash.pod(plan.end);
        hash.pod(plan.bits);
        hash.pod(plan.input_width);
        hash.pod(plan.intermediate);
        hash.pod(plan.output_width);
        hash.pod(plan.input_scale);
        hash.pod(plan.gate_output_scale);
        hash.pod(plan.down_input_scale);
        hash.pod(plan.output_scale);
        const auto& first = graph.nodes()[plan.start];
        hash_tensor(hash, first.inputs()[1]);
        hash_tensor(hash, first.inputs()[2]);
        if (plan.kind == PartitionPlan::Kind::FFN) {
            const auto& down = graph.nodes()[plan.end - 1];
            hash_tensor(hash, down.inputs()[1]);
            hash_tensor(hash, down.inputs()[2]);
        } else if (plan.kind == PartitionPlan::Kind::GATED_FFN) {
            hash_tensor(hash, first.inputs()[3]);
            hash_tensor(hash, first.inputs()[4]);
        }
    }
    return hex(hash.value);
}

struct Metadata {
    std::vector<BuiltPartition> partitions;
};

auto save_metadata(const std::filesystem::path& path, std::string_view id,
                   std::span<const BuiltPartition> partitions) -> void {
    std::ostringstream out;
    out << CACHE_VERSION << '\n' << id << '\n' << partitions.size() << '\n';
    for (const auto& partition : partitions) {
        const auto kind = partition.plan.kind == PartitionPlan::Kind::FFN         ? "ffn"
                          : partition.plan.kind == PartitionPlan::Kind::GATED_FFN ? "gated_ffn"
                                                                                  : "projection";
        out << kind << ' ' << partition.name << ' ' << partition.plan.start << ' ' << partition.plan.end << ' '
            << partition.plan.bits << ' ' << partition.plan.input_width << ' ' << partition.plan.intermediate << ' '
            << partition.plan.output_width << ' ' << partition.plan.input_scale << ' '
            << partition.plan.gate_output_scale << ' ' << partition.plan.down_input_scale << ' '
            << partition.plan.output_scale << ' ' << partition.input_id << ' ' << partition.output_id << '\n';
    }
    write_text(path, out.str());
}

auto load_metadata(const std::filesystem::path& path, std::string_view id) -> std::optional<Metadata> {
    std::ifstream input(path);
    if (!input) return std::nullopt;
    std::string version, file_id;
    std::getline(input, version);
    std::getline(input, file_id);
    if (version != CACHE_VERSION || file_id != id) return std::nullopt;
    std::size_t count = 0;
    input >> count;
    Metadata metadata;
    for (std::size_t index = 0; index < count; ++index) {
        std::string kind;
        BuiltPartition partition;
        input >> kind >> partition.name >> partition.plan.start >> partition.plan.end >> partition.plan.bits >>
            partition.plan.input_width >> partition.plan.intermediate >> partition.plan.output_width >>
            partition.plan.input_scale >> partition.plan.gate_output_scale >> partition.plan.down_input_scale >>
            partition.plan.output_scale >> partition.input_id >> partition.output_id;
        partition.plan.kind = kind == "ffn"         ? PartitionPlan::Kind::FFN
                              : kind == "gated_ffn" ? PartitionPlan::Kind::GATED_FFN
                                                    : PartitionPlan::Kind::PACKED_LINEAR;
        metadata.partitions.push_back(std::move(partition));
    }
    return metadata;
}

class QnnPartition final {
public:
    QnnPartition(BuiltPartition built, Qnn_GraphHandle_t graph)
        : built_(std::move(built)),
          graph_(graph),
          input_dims_{1, static_cast<std::uint32_t>(built_.plan.input_width)},
          output_dims_{1, static_cast<std::uint32_t>(built_.plan.output_width)} {}

    auto start() const noexcept -> std::size_t { return built_.plan.start; }
    auto end() const noexcept -> std::size_t { return built_.plan.end; }

    auto execute(QnnContext& context, std::span<Node> nodes) -> void {
        const auto& input = nodes[built_.plan.start].inputs()[0];
        auto& output = nodes[built_.plan.end - 1].operands[nodes[built_.plan.end - 1].input_count];
        const auto rows = input.numel() / built_.plan.input_width;
        input_dims_ = {static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(built_.plan.input_width)};
        output_dims_ = {static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(built_.plan.output_width)};
        const auto input_values = require(input.data<float>());
        input_buffer_.resize(input_values.size());
        for (std::size_t index = 0; index < input_values.size(); ++index) {
            const auto quantized = std::nearbyint(input_values[index] / built_.plan.input_scale);
            const auto clamped = static_cast<int>(std::clamp(quantized, -128.F, 127.F));
            input_buffer_[index] = static_cast<std::uint8_t>(static_cast<std::int8_t>(clamped)) ^ 0x80U;
        }
        output_buffer_.assign(rows * built_.plan.output_width, 0);
        auto qnn_input = make_tensor("input", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_UFIXED_POINT_8, input_dims_);
        qnn_input.v1.id = built_.input_id;
        qnn_input.v1.quantizeParams = per_tensor(built_.plan.input_scale, -128);
        qnn_input.v1.clientBuf = {input_buffer_.data(), static_cast<std::uint32_t>(input_buffer_.size())};
        auto qnn_output = make_tensor("output", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_UFIXED_POINT_8, output_dims_);
        qnn_output.v1.id = built_.output_id;
        qnn_output.v1.quantizeParams = per_tensor(built_.plan.output_scale, -128);
        qnn_output.v1.clientBuf = {output_buffer_.data(), static_cast<std::uint32_t>(output_buffer_.size())};
        context.execute(graph_, &qnn_input, 1, &qnn_output, 1);
        auto output_values = require(output.data<float>());
        if (output_values.size() != output_buffer_.size())
            throw ops::Failure({ErrorCode::RUNTIME, "QNN partition output shape changed"});
        for (std::size_t index = 0; index < output_buffer_.size(); ++index)
            output_values[index] = (static_cast<int>(output_buffer_[index]) - 128) * built_.plan.output_scale;
    }

private:
    BuiltPartition built_;
    Qnn_GraphHandle_t graph_{};
    std::vector<std::uint32_t> input_dims_, output_dims_;
    std::vector<std::uint8_t> input_buffer_, output_buffer_;
};

class QnnStepExecutable final : public StepExecutable {
public:
    QnnStepExecutable(std::shared_ptr<QnnRuntime> runtime, std::unique_ptr<QnnContext> context, std::vector<Node> nodes,
                      std::vector<Tensor> inputs, std::vector<QnnPartition> partitions)
        : runtime_(std::move(runtime)),
          context_(std::move(context)),
          nodes_(std::move(nodes)),
          inputs_(std::move(inputs)),
          partitions_(std::move(partitions)) {
        std::ranges::sort(partitions_, {}, &QnnPartition::start);
    }

    auto run(std::span<const Tensor> inputs, std::span<Tensor> outputs) -> void override {
        if (inputs.size() != inputs_.size())
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "QNN replay received a different number of inputs"});
        for (std::size_t slot = 0; slot < inputs.size(); ++slot) {
            if (!inputs[slot].defined() || !inputs[slot].is_contiguous() ||
                inputs[slot].dtype() != inputs_[slot].dtype() || inputs[slot].device() != inputs_[slot].device() ||
                !std::ranges::equal(inputs[slot].shape(), inputs_[slot].shape()))
                throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "QNN replay input changed shape or dtype"});
            inputs_[slot] = inputs[slot];
        }
        rebind_inputs();
        std::size_t partition_index = 0;
        for (std::size_t index = 0; index < nodes_.size();) {
            if (partition_index < partitions_.size() && partitions_[partition_index].start() == index) {
                partitions_[partition_index].execute(*context_, nodes_);
                index = partitions_[partition_index].end();
                ++partition_index;
                continue;
            }
            auto& node = nodes_[index];
            const TensorInputs operands(node.inputs());
            auto node_outputs = std::span(node.operands).subspan(node.input_count);
            if (node.mode == graph::Mode::IN_PLACE) {
                node.prepared->run_(operands, node_outputs.front());
            } else {
                node.prepared->run_into(operands, node_outputs);
            }
            ++index;
        }
        (void)outputs;
    }

private:
    auto rebind_inputs() -> void {
        for (auto& node : nodes_) {
            for (std::size_t operand = 0; operand < node.sources.size(); ++operand) {
                const auto& source = node.sources[operand];
                if (source.kind == Source::Kind::INPUT)
                    node.operands[operand] = input_view(inputs_[source.index], source.offset, node.operands[operand]);
            }
        }
    }

    std::shared_ptr<QnnRuntime> runtime_;
    std::unique_ptr<QnnContext> context_;
    std::vector<Node> nodes_;
    std::vector<Tensor> inputs_;
    std::vector<QnnPartition> partitions_;
};

/// Hashes large constant tensors 64 bits at a time; offloaded operators hash their weights once when prepared.
auto hash_words(Hash& hash, const Tensor& tensor) -> void {
    hash.pod(static_cast<int>(tensor.dtype()));
    for (const auto extent : tensor.shape()) hash.pod(extent);
    const auto bytes = require(tensor.host_bytes());
    const auto words = bytes.size() / sizeof(std::uint64_t);
    auto value = hash.value;
    for (std::size_t index = 0; index < words; ++index) {
        std::uint64_t word = 0;
        std::memcpy(&word, bytes.data() + index * sizeof(word), sizeof(word));
        value = (value ^ word) * FNV_PRIME;
    }
    hash.value = value;
    hash.bytes(bytes.subspan(words * sizeof(std::uint64_t)));
}

/// A persistent model identity must be fast even in a debuggable APK. Hash complete small tensors (norms/scales) and
/// evenly sampled blocks of large packed weights; shapes, operation attributes, SDK build ID, and cache version are
/// hashed separately.
auto hash_sampled_tensor(Hash& hash, const Tensor& tensor) -> void {
    hash.pod(static_cast<int>(tensor.dtype()));
    for (const auto extent : tensor.shape()) hash.pod(extent);
    const auto bytes = require(tensor.host_bytes());
    hash.pod(bytes.size());
    constexpr std::size_t COMPLETE_LIMIT = 64 * 1024;
    constexpr std::size_t SAMPLES = 256;
    constexpr std::size_t SAMPLE_BYTES = 32;
    if (bytes.size() <= COMPLETE_LIMIT) return hash.bytes(bytes);
    for (std::size_t sample = 0; sample < SAMPLES; ++sample) {
        const auto offset = sample * (bytes.size() - SAMPLE_BYTES) / (SAMPLES - 1);
        hash.bytes(bytes.subspan(offset, SAMPLE_BYTES));
    }
}

auto environment_size(const char* name, std::size_t fallback) -> std::size_t {
    const auto text = getenv_string(name);
    std::size_t value = 0;
    if (!text.empty() && std::from_chars(text.data(), text.data() + text.size(), value).ec == std::errc{}) return value;
    return fallback;
}

auto ffn_variant() -> std::string {
    auto variant = getenv_string("KIDI_QNN_FFN");
    return variant.empty() ? std::string("u8") : variant;
}

auto structured_tiles_enabled() -> bool { return getenv_string("KIDI_QNN_STRUCTURED_TILES") == "1"; }

/// Plans one eager operator: fused gated feed-forwards always, calibrated projections above a weight-size floor.
auto plan_operator(const Node& node) -> std::optional<PartitionPlan> {
    if (node.operation == Operation::GATED_FEED_FORWARD) return find_gated_ffn(std::span(&node, 1), 0);
    if (!is_calibrated_packed_linear(node) ||
        node.inputs()[1].nbytes() < environment_size("KIDI_QNN_MIN_OPERATOR_BYTES", 4 * 1024 * 1024))
        return std::nullopt;
    return PartitionPlan{PartitionPlan::Kind::PACKED_LINEAR,
                         0,
                         1,
                         static_cast<int>(node.attributes[0]),
                         node.inputs()[0].size(-1),
                         0,
                         node.outputs()[0].size(-1),
                         node.epsilon,
                         0,
                         0,
                         scalar_bits(node.attributes[2])};
}

/// Rounds FP32 onto the symmetric INT8 grid (nearest-even, saturating) and stores it as UINT8 with offset 128.
auto quantize(std::span<const float> values, float inverse, std::span<std::uint8_t> output) -> void {
    std::size_t index = 0;
#if defined(__ARM_NEON)
    const auto scale = vdupq_n_f32(inverse);
    const auto bias = vdupq_n_u8(0x80);
    for (; index + 16 <= values.size(); index += 16) {
        const auto a = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(values.data() + index), scale));
        const auto b = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(values.data() + index + 4), scale));
        const auto c = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(values.data() + index + 8), scale));
        const auto d = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(values.data() + index + 12), scale));
        const auto low = vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
        const auto high = vcombine_s16(vqmovn_s32(c), vqmovn_s32(d));
        const auto bytes = vcombine_s8(vqmovn_s16(low), vqmovn_s16(high));
        vst1q_u8(output.data() + index, veorq_u8(vreinterpretq_u8_s8(bytes), bias));
    }
#endif
    for (; index < values.size(); ++index) {
        const auto quantized = std::clamp(std::nearbyint(values[index] * inverse), -128.F, 127.F);
        output[index] = static_cast<std::uint8_t>(static_cast<int>(quantized) + 128);
    }
}

auto dequantize(std::span<const std::uint8_t> values, float scale, std::span<float> output) -> void {
    std::size_t index = 0;
#if defined(__ARM_NEON)
    const auto factor = vdupq_n_f32(scale);
    const auto bias = vdupq_n_u8(0x80);
    for (; index + 16 <= values.size(); index += 16) {
        const auto bytes = vreinterpretq_s8_u8(veorq_u8(vld1q_u8(values.data() + index), bias));
        const auto low = vmovl_s8(vget_low_s8(bytes)), high = vmovl_s8(vget_high_s8(bytes));
        vst1q_f32(output.data() + index, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(low))), factor));
        vst1q_f32(output.data() + index + 4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(low))), factor));
        vst1q_f32(output.data() + index + 8, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(high))), factor));
        vst1q_f32(output.data() + index + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(high))), factor));
    }
#endif
    for (; index < values.size(); ++index)
        output[index] = static_cast<float>(static_cast<int>(values[index]) - 128) * scale;
}

/// One offloaded operator graph for a fixed block of rows, shared by every operator instance with the same weights.
struct CompiledOperator {
    std::unique_ptr<QnnContext> context;
    BuiltPartition built;
    Qnn_GraphHandle_t graph{};
    std::size_t rows = 0;
    std::mutex mutex; // one execution at a time per graph
};

/// Loads the cached context binary for `id`, or builds, prepares, and caches the graph for `rows`-row blocks.
auto compile_operator(const std::shared_ptr<QnnRuntime>& runtime, const Node& node, const PartitionPlan& plan,
                      const std::string& id, std::size_t rows) -> std::shared_ptr<CompiledOperator> {
    auto result = std::make_shared<CompiledOperator>();
    result->context = std::make_unique<QnnContext>(runtime);
    result->rows = rows;
    const auto directory = cache_directory();
    const auto binary_path = directory / (id + ".bin");
    const auto metadata_path = directory / (id + ".meta");
    const auto started = std::chrono::steady_clock::now();
    auto metadata = load_metadata(metadata_path, id);
    const bool cached = metadata && metadata->partitions.size() == 1 && std::filesystem::exists(binary_path);
    if (cached) {
        result->context->load(read_file(binary_path));
        result->built = metadata->partitions.front();
    } else {
        if (!runtime->prepare_available())
            throw ops::Failure(
                {ErrorCode::UNSUPPORTED, "QNN HTP prepare library is unavailable and no cached context exists"});
        // Build the graph for a full block: the node's own rows may be a shorter prompt tail.
        auto block = node;
        std::vector<std::int64_t> shape{static_cast<std::int64_t>(rows), static_cast<std::int64_t>(plan.input_width)};
        block.operands[0] = require(Tensor::empty(shape, DType::F32));
        std::filesystem::create_directories(directory);
        result->context->create();
        result->built = plan.kind == PartitionPlan::Kind::GATED_FFN
                            ? build_gated_ffn_graph(*result->context, "op", block, plan, ffn_variant())
                            : build_projection_graph(*result->context, "op", block, plan);
        write_file(binary_path, result->context->save());
        save_metadata(metadata_path, id, std::span(&result->built, 1));
    }
    result->graph = result->context->retrieve_graph(result->built.name);
    if (const char* dump = std::getenv("KIDI_QNN_DUMP"); dump && *dump)
        std::cerr << "kidi_qnn_operator|kind="
                  << (plan.kind == PartitionPlan::Kind::GATED_FFN ? "gated_ffn" : "projection") << "|rows=" << rows
                  << "|output=" << plan.output_width << "|cache=" << (cached ? "hit" : "miss") << "|ms="
                  << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count()
                  << '\n';
    return result;
}

/// A many-row calibrated projection or fused gated feed-forward offloaded to the HTP. It runs its CPU kernel until
/// the shared graph is compiled (or if compilation fails), then executes fixed row blocks on the HTP, zero-padding
/// the last. Activations cross the boundary as INT8 on the operator's calibrated scales, as the CPU kernels quantize.
class QnnOperator final : public Operator {
public:
    QnnOperator(std::shared_future<std::shared_ptr<CompiledOperator>> compiled, std::unique_ptr<Operator> fallback,
                const PartitionPlan& plan, std::vector<std::int64_t> output_shape, bool verify)
        : compiled_(std::move(compiled)),
          fallback_(std::move(fallback)),
          plan_(plan),
          output_shape_(std::move(output_shape)),
          verify_(verify) {}

    auto run(TensorInputs inputs) -> Tensor override {
        auto output = require(Tensor::empty(output_shape_, DType::F32));
        dispatch(inputs, output);
        return output;
    }
    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "offloaded QNN operators are not in-place"});
    }
    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { dispatch(inputs, outputs[0]); }

    ~QnnOperator() override {
        const char* dump = std::getenv("KIDI_QNN_DUMP");
        if (!dump || !*dump || !(calls_ + cpu_calls_)) return;
        std::cerr << "kidi_qnn_operator_timing|output=" << plan_.output_width << "|calls=" << calls_
                  << "|cpu_calls=" << cpu_calls_;
        if (calls_)
            std::cerr << "|quantize_us=" << timing_[0] / 1000 / calls_ << "|execute_us=" << timing_[1] / 1000 / calls_
                      << "|dequantize_us=" << timing_[2] / 1000 / calls_;
        if (compared_)
            std::cerr << "|differing=" << static_cast<double>(differing_) / compared_ << "|worst_quanta=" << worst_;
        std::cerr << '\n';
    }

private:
    auto ready() -> CompiledOperator* {
        if (compiled_value_) return compiled_value_.get();
        if (failed_ || compiled_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return nullptr;
        try {
            compiled_value_ = compiled_.get();
        } catch (const std::exception& error) {
            failed_ = true;
            std::cerr << "kidi_qnn_operator|fallback=cpu|error=" << error.what() << '\n';
        }
        return compiled_value_.get();
    }

    auto dispatch(TensorInputs inputs, Tensor& output) -> void {
        auto* compiled = ready();
        if (!compiled) {
            ++cpu_calls_;
            fallback_->run_into(inputs, std::span(&output, 1));
            return;
        }
        execute(*compiled, inputs[0], output);
        if (verify_) compare(inputs, output);
    }

    auto execute(CompiledOperator& compiled, const Tensor& input, Tensor& output) -> void {
        const std::scoped_lock lock(compiled.mutex);
        const auto started = std::chrono::steady_clock::now();
        std::chrono::nanoseconds quantizing{}, executing{}, dequantizing{};
        const auto& plan = compiled.built.plan;
        const auto rows = input.numel() / plan.input_width, block = compiled.rows;
        input_dims_ = {static_cast<std::uint32_t>(block), static_cast<std::uint32_t>(plan.input_width)};
        output_dims_ = {static_cast<std::uint32_t>(block), static_cast<std::uint32_t>(plan.output_width)};
        input_buffer_.resize(block * plan.input_width);
        output_buffer_.resize(block * plan.output_width);
        const auto values = require(input.data<float>());
        auto result = require(output.data<float>());
        if (result.size() != rows * plan.output_width)
            throw ops::Failure({ErrorCode::RUNTIME, "offloaded QNN operator output shape changed"});
        for (std::size_t first = 0; first < rows; first += block) {
            const auto count = std::min(block, rows - first);
            auto mark = std::chrono::steady_clock::now();
            quantize(values.subspan(first * plan.input_width, count * plan.input_width), 1.F / plan.input_scale,
                     input_buffer_);
            // Quantized zero pads the rest of a short block.
            std::fill(input_buffer_.begin() + static_cast<std::ptrdiff_t>(count * plan.input_width),
                      input_buffer_.end(), std::uint8_t{128});
            auto now = std::chrono::steady_clock::now();
            quantizing += now - mark;
            auto qnn_input = make_tensor("input", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_UFIXED_POINT_8, input_dims_);
            qnn_input.v1.id = compiled.built.input_id;
            qnn_input.v1.quantizeParams = per_tensor(plan.input_scale, -128);
            qnn_input.v1.clientBuf = {input_buffer_.data(), static_cast<std::uint32_t>(input_buffer_.size())};
            auto qnn_output =
                make_tensor("output", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_UFIXED_POINT_8, output_dims_);
            qnn_output.v1.id = compiled.built.output_id;
            qnn_output.v1.quantizeParams = per_tensor(plan.output_scale, -128);
            qnn_output.v1.clientBuf = {output_buffer_.data(), static_cast<std::uint32_t>(output_buffer_.size())};
            compiled.context->execute(compiled.graph, &qnn_input, 1, &qnn_output, 1);
            mark = std::chrono::steady_clock::now();
            executing += mark - now;
            dequantize(std::span(output_buffer_).first(count * plan.output_width), plan.output_scale,
                       result.subspan(first * plan.output_width, count * plan.output_width));
            dequantizing += std::chrono::steady_clock::now() - mark;
        }
        timing_[0] += quantizing.count();
        timing_[1] += executing.count();
        timing_[2] += dequantizing.count();
        ++calls_;
        (void)started;
    }

    /// With KIDI_QNN_VERIFY, compares against the CPU kernel in output quanta.
    auto compare(TensorInputs inputs, const Tensor& output) -> void {
        const auto expected = fallback_->run(inputs);
        const auto left = require(expected.data<float>()), right = require(output.data<float>());
        for (std::size_t index = 0; index < left.size(); ++index) {
            const auto quanta =
                static_cast<std::size_t>(std::lround(std::abs(left[index] - right[index]) / plan_.output_scale));
            differing_ += quanta != 0;
            worst_ = std::max(worst_, quanta);
        }
        compared_ += left.size();
    }

    std::shared_future<std::shared_ptr<CompiledOperator>> compiled_;
    std::shared_ptr<CompiledOperator> compiled_value_;
    bool failed_ = false;
    std::unique_ptr<Operator> fallback_;
    PartitionPlan plan_;
    std::vector<std::int64_t> output_shape_;
    bool verify_ = false;
    std::vector<std::uint32_t> input_dims_, output_dims_;
    std::vector<std::uint8_t> input_buffer_, output_buffer_;
    std::array<std::int64_t, 3> timing_{};
    std::size_t calls_ = 0, cpu_calls_ = 0, differing_ = 0, compared_ = 0, worst_ = 0;
};

/// Compiles offloaded operator graphs one at a time on a background thread; libQnnHtpPrepare is CPU- and
/// memory-heavy, so parallel compiles would only compete with inference.
class CompileQueue final {
public:
    CompileQueue() : worker_([this] { work(); }) { worker_.detach(); }
    auto push(std::function<void()> job) -> void {
        {
            const std::scoped_lock lock(mutex_);
            jobs_.push_back(std::move(job));
        }
        ready_.notify_one();
    }

private:
    auto work() -> void {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock lock(mutex_);
                ready_.wait(lock, [this] { return !jobs_.empty(); });
                job = std::move(jobs_.front());
                jobs_.pop_front();
            }
            job();
        }
    }
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::function<void()>> jobs_;
    std::thread worker_;
};

// ---------------------------------------------------------------------------------------------------------------
// Whole-step lowering: every node of a captured decode step runs on the HTP. The step is split into a few chained
// graphs of one context (a single huge graph exhausts libQnnHtpPrepare). KV caches live in registered shared memory
// as INT8 on their calibrated grids; the host writes the small per-step inputs and copies each new K/V row into place.

using Half = _Float16;

/// rpcmem buffers that the CPU and the HTP share without copies.
class SharedMemory final {
public:
    SharedMemory() {
        library_ = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
        if (!library_) throw ops::Failure({ErrorCode::UNSUPPORTED, std::string("load libcdsprpc.so: ") + dlerror()});
        allocate_ = reinterpret_cast<void* (*)(int, std::uint32_t, int)>(dlsym(library_, "rpcmem_alloc"));
        free_ = reinterpret_cast<void (*)(void*)>(dlsym(library_, "rpcmem_free"));
        to_fd_ = reinterpret_cast<int (*)(void*)>(dlsym(library_, "rpcmem_to_fd"));
        if (!allocate_ || !free_ || !to_fd_)
            throw ops::Failure({ErrorCode::UNSUPPORTED, "libcdsprpc.so lacks the rpcmem API"});
    }
    ~SharedMemory() = default;
    SharedMemory(const SharedMemory&) = delete;
    auto operator=(const SharedMemory&) -> SharedMemory& = delete;

    struct Block {
        std::shared_ptr<void> owner;
        std::byte* data = nullptr;
        int fd = -1;
    };
    auto allocate(std::size_t size) -> Block {
        constexpr int HEAP_ID_SYSTEM = 25;
        constexpr std::uint32_t FLAGS = 1;
        auto* data = allocate_(HEAP_ID_SYSTEM, FLAGS, static_cast<int>(size));
        if (!data) throw ops::Failure({ErrorCode::RUNTIME, "rpcmem_alloc failed for " + std::to_string(size)});
        const auto fd = to_fd_(data);
        if (fd < 0) throw ops::Failure({ErrorCode::RUNTIME, "rpcmem_to_fd failed"});
        std::memset(data, 0, size);
        return {std::shared_ptr<void>(data, [free = free_](void* pointer) { free(pointer); }),
                static_cast<std::byte*>(data), fd};
    }

private:
    void* library_{};
    void* (*allocate_)(int, std::uint32_t, int){};
    void (*free_)(void*){};
    int (*to_fd_)(void*){};
};

auto element_count(std::span<const std::uint32_t> dims) -> std::size_t {
    return std::accumulate(dims.begin(), dims.end(), std::size_t{1}, std::multiplies<>());
}

auto dims_of(std::span<const std::int64_t> shape) -> std::vector<std::uint32_t> {
    std::vector<std::uint32_t> result;
    for (const auto extent : shape) result.push_back(static_cast<std::uint32_t>(extent));
    if (result.empty()) result.push_back(1);
    return result;
}

/// Builds one HTP graph from typed values; owns names, shapes, and constant data until the graph is finalized.
class OpBuilder final {
public:
    struct Value {
        Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
        std::vector<std::uint32_t> dims;
        auto type() const -> Qnn_DataType_t { return tensor.v1.dataType; }
    };

    OpBuilder(QnnContext& context, Qnn_GraphHandle_t graph, std::string prefix)
        : context_(context), graph_(graph), prefix_(std::move(prefix)) {}

    auto tensor(Qnn_TensorType_t kind, Qnn_DataType_t type, std::vector<std::uint32_t> dims,
                Qnn_QuantizeParams_t quantize = QNN_QUANTIZE_PARAMS_INIT, const void* data = nullptr,
                std::size_t bytes = 0) -> Value {
        dims_.push_back(dims);
        names_.push_back(prefix_ + "t" + std::to_string(names_.size()));
        auto result = make_tensor(names_.back().c_str(), kind, type, dims_.back());
        result.v1.quantizeParams = quantize;
        if (data) result.v1.clientBuf = {const_cast<void*>(data), static_cast<std::uint32_t>(bytes)};
        check(context_.runtime().api().tensorCreateGraphTensor(graph_, &result),
              "register QNN tensor " + names_.back());
        if (data) result.v1.clientBuf = {const_cast<void*>(data), static_cast<std::uint32_t>(bytes)};
        return {result, std::move(dims)};
    }
    auto native(Qnn_DataType_t type, std::vector<std::uint32_t> dims,
                Qnn_QuantizeParams_t quantize = QNN_QUANTIZE_PARAMS_INIT) -> Value {
        return tensor(QNN_TENSOR_TYPE_NATIVE, type, std::move(dims), quantize);
    }
    auto half(std::span<const float> values, std::vector<std::uint32_t> dims) -> Value {
        auto& data = halves_.emplace_back(values.size());
        for (std::size_t index = 0; index < values.size(); ++index) data[index] = static_cast<Half>(values[index]);
        return tensor(QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, std::move(dims), QNN_QUANTIZE_PARAMS_INIT,
                      data.data(), data.size() * sizeof(Half));
    }
    auto float32(float value, std::vector<std::uint32_t> dims) -> Value {
        auto& data = floats_.emplace_back(element_count(dims), value);
        return tensor(QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_32, std::move(dims), QNN_QUANTIZE_PARAMS_INIT,
                      data.data(), data.size() * sizeof(float));
    }
    auto floats(std::span<const float> values, std::vector<std::uint32_t> dims) -> Value {
        auto& data = floats_.emplace_back(values.begin(), values.end());
        return tensor(QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_32, std::move(dims), QNN_QUANTIZE_PARAMS_INIT,
                      data.data(), data.size() * sizeof(float));
    }
    auto int32(std::int32_t value, std::vector<std::uint32_t> dims) -> Value {
        auto& data = integers_.emplace_back(element_count(dims), value);
        return tensor(QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32, std::move(dims), QNN_QUANTIZE_PARAMS_INIT,
                      data.data(), data.size() * sizeof(std::int32_t));
    }
    auto scalar(float value, std::size_t rank) -> Value {
        const std::array values{value};
        return half(values, std::vector<std::uint32_t>(rank, 1));
    }
    auto weights(Weights weights, std::vector<std::uint32_t> dims) -> Value {
        auto& stored = weights_.emplace_back(std::move(weights));
        // Encodings point into the stored scales, so rebuild them after the move.
        if (stored.params.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET)
            stored.params.axisScaleOffsetEncoding.scaleOffset = stored.scale_offsets.data();
        else if (stored.params.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET)
            stored.params.bwAxisScaleOffsetEncoding.scales = stored.scales.data();
        return tensor(QNN_TENSOR_TYPE_STATIC, stored.data_type, std::move(dims), stored.params, stored.data.data(),
                      stored.data.size());
    }
    /// A static tensor backed by an existing checkpoint buffer, which remains alive through graph finalization.
    auto static_tensor(const Tensor& source, Qnn_DataType_t type, std::vector<std::uint32_t> dims) -> Value {
        const auto bytes = require(source.host_bytes());
        return tensor(QNN_TENSOR_TYPE_STATIC, type, std::move(dims), QNN_QUANTIZE_PARAMS_INIT, bytes.data(),
                      bytes.size());
    }

    auto op(const char* type, std::vector<Value> inputs, std::vector<Value> outputs,
            std::vector<Qnn_Param_t> params = {}, const char* package = QNN_OP_PACKAGE_NAME_QTI_AISW) -> void {
        names_.push_back(prefix_ + "op" + std::to_string(names_.size()) + "_" + type);
        std::vector<Qnn_Tensor_t> in, out;
        for (const auto& value : inputs) in.push_back(value.tensor);
        for (const auto& value : outputs) out.push_back(value.tensor);
        Qnn_OpConfig_t config = QNN_OPCONFIG_INIT;
        config.version = QNN_OPCONFIG_VERSION_1;
        config.v1.name = names_.back().c_str();
        config.v1.packageName = package;
        config.v1.typeName = type;
        config.v1.numOfParams = static_cast<std::uint32_t>(params.size());
        config.v1.params = params.empty() ? nullptr : params.data();
        config.v1.numOfInputs = static_cast<std::uint32_t>(in.size());
        config.v1.inputTensors = in.data();
        config.v1.numOfOutputs = static_cast<std::uint32_t>(out.size());
        config.v1.outputTensors = out.data();
        check(context_.runtime().api().graphAddNode(graph_, config), std::string("add QNN node ") + type);
    }

    static auto boolean(const char* name, bool value) -> Qnn_Param_t {
        Qnn_Param_t param = QNN_PARAM_INIT;
        param.paramType = QNN_PARAMTYPE_SCALAR;
        param.name = name;
        param.scalarParam.dataType = QNN_DATATYPE_BOOL_8;
        param.scalarParam.bool8Value = value ? 1 : 0;
        return param;
    }
    static auto unsigned32(const char* name, std::uint32_t value) -> Qnn_Param_t {
        Qnn_Param_t param = QNN_PARAM_INIT;
        param.paramType = QNN_PARAMTYPE_SCALAR;
        param.name = name;
        param.scalarParam.dataType = QNN_DATATYPE_UINT_32;
        param.scalarParam.uint32Value = value;
        return param;
    }
    static auto float32(const char* name, float value) -> Qnn_Param_t {
        Qnn_Param_t param = QNN_PARAM_INIT;
        param.paramType = QNN_PARAMTYPE_SCALAR;
        param.name = name;
        param.scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
        param.scalarParam.floatValue = value;
        return param;
    }
    auto tensor_param(const char* name, Qnn_DataType_t type, std::vector<std::uint32_t> dims,
                      std::vector<std::int32_t> values) -> Qnn_Param_t {
        auto& data = integers_.emplace_back(std::move(values));
        auto value = tensor(QNN_TENSOR_TYPE_STATIC, type, std::move(dims), QNN_QUANTIZE_PARAMS_INIT, data.data(),
                            data.size() * sizeof(std::int32_t));
        Qnn_Param_t param = QNN_PARAM_INIT;
        param.paramType = QNN_PARAMTYPE_TENSOR;
        param.name = name;
        param.tensorParam = value.tensor;
        return param;
    }

    // ---- composite operations on FP16 values ----
    auto reshape(const Value& input, std::vector<std::uint32_t> dims) -> Value {
        if (input.dims == dims) return input;
        auto output = native(input.type(), std::move(dims), input.tensor.v1.quantizeParams);
        op(QNN_OP_RESHAPE, {input}, {output});
        return output;
    }
    /// Copies `input` into a graph output (Reshape to the same shape) so later graphs or the host can read it.
    auto expose(const Value& input) -> Value { return expose(input, input.dims); }
    auto expose(const Value& input, std::vector<std::uint32_t> dims) -> Value {
        auto output = tensor(QNN_TENSOR_TYPE_APP_READ, input.type(), std::move(dims), input.tensor.v1.quantizeParams);
        op(QNN_OP_RESHAPE, {input}, {output});
        return output;
    }
    auto slice(const Value& input, std::size_t axis, std::uint32_t start, std::uint32_t length) -> Value {
        auto dims = input.dims;
        dims[axis] = length;
        std::vector<std::int32_t> ranges;
        for (std::size_t index = 0; index < input.dims.size(); ++index) {
            const auto begin = index == axis ? start : 0U;
            const auto end = index == axis ? start + length : input.dims[index];
            ranges.insert(ranges.end(), {static_cast<std::int32_t>(begin), static_cast<std::int32_t>(end), 1});
        }
        auto output = native(input.type(), dims, input.tensor.v1.quantizeParams);
        op(QNN_OP_STRIDED_SLICE, {input}, {output},
           {tensor_param(QNN_OP_STRIDED_SLICE_PARAM_RANGES, QNN_DATATYPE_INT_32,
                         {static_cast<std::uint32_t>(input.dims.size()), 3}, std::move(ranges))});
        return output;
    }
    /// A contiguous window of `count` elements at element `offset` of the flattened input, shaped as `dims`.
    auto window(const Value& input, std::size_t offset, std::vector<std::uint32_t> dims) -> Value {
        const auto total = static_cast<std::uint32_t>(element_count(input.dims));
        const auto flat = reshape(input, {total});
        const auto count = static_cast<std::uint32_t>(element_count(dims));
        return reshape(slice(flat, 0, static_cast<std::uint32_t>(offset), count), std::move(dims));
    }
    auto align(const Value& input, std::size_t rank) -> Value {
        if (input.dims.size() >= rank) return input;
        auto dims = std::vector<std::uint32_t>(rank - input.dims.size(), 1);
        dims.insert(dims.end(), input.dims.begin(), input.dims.end());
        return reshape(input, std::move(dims));
    }
    auto binary(const char* type, Value left, Value right) -> Value {
        const auto rank = std::max(left.dims.size(), right.dims.size());
        left = align(left, rank);
        right = align(right, rank);
        std::vector<std::uint32_t> dims(rank);
        for (std::size_t index = 0; index < rank; ++index) dims[index] = std::max(left.dims[index], right.dims[index]);
        auto output = native(QNN_DATATYPE_FLOAT_16, std::move(dims));
        op(type, {left, right}, {output});
        return output;
    }
    auto unary(const char* type, const Value& input) -> Value {
        auto output = native(QNN_DATATYPE_FLOAT_16, input.dims);
        op(type, {input}, {output});
        return output;
    }
    auto select(const Value& condition, const Value& yes, const Value& no) -> Value {
        auto output = native(QNN_DATATYPE_FLOAT_16, yes.dims);
        op(QNN_OP_ELEMENT_WISE_SELECT, {condition, yes, no}, {output});
        return output;
    }
    auto cast(const Value& input, Qnn_DataType_t type) -> Value {
        auto output = native(type, input.dims);
        op(QNN_OP_CAST, {input}, {output});
        return output;
    }
    auto gather(const Value& table, const Value& indices) -> Value {
        auto dims = indices.dims;
        dims.insert(dims.end(), table.dims.begin() + 1, table.dims.end());
        auto output = native(table.type(), std::move(dims), table.tensor.v1.quantizeParams);
        op(QNN_OP_GATHER, {table, indices}, {output}, {unsigned32(QNN_OP_GATHER_PARAM_AXIS, 0)});
        return output;
    }
    auto tile(const Value& input, std::vector<std::int32_t> multiples) -> Value {
        auto dims = input.dims;
        for (std::size_t index = 0; index < dims.size(); ++index)
            dims[index] *= static_cast<std::uint32_t>(multiples[index]);
        auto output = native(input.type(), std::move(dims), input.tensor.v1.quantizeParams);
        op(QNN_OP_TILE, {input}, {output},
           {tensor_param(QNN_OP_TILE_PARAM_MULTIPLES, QNN_DATATYPE_UINT_32,
                         {static_cast<std::uint32_t>(multiples.size())}, std::move(multiples))});
        return output;
    }
    auto reduce(const char* type, const Value& input) -> Value {
        auto dims = input.dims;
        dims.back() = 1;
        auto output = native(QNN_DATATYPE_FLOAT_16, std::move(dims));
        const auto axes_name =
            std::string_view(type) == QNN_OP_REDUCE_MAX ? QNN_OP_REDUCE_MAX_PARAM_AXES : QNN_OP_REDUCE_MEAN_PARAM_AXES;
        const auto keep_name = std::string_view(type) == QNN_OP_REDUCE_MAX ? QNN_OP_REDUCE_MAX_PARAM_KEEP_DIMS
                                                                           : QNN_OP_REDUCE_MEAN_PARAM_KEEP_DIMS;
        op(type, {input}, {output},
           {tensor_param(axes_name, QNN_DATATYPE_UINT_32, {1}, {static_cast<std::int32_t>(input.dims.size() - 1)}),
            boolean(keep_name, true)});
        return output;
    }
    auto quantize(const Value& input, float scale, Qnn_TensorType_t kind = QNN_TENSOR_TYPE_NATIVE) -> Value {
        auto output = tensor(kind, QNN_DATATYPE_UFIXED_POINT_8, input.dims, per_tensor(scale, -128));
        op(QNN_OP_QUANTIZE, {input}, {output});
        return output;
    }
    auto dequantize(const Value& input) -> Value {
        auto output = native(QNN_DATATYPE_FLOAT_16, input.dims);
        op(QNN_OP_DEQUANTIZE, {input}, {output});
        return output;
    }
    auto concat(const Value& left, const Value& right, std::size_t axis) -> Value {
        auto dims = left.dims;
        dims[axis] += right.dims[axis];
        auto output = native(left.type(), std::move(dims));
        op(QNN_OP_CONCAT, {left, right}, {output},
           {unsigned32(QNN_OP_CONCAT_PARAM_AXIS, static_cast<std::uint32_t>(axis))});
        return output;
    }
    auto matmul(const Value& left, const Value& right, bool transpose_right) -> Value {
        auto dims = left.dims;
        dims.back() = transpose_right ? right.dims[right.dims.size() - 2] : right.dims.back();
        auto output = native(QNN_DATATYPE_FLOAT_16, std::move(dims));
        op(QNN_OP_MAT_MUL, {left, right}, {output},
           {boolean(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, false),
            boolean(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, transpose_right)});
        return output;
    }
    auto transpose(const Value& input, std::vector<std::int32_t> permutation) -> Value {
        std::vector<std::uint32_t> dims(permutation.size());
        for (std::size_t index = 0; index < permutation.size(); ++index)
            dims[index] = input.dims[static_cast<std::size_t>(permutation[index])];
        auto output = native(input.type(), std::move(dims), input.tensor.v1.quantizeParams);
        op(QNN_OP_TRANSPOSE, {input}, {output},
           {tensor_param(QNN_OP_TRANSPOSE_PARAM_PERM, QNN_DATATYPE_UINT_32,
                         {static_cast<std::uint32_t>(permutation.size())}, std::move(permutation))});
        return output;
    }
    /// x / sqrt(mean(x^2) + epsilon) * weight, computed on x / max|x| so FP16 squares cannot overflow.
    auto rms_norm(const Value& input, const Value& weight, float epsilon) -> Value {
        const auto rank = input.dims.size();
        auto peak = binary(QNN_OP_ELEMENT_WISE_MAXIMUM,
                           reduce(QNN_OP_REDUCE_MAX, unary(QNN_OP_ELEMENT_WISE_ABS, input)), scalar(1e-2F, rank));
        auto scaled = binary(QNN_OP_ELEMENT_WISE_DIVIDE, input, peak);
        auto mean = reduce(QNN_OP_REDUCE_MEAN, binary(QNN_OP_ELEMENT_WISE_MULTIPLY, scaled, scaled));
        auto floor =
            binary(QNN_OP_ELEMENT_WISE_DIVIDE, binary(QNN_OP_ELEMENT_WISE_DIVIDE, scalar(epsilon, rank), peak), peak);
        auto inverse = unary(QNN_OP_ELEMENT_WISE_RSQRT, binary(QNN_OP_ELEMENT_WISE_ADD, mean, floor));
        return binary(QNN_OP_ELEMENT_WISE_MULTIPLY, binary(QNN_OP_ELEMENT_WISE_MULTIPLY, scaled, inverse), weight);
    }
    /// gelu_tanh(gate) * up; FP16 overflow of the cube saturates tanh to the correct limit.
    auto gelu_multiply(const Value& gate, const Value& up) -> Value {
        const auto rank = gate.dims.size();
        auto cube = binary(QNN_OP_ELEMENT_WISE_MULTIPLY, binary(QNN_OP_ELEMENT_WISE_MULTIPLY, gate, gate), gate);
        auto inner =
            binary(QNN_OP_ELEMENT_WISE_ADD, gate, binary(QNN_OP_ELEMENT_WISE_MULTIPLY, cube, scalar(0.044715F, rank)));
        auto tanh = unary(QNN_OP_TANH, binary(QNN_OP_ELEMENT_WISE_MULTIPLY, inner, scalar(0.7978845608028654F, rank)));
        auto half_gate = binary(QNN_OP_ELEMENT_WISE_MULTIPLY, gate, scalar(0.5F, rank));
        auto gelu =
            binary(QNN_OP_ELEMENT_WISE_MULTIPLY, half_gate, binary(QNN_OP_ELEMENT_WISE_ADD, tanh, scalar(1.F, rank)));
        return binary(QNN_OP_ELEMENT_WISE_MULTIPLY, gelu, up);
    }

private:
    QnnContext& context_;
    Qnn_GraphHandle_t graph_;
    std::string prefix_;
    std::deque<std::string> names_;
    std::deque<std::vector<std::uint32_t>> dims_;
    std::deque<std::vector<Half>> halves_;
    std::deque<std::vector<float>> floats_;
    std::deque<std::vector<std::int32_t>> integers_;
    std::deque<Weights> weights_;
};

/// A graph input or output bound at execution time.
struct Port {
    enum class Kind {
        PROLOGUE,
        INDICES,
        MASK,
        STRUCTURED_MASK,
        FRESH_MASK,
        STEP_INPUT,
        POSITION,
        CACHE,
        CONSTANT,
        IMPORT,
        EXPORT,
        DEBUG,
        ROW,
        TOKEN
    };
    Kind kind{};
    std::size_t index = 0; // node for PROLOGUE, step input slot for MASK/STEP_INPUT, cache, export, or row
    Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
    std::vector<std::uint32_t> dims;
    std::vector<std::byte> buffer; // raw client data (unused for CACHE)
};

struct LoweredGraph {
    std::string name;
    Qnn_GraphHandle_t handle{};
    std::size_t first_node = 0, last_node = 0, weight_bytes = 0;
    std::vector<Port> inputs, outputs;
};

/// One KV cache: its step input slot, width, calibrated grid, and the HTP-side UINT8 copy in shared memory.
struct NpuCache {
    std::size_t slot = 0, width = 0, capacity = 0;
    std::size_t row_node = 0; // STATIC_ROUND or scaled INT8 CAST producing this step's new row
    float scale = 0;
    SharedMemory::Block memory;
    Qnn_MemHandle_t handle{};
    std::vector<std::uint32_t> dims;
    const void* synced_storage = nullptr;
    std::size_t synced = 0;
};

/// Packed model data resident in FastRPC shared memory and bound as a graph input without per-step copies.
struct NpuConstant {
    std::size_t source_node = 0;
    SharedMemory::Block memory;
    Qnn_MemHandle_t handle{};
    std::vector<std::uint32_t> dims;
    Qnn_DataType_t type = QNN_DATATYPE_UNDEFINED;
};

auto whole_fingerprint(const Graph& graph, std::string_view key, const QnnRuntime& runtime) -> std::string {
    Hash hash;
    hash.text(CACHE_VERSION);
    hash.text("whole-step");
    hash.text(key);
    hash.text(runtime.build_id());
    if (runtime.fused_ops()) hash.text("kidi-ops-fused-v1");
    if (runtime.structured_attention()) hash.text("custom-attention-v2");
    if (structured_tiles_enabled()) hash.text("structured-tiles-v1");
    hash.pod(environment_size("KIDI_QNN_PARTITION_MB", 256));
    std::unordered_set<const std::byte*> constants;
    for (const auto& node : graph.nodes()) {
        hash.pod(static_cast<int>(node.operation));
        hash.pod(static_cast<int>(node.dtype));
        hash.pod(node.epsilon);
        hash.pod(static_cast<int>(node.mode));
        for (const auto attribute : node.attributes) hash.pod(attribute);
        for (std::size_t operand = 0; operand < node.input_count; ++operand) {
            const auto& source = node.sources[operand];
            hash.pod(static_cast<int>(source.kind));
            hash.pod(source.index);
            hash.pod(source.output);
            hash.pod(source.offset);
            if (source.kind != Source::Kind::EXTERNAL) continue;
            const auto bytes = require(node.operands[operand].host_bytes());
            if (constants.insert(bytes.data()).second) hash_sampled_tensor(hash, node.operands[operand]);
        }
    }
    return hex(hash.value);
}

/// Runs a complete captured prefill or decode step through chained HTP graphs.
class WholeStepExecutable final : public StepExecutable {
public:
    WholeStepExecutable(std::shared_ptr<QnnRuntime> runtime, std::unique_ptr<QnnContext> context,
                        std::unique_ptr<SharedMemory> memory, std::vector<Node> nodes, std::vector<Tensor> inputs,
                        std::vector<std::size_t> prologue, std::vector<NpuCache> caches,
                        std::vector<NpuConstant> constants, std::vector<LoweredGraph> graphs, std::size_t exports)
        : runtime_(std::move(runtime)),
          context_(std::move(context)),
          memory_(std::move(memory)),
          nodes_(std::move(nodes)),
          inputs_(std::move(inputs)),
          prologue_(std::move(prologue)),
          caches_(std::move(caches)),
          constants_(std::move(constants)),
          graphs_(std::move(graphs)),
          exports_(exports) {}

    ~WholeStepExecutable() override {
        const char* dump = std::getenv("KIDI_QNN_DUMP");
        if (!dump || !*dump || !steps_) return;
        std::ostringstream message;
        message << "kidi_qnn_whole|steps=" << steps_ << "|host_us=" << host_ns_ / 1000 / steps_;
        for (std::size_t index = 0; index < graphs_.size(); ++index)
            message << "|graph" << index << "_us=" << graph_ns_[index] / 1000 / steps_;
        diagnostic_log(message.str());
    }

    auto run(std::span<const Tensor> inputs, std::span<Tensor> outputs) -> void override {
        const auto started = std::chrono::steady_clock::now();
        for (std::size_t slot = 0; slot < inputs.size(); ++slot) inputs_[slot] = inputs[slot];
        const auto position = static_cast<std::size_t>(require(inputs_[1].data<std::int32_t>())[0]);
        for (const auto index : prologue_) {
            auto& node = nodes_[index];
            for (std::size_t operand = 0; operand < node.input_count; ++operand)
                if (node.sources[operand].kind == Source::Kind::INPUT)
                    node.operands[operand] = inputs_[node.sources[operand].index];
            node.prepared->run_into(TensorInputs(node.inputs()), std::span(node.operands).subspan(node.input_count));
        }
        for (auto& cache : caches_) synchronize(cache, position);
        auto host = std::chrono::steady_clock::now() - started;
        graph_ns_.resize(graphs_.size());
        if (!steps_ && std::getenv("KIDI_QNN_DUMP")) {
            std::ostringstream message;
            message << "kidi_qnn_execute|graphs=" << graphs_.size() << "|position=" << position;
            diagnostic_log(message.str());
        }
        for (std::size_t index = 0; index < graphs_.size(); ++index) {
            auto mark = std::chrono::steady_clock::now();
            auto& graph = graphs_[index];
            std::vector<Qnn_Tensor_t> in, out;
            for (auto& port : graph.inputs) in.push_back(bind_input(port, position));
            for (auto& port : graph.outputs) out.push_back(bind_output(port));
            host += std::chrono::steady_clock::now() - mark;
            mark = std::chrono::steady_clock::now();
            context_->execute(graph.handle, in.data(), static_cast<std::uint32_t>(in.size()), out.data(),
                              static_cast<std::uint32_t>(out.size()), graph.name.c_str());
            graph_ns_[index] +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - mark).count();
            mark = std::chrono::steady_clock::now();
            for (auto& port : graph.outputs) collect(port, position, outputs);
            host += std::chrono::steady_clock::now() - mark;
        }
        host_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(host).count();
        ++steps_;
    }

private:
    /// Mirrors CPU cache rows [synced, position) onto the INT8 grid; restarts when the cache storage changes.
    auto synchronize(NpuCache& cache, std::size_t position) -> void {
        const auto& source = inputs_[cache.slot];
        const void* storage = source.storage_identity();
        if (storage != cache.synced_storage || position < cache.synced) cache.synced = 0;
        cache.synced_storage = storage;
        if (cache.synced >= position) return;
        auto* target = reinterpret_cast<std::uint8_t*>(cache.memory.data);
        const auto offset = cache.synced * cache.width;
        const auto count = (position - cache.synced) * cache.width;
        if (source.dtype() == DType::F32) {
            const auto values = require(source.data<float>());
            quantize(values.subspan(offset, count), 1.F / cache.scale, std::span(target + offset, count));
        } else if (source.dtype() == DType::I8) {
            const auto values = require(source.data<std::int8_t>());
            for (std::size_t index = 0; index < count; ++index)
                target[offset + index] =
                    static_cast<std::uint8_t>(static_cast<std::int32_t>(values[offset + index]) + 128);
        } else {
            throw ops::Failure({ErrorCode::UNSUPPORTED, "QNN KV cache source must be FP32 or INT8"});
        }
        cache.synced = position;
    }

    auto bind_input(Port& port, std::size_t position) -> Qnn_Tensor_t {
        auto tensor = port.tensor;
        tensor.v1.clientBuf = {port.buffer.data(), static_cast<std::uint32_t>(port.buffer.size())};
        switch (port.kind) {
            case Port::Kind::PROLOGUE: {
                const auto values = require(nodes_[port.index].outputs()[0].data<float>());
                to_half(values, port.buffer);
                break;
            }
            case Port::Kind::INDICES: {
                const auto bytes = require(inputs_[port.index].host_bytes());
                std::memcpy(port.buffer.data(), bytes.data(), std::min(port.buffer.size(), bytes.size()));
                break;
            }
            case Port::Kind::STEP_INPUT:
                to_half(require(inputs_[port.index].data<float>()), port.buffer);
                break;
            case Port::Kind::POSITION: {
                const auto bytes = require(inputs_[port.index].host_bytes());
                std::memcpy(port.buffer.data(), bytes.data(), std::min(bytes.size(), port.buffer.size()));
                break;
            }
            case Port::Kind::MASK: {
                const auto values = require(inputs_[port.index].data<float>());
                auto* target = reinterpret_cast<Half*>(port.buffer.data());
                const auto extent = port.dims.back();
                const auto rows = values.size() / extent;
                for (std::size_t row = 0; row < rows; ++row)
                    for (std::size_t column = 0; column < extent; ++column) {
                        const auto value = values[row * extent + column];
                        const bool fresh = column >= position && column < position + rows;
                        target[row * extent + column] = static_cast<Half>(fresh || value < -30000.F ? -30000.F : value);
                    }
                break;
            }
            case Port::Kind::STRUCTURED_MASK: {
                const auto values = require(inputs_[port.index].data<float>());
                auto* target = reinterpret_cast<Half*>(port.buffer.data());
                for (std::size_t index = 0; index < values.size(); ++index)
                    target[index] = static_cast<Half>(values[index] < -30000.F ? -30000.F : values[index]);
                break;
            }
            case Port::Kind::FRESH_MASK: {
                const auto values = require(inputs_[port.index].data<float>());
                const auto extent = inputs_[port.index].size(-1);
                const auto rows = port.dims.back();
                auto* target = reinterpret_cast<Half*>(port.buffer.data());
                for (std::size_t row = 0; row < rows; ++row)
                    for (std::size_t column = 0; column < rows; ++column) {
                        const auto value = values[row * extent + position + column];
                        target[row * rows + column] = static_cast<Half>(value < -30000.F ? -30000.F : value);
                    }
                break;
            }
            case Port::Kind::CACHE:
                tensor.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
                tensor.v1.memHandle = caches_[port.index].handle;
                break;
            case Port::Kind::CONSTANT:
                tensor.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
                tensor.v1.memHandle = constants_[port.index].handle;
                break;
            case Port::Kind::IMPORT:
                break; // filled when the exporting graph ran
            default:
                break;
        }
        return tensor;
    }

    auto bind_output(Port& port) -> Qnn_Tensor_t {
        auto tensor = port.tensor;
        tensor.v1.clientBuf = {port.buffer.data(), static_cast<std::uint32_t>(port.buffer.size())};
        return tensor;
    }

    auto collect(Port& port, std::size_t position, std::span<Tensor> outputs) -> void {
        switch (port.kind) {
            case Port::Kind::EXPORT:
                for (auto& graph : graphs_)
                    for (auto& input : graph.inputs)
                        if (input.kind == Port::Kind::IMPORT && input.index == port.index)
                            std::memcpy(input.buffer.data(), port.buffer.data(), input.buffer.size());
                break;
            case Port::Kind::DEBUG: {
                const auto expected = require(nodes_[port.index].outputs()[0].data<float>());
                const auto* actual = reinterpret_cast<const Half*>(port.buffer.data());
                double squared = 0, reference = 0, actual_energy = 0, maximum = 0;
                const auto count = std::min(expected.size(), port.buffer.size() / sizeof(Half));
                for (std::size_t index = 0; index < count; ++index) {
                    const auto delta = static_cast<double>(actual[index]) - expected[index];
                    squared += delta * delta;
                    reference += static_cast<double>(expected[index]) * expected[index];
                    actual_energy += static_cast<double>(actual[index]) * actual[index];
                    maximum = std::max(maximum, std::abs(delta));
                }
                std::cerr << "kidi_qnn_debug|node=" << port.index
                          << "|op=" << operation_name(nodes_[port.index].operation) << "|max=" << maximum
                          << "|rms=" << std::sqrt(squared / std::max<std::size_t>(count, 1))
                          << "|expected_rms=" << std::sqrt(reference / std::max<std::size_t>(count, 1))
                          << "|actual_rms=" << std::sqrt(actual_energy / std::max<std::size_t>(count, 1))
                          << "|relative_rms=" << std::sqrt(squared / std::max(reference, 1e-30)) << '\n';
                break;
            }
            case Port::Kind::ROW: {
                auto& cache = caches_[port.index];
                const auto rows = port.buffer.size() / cache.width;
                auto* target = reinterpret_cast<std::uint8_t*>(cache.memory.data) + position * cache.width;
                std::memcpy(target, port.buffer.data(), rows * cache.width);
                if (inputs_[cache.slot].dtype() == DType::F32) {
                    auto values = require(inputs_[cache.slot].data<float>());
                    dequantize(std::span(reinterpret_cast<const std::uint8_t*>(port.buffer.data()), rows * cache.width),
                               cache.scale, values.subspan(position * cache.width, rows * cache.width));
                } else if (inputs_[cache.slot].dtype() == DType::I8) {
                    auto values = require(inputs_[cache.slot].data<std::int8_t>());
                    const auto* source = reinterpret_cast<const std::uint8_t*>(port.buffer.data());
                    const auto offset = position * cache.width;
                    for (std::size_t index = 0; index < rows * cache.width; ++index)
                        values[offset + index] =
                            static_cast<std::int8_t>(static_cast<std::int32_t>(source[index]) - 128);
                } else {
                    throw ops::Failure({ErrorCode::UNSUPPORTED, "QNN KV cache destination must be FP32 or INT8"});
                }
                if (cache.synced == position) cache.synced = position + rows;
                break;
            }
            case Port::Kind::TOKEN: {
                std::int32_t token = 0;
                std::memcpy(&token, port.buffer.data(), sizeof(token));
                require(outputs[0].data<std::int32_t>())[0] = token;
                break;
            }
            default:
                break;
        }
    }

    static auto to_half(std::span<const float> values, std::vector<std::byte>& buffer) -> void {
        auto* target = reinterpret_cast<Half*>(buffer.data());
        const auto count = std::min(values.size(), buffer.size() / sizeof(Half));
        for (std::size_t index = 0; index < count; ++index) target[index] = static_cast<Half>(values[index]);
    }

    std::shared_ptr<QnnRuntime> runtime_;
    std::unique_ptr<QnnContext> context_;
    std::unique_ptr<SharedMemory> memory_;
    std::vector<Node> nodes_;
    std::vector<Tensor> inputs_;
    std::vector<std::size_t> prologue_;
    std::vector<NpuCache> caches_;
    std::vector<NpuConstant> constants_;
    std::vector<LoweredGraph> graphs_;
    std::size_t exports_ = 0;
    std::size_t steps_ = 0;
    std::int64_t host_ns_ = 0;
    std::vector<std::int64_t> graph_ns_;
};

/// Translates a captured single-token decode step node by node into chained HTP graphs.
class StepLowering final {
public:
    StepLowering(std::shared_ptr<QnnRuntime> runtime, const Graph& graph, bool prefill, std::mutex& constant_mutex,
                 std::map<const void*, SharedMemory::Block>& constant_blocks)
        : runtime_(std::move(runtime)),
          graph_(graph),
          prefill_(prefill),
          constant_mutex_(constant_mutex),
          constant_blocks_(constant_blocks) {}

    auto lower(const std::filesystem::path& binary_path = {}, const std::filesystem::path& metadata_path = {},
               std::string_view id = {}) -> std::unique_ptr<StepExecutable> {
        const auto nodes = graph_.nodes();
        analyze();
        context_ = std::make_unique<QnnContext>(runtime_);
        context_->create();
        memory_ = std::make_unique<SharedMemory>();
        for (auto& cache : caches_) {
            cache.dims = {static_cast<std::uint32_t>(cache.capacity), static_cast<std::uint32_t>(cache.width)};
            cache.memory = memory_->allocate(cache.capacity * cache.width);
            cache.handle = context_->register_memory(cache.memory.fd, cache.dims, QNN_DATATYPE_UFIXED_POINT_8);
        }
        std::size_t current = SIZE_MAX;
        const auto limit = std::min(nodes.size(), environment_size("KIDI_QNN_NODE_LIMIT", nodes.size()));
        for (std::size_t index = 0; index < limit; ++index) {
            if (is_prologue_[index]) continue;
            if (partition_[index] != current) {
                finish();
                current = partition_[index];
                begin(current, index);
            }
            graphs_.back().last_node = index;
            lower_node(index);
        }
        finish();
        if (!binary_path.empty()) {
            write_file(binary_path, context_->save());
            save_whole_metadata(metadata_path, id);
        }
        return executable();
    }

    auto load(std::span<const std::uint8_t> binary, const std::filesystem::path& metadata_path,
              std::string_view id) -> std::unique_ptr<StepExecutable> {
        const auto document = nlohmann::json::parse(read_file(metadata_path));
        if (document.value("version", "") != CACHE_VERSION || document.value("id", "") != id ||
            document.value("prefill", !prefill_) != prefill_)
            throw unsupported("cached whole-step metadata does not match");
        context_ = std::make_unique<QnnContext>(runtime_);
        const auto failures = graph_memory_failures.load(std::memory_order_relaxed);
        context_->load(binary);
        if (graph_memory_failures.load(std::memory_order_relaxed) != failures)
            throw ops::Failure({ErrorCode::RUNTIME, "loading cached QNN context exhausted DSP mapped memory"});
        memory_ = std::make_unique<SharedMemory>();
        prologue_ = document.at("prologue").get<std::vector<std::size_t>>();
        for (const auto& item : document.at("caches")) {
            NpuCache cache;
            cache.slot = item.at("slot");
            cache.width = item.at("width");
            cache.capacity = item.at("capacity");
            cache.row_node = item.at("row_node");
            cache.scale = item.at("scale");
            cache.dims = item.at("dims").get<std::vector<std::uint32_t>>();
            cache.memory = memory_->allocate(cache.capacity * cache.width);
            cache.handle = context_->register_memory(cache.memory.fd, cache.dims, QNN_DATATYPE_UFIXED_POINT_8);
            caches_.push_back(std::move(cache));
        }
        for (const auto& item : document.at("constants")) {
            const auto node = item.at("source_node").get<std::size_t>();
            const auto type = static_cast<Qnn_DataType_t>(item.at("type").get<std::uint32_t>());
            const auto dims = item.at("dims").get<std::vector<std::uint32_t>>();
            constants_.push_back(resident_constant(graph_.nodes()[node].inputs()[1], node, type, dims));
        }
        for (const auto& item : document.at("graphs")) {
            LoweredGraph graph;
            graph.name = item.at("name");
            graph.first_node = item.at("first_node");
            graph.last_node = item.at("last_node");
            graph.handle = context_->retrieve_graph(graph.name);
            for (const auto& port : item.at("inputs")) graph.inputs.push_back(load_port(port, true));
            for (const auto& port : item.at("outputs")) graph.outputs.push_back(load_port(port, false));
            graphs_.push_back(std::move(graph));
        }
        exported_count_ = document.at("exports");
        return executable();
    }

private:
    auto executable() -> std::unique_ptr<StepExecutable> {
        const auto nodes = graph_.nodes();
        return std::make_unique<WholeStepExecutable>(
            runtime_, std::move(context_), std::move(memory_), std::vector<Node>(nodes.begin(), nodes.end()),
            std::vector<Tensor>(graph_.inputs().begin(), graph_.inputs().end()), prologue_, std::move(caches_),
            std::move(constants_), std::move(graphs_), exported_count_ ? exported_count_ : exported_.size());
    }

    using Value = OpBuilder::Value;
    using Key = std::pair<std::size_t, std::size_t>; // node, output

    auto unsupported(const std::string& message) -> ops::Failure {
        return ops::Failure({ErrorCode::UNSUPPORTED, "whole-step QNN lowering: " + message});
    }

    static auto data_bytes(Qnn_DataType_t type) -> std::size_t {
        switch (type) {
            case QNN_DATATYPE_FLOAT_16:
            case QNN_DATATYPE_UFIXED_POINT_16:
                return 2;
            case QNN_DATATYPE_INT_32:
            case QNN_DATATYPE_UINT_32:
            case QNN_DATATYPE_FLOAT_32:
                return 4;
            default:
                return 1;
        }
    }

    static auto save_port(const Port& port) -> nlohmann::json {
        nlohmann::json result{{"kind", static_cast<int>(port.kind)},
                              {"index", port.index},
                              {"id", port.tensor.v1.id},
                              {"type", static_cast<std::uint32_t>(port.tensor.v1.dataType)},
                              {"dims", port.dims}};
        const auto& quantization = port.tensor.v1.quantizeParams;
        if (quantization.encodingDefinition == QNN_DEFINITION_DEFINED) {
            if (quantization.quantizationEncoding != QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
                throw ops::Failure({ErrorCode::UNSUPPORTED, "whole-step port uses non-scalar quantization"});
            result["scale"] = quantization.scaleOffsetEncoding.scale;
            result["offset"] = quantization.scaleOffsetEncoding.offset;
        }
        return result;
    }

    auto load_port(const nlohmann::json& item, bool input_port) -> Port {
        Port result;
        result.kind = static_cast<Port::Kind>(item.at("kind").get<int>());
        result.index = item.at("index");
        result.dims = item.at("dims").get<std::vector<std::uint32_t>>();
        const auto type = static_cast<Qnn_DataType_t>(item.at("type").get<std::uint32_t>());
        result.tensor =
            make_tensor("port", input_port ? QNN_TENSOR_TYPE_APP_WRITE : QNN_TENSOR_TYPE_APP_READ, type, result.dims);
        result.tensor.v1.id = item.at("id");
        if (item.contains("scale"))
            result.tensor.v1.quantizeParams = per_tensor(item.at("scale").get<float>(), item.at("offset").get<int>());
        if (result.kind != Port::Kind::CACHE && result.kind != Port::Kind::CONSTANT)
            result.buffer.assign(element_count(result.dims) * data_bytes(type), std::byte{});
        return result;
    }

    auto save_whole_metadata(const std::filesystem::path& path, std::string_view id) const -> void {
        nlohmann::json document{{"version", std::string(CACHE_VERSION)},
                                {"id", std::string(id)},
                                {"prefill", prefill_},
                                {"exports", exported_.size()},
                                {"prologue", prologue_}};
        document["caches"] = nlohmann::json::array();
        for (const auto& cache : caches_)
            document["caches"].push_back({{"slot", cache.slot},
                                          {"width", cache.width},
                                          {"capacity", cache.capacity},
                                          {"row_node", cache.row_node},
                                          {"scale", cache.scale},
                                          {"dims", cache.dims}});
        document["constants"] = nlohmann::json::array();
        for (const auto& constant : constants_)
            document["constants"].push_back({{"source_node", constant.source_node},
                                             {"type", static_cast<std::uint32_t>(constant.type)},
                                             {"dims", constant.dims}});
        document["graphs"] = nlohmann::json::array();
        for (const auto& graph : graphs_) {
            nlohmann::json item{{"name", graph.name}, {"first_node", graph.first_node}, {"last_node", graph.last_node}};
            item["inputs"] = nlohmann::json::array();
            item["outputs"] = nlohmann::json::array();
            for (const auto& port : graph.inputs) item["inputs"].push_back(save_port(port));
            for (const auto& port : graph.outputs) item["outputs"].push_back(save_port(port));
            document["graphs"].push_back(std::move(item));
        }
        write_text(path, document.dump());
    }

    /// Finds prologue nodes, KV caches, graph partitions, and the values that cross partitions.
    auto analyze() -> void {
        const auto nodes = graph_.nodes();
        is_prologue_.assign(nodes.size(), false);
        partition_.assign(nodes.size(), 0);
        for (std::size_t index = 0; index < nodes.size(); ++index) {
            const auto& node = nodes[index];
            if (node.operation == Operation::EMBEDDING) {
                is_prologue_[index] = true;
                prologue_.push_back(index);
            }
            if (node.operation == Operation::SCATTER) {
                const bool has_row_node = node.sources[1].kind == Source::Kind::NODE;
                const auto row_node = has_row_node ? node.sources[1].index : 0;
                const bool rounded_row = has_row_node && nodes[row_node].operation == Operation::STATIC_ROUND;
                const bool cast_row = has_row_node && nodes[row_node].operation == Operation::CAST &&
                                      nodes[row_node].outputs()[0].dtype() == DType::I8 && nodes[row_node].epsilon > 0;
                if (node.sources[0].kind != Source::Kind::INPUT || (!rounded_row && !cast_row)) {
                    std::ostringstream detail;
                    detail << "cache scatter " << index << " sources=" << static_cast<int>(node.sources[0].kind) << ','
                           << static_cast<int>(node.sources[1].kind);
                    if (node.sources[1].kind == Source::Kind::NODE)
                        detail << " update_node=" << node.sources[1].index
                               << " update_op=" << operation_name(nodes[node.sources[1].index].operation);
                    throw unsupported(detail.str());
                }
                const auto slot = node.sources[0].index;
                const auto& cache = graph_.inputs()[slot];
                if ((cache.dtype() != DType::F32 && cache.dtype() != DType::I8) || cache.dimensions() != 3)
                    throw unsupported("KV caches must be FP32 or INT8 [1, capacity, width]");
                NpuCache entry;
                entry.slot = slot;
                // Set from the ATTENTION key length below. The model state may reserve 9K+ rows while this
                // compiled shape uses only a 512/1024/... prefix bucket.
                entry.capacity = 0;
                entry.width = cache.size(2);
                entry.row_node = row_node;
                entry.scale = nodes[entry.row_node].epsilon;
                cache_of_slot_[slot] = caches_.size();
                row_cache_[entry.row_node] = caches_.size();
                caches_.push_back(entry);
            }
        }
        std::size_t attention_extent = 0;
        for (std::size_t index = 0; index < nodes.size(); ++index) {
            const auto& node = nodes[index];
            if (node.operation != Operation::ATTENTION || node.attributes.size() != 4) continue;
            const auto extent = static_cast<std::size_t>(node.attributes[3]);
            attention_extent = std::max(attention_extent, extent);
            for (std::size_t operand = 1; operand <= 2; ++operand) {
                auto& cache = caches_[cache_of(index, operand)];
                cache.capacity = std::max(cache.capacity, extent);
            }
        }
        if (!attention_extent) throw unsupported("the step contains no attention");
        // Prefill's final producer layer writes a cache row but deliberately skips attention and has no consumer.
        for (auto& cache : caches_)
            if (!cache.capacity) cache.capacity = attention_extent;
        // Finalizing hundreds of megabytes of unpacked low-bit containers in one graph can spend minutes in
        // libQnnHtpPrepare or exhaust its memory. Smaller graphs still keep all arithmetic on the HTP and make
        // compilation observable and recoverable.
        const auto budget = environment_size("KIDI_QNN_PARTITION_MB", 256) * 1024 * 1024;
        std::size_t part = 0, bytes = 0;
        for (std::size_t index = 0; index < nodes.size(); ++index) {
            if (is_prologue_[index]) continue;
            const auto& node = nodes[index];
            std::size_t weight = 0;
            if (node.operation == Operation::PACKED_LINEAR)
                weight = static_cast<std::size_t>(node.inputs()[1].size(0)) * node.inputs()[0].size(-1);
            else if (node.operation == Operation::GATED_FEED_FORWARD)
                weight =
                    static_cast<std::size_t>(node.inputs()[1].size(0)) * static_cast<std::size_t>(node.attributes[1]) +
                    static_cast<std::size_t>(node.inputs()[3].size(0)) * static_cast<std::size_t>(node.attributes[2]);
            else if (node.operation == Operation::LINEAR)
                weight = node.inputs()[1].numel() * sizeof(Half);
            else if (node.operation == Operation::EMBEDDING)
                weight = node.inputs()[1].nbytes();
            if (weight && bytes && bytes + weight > budget) {
                ++part;
                bytes = 0;
            }
            bytes += weight;
            partition_[index] = part;
        }
        const auto consume = [&](std::size_t consumer, std::size_t producer, std::size_t output) {
            if (!is_prologue_[producer] && partition_[producer] < partition_[consumer])
                exported_.try_emplace({producer, output}, exported_.size());
        };
        for (std::size_t index = 0; index < nodes.size(); ++index) {
            if (is_prologue_[index]) continue;
            const auto& node = nodes[index];
            for (std::size_t operand = 0; operand < node.input_count; ++operand) {
                const auto& source = node.sources[operand];
                if (source.kind == Source::Kind::NODE) consume(index, source.index, source.output);
            }
            if (node.operation == Operation::ATTENTION)
                for (std::size_t operand = 1; operand <= 2; ++operand)
                    consume(index, caches_[cache_of(index, operand)].row_node, 0);
        }
        const auto outputs = graph_.outputs();
        if (prefill_) {
            if (!outputs.empty()) throw unsupported("prefill steps must not return tensors");
        } else if (outputs.size() != 1 || outputs[0].dtype() != DType::I32) {
            throw unsupported("decode steps must return a greedy token");
        }
    }

    /// The KV cache an attention operand reads: a cache step input, possibly through its in-place scatter.
    auto cache_of(std::size_t index, std::size_t operand) -> std::size_t {
        const auto nodes = graph_.nodes();
        auto source = nodes[index].sources[operand];
        if (source.kind == Source::Kind::NODE && nodes[source.index].operation == Operation::SCATTER)
            source = nodes[source.index].sources[0];
        if (source.kind != Source::Kind::INPUT || !cache_of_slot_.contains(source.index))
            throw unsupported("attention must read a KV cache step input");
        return cache_of_slot_.at(source.index);
    }

    auto begin(std::size_t part, std::size_t first_node) -> void {
        auto& graph = graphs_.emplace_back();
        graph.name = "step" + std::to_string(part);
        graph.first_node = first_node;
        graph.last_node = first_node;
        graph.handle = context_->create_graph(graph.name, true);
        builder_ = std::make_unique<OpBuilder>(*context_, graph.handle, graph.name + "/");
        values_.clear();
        inputs_.clear();
    }

    auto finish() -> void {
        if (!builder_) return;
        const auto started = std::chrono::steady_clock::now();
        if (const char* dump = std::getenv("KIDI_QNN_DUMP"); dump && *dump)
            std::cerr << "kidi_qnn_finalize|graph=" << graphs_.back().name << "|nodes=" << graphs_.back().first_node
                      << '-' << graphs_.back().last_node << "|state=begin\n";
        const auto failures = graph_memory_failures.load(std::memory_order_relaxed);
        check(runtime_->api().graphFinalize(graphs_.back().handle, nullptr, nullptr),
              "finalize QNN graph " + graphs_.back().name);
        if (graph_memory_failures.load(std::memory_order_relaxed) != failures)
            throw ops::Failure(
                {ErrorCode::RUNTIME, "finalize QNN graph " + graphs_.back().name + " exhausted DSP mapped memory"});
        if (const char* dump = std::getenv("KIDI_QNN_DUMP"); dump && *dump)
            std::cerr << "kidi_qnn_finalize|graph=" << graphs_.back().name << "|nodes=" << graphs_.back().first_node
                      << '-' << graphs_.back().last_node << "|state=end|ms="
                      << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count()
                      << '\n';
        builder_.reset();
    }

    auto add_port(bool input, Port::Kind kind, std::size_t index, const Value& value) -> void {
        Port port;
        port.kind = kind;
        port.index = index;
        port.tensor = value.tensor;
        port.dims = value.dims;
        port.tensor.v1.dimensions = nullptr; // rebound below to the port's own storage
        const auto element = value.type() == QNN_DATATYPE_FLOAT_16                                         ? 2U
                             : value.type() == QNN_DATATYPE_INT_32 || value.type() == QNN_DATATYPE_UINT_32 ? 4U
                             : value.type() == QNN_DATATYPE_UFIXED_POINT_16                                ? 2U
                                                                                                           : 1U;
        if (kind != Port::Kind::CACHE) port.buffer.assign(element_count(value.dims) * element, std::byte{});
        auto& ports = input ? graphs_.back().inputs : graphs_.back().outputs;
        ports.push_back(std::move(port));
        ports.back().tensor.v1.dimensions = ports.back().dims.data();
        ports.back().tensor.v1.name = "port"; // graph-owned names are gone after finalization; ids identify ports
    }

    /// A graph input fed by the host, created once per graph and key.
    auto input(Port::Kind kind, std::size_t index, Qnn_DataType_t type, std::vector<std::uint32_t> dims,
               Qnn_QuantizeParams_t quantize = QNN_QUANTIZE_PARAMS_INIT) -> Value {
        const auto key = std::pair{static_cast<int>(kind), index};
        if (const auto found = inputs_.find(key); found != inputs_.end()) return found->second;
        auto value = builder_->tensor(QNN_TENSOR_TYPE_APP_WRITE, type, std::move(dims), quantize);
        add_port(true, kind, index, value);
        return inputs_.emplace(key, value).first->second;
    }

    /// The value of a node output in the current graph: produced here, imported, or fed by the host prologue.
    auto node_value(std::size_t node, std::size_t output) -> Value {
        if (const auto found = values_.find({node, output}); found != values_.end()) return found->second;
        const auto& tensor = graph_.nodes()[node].outputs()[output];
        if (is_prologue_[node])
            return values_[{node, output}] =
                       input(Port::Kind::PROLOGUE, node, QNN_DATATYPE_FLOAT_16, dims_of(tensor.shape()));
        const auto exported = exported_.find({node, output});
        if (exported == exported_.end()) throw unsupported("value used before it is produced");
        const auto& produced = export_types_.at(exported->second);
        return values_[{node, output}] = input(Port::Kind::IMPORT, exported->second, produced.first,
                                               produced.second.first, produced.second.second);
    }

    /// An operand of `index`, viewed with its recorded shape and offset.
    auto operand(std::size_t index, std::size_t position) -> Value {
        const auto& node = graph_.nodes()[index];
        const auto& source = node.sources[position];
        const auto& tensor = node.operands[position];
        auto dims = dims_of(tensor.shape());
        Value base;
        std::size_t element = sizeof(float);
        switch (source.kind) {
            case Source::Kind::EXTERNAL: {
                if (tensor.dtype() == DType::F32) return builder_->half(require(tensor.data<float>()), dims);
                if (tensor.dtype() == DType::BF16) {
                    const auto bytes = require(tensor.host_bytes());
                    std::vector<float> values(tensor.numel());
                    for (std::size_t item = 0; item < values.size(); ++item) {
                        std::uint16_t bits = 0;
                        std::memcpy(&bits, bytes.data() + item * 2, 2);
                        values[item] = std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
                    }
                    return builder_->half(values, dims);
                }
                throw unsupported("unsupported constant dtype");
            }
            case Source::Kind::INPUT: {
                const auto slot = source.index;
                if (slot >= 4 && slot < 8)
                    base = input(Port::Kind::STEP_INPUT, slot, QNN_DATATYPE_FLOAT_16,
                                 dims_of(graph_.inputs()[slot].shape()));
                else
                    throw unsupported("step input " + std::to_string(slot) + " read outside attention");
                break;
            }
            case Source::Kind::NODE: {
                base = node_value(source.index, source.output);
                element = tensor::element_size(graph_.nodes()[source.index].outputs()[source.output].dtype());
                break;
            }
        }
        if (source.offset == 0 && element_count(base.dims) == element_count(dims)) return builder_->reshape(base, dims);
        return builder_->window(base, static_cast<std::size_t>(source.offset) / element, dims);
    }

    auto record(std::size_t index, Value value) -> void {
        const auto& node = graph_.nodes()[index];
        const auto shape = dims_of(node.outputs()[0].shape());
        const bool debug_attention = !getenv_string("KIDI_QNN_VERIFY_CAPTURE").empty() &&
                                     graph_.nodes()[index].operation == Operation::ATTENTION;
        if (debug_attention) {
            value = builder_->expose(value, shape);
            add_port(false, Port::Kind::DEBUG, index, value);
        } else {
            value = builder_->reshape(value, shape);
        }
        values_[{index, 0}] = value;
        if (const auto found = exported_.find({index, 0}); found != exported_.end()) {
            const auto exposed = builder_->expose(value);
            export_types_[found->second] = {exposed.type(), {exposed.dims, exposed.tensor.v1.quantizeParams}};
            add_port(false, Port::Kind::EXPORT, found->second, exposed);
        }
    }

    auto calibrated_projection(const Node& node, const Value& input) -> Value {
        const auto width = static_cast<std::uint32_t>(input.dims.back());
        const auto rows = static_cast<std::uint32_t>(element_count(input.dims) / width);
        const auto outputs = static_cast<std::uint32_t>(node.inputs()[1].size(0));
        const auto bits = static_cast<int>(node.attributes[0]);
        const auto flat = builder_->reshape(input, {rows, width});
        auto values = unpack_packed_weight(node.inputs()[1], bits, width);
        auto weight =
            builder_->weights(make_weights(values, require(node.inputs()[2].data<float>()), bits), {outputs, width});
        const auto output_scale = scalar_bits(node.attributes[2]);
        if (node.epsilon > 0 && output_scale > 0) {
            auto quantized = builder_->quantize(flat, node.epsilon);
            auto output =
                builder_->native(QNN_DATATYPE_UFIXED_POINT_8, {rows, outputs}, per_tensor(output_scale, -128));
            builder_->op(QNN_OP_FULLY_CONNECTED, {quantized, weight}, {output});
            return builder_->dequantize(output);
        }
        // Output head: CPU quantizes its input dynamically. Dividing by max|x| keeps greedy argmax unchanged.
        const auto peak =
            builder_->binary(QNN_OP_ELEMENT_WISE_MAXIMUM,
                             builder_->reduce(QNN_OP_REDUCE_MAX, builder_->unary(QNN_OP_ELEMENT_WISE_ABS, flat)),
                             builder_->scalar(1e-2F, 2));
        auto quantized = builder_->quantize(builder_->binary(QNN_OP_ELEMENT_WISE_DIVIDE, flat, peak), 1.F / 127.F);
        // HTP's quantized FullyConnected does not accept FP16 or UINT16 output for this 2-bit vocabulary matrix.
        // Calibrate UINT8 from the second CPU capture with 25% headroom. The positive input normalization changes
        // every logit by the same factor, and ArgMax consumes this tensor directly without dequantizing it.
        const auto captured = require(node.outputs()[0].data<float>());
        float maximum = 0;
        for (const auto value : captured) maximum = std::max(maximum, std::abs(value));
        const auto head_scale = std::max(maximum * 1.25F / 127.F, 1e-4F);
        if (const char* dump = std::getenv("KIDI_QNN_DUMP"); dump && *dump)
            std::cerr << "kidi_qnn_head|max_abs=" << maximum << "|scale=" << head_scale << '\n';
        auto output = builder_->native(QNN_DATATYPE_UFIXED_POINT_8, {rows, outputs}, per_tensor(head_scale, -128));
        builder_->op(QNN_OP_FULLY_CONNECTED, {quantized, weight}, {output});
        return output;
    }

    /// One custom op for RMS_NORM and RMS_NORM_RESIDUAL (x, epsilon, weight[, residual[, scale]]).
    auto fused_rms_norm(std::size_t index) -> std::optional<Value> {
        const auto& node = graph_.nodes()[index];
        const auto input = operand(index, 0);
        const auto width = input.dims.back();
        if (!runtime_->fused_ops() || width % 64 != 0) return std::nullopt;
        const auto rows = static_cast<std::uint32_t>(element_count(input.dims) / width);
        const bool residual = node.operation == Operation::RMS_NORM_RESIDUAL;
        auto x = builder_->reshape(input, {1, 1, rows, width});
        const std::array parameters{node.epsilon, residual ? 1.0F : 0.0F};
        std::vector<Value> inputs{x, builder_->floats(parameters, {1, 1, 1, 2}),
                                  builder_->reshape(operand(index, 1), {1, 1, 1, width}),
                                  residual ? builder_->reshape(operand(index, 2), {1, 1, rows, width}) : x,
                                  residual && node.input_count == 4 ? builder_->reshape(operand(index, 3), {1, 1, 1, 1})
                                                                    : builder_->scalar(1.0F, 4)};
        auto output = builder_->native(QNN_DATATYPE_FLOAT_16, {1, 1, rows, width});
        builder_->op("RmsNorm", std::move(inputs), {output}, {}, CUSTOM_PACKAGE);
        return builder_->reshape(output, input.dims);
    }

    auto gated_feed_forward(const Node& node, const Value& input) -> Value {
        if (node.attributes.size() != 6 || node.inputs().size() != 5)
            throw unsupported("invalid fused gated feed-forward");
        const auto bits = static_cast<int>(node.attributes[0]);
        const auto input_width = static_cast<std::uint32_t>(node.attributes[1]);
        const auto intermediate = static_cast<std::uint32_t>(node.attributes[2]);
        const auto gate_scale = scalar_bits(node.attributes[3]);
        const auto hidden_scale = scalar_bits(node.attributes[4]);
        const auto output_scale = scalar_bits(node.attributes[5]);
        const auto rows = static_cast<std::uint32_t>(element_count(input.dims) / input_width);
        auto flat = builder_->reshape(input, {rows, input_width});
        auto quantized = builder_->quantize(flat, node.epsilon);
        auto gate_values = unpack_packed_weight(node.inputs()[1], bits, input_width);
        auto gate_weight = builder_->weights(make_weights(gate_values, require(node.inputs()[2].data<float>()), bits),
                                             {2 * intermediate, input_width});
        auto projected =
            builder_->native(QNN_DATATYPE_UFIXED_POINT_8, {rows, 2 * intermediate}, per_tensor(gate_scale, -128));
        builder_->op(QNN_OP_FULLY_CONNECTED, {quantized, gate_weight}, {projected});
        Value hidden;
        if (runtime_->fused_ops() && intermediate % 128 == 0) {
            // One HVX pass: gate/up codes -> GELU(gate) * up -> down-projection codes.
            auto fused = builder_->native(QNN_DATATYPE_UFIXED_POINT_8, {1, 1, rows, intermediate},
                                          per_tensor(hidden_scale, -128));
            builder_->op("GeluMultiply", {builder_->reshape(projected, {1, 1, rows, 2 * intermediate})}, {fused}, {},
                         CUSTOM_PACKAGE);
            hidden = builder_->reshape(fused, {rows, intermediate});
        } else {
            auto real = builder_->dequantize(projected);
            auto gate = builder_->slice(real, 1, 0, intermediate);
            auto up = builder_->slice(real, 1, intermediate, intermediate);
            hidden = builder_->quantize(builder_->gelu_multiply(gate, up), hidden_scale);
        }
        auto down_values = unpack_packed_weight(node.inputs()[3], bits, intermediate);
        auto down_weight = builder_->weights(make_weights(down_values, require(node.inputs()[4].data<float>()), bits),
                                             {static_cast<std::uint32_t>(node.inputs()[3].size(0)), intermediate});
        auto output =
            builder_->native(QNN_DATATYPE_UFIXED_POINT_8, {rows, static_cast<std::uint32_t>(node.inputs()[3].size(0))},
                             per_tensor(output_scale, -128));
        builder_->op(QNN_OP_FULLY_CONNECTED, {hidden, down_weight}, {output});
        return builder_->dequantize(output);
    }

    auto resident_constant(const Tensor& source, std::size_t source_node, Qnn_DataType_t type,
                           const std::vector<std::uint32_t>& dims) -> NpuConstant {
        const auto identity = source.storage_identity();
        NpuConstant entry;
        entry.source_node = source_node;
        entry.dims = dims;
        entry.type = type;
        const auto bytes = require(source.host_bytes());
        {
            const std::scoped_lock lock(constant_mutex_);
            const auto found = constant_blocks_.find(identity);
            if (found != constant_blocks_.end()) {
                entry.memory = found->second;
            } else {
                entry.memory = memory_->allocate(bytes.size());
                std::memcpy(entry.memory.data, bytes.data(), bytes.size());
                constant_blocks_.emplace(identity, entry.memory);
            }
        }
        entry.handle = context_->register_memory(entry.memory.fd, entry.dims, entry.type);
        return entry;
    }

    auto constant(const Tensor& source, std::size_t source_node, Qnn_DataType_t type,
                  std::vector<std::uint32_t> dims) -> Value {
        const auto identity = source.storage_identity();
        std::size_t index = 0;
        if (const auto found = constant_of_.find(identity); found != constant_of_.end()) {
            index = found->second;
        } else {
            index = constants_.size();
            constants_.push_back(resident_constant(source, source_node, type, dims));
            constant_of_[identity] = index;
        }
        return input(Port::Kind::CONSTANT, index, type, std::move(dims));
    }

    /// Gathers a packed 2/4/8-bit embedding row and reconstructs it on the HTP.
    auto embedding(std::size_t index) -> Value {
        const auto& node = graph_.nodes()[index];
        if (node.attributes.size() != 3 || node.inputs().size() != 3)
            throw unsupported("only packed embeddings are supported");
        const auto width = static_cast<std::uint32_t>(node.attributes[0]);
        const auto bits = static_cast<int>(node.attributes[1]);
        const auto groups = static_cast<std::uint32_t>(node.attributes[2]);
        if ((bits != 2 && bits != 4 && bits != 8) || !groups || width % groups)
            throw unsupported("invalid packed embedding attributes");
        const auto per_byte = static_cast<std::uint32_t>(8 / bits);
        const auto rows = static_cast<std::uint32_t>(node.inputs()[1].size(0));
        const auto stored = width / per_byte;
        const auto length = static_cast<std::uint32_t>(node.inputs()[0].numel());
        auto indices = input(Port::Kind::INDICES, 0, QNN_DATATYPE_INT_32, {length});
        auto table = constant(node.inputs()[1], index, QNN_DATATYPE_UINT_8, {rows, stored});
        auto packed = builder_->cast(builder_->gather(table, indices), QNN_DATATYPE_FLOAT_16);
        Value expanded;
        const auto radix = static_cast<float>(1 << bits);
        const auto sign = static_cast<float>(1 << (bits - 1));
        for (std::uint32_t part = 0; part < per_byte; ++part) {
            const auto divisor = static_cast<float>(1U << (part * bits));
            auto digit = builder_->unary(QNN_OP_ELEMENT_WISE_FLOOR, builder_->binary(QNN_OP_ELEMENT_WISE_DIVIDE, packed,
                                                                                     builder_->scalar(divisor, 2)));
            auto quotient =
                builder_->unary(QNN_OP_ELEMENT_WISE_FLOOR,
                                builder_->binary(QNN_OP_ELEMENT_WISE_DIVIDE, digit, builder_->scalar(radix, 2)));
            digit =
                builder_->binary(QNN_OP_ELEMENT_WISE_SUBTRACT, digit,
                                 builder_->binary(QNN_OP_ELEMENT_WISE_MULTIPLY, quotient, builder_->scalar(radix, 2)));
            auto condition = builder_->native(QNN_DATATYPE_BOOL_8, digit.dims);
            builder_->op(QNN_OP_ELEMENT_WISE_GREATER_EQUAL, {digit, builder_->scalar(sign, 2)}, {condition});
            auto signed_digit = builder_->select(
                condition, builder_->binary(QNN_OP_ELEMENT_WISE_SUBTRACT, digit, builder_->scalar(radix, 2)), digit);
            signed_digit = builder_->reshape(signed_digit, {length, stored, 1});
            expanded = part ? builder_->concat(expanded, signed_digit, 2) : signed_digit;
        }
        expanded = builder_->reshape(expanded, {length, width});
        const auto scales = require(node.inputs()[2].data<float>());
        auto scale_table = builder_->half(scales, {rows, groups});
        auto row_scales = builder_->gather(scale_table, indices);
        if (groups > 1)
            row_scales = builder_->reshape(builder_->tile(builder_->reshape(row_scales, {length, groups, 1}),
                                                          {1, 1, static_cast<std::int32_t>(width / groups)}),
                                           {length, width});
        auto output = builder_->binary(QNN_OP_ELEMENT_WISE_MULTIPLY, expanded, row_scales);
        return builder_->binary(QNN_OP_ELEMENT_WISE_MULTIPLY, output, builder_->scalar(node.epsilon, 2));
    }

    auto attention(std::size_t index) -> Value {
        const auto& node = graph_.nodes()[index];
        const auto heads = static_cast<std::uint32_t>(node.attributes[0]);
        const auto key_heads = static_cast<std::uint32_t>(node.attributes[1]);
        if (node.attributes.size() != 4 || node.attributes[2] != 0 || key_heads != 1)
            throw unsupported("structured attention requires one key head and no key offset");
        const auto key_cache = cache_of(index, 1), value_cache = cache_of(index, 2);
        const auto& keys = caches_[key_cache];
        if (static_cast<std::size_t>(node.attributes[3]) != keys.capacity)
            throw unsupported("attention must span the whole cache");
        const auto width = static_cast<std::uint32_t>(keys.width);
        const auto capacity = static_cast<std::uint32_t>(keys.capacity);
        const auto query_rows = static_cast<std::uint32_t>(node.inputs()[0].size(1));
        const auto cached = [&](std::size_t cache) {
            const auto& entry = caches_[cache];
            return builder_->dequantize(input(Port::Kind::CACHE, cache, QNN_DATATYPE_UFIXED_POINT_8, entry.dims,
                                              per_tensor(entry.scale, -128)));
        };
        const auto fresh = [&](std::size_t cache) {
            return builder_->reshape(node_value(caches_[cache].row_node, 0), {query_rows, key_heads * width});
        };
        const auto mask_slot = node.sources[3].index;
        if (node.sources[3].kind != Source::Kind::INPUT) throw unsupported("attention masks must be step inputs");
        if (runtime_->structured_attention()) {
            auto output = builder_->native(QNN_DATATYPE_FLOAT_16, {1, query_rows, heads, width});
            auto positions = input(Port::Kind::POSITION, 1, QNN_DATATYPE_INT_32, {1, 1, 1, query_rows});
            auto key_memory =
                builder_->reshape(input(Port::Kind::CACHE, key_cache, QNN_DATATYPE_UFIXED_POINT_8,
                                        caches_[key_cache].dims, per_tensor(caches_[key_cache].scale, -128)),
                                  {1, capacity, 1, width});
            auto value_memory =
                builder_->reshape(input(Port::Kind::CACHE, value_cache, QNN_DATATYPE_UFIXED_POINT_8,
                                        caches_[value_cache].dims, per_tensor(caches_[value_cache].scale, -128)),
                                  {1, capacity, 1, width});
            builder_->op("StructuredAttention",
                         {builder_->reshape(operand(index, 0), {1, query_rows, heads, width}),
                          builder_->reshape(fresh(key_cache), {1, query_rows, 1, width}),
                          builder_->reshape(fresh(value_cache), {1, query_rows, 1, width}), key_memory, value_memory,
                          positions, builder_->float32(node.epsilon, {1, 1, 1, 1}),
                          builder_->int32(mask_slot == 2 ? 512 : 0, {1, 1, 1, 1})},
                         {output}, {}, CUSTOM_PACKAGE);
            return builder_->reshape(output, {1, query_rows, heads * width});
        }
        auto query = builder_->transpose(builder_->reshape(operand(index, 0), {query_rows, heads, width}), {1, 0, 2});
        query = builder_->binary(QNN_OP_ELEMENT_WISE_MULTIPLY, query, builder_->scalar(node.epsilon, 3));
        if (query_rows > 1 && structured_tiles_enabled()) {
            constexpr std::uint32_t TILE = 64;
            constexpr std::uint32_t WINDOW = 512;
            const auto past = capacity - query_rows;
            auto old_keys = cached(key_cache);
            auto old_values = cached(value_cache);
            auto new_keys = fresh(key_cache);
            auto new_values = fresh(value_cache);
            if (past) {
                old_keys = builder_->slice(old_keys, 0, 0, past);
                old_values = builder_->slice(old_values, 0, 0, past);
                new_keys = builder_->concat(old_keys, new_keys, 0);
                new_values = builder_->concat(old_values, new_values, 0);
            }
            const auto full_mask =
                input(Port::Kind::STRUCTURED_MASK, mask_slot, QNN_DATATYPE_FLOAT_16, {query_rows, capacity});
            const bool local = mask_slot == 2;
            Value combined;
            for (std::uint32_t query_start = 0; query_start < query_rows; query_start += TILE) {
                const auto rows = std::min(TILE, query_rows - query_start);
                const auto key_end = past + query_start + rows;
                const auto earliest_end = past + query_start + 1;
                const auto key_start = local && earliest_end > WINDOW ? earliest_end - WINDOW : 0;
                const auto keys = key_end - key_start;
                auto query_tile = builder_->slice(query, 1, query_start, rows);
                auto key_tile = builder_->slice(new_keys, 0, key_start, keys);
                auto value_tile = builder_->slice(new_values, 0, key_start, keys);
                auto mask_tile = builder_->slice(builder_->slice(full_mask, 0, query_start, rows), 1, key_start, keys);
                auto scores =
                    builder_->binary(QNN_OP_ELEMENT_WISE_ADD, builder_->matmul(query_tile, key_tile, true), mask_tile);
                auto probabilities = builder_->native(QNN_DATATYPE_FLOAT_16, {heads, rows, keys});
                builder_->op(QNN_OP_SOFTMAX, {scores}, {probabilities},
                             {OpBuilder::unsigned32(QNN_OP_SOFTMAX_PARAM_AXIS, 2),
                              OpBuilder::float32(QNN_OP_SOFTMAX_PARAM_BETA, 1.F)});
                auto output = builder_->matmul(probabilities, value_tile, false);
                combined = query_start ? builder_->concat(combined, output, 1) : output;
            }
            return builder_->reshape(builder_->transpose(combined, {1, 0, 2}), {1, query_rows, heads * width});
        }
        const auto mask = input(Port::Kind::MASK, mask_slot, QNN_DATATYPE_FLOAT_16, {query_rows, capacity});
        const auto fresh_mask =
            input(Port::Kind::FRESH_MASK, mask_slot, QNN_DATATYPE_FLOAT_16, {query_rows, query_rows});
        auto old_scores =
            builder_->binary(QNN_OP_ELEMENT_WISE_ADD, builder_->matmul(query, cached(key_cache), true), mask);
        auto new_scores =
            builder_->binary(QNN_OP_ELEMENT_WISE_ADD, builder_->matmul(query, fresh(key_cache), true), fresh_mask);
        const auto axis = std::size_t{2};
        auto probabilities = builder_->native(QNN_DATATYPE_FLOAT_16, {heads, query_rows, capacity + query_rows});
        builder_->op(QNN_OP_SOFTMAX, {builder_->concat(old_scores, new_scores, axis)}, {probabilities},
                     {OpBuilder::unsigned32(QNN_OP_SOFTMAX_PARAM_AXIS, static_cast<std::uint32_t>(axis)),
                      OpBuilder::float32(QNN_OP_SOFTMAX_PARAM_BETA, 1.F)});
        const auto old_weights = builder_->slice(probabilities, axis, 0, capacity);
        const auto new_weights = builder_->slice(probabilities, axis, capacity, query_rows);
        auto output =
            builder_->binary(QNN_OP_ELEMENT_WISE_ADD, builder_->matmul(old_weights, cached(value_cache), false),
                             builder_->matmul(new_weights, fresh(value_cache), false));
        return builder_->reshape(builder_->transpose(output, {1, 0, 2}), {1, query_rows, heads * width});
    }

    auto lower_node(std::size_t index) -> void {
        const auto& node = graph_.nodes()[index];
        switch (node.operation) {
            case Operation::EMBEDDING:
                return record(index, embedding(index));
            case Operation::CAST: {
                const auto input = operand(index, 0);
                const auto row = row_cache_.find(index);
                if (row == row_cache_.end()) return record(index, input);
                const auto width = static_cast<std::uint32_t>(caches_[row->second].width);
                const auto rows =
                    static_cast<std::uint32_t>(input.dims.size() > 1 ? element_count(input.dims) / width : 1);
                auto quantized =
                    builder_->quantize(builder_->reshape(input, {rows, width}), node.epsilon, QNN_TENSOR_TYPE_APP_READ);
                add_port(false, Port::Kind::ROW, row->second, quantized);
                return record(index, builder_->dequantize(quantized));
            }
            case Operation::LINEAR: {
                const auto input = operand(index, 0);
                const auto& weight = node.inputs()[1];
                const bool transposed = !node.attributes.empty() && node.attributes[0];
                const auto outputs = static_cast<std::uint32_t>(transposed ? weight.size(0) : weight.size(1));
                const auto width = static_cast<std::uint32_t>(input.dims.back());
                const auto rows = static_cast<std::uint32_t>(element_count(input.dims) / width);
                std::vector<float> values(weight.numel());
                const auto bytes = require(weight.host_bytes());
                for (std::size_t item = 0; item < values.size(); ++item) {
                    if (weight.dtype() == DType::F32) {
                        std::memcpy(&values[item], bytes.data() + item * 4, 4);
                    } else {
                        std::uint16_t bits = 0;
                        std::memcpy(&bits, bytes.data() + item * 2, 2);
                        values[item] = std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
                    }
                }
                if (!transposed) {
                    std::vector<float> swapped(values.size());
                    for (std::size_t row = 0; row < width; ++row)
                        for (std::size_t column = 0; column < outputs; ++column)
                            swapped[column * width + row] = values[row * outputs + column];
                    values = std::move(swapped);
                }
                auto output = builder_->native(QNN_DATATYPE_FLOAT_16, {rows, outputs});
                builder_->op(QNN_OP_FULLY_CONNECTED,
                             {builder_->reshape(input, {rows, width}), builder_->half(values, {outputs, width})},
                             {output});
                return record(index, output);
            }
            case Operation::MULTIPLY:
                return record(index,
                              builder_->binary(QNN_OP_ELEMENT_WISE_MULTIPLY, operand(index, 0), operand(index, 1)));
            case Operation::ADD:
                return record(index, builder_->binary(QNN_OP_ELEMENT_WISE_ADD, operand(index, 0), operand(index, 1)));
            case Operation::RMS_NORM:
                if (auto fused = fused_rms_norm(index)) return record(index, *fused);
                return record(index, builder_->rms_norm(operand(index, 0), operand(index, 1), node.epsilon));
            case Operation::RMS_NORM_RESIDUAL: {
                if (auto fused = fused_rms_norm(index)) return record(index, *fused);
                auto output = builder_->binary(QNN_OP_ELEMENT_WISE_ADD, operand(index, 2),
                                               builder_->rms_norm(operand(index, 0), operand(index, 1), node.epsilon));
                if (node.input_count == 4)
                    output = builder_->binary(QNN_OP_ELEMENT_WISE_MULTIPLY, output, operand(index, 3));
                return record(index, output);
            }
            case Operation::PACKED_LINEAR:
                return record(index, calibrated_projection(node, operand(index, 0)));
            case Operation::GATED_FEED_FORWARD:
                return record(index, gated_feed_forward(node, operand(index, 0)));
            case Operation::ROTARY: {
                const auto input = operand(index, 0);
                const auto cosine = operand(index, 1), sine = operand(index, 2);
                const std::size_t axis = input.dims.size() - 1;
                const std::uint32_t half = input.dims.back() / 2;
                const auto first = builder_->slice(input, axis, 0, half),
                           second = builder_->slice(input, axis, half, half);
                const auto left = builder_->binary(QNN_OP_ELEMENT_WISE_SUBTRACT,
                                                   builder_->binary(QNN_OP_ELEMENT_WISE_MULTIPLY, first, cosine),
                                                   builder_->binary(QNN_OP_ELEMENT_WISE_MULTIPLY, second, sine));
                const auto right = builder_->binary(QNN_OP_ELEMENT_WISE_ADD,
                                                    builder_->binary(QNN_OP_ELEMENT_WISE_MULTIPLY, second, cosine),
                                                    builder_->binary(QNN_OP_ELEMENT_WISE_MULTIPLY, first, sine));
                return record(index, builder_->concat(left, right, axis));
            }
            case Operation::STATIC_ROUND: {
                const auto input = operand(index, 0);
                const auto row = row_cache_.find(index);
                if (row == row_cache_.end())
                    return record(index, builder_->dequantize(builder_->quantize(input, node.epsilon)));
                const auto width = static_cast<std::uint32_t>(caches_[row->second].width);
                const auto rows =
                    static_cast<std::uint32_t>(input.dims.size() > 1 ? element_count(input.dims) / width : 1);
                auto quantized =
                    builder_->quantize(builder_->reshape(input, {rows, width}), node.epsilon, QNN_TENSOR_TYPE_APP_READ);
                add_port(false, Port::Kind::ROW, row->second, quantized);
                return record(index, builder_->dequantize(quantized));
            }
            case Operation::SCATTER:
                return; // the host writes the row after the step; attention scores it separately meanwhile
            case Operation::ATTENTION:
                return record(index, attention(index));
            case Operation::SLICE: {
                const auto input = operand(index, 0);
                auto axis = node.attributes[0];
                if (axis < 0) axis += static_cast<std::int64_t>(input.dims.size());
                return record(index, builder_->slice(input, static_cast<std::size_t>(axis),
                                                     static_cast<std::uint32_t>(node.attributes[1]),
                                                     static_cast<std::uint32_t>(node.attributes[2])));
            }
            case Operation::GELU_MULTIPLY:
                return record(index, builder_->gelu_multiply(operand(index, 0), operand(index, 1)));
            case Operation::GREEDY_TOKEN: {
                auto logits = operand(index, 0);
                const auto vocabulary = logits.dims.back();
                logits = builder_->reshape(logits, {1, vocabulary});
                auto token = builder_->tensor(QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_INT_32, {1});
                builder_->op(QNN_OP_ARGMAX, {logits}, {token},
                             {OpBuilder::unsigned32(QNN_OP_ARGMAX_PARAM_AXIS, 1),
                              OpBuilder::boolean(QNN_OP_ARGMAX_PARAM_KEEP_DIMS, false)});
                add_port(false, Port::Kind::TOKEN, 0, token);
                values_[{index, 0}] = token;
                return;
            }
            default:
                throw unsupported(std::string("operation ") + std::string(operation_name(node.operation)));
        }
    }

    std::shared_ptr<QnnRuntime> runtime_;
    const Graph& graph_;
    bool prefill_;
    std::mutex& constant_mutex_;
    std::map<const void*, SharedMemory::Block>& constant_blocks_;
    std::unique_ptr<QnnContext> context_;
    std::unique_ptr<SharedMemory> memory_;
    std::unique_ptr<OpBuilder> builder_;
    std::vector<bool> is_prologue_;
    std::vector<std::size_t> partition_, prologue_;
    std::vector<NpuCache> caches_;
    std::vector<NpuConstant> constants_;
    std::map<std::size_t, std::size_t> cache_of_slot_, row_cache_;
    std::map<const void*, std::size_t> constant_of_;
    std::map<Key, std::size_t> exported_;
    std::map<std::size_t, std::pair<Qnn_DataType_t, std::pair<std::vector<std::uint32_t>, Qnn_QuantizeParams_t>>>
        export_types_;
    std::map<Key, Value> values_;
    std::map<std::pair<int, std::size_t>, Value> inputs_;
    std::vector<LoweredGraph> graphs_;
    std::size_t exported_count_ = 0;
};

class QnnStepCompiler final : public StepCompiler {
public:
    explicit QnnStepCompiler(std::shared_ptr<QnnRuntime> runtime) : runtime_(std::move(runtime)) {}

    auto name() const -> std::string_view override { return "qnn-htp"; }

    // Whole-step graphs use power-of-two cache prefixes. This keeps early decode from reading a 9K-token cache while
    // bounding the number of compiled shapes. A new bucket synchronizes the existing CPU cache prefix once.
    auto key_extent(std::size_t required, std::size_t capacity) const -> std::size_t override {
        if (!whole_step()) return std::min(capacity, (required + 127) / 128 * 128);
        if (runtime_->structured_attention()) return capacity;
        if (structured_tiles_enabled()) {
            constexpr std::size_t TILE = 64;
            return std::min(capacity, std::max(TILE, (required + TILE - 1) / TILE * TILE));
        }
        return std::min(capacity, std::bit_ceil(std::max(required, std::size_t{512})));
    }

    auto crop_local_attention() const -> bool override { return !whole_step(); }

    static auto whole_step() -> bool { return getenv_string("KIDI_QNN_WHOLE") != "0"; }

    auto compiles_in_background() const -> bool override { return true; }
    auto compiles_in_background(const Graph& graph, std::string_view key) const -> bool override {
        if (!whole_step()) return true;
        const auto id = whole_fingerprint(graph, key, *runtime_);
        const auto directory = cache_directory();
        return !std::filesystem::exists(directory / (id + ".whole.bin")) ||
               !std::filesystem::exists(directory / (id + ".whole.json"));
    }

    auto captures_prefill() const -> bool override { return whole_step(); }
    auto prefill_chunk_size(std::size_t requested) const -> std::size_t override {
        return whole_step() && requested ? std::max(requested, std::size_t{128}) : requested;
    }

    auto prepare_operator(const OperatorSpec& spec, TensorInputs inputs) -> std::unique_ptr<Operator> override {
        // Whole-step mode captures the CPU reference twice, then replaces every operator with one NPU executable.
        if (whole_step()) return nullptr;
        if ((spec.operation != Operation::GATED_FEED_FORWARD && spec.operation != Operation::PACKED_LINEAR) ||
            inputs.size() < 3 || inputs[0].dtype() != DType::F32 || inputs[0].dimensions() == 0)
            return nullptr;
        const auto rows = inputs[0].numel() / inputs[0].size(-1);
        if (rows < environment_size("KIDI_QNN_MIN_ROWS", 16)) return nullptr;
        try {
            Node node{};
            node.operation = spec.operation;
            node.dtype = spec.dtype;
            node.epsilon = spec.epsilon;
            node.attributes.assign(spec.attributes.begin(), spec.attributes.end());
            for (std::size_t index = 0; index < inputs.size(); ++index) node.operands.push_back(inputs[index]);
            node.input_count = inputs.size();
            std::vector<std::int64_t> shape(inputs[0].shape().begin(), inputs[0].shape().end());
            shape.back() = spec.operation == Operation::GATED_FEED_FORWARD && inputs.size() == 5 ? inputs[3].size(0)
                                                                                                 : inputs[1].size(0);
            node.operands.push_back(require(Tensor::empty(shape, DType::F32)));
            const auto plan = plan_operator(node);
            if (!plan) return nullptr;
            const auto block = environment_size("KIDI_QNN_BLOCK_ROWS", 128);
            // Weights are hashed once per distinct operator; the result keys both the disk cache and shared graphs.
            Hash hash;
            hash.text(CACHE_VERSION);
            hash.text("operator");
            hash.text(ffn_variant());
            hash.text(runtime_->build_id());
            hash.pod(static_cast<int>(plan->kind));
            hash.pod(block);
            hash.pod(plan->bits);
            hash.pod(plan->input_width);
            hash.pod(plan->intermediate);
            hash.pod(plan->output_width);
            hash.pod(plan->input_scale);
            hash.pod(plan->gate_output_scale);
            hash.pod(plan->down_input_scale);
            hash.pod(plan->output_scale);
            for (std::size_t index = 1; index < inputs.size(); ++index) hash_words(hash, inputs[index]);
            const auto id = hex(hash.value);
            std::shared_future<std::shared_ptr<CompiledOperator>> compiled;
            {
                const std::scoped_lock lock(operators_mutex_);
                auto found = operators_.find(id);
                if (found == operators_.end()) {
                    if (offloaded_ >=
                        environment_size("KIDI_QNN_MAX_OPERATORS", std::numeric_limits<std::size_t>::max()))
                        return nullptr;
                    ++offloaded_;
                    auto promise = std::make_shared<std::promise<std::shared_ptr<CompiledOperator>>>();
                    found = operators_.emplace(id, promise->get_future().share()).first;
                    auto job = [runtime = runtime_, node, plan = *plan, id, block, promise] {
                        try {
                            promise->set_value(compile_operator(runtime, node, plan, id, block));
                        } catch (...) {
                            promise->set_exception(std::current_exception());
                        }
                    };
                    if (getenv_string("KIDI_BACKGROUND_COMPILE") == "0")
                        job();
                    else
                        queue_.push(std::move(job));
                }
                compiled = found->second;
            }
            if (!reference_backend_) reference_backend_ = cpu_operators();
            return std::make_unique<QnnOperator>(std::move(compiled), reference_backend_->prepare(spec, inputs), *plan,
                                                 std::move(shape), !getenv_string("KIDI_QNN_VERIFY").empty());
        } catch (const std::exception& error) {
            std::cerr << "kidi_qnn_operator|fallback=cpu|error=" << error.what() << '\n';
            return nullptr;
        }
    }

    auto compile(const Graph& graph, std::string_view key) -> std::unique_ptr<StepExecutable> override {
        const bool decode = key.starts_with("gemma4_decode:") && key.ends_with(":token");
        const bool prefill = key.starts_with("gemma4_prefill:");
        if ((!decode && !prefill) || getenv_string("KIDI_QNN_STEPS") == "0") return nullptr;
        if (whole_step()) {
            const auto started = std::chrono::steady_clock::now();
            try {
                const std::scoped_lock compile_lock(whole_compile_mutex_);
                const auto id = whole_fingerprint(graph, key, *runtime_);
                const auto directory = cache_directory();
                const auto binary_path = directory / (id + ".whole.bin");
                const auto metadata_path = directory / (id + ".whole.json");
                if (const char* dump = std::getenv("KIDI_QNN_DUMP"); dump && *dump) {
                    std::ostringstream message;
                    message << "kidi_qnn_cache|key=" << key << "|id=" << id << "|present="
                            << (std::filesystem::exists(binary_path) && std::filesystem::exists(metadata_path));
                    diagnostic_log(message.str());
                }
                const bool diagnostics =
                    !getenv_string("KIDI_QNN_NODE_LIMIT").empty() || !getenv_string("KIDI_QNN_VERIFY_CAPTURE").empty();
                std::unique_ptr<StepExecutable> executable;
                bool loaded = false;
                if (!diagnostics && std::filesystem::exists(binary_path) && std::filesystem::exists(metadata_path)) {
                    executable = StepLowering(runtime_, graph, prefill, constant_mutex_, constant_blocks_)
                                     .load(read_file(binary_path), metadata_path, id);
                    loaded = true;
                } else {
                    if (!runtime_->prepare_available())
                        throw ops::Failure({ErrorCode::UNSUPPORTED,
                                            "QNN HTP prepare library is unavailable and no whole-step cache exists"});
                    if (!diagnostics) std::filesystem::create_directories(directory);
                    executable = StepLowering(runtime_, graph, prefill, constant_mutex_, constant_blocks_)
                                     .lower(diagnostics ? std::filesystem::path{} : binary_path,
                                            diagnostics ? std::filesystem::path{} : metadata_path, id);
                }
                if (!getenv_string("KIDI_QNN_VERIFY_CAPTURE").empty()) {
                    std::vector<Tensor> outputs(graph.outputs().begin(), graph.outputs().end());
                    executable->run(graph.inputs(), outputs);
                }
                if (const char* dump = std::getenv("KIDI_QNN_DUMP"); dump && *dump) {
                    std::ostringstream message;
                    message
                        << "kidi_qnn_whole|key=" << key << "|compile_ms="
                        << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count()
                        << "|cache=" << (loaded ? "hit" : "miss") << "|id=" << id;
                    diagnostic_log(message.str());
                }
                return executable;
            } catch (const std::exception& error) {
                diagnostic_log("kidi_qnn_whole|key=" + std::string(key) + "|fallback=partitions|error=" + error.what());
                if (getenv_string("KIDI_QNN_WHOLE") == "1") throw;
            }
        }
        const auto plans = plan_partitions(graph);
        if (const char* dump = std::getenv("KIDI_QNN_DUMP_VERBOSE"); dump && *dump) {
            std::cerr << "kidi_qnn_graph|key=" << key << "|nodes=" << graph.nodes().size() << '\n';
            for (std::size_t index = 0; index < graph.nodes().size(); ++index) {
                const auto& node = graph.nodes()[index];
                if (std::getenv("KIDI_QNN_DUMP_ALL")) {
                    std::cerr << "kidi_qnn_all|index=" << index << "|op=" << operation_name(node.operation)
                              << "|inputs=" << node.input_count << "|outputs=" << node.outputs().size();
                    if (!node.outputs().empty()) {
                        std::cerr << "|shape=";
                        for (const auto extent : node.outputs()[0].shape()) std::cerr << extent << ',';
                    }
                    std::cerr << '\n';
                }
                if (node.operation != Operation::PACKED_LINEAR) continue;
                std::cerr << "kidi_qnn_node|index=" << index << "|op=packed_linear|input=" << node.inputs()[0].size(-1)
                          << "|output=" << node.outputs()[0].size(-1) << "|weight_bytes=" << node.inputs()[1].nbytes()
                          << "|bits=" << (node.attributes.empty() ? 0 : node.attributes[0])
                          << "|group=" << (node.attributes.size() < 2 ? 0 : node.attributes[1])
                          << "|input_scale=" << node.epsilon
                          << "|output_scale=" << (node.attributes.size() < 3 ? 0.F : scalar_bits(node.attributes[2]))
                          << '\n';
            }
        }
        if (plans.empty()) return nullptr;
        const auto id = fingerprint(graph, plans, key, runtime_->build_id());
        auto context = std::make_unique<QnnContext>(runtime_);
        const auto cache_dir = cache_directory();
        const auto binary_path = cache_dir / (id + ".bin");
        const auto metadata_path = cache_dir / (id + ".meta");
        std::vector<BuiltPartition> built;
        bool loaded = false;
        if (auto metadata = load_metadata(metadata_path, id); metadata && std::filesystem::exists(binary_path)) {
            context->load(read_file(binary_path));
            built = std::move(metadata->partitions);
            loaded = true;
        } else {
            if (!runtime_->prepare_available())
                throw ops::Failure(
                    {ErrorCode::UNSUPPORTED, "QNN HTP prepare library is unavailable and no cached context exists"});
            std::filesystem::create_directories(cache_dir);
            context->create();
            const auto nodes = graph.nodes();
            for (std::size_t index = 0; index < plans.size(); ++index) {
                const auto name = "step" + std::to_string(index);
                if (plans[index].kind == PartitionPlan::Kind::FFN)
                    built.push_back(build_ffn_graph(*context, name, nodes, plans[index]));
                else if (plans[index].kind == PartitionPlan::Kind::GATED_FFN)
                    built.push_back(build_gated_ffn_graph(*context, name, nodes[plans[index].start], plans[index]));
                else
                    built.push_back(build_projection_graph(*context, name, nodes[plans[index].start], plans[index]));
            }
            write_file(binary_path, context->save());
            save_metadata(metadata_path, id, built);
        }
        std::vector<QnnPartition> partitions;
        partitions.reserve(built.size());
        for (const auto& partition : built) partitions.emplace_back(partition, context->retrieve_graph(partition.name));
        if (const char* dump = std::getenv("KIDI_QNN_DUMP"); dump && *dump) {
            std::cerr << "kidi_qnn|key=" << key << "|cache=" << (loaded ? "hit" : "miss") << "|id=" << id
                      << "|partitions=" << partitions.size() << "|api=" << runtime_->api_version()
                      << "|build=" << runtime_->build_id() << "|burst=" << runtime_->burst() << '\n';
        }
        return std::make_unique<QnnStepExecutable>(
            runtime_, std::move(context), std::vector<Node>(graph.nodes().begin(), graph.nodes().end()),
            std::vector<Tensor>(graph.inputs().begin(), graph.inputs().end()), std::move(partitions));
    }

private:
    std::shared_ptr<QnnRuntime> runtime_;
    std::mutex whole_compile_mutex_;
    std::mutex constant_mutex_;
    std::map<const void*, SharedMemory::Block> constant_blocks_;
    std::mutex operators_mutex_;
    std::map<std::string, std::shared_future<std::shared_ptr<CompiledOperator>>> operators_;
    std::size_t offloaded_ = 0;
    std::unique_ptr<OperatorBackend> reference_backend_;
    CompileQueue queue_;
};

auto cache_has_entries() -> bool {
    std::error_code error;
    const auto directory = cache_directory();
    if (!std::filesystem::is_directory(directory, error)) return false;
    for (std::filesystem::directory_iterator it(directory, error), end; !error && it != end; it.increment(error))
        if (it->path().extension() == ".bin") return true;
    return false;
}

} // namespace

auto npu_step_compiler() -> Result<std::shared_ptr<StepCompiler>> {
    static std::mutex mutex;
    // QNN/FastRPC teardown can outlive C++ static destruction; keep the singleton alive until process exit.
    static auto* compiler = new std::shared_ptr<StepCompiler>;
    static auto* error = new std::optional<Error>;
    std::scoped_lock lock(mutex);
    if (*compiler) return *compiler;
    if (*error) return std::unexpected(**error);
    try {
        auto runtime = std::make_shared<QnnRuntime>();
        if (!runtime->prepare_available() && !cache_has_entries()) {
            *error =
                Error{ErrorCode::UNSUPPORTED,
                      "QNN HTP runtime is loadable, but libQnnHtpPrepare.so is unavailable and the cache is empty"};
            return std::unexpected(**error);
        }
        *compiler = std::make_shared<QnnStepCompiler>(std::move(runtime));
        return *compiler;
    } catch (const ops::Failure& failure) {
        *error = failure.error();
        return std::unexpected(**error);
    } catch (const std::exception& exception) {
        *error = Error{ErrorCode::UNSUPPORTED, exception.what()};
        return std::unexpected(**error);
    }
}

} // namespace kidi::runtime
