#include "HTP/QnnHtpCommon.h"
#include "HTP/core/qhpi.h"
#include "QnnOpPackage.h"
#include "QnnSdkBuildId.h"

#include <array>
#include <string>
#include <utility>

#define KIDI_STRINGIZE_DETAIL(value) #value
#define KIDI_STRINGIZE(value) KIDI_STRINGIZE_DETAIL(value)
#define KIDI_PACKAGE_NAME KIDI_STRINGIZE(THIS_PKG_NAME)

auto describeStructuredAttention() -> QHPI_OpInfo_v1;
auto describeFusedOps() -> std::pair<QHPI_OpInfo_v1, QHPI_OpInfo_v1>;

namespace {

constexpr auto PACKAGE_NAME = KIDI_PACKAGE_NAME;
constexpr auto ATTENTION = "StructuredAttention";
constexpr auto GELU_MULTIPLY = "GeluMultiply";
constexpr auto RMS_NORM = "RmsNorm";
std::array<const char*, 3> operation_names{{ATTENTION, GELU_MULTIPLY, RMS_NORM}};
Qnn_ApiVersion_t sdk_version = QNN_HTP_API_VERSION_INIT;
QnnOpPackage_Info_t package_info = QNN_OP_PACKAGE_INFO_INIT;
QnnOpPackage_GlobalInfrastructure_t infrastructure = nullptr;
bool initialized = false;

auto KidiOpsInit(QnnOpPackage_GlobalInfrastructure_t global) -> Qnn_ErrorHandle_t {
    if (initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
    infrastructure = global;
    initialized = true;
    return QNN_SUCCESS;
}

auto KidiOpsGetInfo(const QnnOpPackage_Info_t** info) -> Qnn_ErrorHandle_t {
    if (!initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    if (!info) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    package_info = QNN_OP_PACKAGE_INFO_INIT;
    package_info.packageName = PACKAGE_NAME;
    package_info.operationNames = operation_names.data();
    package_info.numOperations = operation_names.size();
    package_info.sdkBuildId = QNN_SDK_BUILD_ID;
    package_info.sdkApiVersion = &sdk_version;
    *info = &package_info;
    return QNN_SUCCESS;
}

auto KidiOpsValidateOpConfig(Qnn_OpConfig_t config) -> Qnn_ErrorHandle_t {
    if (config.version != QNN_OPCONFIG_VERSION_1 || !config.v1.packageName || !config.v1.typeName ||
        std::string(config.v1.packageName) != PACKAGE_NAME || config.v1.numOfParams != 0 || config.v1.numOfOutputs != 1)
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    const std::string type = config.v1.typeName;
    const auto inputs = config.v1.numOfInputs;
    if ((type == ATTENTION && inputs == 8) || (type == GELU_MULTIPLY && inputs == 1) ||
        (type == RMS_NORM && inputs == 5))
        return QNN_SUCCESS;
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
}

auto KidiOpsTerminate() -> Qnn_ErrorHandle_t {
    if (!initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    infrastructure = nullptr;
    initialized = false;
    return QNN_SUCCESS;
}

} // namespace

extern "C" Qnn_ErrorHandle_t KidiOpsInterfaceProvider(QnnOpPackage_Interface_t* interface) {
    if (!interface) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    interface->interfaceVersion = {1, 4, 0};
    interface->v1_4.init = KidiOpsInit;
    interface->v1_4.terminate = KidiOpsTerminate;
    interface->v1_4.getInfo = KidiOpsGetInfo;
    interface->v1_4.validateOpConfig = KidiOpsValidateOpConfig;
    interface->v1_4.createOpImpl = nullptr;
    interface->v1_4.freeOpImpl = nullptr;
    interface->v1_4.logInitialize = nullptr;
    interface->v1_4.logSetLevel = nullptr;
    interface->v1_4.logTerminate = nullptr;
    return QNN_SUCCESS;
}

extern "C" const char* qhpi_init() {
    static bool registered = false;
    if (!registered) {
        static std::array<QHPI_OpInfo_v1, 3> operations{};
        const auto [gelu, norm] = describeFusedOps();
        operations = {describeStructuredAttention(), gelu, norm};
        qhpi_register_ops_v1(static_cast<std::uint32_t>(operations.size()), operations.data(), PACKAGE_NAME);
        registered = true;
    }
    return PACKAGE_NAME;
}
