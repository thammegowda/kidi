#include "kidi/graph/compiler.h"

#include <mutex>

namespace kidi::runtime {
namespace {
std::mutex npu_mutex;
NpuOptions configured_npu;
} // namespace

auto set_npu_options(NpuOptions options) -> void {
    const std::lock_guard lock(npu_mutex);
    configured_npu = std::move(options);
}
auto npu_options() -> NpuOptions {
    const std::lock_guard lock(npu_mutex);
    return configured_npu;
}
#if !defined(KIDI_HAS_QNN)
auto npu_step_compiler() -> Result<std::shared_ptr<graph::Compiler>> {
    return std::unexpected(Error{ErrorCode::UNSUPPORTED, "the Qualcomm NPU backend is not built"});
}
#endif
} // namespace kidi::runtime
