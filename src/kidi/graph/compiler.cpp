#include "kidi/graph/compiler.h"

namespace kidi::runtime {
#if !defined(KIDI_HAS_QNN)
auto npu_step_compiler() -> Result<std::shared_ptr<graph::Compiler>> {
    return std::unexpected(Error{ErrorCode::UNSUPPORTED, "the Qualcomm NPU backend is not built"});
}
#endif
} // namespace kidi::runtime