#pragma once

#include "kidi/graph/compiler.h"

#include <string_view>

/// Help text for the Qualcomm NPU options shared by the Android benchmark tools.
constexpr std::string_view NPU_FLAGS_HELP =
    "  --qnn-dir DIR             QNN runtime libraries; compiled contexts are cached in DIR/cache\n"
    "  --qnn-op-package LIB      Kidi HTP op package (host build) for the fused NPU operators\n"
    "  --qnn-op-package-htp LIB  DSP build of the op package (default: --qnn-op-package)\n"
    "  --qnn-diagnostics         log NPU graph construction and QNN messages\n";

/// Reads one Qualcomm NPU option into `options`; `value()` consumes the option's argument. Returns false for other
/// options.
template <typename Value>
auto parse_npu_flag(std::string_view name, Value&& value, kidi::runtime::NpuOptions& options) -> bool {
    if (name == "--qnn-dir") {
        options.library_dir = value();
        options.cache_dir = options.library_dir / "cache";
    } else if (name == "--qnn-op-package") {
        options.op_package = value();
    } else if (name == "--qnn-op-package-htp") {
        options.op_package_htp = value();
    } else if (name == "--qnn-diagnostics") {
        options.diagnostics = true;
    } else {
        return false;
    }
    return true;
}
