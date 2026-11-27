#pragma once

#include <algorithm>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

#include "kidi/runtime/operator.h"

namespace kidi::graph {
class Graph;

class Executable {
public:
    virtual ~Executable() = default;
    virtual auto run(std::span<const tensor::Tensor> inputs, std::span<tensor::Tensor> outputs) -> void = 0;
};

class Compiler {
public:
    virtual ~Compiler() = default;
    virtual auto name() const -> std::string_view = 0;
    virtual auto compile(const Graph& graph, std::string_view key) -> std::unique_ptr<Executable> = 0;
    virtual auto requires_second_capture(std::string_view) const -> std::optional<bool> { return std::nullopt; }
    virtual auto key_extent(std::size_t required, std::size_t capacity) const -> std::size_t {
        return std::min(capacity, (required + 127) / 128 * 128);
    }
    virtual auto crop_local_attention() const -> bool { return true; }
    virtual auto compiles_in_background() const -> bool { return false; }
    virtual auto compiles_in_background(const Graph&, std::string_view) const -> bool {
        return compiles_in_background();
    }
    virtual auto captures_prefill() const -> bool { return true; }
    virtual auto prefill_chunk_size(std::size_t requested) const -> std::size_t { return requested; }
    virtual auto supports_step_operation(runtime::Operation) const -> bool { return true; }
    virtual auto prepare_operator(const runtime::OperatorSpec&, runtime::TensorInputs)
        -> std::unique_ptr<runtime::Operator> {
        return nullptr;
    }
};
} // namespace kidi::graph

namespace kidi::runtime {
/// Qualcomm NPU settings for the QNN step compiler. Set them before the first NPU model loads; empty paths keep the
/// defaults. Diagnostics may change at any time.
struct NpuOptions {
    std::filesystem::path library_dir;    ///< QNN runtime libraries (default: the dynamic loader's search path).
    std::filesystem::path cache_dir;      ///< Compiled-context cache (default: .kidi-qnn-cache).
    std::filesystem::path skel_dir;       ///< Hexagon skel libraries, prepended to ADSP_LIBRARY_PATH.
    std::filesystem::path op_package;     ///< Kidi HTP op package (host build); enables the fused NPU operators.
    std::filesystem::path op_package_htp; ///< Device build of the op package (default: op_package).
    bool diagnostics = false;             ///< Log graph construction and QNN messages.
};
auto set_npu_options(NpuOptions options) -> void;
auto npu_options() -> NpuOptions;
auto npu_step_compiler() -> Result<std::shared_ptr<graph::Compiler>>;
} // namespace kidi::runtime