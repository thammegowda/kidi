#pragma once

#include <algorithm>
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
auto npu_step_compiler() -> Result<std::shared_ptr<graph::Compiler>>;
} // namespace kidi::runtime