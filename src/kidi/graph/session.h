#pragma once

#include <cstdint>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "kidi/graph/compiler.h"
#include "kidi/graph/graph.h"

namespace kidi::graph {

struct SessionOptions {
    bool enabled = false;
    bool background_compile = true;
    std::size_t capacity = 8;
};

enum class ReplayMode { EAGER, CAPTURE, REPLAY, COMPILED };

struct ReplayResult {
    std::span<const tensor::Tensor> outputs;
    ReplayMode mode;
};

class Session {
public:
    using Step = std::function<std::vector<tensor::Tensor>(std::span<const tensor::Tensor>)>;

    explicit Session(std::shared_ptr<Compiler> compiler = {}, SessionOptions options = {});
    Session(Session&&) noexcept = default;
    auto operator=(Session&&) noexcept -> Session& = default;
    Session(const Session&) = delete;
    auto operator=(const Session&) -> Session& = delete;

    auto replay(std::string_view key, std::span<const tensor::Tensor> inputs, const Step& step,
                bool validate_structure = true, const std::function<void()>& on_compiled = {}) -> ReplayResult;
    auto enabled() const noexcept -> bool { return options_.enabled; }
    auto recorder() const noexcept -> Recorder* { return recorder_; }
    auto set_capacity(std::size_t capacity) -> void;
    auto clear(std::string_view prefix = {}) noexcept -> void;
    auto clear_outputs() noexcept -> void { eager_outputs_.clear(); }

private:
    struct InputLayout {
        tensor::DType dtype;
        tensor::Device device;
        std::vector<std::int64_t> shape;
    };

    struct Captured {
        std::optional<Graph> first, graph;
        std::unique_ptr<Executable> executable;
        std::future<std::unique_ptr<Executable>> pending;
        std::vector<InputLayout> inputs;
        std::vector<tensor::Tensor> outputs;
        std::uint64_t used = 0;
    };

    auto evict_oldest() -> void;

    std::shared_ptr<Compiler> compiler_;
    SessionOptions options_;
    Recorder* recorder_ = nullptr;
    std::map<std::string, Captured, std::less<>> steps_;
    std::uint64_t clock_ = 0;
    std::vector<tensor::Tensor> eager_outputs_;
};

} // namespace kidi::graph