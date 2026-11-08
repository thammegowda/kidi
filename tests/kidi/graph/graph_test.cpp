#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "kidi/graph/graph.h"
#include "kidi/graph/compiler.h"
#include "kidi/ops/context.h"
#include "kidi/runtime/operator.h"

namespace {
using kidi::ops::Context;
using kidi::ops::require;
using kidi::tensor::DType;
using kidi::tensor::Tensor;

auto tensor(std::vector<std::int64_t> shape, std::vector<float> values) -> Tensor {
    return require(Tensor::from_host(std::move(shape), std::span<const float>(values)));
}

auto values(const Tensor& tensor) -> std::vector<float> {
    const auto data = require(tensor.data<float>());
    return {data.begin(), data.end()};
}

auto fill(Tensor& tensor, float start) -> void {
    auto data = require(tensor.data<float>());
    for (std::size_t index = 0; index < data.size(); ++index) data[index] = start + 0.5F * static_cast<float>(index);
}

auto expect(bool condition, const char* message) -> bool {
    if (!condition) std::cerr << "FAILED: " << message << '\n';
    return condition;
}

auto tensor_inputs_borrow_operands() -> bool {
    const std::array<Tensor, 2> operands{};
    const auto check = [&](kidi::runtime::TensorInputs inputs) {
        const auto copied = inputs;
        const auto prefix = copied.first(1);
        return expect(inputs.size() == 2 && &inputs[0] == &operands[0] && &inputs[1] == &operands[1],
                      "tensor inputs borrow the original operands") &&
               expect(copied.size() == 2 && &copied.front() == &operands[0] && &copied.back() == &operands[1],
                      "copied tensor inputs preserve borrowed operands") &&
               expect(prefix.size() == 1 && &prefix.front() == &operands[0] && &prefix.back() == &operands[0],
                      "tensor input prefixes preserve borrowing without changing the original view");
    };
    const kidi::runtime::TensorInputs empty{};
    return check(operands) && check({&operands[0], &operands[1]}) &&
           expect(empty.size() == 0, "tensor inputs support an empty operand list");
}

auto fails_with(const std::function<void()>& call, std::string_view text) -> bool {
    try {
        call();
    } catch (const kidi::ops::Failure& error) {
        return error.error().message.find(text) != std::string::npos;
    }
    return false;
}

auto replay_matches_eager() -> bool {
    Context context;
    const auto weight = tensor({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    const auto bias = tensor({2, 4}, {-1, -2, -3, -4, 1, 2, 3, 4});
    auto input = require(Tensor::empty({2, 4}, DType::F32));
    int calls = 0;
    const auto step = [&](std::span<const Tensor> inputs) {
        ++calls;
        const auto hidden = context.add(context.multiply(inputs[0], weight), bias);
        return std::vector{context.softmax(hidden)};
    };
    const std::array inputs{input};
    std::vector<const void*> storage;
    for (int call = 0; call < 5; ++call) {
        fill(input, static_cast<float>(call));
        const auto outputs = context.replay("affine", inputs, step);
        Context reference;
        const auto expected = reference.softmax(reference.add(reference.multiply(input, weight), bias));
        if (!expect(values(outputs[0]) == values(expected), "replayed values match eager execution")) return false;
        storage.push_back(outputs[0].storage_identity());
    }
    if (!expect(calls == 2, "a step body runs only for its two captures") ||
        !expect(storage[2] == storage[4], "replay writes pointer-stable outputs"))
        return false;
    const auto retained = context.replay("affine", inputs, step)[0];
    const auto saved = values(retained);
    context.release_workspaces();
    const auto fresh = context.replay("affine", inputs, step)[0];
    return expect(values(retained) == saved, "workspace release preserves externally owned tensor storage") &&
           expect(values(fresh) == saved, "operators rebuild correctly after workspace release") &&
           expect(calls == 3, "workspace release discards the old captured step") && context.replay_enabled();
}

auto replay_rebinds_inputs_and_views() -> bool {
    Context context;
    const auto scale = tensor({3}, {2, 3, 4});
    const auto step = [&](std::span<const Tensor> inputs) {
        const auto rows = context.reshape(inputs[0], {2, 3});
        const auto tail = context.slice(inputs[0], 0, 3, 3);
        return std::vector{context.multiply(context.add(rows, context.reshape(tail, {1, 3})), scale)};
    };
    auto first = tensor({6}, {1, 2, 3, 4, 5, 6});
    auto second = tensor({6}, {1, 2, 3, 4, 5, 6});
    context.replay("view", std::array{first}, step);
    context.replay("view", std::array{second}, step);
    const auto third = tensor({6}, {10, 20, 30, 40, 50, 60});
    const auto outputs = context.replay("view", std::array{third}, step);
    const std::vector<float> expected{100, 210, 360, 160, 300, 480};
    const auto rebound = values(outputs[0]) == expected;
    const auto again = values(context.replay("view", std::array{first}, step)[0]);
    return expect(rebound, "replay reads a rebound input through reshaped and sliced views") &&
           expect(again == std::vector<float>{10, 21, 36, 16, 30, 48}, "replay rebinds back to an earlier input") &&
           expect(fails_with([&] { context.replay("view", std::array{tensor({5}, {1, 2, 3, 4, 5})}, step); },
                             "changed shape"),
                  "rebinding rejects a different shape");
}

auto replay_updates_state_in_place() -> bool {
    Context context;
    auto cache = require(Tensor::zeros({1, 4, 2}, DType::F32));
    auto index = require(Tensor::empty({1}, DType::I32));
    auto token = require(Tensor::empty({1, 1, 2}, DType::F32));
    const auto offset = tensor({1, 1, 2}, {0.25F, 0.5F});
    const auto step = [&](std::span<const Tensor> inputs) {
        auto memory = inputs[1];
        context.scatter_(memory, context.add(inputs[0], offset), inputs[2]);
        return std::vector{context.multiply(memory, memory)};
    };
    const std::array inputs{token, cache, index};
    std::vector<float> last;
    for (int position = 0; position < 4; ++position) {
        require(index.data<std::int32_t>())[0] = position;
        fill(token, static_cast<float>(position));
        last = values(context.replay("cache", inputs, step)[0]);
    }
    const std::vector<float> stored{0.25F, 1.F, 1.25F, 2.F, 2.25F, 3.F, 3.25F, 4.F};
    std::vector<float> squared;
    for (const auto value : stored) squared.push_back(value * value);
    return expect(values(cache) == stored, "replayed scatter_ writes each step's row into the state input") &&
           expect(last == squared, "later replayed operators observe the in-place update");
}

auto replay_supports_paired_and_selection_outputs() -> bool {
    Context context;
    const auto scale = tensor({4}, {1, 1, 1, 1});
    const auto bias = tensor({4}, {0, 0, 0, 0});
    auto hidden = require(Tensor::empty({1, 4}, DType::F32));
    auto residual = require(Tensor::empty({1, 4}, DType::F32));
    const auto step = [&](std::span<const Tensor> inputs) {
        auto [sum, normalized] = context.residual_layer_norm(inputs[0], inputs[1], scale, bias, 1e-5F);
        return std::vector{sum, normalized, context.greedy_token(normalized)};
    };
    const std::array inputs{hidden, residual};
    for (int call = 0; call < 4; ++call) {
        fill(hidden, static_cast<float>(call));
        auto data = require(residual.data<float>());
        for (std::size_t index = 0; index < data.size(); ++index) data[index] = call % 2 ? -float(index) : float(index);
        const auto outputs = context.replay("paired", inputs, step);
        Context reference;
        const auto [sum, normalized] = reference.residual_layer_norm(hidden, residual, scale, bias, 1e-5F);
        const auto token = reference.greedy_token(normalized);
        if (!expect(values(outputs[0]) == values(sum) && values(outputs[1]) == values(normalized) &&
                        require(outputs[2].data<std::int32_t>())[0] == require(token.data<std::int32_t>())[0],
                    "paired and selection operators replay exactly"))
            return false;
    }
    return true;
}

auto capture_rejects_per_call_host_values() -> bool {
    Context context;
    auto input = tensor({2}, {1, 2});
    int call = 0;
    const auto temporary = [&](std::span<const Tensor> inputs) {
        const auto host = tensor({2}, {static_cast<float>(++call), 0});
        return std::vector{context.add(inputs[0], host)};
    };
    context.replay("temporary", std::array{input}, temporary);
    const bool temporary_rejected =
        fails_with([&] { context.replay("temporary", std::array{input}, temporary); }, "storage changed");
    float scale = 0.5F;
    const auto argument = [&](std::span<const Tensor> inputs) {
        scale *= 2;
        return std::vector{context.static_round(inputs[0], scale)};
    };
    context.replay("argument", std::array{input}, argument);
    const bool argument_rejected =
        fails_with([&] { context.replay("argument", std::array{input}, argument); }, "arguments");
    auto destination = require(Tensor::zeros({4}, DType::F32));
    const bool copy_rejected = fails_with(
        [&] {
            context.replay("copy", std::array{input}, [&](std::span<const Tensor> inputs) {
                context.copy_slice_(destination, inputs[0], 0, 1);
                return std::vector{destination};
            });
        },
        "scatter_");
    return expect(temporary_rejected, "a per-call host temporary is rejected") &&
           expect(argument_rejected, "a per-call operator argument is rejected") &&
           expect(copy_rejected, "copy_slice_ is rejected inside a captured step");
}

auto least_recently_used_steps_are_recaptured() -> bool {
    Context context;
    auto input = tensor({2}, {1, 2});
    int calls = 0;
    const auto step = [&](std::span<const Tensor> inputs) {
        ++calls;
        return std::vector{context.add(inputs[0], inputs[0])};
    };
    for (int key = 0; key < 12; ++key)
        for (int call = 0; call < 3; ++call) context.replay("step" + std::to_string(key), std::array{input}, step);
    const auto captured = calls;
    const auto outputs = context.replay("step0", std::array{input}, step);
    return expect(captured == 24, "each step body runs for its two captures") &&
           expect(calls == 25 && values(outputs[0]) == std::vector<float>{2, 4},
                  "an evicted step is captured again and still computes correctly");
}

/// Doubles its input on the host, standing in for an accelerator executable.
class Doubler final : public kidi::graph::Executable {
public:
    explicit Doubler(std::atomic<int>& runs) : runs_(runs) {}
    auto run(std::span<const Tensor> inputs, std::span<Tensor> outputs) -> void override {
        ++runs_;
        const auto source = require(inputs[0].data<float>());
        auto target = require(outputs[0].data<float>());
        for (std::size_t index = 0; index < source.size(); ++index) target[index] = 2 * source[index];
    }

private:
    std::atomic<int>& runs_;
};

/// Compiles on a background thread once released, or fails there when `fail` is set.
class BackgroundCompiler final : public kidi::graph::Compiler {
public:
    explicit BackgroundCompiler(bool fail) : fail_(fail) {}
    auto name() const -> std::string_view override { return "background-test"; }
    auto compiles_in_background() const -> bool override { return true; }
    auto compile(const kidi::graph::Graph& graph, std::string_view)
        -> std::unique_ptr<kidi::graph::Executable> override {
        ++compiles;
        released_.wait();
        ++completed;
        if (fail_ || graph.nodes().empty())
            throw kidi::ops::Failure({kidi::core::ErrorCode::UNSUPPORTED, "test compile failure"});
        return std::make_unique<Doubler>(runs);
    }
    auto release() -> void { release_.set_value(); }
    std::atomic<int> compiles{0}, completed{0}, runs{0};

private:
    bool fail_;
    std::promise<void> release_;
    std::shared_future<void> released_ = release_.get_future().share();
};

class SingleCaptureCompiler final : public kidi::graph::Compiler {
public:
    explicit SingleCaptureCompiler(std::optional<bool> validation = false) : validation_(validation) {}
    auto name() const -> std::string_view override { return "single-capture-test"; }
    auto requires_second_capture(std::string_view) const -> std::optional<bool> override { return validation_; }
    auto captures_prefill() const -> bool override { return true; }
    auto compile(const kidi::graph::Graph& graph, std::string_view)
        -> std::unique_ptr<kidi::graph::Executable> override {
        ++compiles;
        if (graph.nodes().empty())
            throw kidi::ops::Failure({kidi::core::ErrorCode::UNSUPPORTED, "empty single-capture test graph"});
        return std::make_unique<Doubler>(runs);
    }
    int compiles = 0;
    std::atomic<int> runs{0};

private:
    std::optional<bool> validation_;
};

auto compiler_can_defer_capture_validation_to_caller() -> bool {
    for (const bool validation : {false, true}) {
        auto compiler = std::make_shared<SingleCaptureCompiler>(std::nullopt);
        const kidi::ops::StepCompilerScope scope(compiler);
        Context context;
        const auto input = tensor({2}, {1, 2});
        int captures = 0;
        const auto step = [&](std::span<const Tensor> inputs) {
            ++captures;
            return std::vector{context.add(inputs[0], inputs[0])};
        };
        for (int call = 0; call < 3; ++call) {
            const auto output = context.replay("validation", std::array{input}, step, validation);
            if (!expect(values(output[0]) == std::vector<float>{2, 4}, "caller-selected captures preserve outputs"))
                return false;
        }
        if (!expect(captures == (validation ? 2 : 1) && compiler->compiles == 1,
                    "a compiler without a capture override follows caller validation"))
            return false;
    }
    return true;
}

auto compiled_replay_validates_bindings() -> bool {
    auto compiler = std::make_shared<SingleCaptureCompiler>();
    const kidi::ops::StepCompilerScope scope(compiler);
    Context context;
    const auto input = tensor({2}, {1, 2});
    const auto step = [&](std::span<const Tensor> inputs) { return std::vector{context.add(inputs[0], inputs[0])}; };
    context.replay("compiled", std::array{input}, step);
    const auto wrong_shape = tensor({3}, {1, 2, 3});
    const auto wrong_dtype = require(Tensor::zeros({2}, DType::I32));
    const auto non_contiguous = require(tensor({2, 2}, {1, 2, 3, 4}).narrow(1, 0, 1));
    const bool rejected =
        fails_with([&] { context.replay("compiled", {}, step); }, "input count") &&
        fails_with([&] { context.replay("compiled", std::array{wrong_shape}, step); }, "changed shape") &&
        fails_with([&] { context.replay("compiled", std::array{wrong_dtype}, step); }, "dtype") &&
        fails_with([&] { context.replay("compiled", std::array{non_contiguous}, step); }, "changed shape");
    const auto output = context.replay("compiled", std::array{input}, step);
    return expect(rejected && compiler->runs == 1, "compiled replay rejects incompatible bindings before execution") &&
           expect(values(output[0]) == std::vector<float>{2, 4}, "invalid bindings do not poison a compiled step");
}

auto capture_recovers_after_exception() -> bool {
    Context context;
    const auto input = tensor({2}, {1, 2});
    const bool rejected = fails_with(
        [&] {
            context.replay("exception", std::array{input}, [&](std::span<const Tensor>) -> std::vector<Tensor> {
                throw kidi::ops::Failure({kidi::ErrorCode::RUNTIME, "capture failure"});
            });
        },
        "capture failure");
    const auto step = [&](std::span<const Tensor> inputs) { return std::vector{context.add(inputs[0], inputs[0])}; };
    for (int call = 0; call < 3; ++call)
        if (!expect(values(context.replay("exception", std::array{input}, step)[0]) == std::vector<float>{2, 4},
                    "failed capture restores recording state"))
            return false;
    return rejected;
}

auto compiled_sessions_keep_independent_outputs() -> bool {
    auto compiler = std::make_shared<SingleCaptureCompiler>();
    const kidi::ops::StepCompilerScope scope(compiler);
    Context first, second;
    const auto input = tensor({2}, {1, 2}), other = tensor({2}, {3, 4});
    const auto first_step = [&](std::span<const Tensor> inputs) {
        return std::vector{first.add(inputs[0], inputs[0])};
    };
    const auto second_step = [&](std::span<const Tensor> inputs) {
        return std::vector{second.add(inputs[0], inputs[0])};
    };
    const auto first_output = first.replay("shared", std::array{input}, first_step)[0];
    const auto second_output = second.replay("shared", std::array{other}, second_step)[0];
    second.replay("shared", std::array{input}, second_step);
    first.replay("shared", std::array{other}, first_step);
    return expect(first_output.storage_identity() != second_output.storage_identity(),
                  "sessions sharing a compiler retain separate output storage") &&
           expect(values(first_output) == std::vector<float>{6, 8} && values(second_output) == std::vector<float>{2, 4},
                  "compiled session output updates do not overwrite another session");
}

auto single_capture_compiles_and_rebinds_inputs() -> bool {
    auto compiler = std::make_shared<SingleCaptureCompiler>();
    const kidi::ops::StepCompilerScope scope(compiler);
    Context context;
    int captures = 0;
    const auto step = [&](std::span<const Tensor> inputs) {
        ++captures;
        return std::vector{context.add(inputs[0], inputs[0])};
    };
    auto first = tensor({2}, {1, 2});
    const auto initial = values(context.replay("single", std::array{first}, step)[0]);
    auto second = tensor({2}, {3, 5});
    const auto rebound = values(context.replay("single", std::array{second}, step)[0]);
    return expect(initial == std::vector<float>{2, 4}, "single capture returns the eager capture output") &&
           expect(rebound == std::vector<float>{6, 10}, "single-capture executable reads a rebound input") &&
           expect(captures == 1, "a single-capture compiler runs the step body once") &&
           expect(compiler->compiles == 1 && compiler->runs == 1,
                  "a single-capture compiler compiles once and executes on the next call");
}

auto attached_accelerator_enables_replay() -> bool {
    auto compiler = std::make_shared<SingleCaptureCompiler>();
    const kidi::ops::StepCompilerScope scope(compiler);
    Context context(kidi::tensor::Device::cpu(), false, false);
    auto input = tensor({2}, {1, 2});
    const auto step = [&](std::span<const Tensor> inputs) { return std::vector{context.add(inputs[0], inputs[0])}; };
    (void)context.replay("accelerated", std::array{input}, step);
    (void)context.replay("accelerated", std::array{input}, step);
    return expect(context.replay_enabled(), "an attached whole-step accelerator enables capture") &&
           expect(compiler->compiles == 1 && compiler->runs == 1,
                  "the attached whole-step accelerator compiles and replays");
}

auto background_compilation_hands_over_from_cpu_replay(bool fail) -> bool {
    auto compiler = std::make_shared<BackgroundCompiler>(fail);
    const kidi::ops::StepCompilerScope scope(compiler);
    Context context;
    auto input = tensor({2}, {1, 2});
    const auto step = [&](std::span<const Tensor> inputs) { return std::vector{context.add(inputs[0], inputs[0])}; };
    const auto doubled = [&] {
        return values(context.replay("double", std::array{input}, step)[0]) == std::vector<float>{2, 4};
    };
    for (int call = 0; call < 4; ++call)
        if (!expect(doubled(), "CPU replay serves the step while it compiles")) return false;
    const bool waited = compiler->runs == 0;
    compiler->release();
    for (int attempt = 0; attempt < 2000 && compiler->runs == 0; ++attempt) {
        if (!expect(doubled(), "the step stays correct while compilation finishes")) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!expect(waited && compiler->compiles == 1, "a captured step compiles once, off the calling thread"))
        return false;
    if (fail) return expect(compiler->runs == 0 && doubled(), "a failed background compile keeps CPU replay");
    return expect(compiler->runs > 0 && doubled(), "the compiled step takes over once ready");
}

auto pending_background_compilation_does_not_block_eviction() -> bool {
    auto compiler = std::make_shared<BackgroundCompiler>(false);
    const kidi::ops::StepCompilerScope scope(compiler);
    Context context;
    auto input = tensor({2}, {1, 2});
    const auto step = [&](std::span<const Tensor> inputs) { return std::vector{context.add(inputs[0], inputs[0])}; };
    (void)context.replay("prefill:128", std::array{input}, step);
    (void)context.replay("prefill:128", std::array{input}, step);
    for (int attempt = 0; attempt < 2000 && compiler->compiles == 0; ++attempt)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const auto started = std::chrono::steady_clock::now();
    context.clear_replays("prefill:");
    const auto elapsed = std::chrono::steady_clock::now() - started;
    compiler->release();
    for (int attempt = 0; attempt < 2000 && compiler->completed == 0; ++attempt)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return expect(compiler->compiles == 1, "the background compiler started before eviction") &&
           expect(elapsed < std::chrono::milliseconds(500), "eviction does not join a pending compiler") &&
           expect(compiler->completed == 1, "an evicted compile may finish and populate its persistent cache");
}

auto disabled_replay_runs_eagerly() -> bool {
    auto compiler = std::make_shared<SingleCaptureCompiler>();
    const kidi::ops::StepCompilerScope scope(compiler);
    setenv("KIDI_REPLAY", "0", 1);
    Context context;
    unsetenv("KIDI_REPLAY");
    int calls = 0;
    auto input = tensor({2}, {1, 2});
    for (int call = 0; call < 3; ++call)
        context.replay("eager", std::array{input}, [&](std::span<const Tensor> inputs) {
            ++calls;
            return std::vector{context.add(inputs[0], inputs[0])};
        });
    return expect(!context.replay_enabled() && calls == 3 && compiler->compiles == 0,
                  "KIDI_REPLAY=0 runs the step eagerly even with an attached accelerator");
}

auto bounded_replay_cache_evicts_old_shapes() -> bool {
    setenv("KIDI_REPLAY_CAPACITY", "1", 1);
    Context context;
    unsetenv("KIDI_REPLAY_CAPACITY");
    auto input = tensor({2}, {1, 2});
    int first_calls = 0, second_calls = 0;
    const auto replay = [&](std::string_view key, int& calls) {
        return context.replay(key, std::array{input}, [&](std::span<const Tensor> inputs) {
            ++calls;
            return std::vector{context.add(inputs[0], inputs[0])};
        });
    };
    (void)replay("first", first_calls);
    (void)replay("first", first_calls);
    (void)replay("second", second_calls);
    (void)replay("second", second_calls);
    (void)replay("first", first_calls);
    return expect(first_calls == 3 && second_calls == 2,
                  "a bounded replay cache recaptures the least-recently-used shape");
}

auto replay_capacity_can_be_changed() -> bool {
    Context context;
    context.set_replay_capacity(2);
    const auto input = tensor({2}, {1, 2});
    int first_calls = 0, second_calls = 0, third_calls = 0;
    const auto replay = [&](std::string_view key, int& calls) {
        return context.replay(
            key, std::array{input},
            [&](std::span<const Tensor> inputs) {
                ++calls;
                return std::vector{context.add(inputs[0], inputs[0])};
            },
            false);
    };
    (void)replay("first", first_calls);
    (void)replay("second", second_calls);
    (void)replay("first", first_calls);
    (void)replay("third", third_calls);
    (void)replay("first", first_calls);
    (void)replay("second", second_calls);
    if (!expect(first_calls == 1 && second_calls == 2 && third_calls == 1,
                "configured capacity retains hot regions and evicts the least recent region"))
        return false;
    context.set_replay_capacity(1);
    (void)replay("second", second_calls);
    const auto output = values(replay("first", first_calls)[0]);
    return expect(first_calls == 2 && second_calls == 2 && output == std::vector<float>{2, 4},
                  "shrinking replay capacity immediately evicts cold regions") &&
           fails_with([&] { context.set_replay_capacity(0); }, "positive");
}
} // namespace

namespace {
auto planned_intermediates_preserve_rebinding() -> bool {
    auto input = tensor({4}, {1, 2, 3, 4});
    const auto increment = tensor({4}, {1, 1, 1, 1});
    const std::array inputs{input};
    kidi::graph::Recorder recorder(inputs);
    auto backend = kidi::runtime::cpu_operators();
    auto previous = input;
    for (int index = 0; index < 8; ++index) {
        const kidi::runtime::OperatorSpec spec{kidi::runtime::Operation::ADD, {}, DType::F32};
        const std::array operands{previous, increment};
        std::shared_ptr<kidi::runtime::Operator> operation = backend->prepare(spec, operands);
        auto output = operation->run(operands);
        output = tensor({4}, values(output));
        recorder.record(spec, operation, kidi::graph::Mode::RESULT, operands, {&output, 1});
        previous = std::move(output);
    }
    auto graph = std::move(recorder).finish(std::array{previous});
    const auto memory = graph.plan_intermediates();
    const auto rebound = tensor({4}, {10, 20, 30, 40});
    const auto outputs = graph.replay(std::array{rebound});
    return expect(memory.planned_bytes < memory.original_bytes, "liveness planning reuses intermediate buffers") &&
           expect(values(outputs[0]) == std::vector<float>{18, 28, 38, 48},
                  "planned buffers preserve numerical output and input rebinding");
}
} // namespace

auto main() -> int {
    if (!tensor_inputs_borrow_operands()) return 1;
    try {
        return planned_intermediates_preserve_rebinding() && replay_matches_eager() &&
                       replay_rebinds_inputs_and_views() && replay_updates_state_in_place() &&
                       replay_supports_paired_and_selection_outputs() && capture_rejects_per_call_host_values() &&
                       least_recently_used_steps_are_recaptured() && bounded_replay_cache_evicts_old_shapes() &&
                       replay_capacity_can_be_changed() && disabled_replay_runs_eagerly() &&
                       single_capture_compiles_and_rebinds_inputs() && attached_accelerator_enables_replay() &&
                       compiler_can_defer_capture_validation_to_caller() && compiled_replay_validates_bindings() &&
                       capture_recovers_after_exception() && compiled_sessions_keep_independent_outputs() &&
                       background_compilation_hands_over_from_cpu_replay(false) &&
                       background_compilation_hands_over_from_cpu_replay(true) &&
                       pending_background_compilation_does_not_block_eviction()
                   ? 0
                   : 1;
    } catch (const kidi::ops::Failure& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
