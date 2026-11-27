#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <future>
#include <string>
#include <thread>
#include <vector>

#include "kidi/graph/graph.h"
#include "kidi/graph/compiler.h"
#include "kidi/ops/context.h"
#include "kidi/runtime/operator.h"

#include <gtest/gtest.h>

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

auto fails_with(const std::function<void()>& call, std::string_view text) -> ::testing::AssertionResult {
    try {
        call();
    } catch (const kidi::ops::Failure& error) {
        if (error.error().message.find(text) != std::string::npos) return ::testing::AssertionSuccess();
        return ::testing::AssertionFailure() << "failure \"" << error.error().message << "\" does not mention \""
                                             << text << '"';
    }
    return ::testing::AssertionFailure() << "call did not fail";
}

TEST(GraphTest, TensorInputsBorrowOperands) {
    const std::array<Tensor, 2> operands{};
    const auto check = [&](kidi::runtime::TensorInputs inputs) {
        const auto copied = inputs;
        const auto prefix = copied.first(1);
        EXPECT_TRUE(inputs.size() == 2 && &inputs[0] == &operands[0] && &inputs[1] == &operands[1])
            << "tensor inputs borrow the original operands";
        EXPECT_TRUE(copied.size() == 2 && &copied.front() == &operands[0] && &copied.back() == &operands[1])
            << "copied tensor inputs preserve borrowed operands";
        EXPECT_TRUE(prefix.size() == 1 && &prefix.front() == &operands[0] && &prefix.back() == &operands[0])
            << "tensor input prefixes preserve borrowing without changing the original view";
    };
    const kidi::runtime::TensorInputs empty{};
    {
        SCOPED_TRACE("span");
        check(operands);
    }
    {
        SCOPED_TRACE("initializer list");
        check({&operands[0], &operands[1]});
    }
    EXPECT_EQ(empty.size(), 0U) << "tensor inputs support an empty operand list";
}

TEST(GraphTest, ReplayMatchesEager) {
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
        ASSERT_EQ(values(outputs[0]), values(expected)) << "replayed values match eager execution (call " << call << ')';
        storage.push_back(outputs[0].storage_identity());
    }
    EXPECT_EQ(calls, 2) << "a step body runs only for its two captures";
    EXPECT_EQ(storage[2], storage[4]) << "replay writes pointer-stable outputs";
    const auto retained = context.replay("affine", inputs, step)[0];
    const auto saved = values(retained);
    context.release_workspaces();
    const auto fresh = context.replay("affine", inputs, step)[0];
    EXPECT_EQ(values(retained), saved) << "workspace release preserves externally owned tensor storage";
    EXPECT_EQ(values(fresh), saved) << "operators rebuild correctly after workspace release";
    EXPECT_EQ(calls, 3) << "workspace release discards the old captured step";
    EXPECT_TRUE(context.replay_enabled());
}

TEST(GraphTest, ReplayRebindsInputsAndViews) {
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
    EXPECT_EQ(values(outputs[0]), expected) << "replay reads a rebound input through reshaped and sliced views";
    const auto again = values(context.replay("view", std::array{first}, step)[0]);
    EXPECT_EQ(again, (std::vector<float>{10, 21, 36, 16, 30, 48})) << "replay rebinds back to an earlier input";
    EXPECT_TRUE(fails_with([&] { context.replay("view", std::array{tensor({5}, {1, 2, 3, 4, 5})}, step); },
                           "changed shape"))
        << "rebinding rejects a different shape";
}

TEST(GraphTest, ReplayUpdatesStateInPlace) {
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
    EXPECT_EQ(values(cache), stored) << "replayed scatter_ writes each step's row into the state input";
    EXPECT_EQ(last, squared) << "later replayed operators observe the in-place update";
}

TEST(GraphTest, ReplayPairedOutputs) {
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
        SCOPED_TRACE(call);
        fill(hidden, static_cast<float>(call));
        auto data = require(residual.data<float>());
        for (std::size_t index = 0; index < data.size(); ++index) data[index] = call % 2 ? -float(index) : float(index);
        const auto outputs = context.replay("paired", inputs, step);
        Context reference;
        const auto [sum, normalized] = reference.residual_layer_norm(hidden, residual, scale, bias, 1e-5F);
        const auto token = reference.greedy_token(normalized);
        ASSERT_EQ(values(outputs[0]), values(sum)) << "paired and selection operators replay exactly";
        ASSERT_EQ(values(outputs[1]), values(normalized)) << "paired and selection operators replay exactly";
        ASSERT_EQ(require(outputs[2].data<std::int32_t>())[0], require(token.data<std::int32_t>())[0])
            << "paired and selection operators replay exactly";
    }
}

TEST(GraphTest, RejectsHostValues) {
    Context context;
    auto input = tensor({2}, {1, 2});
    int call = 0;
    const auto temporary = [&](std::span<const Tensor> inputs) {
        const auto host = tensor({2}, {static_cast<float>(++call), 0});
        return std::vector{context.add(inputs[0], host)};
    };
    context.replay("temporary", std::array{input}, temporary);
    EXPECT_TRUE(fails_with([&] { context.replay("temporary", std::array{input}, temporary); }, "storage changed"))
        << "a per-call host temporary is rejected";
    float scale = 0.5F;
    const auto argument = [&](std::span<const Tensor> inputs) {
        scale *= 2;
        return std::vector{context.static_round(inputs[0], scale)};
    };
    context.replay("argument", std::array{input}, argument);
    EXPECT_TRUE(fails_with([&] { context.replay("argument", std::array{input}, argument); }, "arguments"))
        << "a per-call operator argument is rejected";
    auto destination = require(Tensor::zeros({4}, DType::F32));
    EXPECT_TRUE(fails_with(
        [&] {
            context.replay("copy", std::array{input}, [&](std::span<const Tensor> inputs) {
                context.copy_slice_(destination, inputs[0], 0, 1);
                return std::vector{destination};
            });
        },
        "scatter_"))
        << "copy_slice_ is rejected inside a captured step";
}

TEST(GraphTest, LruRecapture) {
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
    EXPECT_EQ(captured, 24) << "each step body runs for its two captures";
    EXPECT_EQ(calls, 25) << "an evicted step is captured again and still computes correctly";
    EXPECT_EQ(values(outputs[0]), (std::vector<float>{2, 4}))
        << "an evicted step is captured again and still computes correctly";
}

TEST(GraphTest, ReplayCacheEvicts) {
    Context context;
    context.set_replay_capacity(1);
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
    EXPECT_EQ(first_calls, 3) << "a bounded replay cache recaptures the least-recently-used shape";
    EXPECT_EQ(second_calls, 2) << "a bounded replay cache recaptures the least-recently-used shape";
}

TEST(GraphTest, ReplayCapacityCanBeChanged) {
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
    constexpr auto RETAINS = "configured capacity retains hot regions and evicts the least recent region";
    ASSERT_EQ(first_calls, 1) << RETAINS;
    ASSERT_EQ(second_calls, 2) << RETAINS;
    ASSERT_EQ(third_calls, 1) << RETAINS;
    context.set_replay_capacity(1);
    (void)replay("second", second_calls);
    const auto output = values(replay("first", first_calls)[0]);
    constexpr auto SHRINKS = "shrinking replay capacity immediately evicts cold regions";
    EXPECT_EQ(first_calls, 2) << SHRINKS;
    EXPECT_EQ(second_calls, 2) << SHRINKS;
    EXPECT_EQ(output, (std::vector<float>{2, 4})) << SHRINKS;
    EXPECT_TRUE(fails_with([&] { context.set_replay_capacity(0); }, "positive"));
}

TEST(GraphTest, PlannedRebinding) {
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
    EXPECT_LT(memory.planned_bytes, memory.original_bytes) << "liveness planning reuses intermediate buffers";
    EXPECT_EQ(values(outputs[0]), (std::vector<float>{18, 28, 38, 48}))
        << "planned buffers preserve numerical output and input rebinding";
}

TEST(GraphTest, CaptureRecoversAfterException) {
    Context context;
    const auto input = tensor({2}, {1, 2});
    const auto rejected = fails_with(
        [&] {
            context.replay("exception", std::array{input}, [&](std::span<const Tensor>) -> std::vector<Tensor> {
                throw kidi::ops::Failure({kidi::ErrorCode::RUNTIME, "capture failure"});
            });
        },
        "capture failure");
    const auto step = [&](std::span<const Tensor> inputs) { return std::vector{context.add(inputs[0], inputs[0])}; };
    for (int call = 0; call < 3; ++call)
        ASSERT_EQ(values(context.replay("exception", std::array{input}, step)[0]), (std::vector<float>{2, 4}))
            << "failed capture restores recording state (call " << call << ')';
    EXPECT_TRUE(rejected);
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


TEST(GraphTest, DeferredValidation) {
    for (const bool validation : {false, true}) {
        SCOPED_TRACE(validation);
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
            ASSERT_EQ(values(output[0]), (std::vector<float>{2, 4})) << "caller-selected captures preserve outputs";
        }
        constexpr auto MESSAGE = "a compiler without a capture override follows caller validation";
        ASSERT_EQ(captures, validation ? 2 : 1) << MESSAGE;
        ASSERT_EQ(compiler->compiles, 1) << MESSAGE;
    }
}

TEST(GraphTest, ValidatesBindings) {
    auto compiler = std::make_shared<SingleCaptureCompiler>();
    const kidi::ops::StepCompilerScope scope(compiler);
    Context context;
    const auto input = tensor({2}, {1, 2});
    const auto step = [&](std::span<const Tensor> inputs) { return std::vector{context.add(inputs[0], inputs[0])}; };
    context.replay("compiled", std::array{input}, step);
    const auto wrong_shape = tensor({3}, {1, 2, 3});
    const auto wrong_dtype = require(Tensor::zeros({2}, DType::I32));
    const auto non_contiguous = require(tensor({2, 2}, {1, 2, 3, 4}).narrow(1, 0, 1));
    constexpr auto REJECTS = "compiled replay rejects incompatible bindings before execution";
    EXPECT_TRUE(fails_with([&] { context.replay("compiled", {}, step); }, "input count")) << REJECTS;
    EXPECT_TRUE(fails_with([&] { context.replay("compiled", std::array{wrong_shape}, step); }, "changed shape"))
        << REJECTS;
    EXPECT_TRUE(fails_with([&] { context.replay("compiled", std::array{wrong_dtype}, step); }, "dtype")) << REJECTS;
    EXPECT_TRUE(fails_with([&] { context.replay("compiled", std::array{non_contiguous}, step); }, "changed shape"))
        << REJECTS;
    const auto output = context.replay("compiled", std::array{input}, step);
    EXPECT_EQ(compiler->runs, 1) << REJECTS;
    EXPECT_EQ(values(output[0]), (std::vector<float>{2, 4})) << "invalid bindings do not poison a compiled step";
}

TEST(GraphTest, IndependentSessions) {
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
    EXPECT_NE(first_output.storage_identity(), second_output.storage_identity())
        << "sessions sharing a compiler retain separate output storage";
    EXPECT_EQ(values(first_output), (std::vector<float>{6, 8}))
        << "compiled session output updates do not overwrite another session";
    EXPECT_EQ(values(second_output), (std::vector<float>{2, 4}))
        << "compiled session output updates do not overwrite another session";
}

TEST(GraphTest, SingleCaptureRebinds) {
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
    EXPECT_EQ(initial, (std::vector<float>{2, 4})) << "single capture returns the eager capture output";
    EXPECT_EQ(rebound, (std::vector<float>{6, 10})) << "single-capture executable reads a rebound input";
    EXPECT_EQ(captures, 1) << "a single-capture compiler runs the step body once";
    EXPECT_EQ(compiler->compiles, 1) << "a single-capture compiler compiles once and executes on the next call";
    EXPECT_EQ(compiler->runs, 1) << "a single-capture compiler compiles once and executes on the next call";
}

TEST(GraphTest, AcceleratorReplay) {
    auto compiler = std::make_shared<SingleCaptureCompiler>();
    const kidi::ops::StepCompilerScope scope(compiler);
    Context context(kidi::tensor::Device::cpu(), false, false);
    auto input = tensor({2}, {1, 2});
    const auto step = [&](std::span<const Tensor> inputs) { return std::vector{context.add(inputs[0], inputs[0])}; };
    (void)context.replay("accelerated", std::array{input}, step);
    (void)context.replay("accelerated", std::array{input}, step);
    EXPECT_TRUE(context.replay_enabled()) << "an attached whole-step accelerator enables capture";
    EXPECT_EQ(compiler->compiles, 1) << "the attached whole-step accelerator compiles and replays";
    EXPECT_EQ(compiler->runs, 1) << "the attached whole-step accelerator compiles and replays";
}

TEST(GraphTest, DisabledReplayRunsEagerly) {
    auto compiler = std::make_shared<SingleCaptureCompiler>();
    const kidi::ops::StepCompilerScope scope(compiler);
    const kidi::ops::EagerScope eager;
    Context context;
    int calls = 0;
    auto input = tensor({2}, {1, 2});
    for (int call = 0; call < 3; ++call)
        context.replay("eager", std::array{input}, [&](std::span<const Tensor> inputs) {
            ++calls;
            return std::vector{context.add(inputs[0], inputs[0])};
        });
    constexpr auto MESSAGE = "an eager scope runs the step eagerly even with an attached accelerator";
    EXPECT_FALSE(context.replay_enabled()) << MESSAGE;
    EXPECT_EQ(calls, 3) << MESSAGE;
    EXPECT_EQ(compiler->compiles, 0) << MESSAGE;
}

class GraphAsyncTest : public ::testing::TestWithParam<bool> {};

TEST_P(GraphAsyncTest, HandsOver) {
    const bool fail = GetParam();
    auto compiler = std::make_shared<BackgroundCompiler>(fail);
    const kidi::ops::StepCompilerScope scope(compiler);
    Context context;
    auto input = tensor({2}, {1, 2});
    const auto step = [&](std::span<const Tensor> inputs) { return std::vector{context.add(inputs[0], inputs[0])}; };
    const auto doubled = [&] {
        return values(context.replay("double", std::array{input}, step)[0]) == std::vector<float>{2, 4};
    };
    for (int call = 0; call < 4; ++call) EXPECT_TRUE(doubled()) << "CPU replay serves the step while it compiles";
    const bool waited = compiler->runs == 0;
    compiler->release();
    for (int attempt = 0; attempt < 2000 && compiler->runs == 0; ++attempt) {
        ASSERT_TRUE(doubled()) << "the step stays correct while compilation finishes";
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(waited) << "a captured step compiles once, off the calling thread";
    ASSERT_EQ(compiler->compiles, 1) << "a captured step compiles once, off the calling thread";
    if (fail) {
        EXPECT_EQ(compiler->runs, 0) << "a failed background compile keeps CPU replay";
        EXPECT_TRUE(doubled()) << "a failed background compile keeps CPU replay";
    } else {
        EXPECT_GT(compiler->runs, 0) << "the compiled step takes over once ready";
        EXPECT_TRUE(doubled()) << "the compiled step takes over once ready";
    }
}

INSTANTIATE_TEST_SUITE_P(, GraphAsyncTest, ::testing::Values(false, true),
                         [](const ::testing::TestParamInfo<bool>& info) { return info.param ? "fail" : "ok"; });

TEST(GraphTest, PendingCompileAllowsEvict) {
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
    EXPECT_EQ(compiler->compiles, 1) << "the background compiler started before eviction";
    EXPECT_LT(elapsed, std::chrono::milliseconds(500)) << "eviction does not join a pending compiler";
    EXPECT_EQ(compiler->completed, 1) << "an evicted compile may finish and populate its persistent cache";
}

} // namespace
