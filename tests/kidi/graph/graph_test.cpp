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
#include "kidi/ops/context.h"

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
    return expect(calls == 2, "a step body runs only for its two captures") &&
           expect(storage[2] == storage[4], "replay writes pointer-stable outputs") && context.replay_enabled();
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
class Doubler final : public kidi::runtime::StepExecutable {
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
class BackgroundCompiler final : public kidi::runtime::StepCompiler {
public:
    explicit BackgroundCompiler(bool fail) : fail_(fail) {}
    auto name() const -> std::string_view override { return "background-test"; }
    auto compiles_in_background() const -> bool override { return true; }
    auto compile(const kidi::graph::Graph& graph,
                 std::string_view) -> std::unique_ptr<kidi::runtime::StepExecutable> override {
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
    return expect(!context.replay_enabled() && calls == 3, "KIDI_REPLAY=0 runs the step eagerly every call");
}
} // namespace

auto main() -> int {
    try {
        return replay_matches_eager() && replay_rebinds_inputs_and_views() && replay_updates_state_in_place() &&
                       replay_supports_paired_and_selection_outputs() && capture_rejects_per_call_host_values() &&
                       least_recently_used_steps_are_recaptured() && disabled_replay_runs_eagerly() &&
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
