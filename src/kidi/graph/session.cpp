#include "kidi/graph/session.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>
#include <utility>

#include "kidi/ops/context.h"

namespace kidi::graph {

Session::Session(std::shared_ptr<Compiler> compiler, SessionOptions options)
    : compiler_(std::move(compiler)), options_(options) {
    if (!options_.capacity) throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "replay capacity must be positive"});
}

auto Session::evict_oldest() -> void {
    steps_.erase(std::ranges::min_element(steps_, {}, [](const auto& item) { return item.second.used; }));
}

auto Session::set_capacity(std::size_t capacity) -> void {
    if (!capacity || recorder_)
        throw ops::Failure(
            {ErrorCode::INVALID_ARGUMENT, "replay capacity must be positive and changed outside a capture"});
    options_.capacity = capacity;
    while (steps_.size() > capacity) evict_oldest();
}

auto Session::clear(std::string_view prefix) noexcept -> void {
    std::erase_if(steps_, [&](const auto& item) { return item.first.starts_with(prefix); });
}

auto Session::replay(std::string_view key, std::span<const tensor::Tensor> inputs, const Step& step,
                     bool validate_structure, const std::function<void()>& on_compiled) -> ReplayResult {
    if (recorder_) throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "captured steps cannot be nested"});
    if (!options_.enabled) {
        eager_outputs_ = step(inputs);
        return {eager_outputs_, ReplayMode::EAGER};
    }
    auto found = steps_.find(key);
    if (found == steps_.end()) {
        if (steps_.size() >= options_.capacity) evict_oldest();
        found = steps_.try_emplace(std::string(key)).first;
    }
    auto& entry = found->second;
    entry.used = ++clock_;
    const auto adopt = [&](std::unique_ptr<Executable> executable) {
        if (!executable) return;
        entry.executable = std::move(executable);
        entry.inputs.clear();
        for (const auto& input : entry.graph->inputs())
            entry.inputs.push_back({input.dtype(), input.device(), {input.shape().begin(), input.shape().end()}});
        entry.outputs.assign(entry.graph->outputs().begin(), entry.graph->outputs().end());
        entry.graph.reset();
        if (on_compiled) on_compiled();
    };
    if (entry.pending.valid() && entry.pending.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        const auto fallback = [&](std::string_view reason) {
            std::cerr << "kidi_step|key=" << key << "|accelerator=" << compiler_->name()
                      << "|fallback=replay|error=" << reason << '\n';
        };
        try {
            adopt(entry.pending.get());
        } catch (const std::exception& error) {
            fallback(error.what());
        } catch (...) {
            fallback("unknown error");
        }
    }
    if (entry.executable) {
        if (inputs.size() != entry.inputs.size())
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "compiled step input count changed"});
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            const auto& layout = entry.inputs[index];
            const auto& input = inputs[index];
            if (!input.defined() || !input.is_contiguous() || input.dtype() != layout.dtype ||
                input.device() != layout.device || !std::ranges::equal(input.shape(), layout.shape))
                throw ops::Failure(
                    {ErrorCode::INVALID_ARGUMENT, "compiled step input changed shape, dtype, or device"});
        }
        entry.executable->run(inputs, entry.outputs);
        return {entry.outputs, ReplayMode::COMPILED};
    }
    if (entry.graph) return {entry.graph->replay(inputs), ReplayMode::REPLAY};

    Recorder recorder(inputs);
    recorder_ = &recorder;
    struct Reset {
        Recorder*& active;
        ~Reset() { active = nullptr; }
    } reset{recorder_};
    const auto outputs = step(inputs);
    recorder_ = nullptr;
    auto graph = std::move(recorder).finish(outputs);
    const auto requires_second_capture =
        compiler_ ? compiler_->requires_second_capture(key).value_or(validate_structure) : validate_structure;
    if (!entry.first && requires_second_capture) {
        entry.first = std::move(graph);
        return {entry.first->outputs(), ReplayMode::CAPTURE};
    }
    if (entry.first) require_same_structure(*entry.first, graph, key);
    entry.first.reset();
    entry.graph = std::move(graph);
    if (compiler_ && options_.background_compile && compiler_->compiles_in_background(*entry.graph, key)) {
        std::promise<std::unique_ptr<Executable>> result;
        entry.pending = result.get_future();
        std::thread([compiler = compiler_, snapshot = *entry.graph, name = std::string(key),
                     result = std::move(result)]() mutable {
            try {
                result.set_value(compiler->compile(snapshot, name));
            } catch (...) {
                result.set_exception(std::current_exception());
            }
        }).detach();
    } else if (compiler_) {
        adopt(compiler_->compile(*entry.graph, key));
        if (entry.executable) return {entry.outputs, ReplayMode::CAPTURE};
    }
    return {entry.graph->outputs(), ReplayMode::CAPTURE};
}

} // namespace kidi::graph