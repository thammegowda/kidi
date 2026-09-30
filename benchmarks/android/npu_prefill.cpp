// Fast CPU-vs-NPU Gemma 4 prefill check for iterating on the NPU path on a phone:
//
//   kidi_npu_prefill MODEL_DIR [tokens=512] [chunk=128] [decode=16] [capacity=9216]
//
// Loads the model twice (CPU, and CPU with the NPU step compiler), prefills the same prompt in `chunk`-token
// pieces, compares every producer layer's K/V rows, times each chunk, and compares greedy continuations.
#include "kidi/checkpoint/config.h"
#include "kidi/core/module.h"
#include "kidi/model/gemma4.h"
#include "kidi/ops/context.h"
#include "kidi/runtime/operator.h"
#include "kidi/text/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {
using kidi::ops::require;
using kidi::tensor::DType;
using kidi::tensor::Tensor;
using Clock = std::chrono::steady_clock;

auto milliseconds(Clock::time_point start) -> double {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

auto load(const std::filesystem::path& directory, bool npu) -> kidi::model::Gemma4 {
    auto config = require(kidi::checkpoint::load_config(directory / "model.yaml"));
    auto weights = require(kidi::checkpoint::Weights::load(config["weights_file"].as<std::string>()));
    const auto parameter =
        require(weights.tensor(config["model"]["quantization_config"] ? "model.language_model.norm.weight"
                                                                      : "model.language_model.embed_tokens.weight"));
    std::optional<kidi::ops::StepCompilerScope> accelerator;
    if (npu) accelerator.emplace(require(kidi::runtime::npu_step_compiler()));
    const kidi::ModuleScope construction(parameter.dtype(), false, kidi::tensor::Device::cpu());
    auto model = require(kidi::model::Gemma4Impl::create(config["model"]));
    require(model->set_checkpoint(weights, 0, 128, true));
    return model;
}

auto prompt(const std::filesystem::path& directory, std::size_t count) -> std::vector<std::int32_t> {
    auto config = require(kidi::checkpoint::load_config(directory / "model.yaml"));
    const auto tokenizer = require(kidi::text::Tokenizer::load(config["tokenizer_file"].as<std::string>()));
    std::string text = "<bos><|turn>user\n";
    const std::string paragraph =
        "Summarize the following notes about river ecosystems. Rivers carry water, sediment, and nutrients from "
        "mountains to the sea. Along the way they shape valleys, feed wetlands, and support fish, insects, birds, "
        "and people. Dams change the timing of floods, trap sediment, and warm the water downstream. Restoration "
        "projects reconnect floodplains, remove barriers, and replant banks so that shade and wood return. ";
    auto tokens = require(tokenizer.encode(text));
    for (int repeat = 0; tokens.size() < count; ++repeat) {
        text += paragraph + "Section " + std::to_string(repeat + 1) + ". ";
        tokens = require(tokenizer.encode(text));
    }
    tokens.resize(count);
    return tokens;
}

struct Difference {
    double max_abs = 0, rms = 0, reference_rms = 0;
    std::size_t mismatched = 0, count = 0;
};

auto compare(const Tensor& expected, const Tensor& actual, std::size_t first_row, std::size_t rows) -> Difference {
    Difference result;
    const auto width = expected.size(-1);
    const auto begin = first_row * width, end = (first_row + rows) * width;
    if (expected.dtype() == DType::I8) {
        const auto left = require(expected.data<std::int8_t>()), right = require(actual.data<std::int8_t>());
        for (auto index = begin; index < end; ++index) {
            const auto delta = std::abs(static_cast<int>(left[index]) - static_cast<int>(right[index]));
            result.max_abs = std::max(result.max_abs, static_cast<double>(delta));
            result.rms += static_cast<double>(delta) * delta;
            result.reference_rms += static_cast<double>(left[index]) * left[index];
            result.mismatched += delta != 0;
        }
    } else {
        const auto left = require(expected.data<float>()), right = require(actual.data<float>());
        for (auto index = begin; index < end; ++index) {
            const auto delta = std::abs(static_cast<double>(left[index]) - right[index]);
            result.max_abs = std::max(result.max_abs, delta);
            result.rms += delta * delta;
            result.reference_rms += static_cast<double>(left[index]) * left[index];
            result.mismatched += delta != 0;
        }
    }
    result.count = end - begin;
    result.rms = std::sqrt(result.rms / std::max<std::size_t>(1, result.count));
    result.reference_rms = std::sqrt(result.reference_rms / std::max<std::size_t>(1, result.count));
    return result;
}
} // namespace

auto main(int argc, char** argv) -> int {
    if (argc < 2) {
        std::cerr << "usage: kidi_npu_prefill MODEL_DIR [tokens=512] [chunk=128] [decode=16] [capacity=9216]\n";
        return 2;
    }
    try {
        const std::filesystem::path directory = argv[1];
        const auto count = argc > 2 ? std::stoul(argv[2]) : 512UL;
        const auto chunk = argc > 3 ? std::stoul(argv[3]) : 128UL;
        const auto decode = argc > 4 ? std::stoul(argv[4]) : 16UL;
        const auto capacity = argc > 5 ? std::stoul(argv[5]) : 9216UL;
        auto started = Clock::now();
        const auto tokens = prompt(directory, count);
        // KIDI_PREFILL_SIDE=npu skips the CPU reference, e.g. to profile the NPU model alone.
        const auto* side = std::getenv("KIDI_PREFILL_SIDE");
        const bool npu_only = side && std::string_view(side) == "npu";
        const bool cpu_only = side && std::string_view(side) == "cpu";
        const bool reference = !npu_only && !cpu_only;
        auto cpu = load(directory, false);
        std::printf("load cpu %.0f ms\n", milliseconds(started));
        started = Clock::now();
        auto npu = load(directory, true);
        std::printf("load npu %.0f ms\n", milliseconds(started));
        auto cpu_state = require(cpu->create_state(capacity));
        auto npu_state = require(npu->create_state(capacity));
        double cpu_total = 0, npu_total = 0, npu_steady = 0;
        std::size_t steady_tokens = 0;
        for (std::size_t start = 0; start < tokens.size(); start += chunk) {
            const auto span = std::span(tokens).subspan(start, std::min(chunk, tokens.size() - start));
            started = Clock::now();
            if (!npu_only) require(cpu->prefill(span, cpu_state));
            const auto cpu_ms = milliseconds(started);
            started = Clock::now();
            if (!cpu_only) require(npu->prefill(span, npu_state));
            const auto npu_ms = milliseconds(started);
            cpu_total += cpu_ms;
            npu_total += npu_ms;
            if (start) {
                npu_steady += npu_ms;
                steady_tokens += span.size();
            }
            std::printf("chunk %4zu+%-4zu cpu %8.1f ms  npu %8.1f ms  speedup %.2fx\n", start, span.size(), cpu_ms,
                        npu_ms, cpu_ms / npu_ms);
        }
        std::printf("prefill cpu %.1f tok/s  npu %.1f tok/s  npu-after-first-chunk %.1f tok/s\n",
                    1000.0 * tokens.size() / cpu_total, 1000.0 * tokens.size() / npu_total,
                    steady_tokens ? 1000.0 * steady_tokens / npu_steady : 0.0);
        const auto replay_count =
            std::getenv("KIDI_PREFILL_REPLAY") ? std::atoi(std::getenv("KIDI_PREFILL_REPLAY")) : 0;
        for (int attempt = 0; attempt < replay_count; ++attempt) {
            auto replay = require(npu->create_state(capacity));
            started = Clock::now();
            require(npu->prefill(std::span(tokens).first(std::min(chunk, tokens.size())), replay));
            std::printf("prefill replay %d %zu tokens %.1f ms\n", attempt + 1, std::min(chunk, tokens.size()),
                        milliseconds(started));
        }
        if (cpu_only) {
            std::int32_t next = tokens.back();
            std::vector<std::int32_t> generated;
            double elapsed_ms = 0;
            for (std::size_t step = 0; step < decode; ++step) {
                started = Clock::now();
                next = require(cpu->forward_token(std::span(&next, 1), cpu_state));
                const auto step_ms = milliseconds(started);
                if (step >= 4)
                    elapsed_ms += step_ms;
                else
                    std::printf("decode step %zu cpu %.1f ms\n", step, step_ms);
                generated.push_back(next);
            }
            if (decode > 4) std::printf("decode cpu %.2f tok/s\n", 1000.0 * (decode - 4) / elapsed_ms);
            std::printf("cpu tokens");
            for (const auto token : generated) std::printf(" %d", token);
            std::printf("\n");
            return 0;
        }
        if (npu_only) {
            std::int32_t next = tokens.back();
            std::vector<std::int32_t> generated;
            double elapsed_ms = 0;
            for (std::size_t step = 0; step < decode; ++step) {
                started = Clock::now();
                next = require(npu->forward_token(std::span(&next, 1), npu_state));
                const auto step_ms = milliseconds(started);
                if (step >= 4)
                    elapsed_ms += step_ms;
                else
                    std::printf("decode step %zu npu %.1f ms\n", step, step_ms);
                generated.push_back(next);
            }
            if (decode > 4) std::printf("decode npu %.2f tok/s\n", 1000.0 * (decode - 4) / elapsed_ms);
            std::printf("npu tokens");
            for (const auto token : generated) std::printf(" %d", token);
            std::printf("\n");
            return 0;
        }
        std::size_t worst_layer = 0;
        double worst = 0;
        for (std::size_t layer = 0; layer < cpu_state.layers.size(); ++layer) {
            const auto key = compare(cpu_state.layers[layer].key, npu_state.layers[layer].key, 0, tokens.size());
            const auto value = compare(cpu_state.layers[layer].value, npu_state.layers[layer].value, 0, tokens.size());
            const auto relative = std::max(key.rms / std::max(1e-12, key.reference_rms),
                                           value.rms / std::max(1e-12, value.reference_rms));
            if (relative > worst) worst = relative, worst_layer = layer;
            std::printf("layer %2zu  K max %.4g rel-rms %.3g mism %.3f  V max %.4g rel-rms %.3g mism %.3f\n", layer,
                        key.max_abs, key.rms / std::max(1e-12, key.reference_rms),
                        static_cast<double>(key.mismatched) / key.count, value.max_abs,
                        value.rms / std::max(1e-12, value.reference_rms),
                        static_cast<double>(value.mismatched) / value.count);
        }
        std::printf("worst layer %zu rel-rms %.3g\n", worst_layer, worst);
        std::vector<std::int32_t> cpu_tokens, npu_tokens;
        std::int32_t cpu_next = tokens.back(), npu_next = tokens.back();
        // The prompt's last token was prefilled; decode from a fresh copy of it at the next position.
        // The first steps capture the decode step twice and compile it; steady-state rates skip them.
        constexpr std::size_t WARMUP = 4;
        double cpu_decode = 0, npu_decode = 0;
        for (std::size_t step = 0; step < decode; ++step) {
            started = Clock::now();
            cpu_next = require(cpu->forward_token(std::span(&cpu_next, 1), cpu_state));
            const auto cpu_ms = milliseconds(started);
            started = Clock::now();
            npu_next = require(npu->forward_token(std::span(&npu_next, 1), npu_state));
            const auto npu_ms = milliseconds(started);
            if (step >= WARMUP)
                cpu_decode += cpu_ms, npu_decode += npu_ms;
            else
                std::printf("decode step %zu cpu %.1f ms npu %.1f ms\n", step, cpu_ms, npu_ms);
            cpu_tokens.push_back(cpu_next);
            npu_tokens.push_back(npu_next);
        }
        if (decode > WARMUP)
            std::printf("decode cpu %.2f tok/s  npu %.2f tok/s\n", 1000.0 * (decode - WARMUP) / cpu_decode,
                        1000.0 * (decode - WARMUP) / npu_decode);
        const auto agree =
            static_cast<std::size_t>(std::ranges::mismatch(cpu_tokens, npu_tokens).in1 - cpu_tokens.begin());
        std::printf("greedy agreement %zu/%zu\n", agree, cpu_tokens.size());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
