// Inkling-2 checks on a checkpoint in the default local model hub; skips when missing.
// Optionally set KIDI_INKLING2_DUMP to write the prompt's FP32 logits [tokens, vocabulary] for reference comparison
// (tools/reference/inkling2.py).
#include "kidi/checkpoint/config.h"
#include "kidi/checkpoint/hub.h"
#include "kidi/checkpoint/weights.h"
#include "kidi/inference/generator.h"
#include "kidi/model/inkling2.h"
#include "kidi/model/registry.h"
#include "kidi/ops/context.h"
#include "kidi/text/tokenizer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include <gtest/gtest.h>

namespace {
using namespace kidi;
using ops::require;

auto argmax(std::span<const float> row) -> std::int32_t {
    return static_cast<std::int32_t>(std::ranges::max_element(row) - row.begin());
}

auto host_logits(const tensor::Tensor& logits) -> std::vector<float> {
    const auto copied = require(logits.to(tensor::Device::cpu()));
    const auto values = require(copied.data<float>());
    return {values.begin(), values.end()};
}

auto chat(inference::Generator& generator, std::string_view question) -> inference::TextGeneration {
    const nlohmann::ordered_json messages{{{"role", "system"}, {"content", "Answer in one word."}},
                                          {{"role", "user"}, {"content", question}}};
    const auto id = require(generator.enqueue_chat(
        messages, {.maximum_new_tokens = 48, .context_size = 512, .prefix_cache_bytes = std::size_t{1} << 30}));
    while (true)
        for (auto& event : require(generator.step()).events)
            if (event.request_id == id && event.completed) return std::move(*event.completed);
}

auto run_core_benchmark(bool quick_512) -> void {
    const auto directory = checkpoint::Hub{}.resolve("@thinkingmachines/Inkling-2-Mini");
    if (!directory && directory.error().code == ErrorCode::IO) GTEST_SKIP() << directory.error().message;
    ASSERT_TRUE(directory.has_value()) << directory.error().message;
    const auto config = require(checkpoint::load_config(*directory));
    const auto device = tensor::Device::cuda();
    ops::Context context(device);
    using Clock = std::chrono::steady_clock;
    const auto load_started = Clock::now();
    auto model = require(model::create_model_as<model::CausalLanguageModel>(
        config["model"], {.device = device, .parameter_dtype = tensor::DType::BF16}));
    require(model->set_checkpoint(require(checkpoint::Weights::load(config["weights_file"].as<std::string>())),
                                 {.precision = core::InferencePrecision::CHECKPOINT}));
    context.synchronize();
    std::cout << "core_benchmark load_ms="
              << std::chrono::duration<double, std::milli>(Clock::now() - load_started).count() << '\n';

    const auto tokenizer = require(text::Tokenizer::load(config["tokenizer_file"].as<std::string>()));
    const auto text = require(tokenizer.encode(
        "Mixture-of-experts models route each token to selected experts. "
        "Efficient inference reuses cached attention state and reads compact weights directly. "));
    ASSERT_FALSE(text.empty());
    constexpr std::size_t CAPACITY = 1024, REPETITIONS = 3, DECODE_STEPS = 128, WARMUP_STEPS = 8;
    std::vector<std::int32_t> tokens(CAPACITY);
    for (std::size_t index = 0; index < tokens.size(); ++index) tokens[index] = text[index % text.size()];
    auto state = require(model->create_state(CAPACITY));
    context.synchronize();
    const auto report = [](std::string_view phase, std::size_t prompt_tokens, std::size_t measured_tokens,
                           std::array<double, REPETITIONS> seconds) {
        std::ranges::sort(seconds);
        const auto median = seconds[REPETITIONS / 2];
        std::cout << "core_benchmark phase=" << phase << " prompt_tokens=" << prompt_tokens
                  << " measured_tokens=" << measured_tokens << " repetitions=" << REPETITIONS
                  << " median_ms=" << median * 1000 << " tokens_per_second=" << measured_tokens / median
                  << " ms_per_token=" << median * 1000 / measured_tokens
                  << " min_ms=" << seconds.front() * 1000 << " max_ms=" << seconds.back() * 1000 << '\n';
    };

    const std::vector<std::size_t> prefill_lengths = quick_512 ? std::vector<std::size_t>{512}
                                                               : std::vector<std::size_t>{32, 128, 512};
    for (const std::size_t length : prefill_lengths) {
        const auto prompt = std::span(tokens).first(length);
        require(model->rewind(*state, 0));
        require(model->prefill(prompt, *state));
        context.synchronize();
        std::array<double, REPETITIONS> seconds{};
        for (auto& elapsed : seconds) {
            require(model->rewind(*state, 0));
            context.synchronize();
            const auto started = Clock::now();
            require(model->prefill(prompt, *state));
            context.synchronize();
            elapsed = std::chrono::duration<double>(Clock::now() - started).count();
            ASSERT_EQ(state->sequence_position(), length);
        }
        report("prefill", length, length, seconds);
    }

    const std::vector<std::size_t> decode_lengths = quick_512 ? std::vector<std::size_t>{512}
                                                              : std::vector<std::size_t>{128, 512};
    const std::vector<int> decode_modes =
        quick_512 ? std::vector<int>{model->supports_device_tokens() ? 2 : 1} : std::vector<int>{0, 1, 2};
    for (const std::size_t length : decode_lengths)
        for (const int mode : decode_modes) {
            if (mode == 2 && !model->supports_device_tokens()) continue;
            std::array<double, REPETITIONS> seconds{};
            for (auto& elapsed : seconds) {
                require(model->rewind(*state, 0));
                require(model->prefill(std::span(tokens).first(length), *state));
                auto next = tokens[length];
                auto device_token =
                    mode == 2
                        ? require(tensor::Tensor::from_host({1}, std::span<const std::int32_t>(&next, 1), device))
                        : tensor::Tensor{};
                tensor::Tensor logits;
                const auto step = [&](std::size_t index) {
                    if (mode == 2) {
                        device_token = require(model->forward_device_token(device_token, *state));
                    } else if (mode == 1) {
                        next = require(model->forward_token(std::span(&next, 1), *state));
                    } else {
                        logits = require(model->forward(std::span(tokens).subspan(length + index, 1), *state));
                    }
                };
                for (std::size_t index = 0; index < WARMUP_STEPS; ++index) step(index);
                context.synchronize();
                const auto started = Clock::now();
                for (std::size_t index = 0; index < DECODE_STEPS; ++index) step(WARMUP_STEPS + index);
                context.synchronize();
                elapsed = std::chrono::duration<double>(Clock::now() - started).count();
                ASSERT_EQ(state->sequence_position(), length + WARMUP_STEPS + DECODE_STEPS);
                if (mode == 2) {
                    const auto host = require(device_token.to(tensor::Device::cpu()));
                    const auto value = require(host.data<std::int32_t>())[0];
                    ASSERT_GE(value, 0);
                    ASSERT_LT(value, model->vocabulary_size());
                } else if (mode == 1) {
                    ASSERT_GE(next, 0);
                    ASSERT_LT(next, model->vocabulary_size());
                } else {
                    ASSERT_EQ(logits.numel(), static_cast<std::size_t>(model->vocabulary_size()));
                    const auto values = host_logits(logits);
                    ASSERT_TRUE(std::ranges::all_of(values, [](float value) { return std::isfinite(value); }));
                }
            }
            report(mode == 2 ? "decode_chained" : (mode == 1 ? "decode_greedy" : "decode_forward_only"), length,
                   DECODE_STEPS, seconds);
        }
}

// Opt-in performance workloads, not timing assertions in the correctness suite.
TEST(Inkling2Test, DISABLED_CoreBenchmark) { run_core_benchmark(false); }

TEST(Inkling2Test, DISABLED_Quick512Benchmark) { run_core_benchmark(true); }

TEST(Inkling2Test, PrefillMatchesDecode) {
    const auto directory = checkpoint::Hub{}.resolve("@thinkingmachines/Inkling-2-Mini");
    if (!directory && directory.error().code == ErrorCode::IO) GTEST_SKIP() << directory.error().message;
    ASSERT_TRUE(directory.has_value()) << directory.error().message;
    const auto config = require(checkpoint::load_config(*directory));
    const auto started = std::chrono::steady_clock::now();
    auto model = require(model::create_model_as<model::CausalLanguageModel>(
        config["model"], {.device = tensor::Device::cuda(), .parameter_dtype = tensor::DType::BF16}));
    require(model->set_checkpoint(require(checkpoint::Weights::load(config["weights_file"].as<std::string>()))));
    std::cout << "loaded in " << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
              << " s\n";
    const auto tokenizer = require(text::Tokenizer::load(config["tokenizer_file"].as<std::string>()));
    const auto prompt = require(tokenizer.encode("The capital of France is Paris. The capital of Japan is"));
    const auto vocabulary = static_cast<std::size_t>(model->vocabulary_size());

    // Whole-prompt prefill must agree with token-by-token decoding through the caches.
    auto full_state = require(model->create_state(256));
    const auto full = host_logits(require(model->forward(prompt, *full_state, true)));
    require(model->rewind(*full_state, 0));
    const auto recycled = host_logits(require(model->forward(prompt, *full_state, true)));
    ASSERT_TRUE(
        recycled.size() == full.size() &&
        std::ranges::equal(recycled, full, [](float left, float right) { return std::abs(left - right) < 1e-5F; }))
        << "rewound state changed the prompt logits";
    if (const auto* dump = std::getenv("KIDI_INKLING2_DUMP")) {
        std::ofstream file(dump, std::ios::binary);
        file.write(reinterpret_cast<const char*>(full.data()), static_cast<std::streamsize>(full.size() * 4));
    }
    auto step_state = require(model->create_state(256));
    float worst = 0;
    for (std::size_t index = 0; index < prompt.size(); ++index) {
        const auto step = host_logits(require(model->forward(std::span(prompt).subspan(index, 1), *step_state)));
        for (std::size_t column = 0; column < vocabulary; ++column)
            worst = std::max(worst, std::abs(step[column] - full[index * vocabulary + column]));
        ASSERT_EQ(argmax(step), argmax(std::span(full).subspan(index * vocabulary, vocabulary)))
            << "incremental decoding changed the top token at position " << index;
    }
    std::cout << "prefill vs incremental max |logit diff| = " << worst << '\n';

    std::vector<std::int32_t> generated;
    auto next = argmax(std::span(full).subspan((prompt.size() - 1) * vocabulary, vocabulary));
    const auto decode_started = std::chrono::steady_clock::now();
    for (int count = 0; count < 16 && next != 200006; ++count) {
        generated.push_back(next);
        next = argmax(host_logits(require(model->forward(std::span(&generated.back(), 1), *full_state))));
    }
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_started).count();
    std::cout << "greedy:";
    for (const auto token : generated) std::cout << ' ' << token;
    std::cout << "\ntext: " << require(tokenizer.decode(generated)) << '\n'
              << "decode: " << static_cast<double>(generated.size()) / seconds << " tokens/s\n";
}

TEST(Inkling2Test, CompactWeightsMatchBfloatReference) {
    const auto directory = checkpoint::Hub{}.resolve("@thinkingmachines/Inkling-2-Mini");
    if (!directory && directory.error().code == ErrorCode::IO) GTEST_SKIP() << directory.error().message;
    ASSERT_TRUE(directory.has_value()) << directory.error().message;
    const auto config = require(checkpoint::load_config(*directory));
    const auto weights = require(checkpoint::Weights::load(config["weights_file"].as<std::string>()));
    const auto tokenizer = require(text::Tokenizer::load(config["tokenizer_file"].as<std::string>()));
    const auto prompt = require(tokenizer.encode("The capital of France is Paris. The capital of Japan is"));
    std::vector<float> reference;
    {
        auto model = require(model::create_model_as<model::CausalLanguageModel>(
            config["model"], {.device = tensor::Device::cuda(), .parameter_dtype = tensor::DType::BF16}));
        require(model->set_checkpoint(weights, {.precision = core::InferencePrecision::BF16}));
        auto state = require(model->create_state(256));
        reference = host_logits(require(model->forward(prompt, *state, true)));
    }
    for (const auto precision : {core::InferencePrecision::CHECKPOINT, core::InferencePrecision::Q8A16}) {
        auto model = require(model::create_model_as<model::CausalLanguageModel>(
            config["model"], {.device = tensor::Device::cuda(), .parameter_dtype = tensor::DType::BF16}));
        const auto conflicting =
            model->set_checkpoint(weights, {.weight_bits = 8, .precision = core::InferencePrecision::BF16});
        ASSERT_FALSE(conflicting);
        EXPECT_EQ(conflicting.error().code, ErrorCode::INVALID_ARGUMENT);
        require(model->set_checkpoint(weights, {.precision = precision}));
        auto state = require(model->create_state(256));
        const auto actual = host_logits(require(model->forward(prompt, *state, true)));
        ASSERT_EQ(actual.size(), reference.size());
        const auto vocabulary = static_cast<std::size_t>(model->vocabulary_size());
        double squared_error = 0, squared_reference = 0;
        std::size_t matches = 0;
        for (std::size_t index = 0; index < actual.size(); ++index) {
            ASSERT_TRUE(std::isfinite(actual[index]));
            const auto difference = static_cast<double>(actual[index]) - reference[index];
            squared_error += difference * difference;
            squared_reference += static_cast<double>(reference[index]) * reference[index];
        }
        for (std::size_t index = 0; index < prompt.size(); ++index)
            matches += argmax(std::span(actual).subspan(index * vocabulary, vocabulary)) ==
                       argmax(std::span(reference).subspan(index * vocabulary, vocabulary));
        const auto relative_error = std::sqrt(squared_error / squared_reference);
        std::cout << (precision == core::InferencePrecision::CHECKPOINT ? "Native" : "INT8")
                  << " vs BF16: relative RMS error=" << relative_error << ", top-token agreement=" << matches << '/'
                  << prompt.size() << '\n';
        EXPECT_LT(relative_error, 0.05);
        EXPECT_GE(matches * 10, prompt.size() * 9);
        EXPECT_EQ(argmax(std::span(actual).last(vocabulary)), argmax(std::span(reference).last(vocabulary)));

        const auto last_layer = config["model"]["num_hidden_layers"].as<std::int32_t>() - 1;
        const std::array mappings{checkpoint::StateMappingSpec{
            .sources = {"model\\.llm\\.layers\\." + std::to_string(last_layer) + "\\.mlp\\.experts\\.w2_weight"},
            .op = checkpoint::StateMappingOp::IGNORE}};
        const auto incomplete = require(weights.mapped(mappings));
        ASSERT_FALSE(model->set_checkpoint(incomplete, {.precision = precision}));
        auto retained_state = require(model->create_state(256));
        const auto retained = host_logits(require(model->forward(prompt, *retained_state, true)));
        EXPECT_TRUE(std::ranges::equal(retained, actual)) << "failed checkpoint replacement changed the loaded model";
    }
}

// A retained prefix whose recurrent state cannot rewind to the shared token count must fall back to a fresh start
// instead of failing the serving loop, and must not change the answer.
TEST(Inkling2Test, PrefixReuseFallback) {
    const auto directory = checkpoint::Hub{}.resolve("@thinkingmachines/Inkling-2-Mini");
    if (!directory && directory.error().code == ErrorCode::IO) GTEST_SKIP() << directory.error().message;
    ASSERT_TRUE(directory.has_value()) << directory.error().message;
    auto generator = require(inference::Generator::load(*directory, {.device = tensor::Device::cuda()}));
    require(generator.configure_serving({.maximum_active = 1, .maximum_requests = 1, .cache_token_budget = 512}));
    const auto fresh = chat(generator, "What color is the sky?");
    (void)chat(generator, "What color is grass?");
    const auto reused = chat(generator, "What color is the sky?");
    EXPECT_TRUE(reused.text == fresh.text && reused.stats.reused_prompt_tokens == 0)
        << "prefix reuse changed the answer or reused a non-rewindable prefix";
    std::cout << "prefix fallback: " << reused.text << '\n';
}
} // namespace
