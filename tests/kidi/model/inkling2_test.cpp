// Inkling-2 checks on a checkpoint in the default local model hub; skips when missing.
// Optionally set KIDI_INKLING2_DUMP to write the prompt's FP32 logits [tokens, vocabulary] for reference comparison
// (tools/reference/inkling2.py).
#include "kidi/checkpoint/config.h"
#include "kidi/checkpoint/hub.h"
#include "kidi/checkpoint/weights.h"
#include "kidi/inference/generator.h"
#include "kidi/model/inkling2.h"
#include "kidi/model/registry.h"
#include "kidi/text/tokenizer.h"

#include <algorithm>
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
