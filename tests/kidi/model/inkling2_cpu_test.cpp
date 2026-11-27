#include "kidi/checkpoint/config.h"
#include "kidi/checkpoint/hub.h"
#include "kidi/checkpoint/weights.h"
#include "kidi/inference/generator.h"
#include "kidi/model/inkling2.h"
#include "kidi/model/registry.h"
#include "kidi/ops/context.h"
#include "kidi/runtime/ynn/graph.h"
#include "kidi/text/tokenizer.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <limits>

#include <gtest/gtest.h>

namespace {
using namespace kidi;
using ops::require;

TEST(Inkling2CpuTest, Int8PrefillDecodeAndRewind) {
    const auto directory = checkpoint::Hub{}.resolve("@thinkingmachines/Inkling-2-Mini");
    if (!directory && directory.error().code == ErrorCode::IO) GTEST_SKIP() << directory.error().message;
    ASSERT_TRUE(directory.has_value()) << directory.error().message;
    runtime::ynn::set_thread_count(16);
    const auto config = require(checkpoint::load_config(*directory));
    auto model = require(model::create_model_as<model::CausalLanguageModel>(
        config["model"], {.device = tensor::Device::cpu(), .parameter_dtype = tensor::DType::BF16}));
    // The default CPU policy must pack experts even without an explicit --precision q8.
    require(model->set_checkpoint(require(checkpoint::Weights::load(config["weights_file"].as<std::string>()))));
    const auto tokenizer = require(text::Tokenizer::load(config["tokenizer_file"].as<std::string>()));
    const auto tokens = require(tokenizer.encode("The capital of France is"));
    ASSERT_GT(tokens.size(), 4);
    const auto vocabulary = static_cast<std::size_t>(model->vocabulary_size());
    const auto logits = [&](std::span<const std::int32_t> input, model::LanguageModelState& state) {
        const auto output = require(model->forward(input, state, true));
        const auto data = require(output.data<float>());
        return std::vector<float>(data.begin(), data.end());
    };
    auto full_state = require(model->create_state(32));
    ASSERT_EQ(require(model->kv_cache_dtype(32)), tensor::DType::I8);
    const auto* caches = dynamic_cast<const model::Inkling2State*>(full_state.get());
    ASSERT_NE(caches, nullptr);
    std::size_t bytes = 0;
    for (const auto& visit : caches->visits) {
        ASSERT_EQ(visit.key.dtype(), tensor::DType::I8);
        ASSERT_EQ(visit.value.dtype(), tensor::DType::I8);
        ASSERT_EQ(visit.key_scales.dtype(), tensor::DType::F32);
        for (const auto* tensor : {&visit.key, &visit.value, &visit.key_scales, &visit.value_scales,
                                   &visit.key_conv, &visit.value_conv, &visit.attention_conv, &visit.mlp_conv})
            bytes += tensor->nbytes();
        EXPECT_LT(visit.key.nbytes() + visit.key_scales.nbytes(), visit.key.numel() * 2);
    }
    EXPECT_EQ(full_state->cache_bytes(), bytes);
    const auto full = logits(tokens, *full_state);
    ASSERT_EQ(full.size(), tokens.size() * vocabulary);
    ASSERT_TRUE(std::ranges::all_of(full, [](float value) { return std::isfinite(value); }));
    {
        const auto prepared = require(checkpoint::prepare(*directory, model::Inkling2Impl::int8_preparation));
        auto cached = require(model::create_model_as<model::CausalLanguageModel>(
            config["model"], {.device = tensor::Device::cpu(), .parameter_dtype = tensor::DType::BF16}));
        require(cached->set_checkpoint(require(checkpoint::Weights::load(prepared / "model.safetensors"))));
        auto state = require(cached->create_state(32));
        const auto output = require(cached->forward(tokens, *state, true));
        const auto data = require(output.data<float>());
        EXPECT_EQ(std::vector<float>(data.begin(), data.end()), full);
    }
    auto incremental = require(model->create_state(32));
    double squared_error = 0, squared_reference = 0;
    std::vector<float> last;
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        last = logits(std::span(tokens).subspan(index, 1), *incremental);
        for (std::size_t column = 0; column < vocabulary; ++column) {
            const auto reference = full[index * vocabulary + column];
            const auto difference = static_cast<double>(last[column]) - reference;
            squared_error += difference * difference;
            squared_reference += static_cast<double>(reference) * reference;
        }
    }
    EXPECT_EQ(incremental->sequence_position(), tokens.size());
    const auto relative_error = std::sqrt(squared_error / squared_reference);
    std::cout << "CPU INT8 prefill/decode relative RMS error=" << relative_error << '\n';
    EXPECT_LT(relative_error, 0.03);
    const auto argmax = [](std::span<const float> row) {
        return static_cast<std::int32_t>(std::ranges::max_element(row) - row.begin());
    };
    const auto next = argmax(std::span(full).last(vocabulary));
    EXPECT_EQ(argmax(last), next);
    const std::array next_token{next};
    EXPECT_NE(require(tokenizer.decode(next_token)).find("Paris"), std::string::npos);

    require(model->rewind(*incremental, 0));
    constexpr std::size_t PREFIX = 3;
    require(model->prefill(std::span(tokens).first(PREFIX), *incremental));
    require(model->mark_rewind_point(*incremental));
    const auto tail = std::span(tokens).subspan(PREFIX);
    const auto chunked = logits(tail, *incremental);
    require(model->rewind(*incremental, PREFIX));
    EXPECT_EQ(logits(tail, *incremental), chunked);
    EXPECT_EQ(argmax(std::span(chunked).last(vocabulary)), next);

    // Verification can reject anywhere in a block, including after the convolution ring has wrapped.
    std::vector<std::int32_t> block;
    for (std::size_t i = 0; i < 8; ++i) block.push_back(tokens[i % tokens.size()]);
    for (const std::size_t accepted : {1, 3, 7}) {
        require(model->rewind(*incremental, PREFIX));
        logits(block, *incremental);
        const auto position = PREFIX + accepted;
        EXPECT_EQ(model->rewind_floor(*incremental, position), position);
        require(model->rewind(*incremental, position));
        auto reference = require(model->create_state(32));
        require(model->prefill(std::span(tokens).first(PREFIX), *reference));
        require(model->prefill(std::span(block).first(accepted), *reference));
        const auto actual = logits(next_token, *incremental);
        const auto expected = logits(next_token, *reference);
        double error = 0, norm = 0;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const auto delta = static_cast<double>(actual[i]) - expected[i];
            error += delta * delta;
            norm += static_cast<double>(expected[i]) * expected[i];
        }
        EXPECT_LT(std::sqrt(error / norm), 0.03) << "accepted prefix " << accepted;
    }
    // Verification blocks must not move the prompt rewind point that the next chat turn reuses.
    require(model->rewind(*incremental, PREFIX));
    require(model->prefill(std::span(block).first(2), *incremental));
    logits(block, *incremental);
    EXPECT_EQ(model->rewind_floor(*incremental, PREFIX + 1), PREFIX);
    require(model->rewind(*incremental, PREFIX));
    EXPECT_EQ(logits(tail, *incremental), chunked);
}

// Prefill chunks of 16+ tokens use the INT8 flash attention; decode keeps the legacy float-query kernel. INT8
// activation quantization amplifies any bit-level change (scaling attention outputs by 1+1e-6 already gives top-1
// ~95.6% and mean KL ~9e-3 on ARCHITECTURE.md), so the bounds are that noise floor, not exact agreement.
TEST(Inkling2CpuTest, Int8FlashPrefillTracksDecodeQuality) {
    const auto directory = checkpoint::Hub{}.resolve("@thinkingmachines/Inkling-2-Mini");
    if (!directory && directory.error().code == ErrorCode::IO) GTEST_SKIP() << directory.error().message;
    ASSERT_TRUE(directory.has_value()) << directory.error().message;
    runtime::ynn::set_thread_count(16);
    const auto config = require(checkpoint::load_config(*directory));
    const auto prepared = require(checkpoint::prepare(*directory, model::Inkling2Impl::int8_preparation));
    auto model = require(model::create_model_as<model::CausalLanguageModel>(
        config["model"], {.device = tensor::Device::cpu(), .parameter_dtype = tensor::DType::BF16}));
    require(model->set_checkpoint(require(checkpoint::Weights::load(prepared / "model.safetensors"))));
    const auto tokenizer = require(text::Tokenizer::load(config["tokenizer_file"].as<std::string>()));
    std::stringstream source;
    source << std::ifstream(KIDI_INKLING2_TEXT_FIXTURE).rdbuf();
    // Skip the license header, which the model has memorized.
    auto tokens = require(tokenizer.encode(source.str().substr(20000, 4000)));
    constexpr std::size_t TOKENS = 640, CHUNK = 128;
    ASSERT_GE(tokens.size(), TOKENS);
    tokens.resize(TOKENS);
    const auto vocabulary = static_cast<std::size_t>(model->vocabulary_size());
    const auto logits = [&](std::span<const std::int32_t> input, model::LanguageModelState& state) {
        const auto output = require(model->forward(input, state, true));
        const auto data = require(output.data<float>());
        return std::vector<float>(data.begin(), data.end());
    };
    const auto log_softmax = [](std::span<const float> row) {
        const double maximum = *std::ranges::max_element(row);
        double sum = 0;
        for (const auto value : row) sum += std::exp(value - maximum);
        std::vector<double> out(row.size());
        for (std::size_t i = 0; i < row.size(); ++i) out[i] = row[i] - maximum - std::log(sum);
        return out;
    };
    auto prefill = require(model->create_state(TOKENS));
    auto decode = require(model->create_state(TOKENS));
    double kl = 0, prefill_nll = 0, decode_nll = 0;
    std::size_t agree = 0;
    for (std::size_t begin = 0; begin < TOKENS; begin += CHUNK) {
        const auto chunk = std::span(tokens).subspan(begin, CHUNK);
        const auto batched = logits(chunk, *prefill);
        for (std::size_t i = 0; i < CHUNK; ++i) {
            const auto single = logits(chunk.subspan(i, 1), *decode);
            const auto row = std::span(batched).subspan(i * vocabulary, vocabulary);
            const auto p = log_softmax(single), q = log_softmax(row);
            for (std::size_t v = 0; v < vocabulary; ++v) kl += std::exp(p[v]) * (p[v] - q[v]);
            agree += std::ranges::max_element(single) - single.begin() == std::ranges::max_element(row) - row.begin();
            if (begin + i + 1 < TOKENS) {
                const auto next = static_cast<std::size_t>(tokens[begin + i + 1]);
                decode_nll -= p[next];
                prefill_nll -= q[next];
            }
        }
    }
    const auto agreement = static_cast<double>(agree) / TOKENS;
    kl /= TOKENS;
    std::cout << "INT8 flash prefill vs decode: top1=" << agreement << " meanKL=" << kl
              << " nll prefill=" << prefill_nll / (TOKENS - 1) << " decode=" << decode_nll / (TOKENS - 1) << '\n';
    EXPECT_GE(agreement, 0.93);
    EXPECT_LT(kl, 0.02);
    EXPECT_LT(std::abs(prefill_nll - decode_nll) / (TOKENS - 1), 0.02) << "nats per token";
}

TEST(Inkling2CpuTest, AdaptivePromptLookup) {
    const auto directory = checkpoint::Hub{}.resolve("@thinkingmachines/Inkling-2-Mini");
    if (!directory && directory.error().code == ErrorCode::IO) GTEST_SKIP() << directory.error().message;
    ASSERT_TRUE(directory.has_value()) << directory.error().message;
    runtime::ynn::set_thread_count(16);
    std::string prompt;
    for (int i = 0; i < 8; ++i) prompt += "int value = 0;\n";
    prompt += "int value =";
    const inference::GenerationOptions options{.maximum_new_tokens = 32,
                                               .context_size = 256,
                                               .prefill_chunk_size = 32,
                                               .raw_prompt = true,
                                               .ignore_eos = true};
    std::vector<std::int32_t> expected;
    {
        auto plain = require(inference::Generator::load(*directory, {.dense_lm_head = true}));
        expected = require(plain.generate(prompt, options)).generation.token_ids;
    }
    auto speculative = require(inference::Generator::load(
        *directory, {.dense_lm_head = true, .prompt_lookup = true, .draft_tokens = 8}));
    const auto actual = require(speculative.generate(prompt, options));
    EXPECT_EQ(actual.generation.token_ids, expected);
    EXPECT_TRUE(actual.stats.speculative);
    EXPECT_GT(actual.stats.accepted_draft_tokens, 0);
    // At least one round must grow beyond one draft.
    EXPECT_GT(actual.stats.accepted_draft_tokens, actual.stats.verification_steps);
    EXPECT_LT(actual.stats.verification_steps, expected.size() - 1);
    require(speculative.configure_serving({.maximum_active = 1,
                                           .maximum_requests = 1,
                                           .cache_token_budget = 256,
                                           .prefill_tokens_per_step = 32}));
    auto streaming = options;
    streaming.prefix_cache_bytes = std::numeric_limits<std::size_t>::max();
    streaming.stream_text = true;
    for (int turn = 0; turn < 2; ++turn) {
        require(speculative.enqueue(prompt, streaming));
        std::vector<std::int32_t> emitted;
        std::string text;
        std::optional<inference::TextGeneration> completed;
        while (speculative.pending_requests())
            for (auto& event : require(speculative.step()).events) {
                if (event.token) emitted.push_back(*event.token);
                text += event.text;
                if (event.completed) completed = std::move(event.completed);
            }
        ASSERT_TRUE(completed);
        EXPECT_EQ(emitted, expected);
        EXPECT_EQ(completed->generation.token_ids, expected);
        EXPECT_EQ(text, completed->text);
    }
    auto sampled = options;
    sampled.temperature = 0.6F;
    EXPECT_FALSE(speculative.enqueue(prompt, sampled));
}
} // namespace
