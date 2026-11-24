#include "kidi/checkpoint/config.h"
#include "kidi/checkpoint/hub.h"
#include "kidi/checkpoint/weights.h"
#include "kidi/model/inkling2.h"
#include "kidi/model/registry.h"
#include "kidi/ops/context.h"
#include "kidi/runtime/ynn/graph.h"
#include "kidi/text/tokenizer.h"

#include <algorithm>
#include <cmath>
#include <iostream>

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
    // The default CPU policy must pack experts even without an explicit --precision q8a16.
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
}
} // namespace
