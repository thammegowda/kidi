#include "kidi/model/gemma4.h"
#include "kidi/inference/decoder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>

#include <gtest/gtest.h>

namespace {
using namespace kidi;
using ops::require;
auto snapshot(const tensor::Tensor& tensor) -> tensor::Tensor {
    return require(tensor::Tensor::from_bytes({tensor.shape().begin(), tensor.shape().end()}, tensor.dtype(),
                                              require(tensor.copy_to_host())));
}
auto close(const tensor::Tensor& left, const tensor::Tensor& right) -> ::testing::AssertionResult {
    if (!std::ranges::equal(left.shape(), right.shape())) return ::testing::AssertionFailure() << "shapes differ";
    const auto a = require(left.data<float>()), b = require(right.data<float>());
    for (std::size_t index = 0; index < a.size(); ++index)
        if (!std::isfinite(a[index]) || !std::isfinite(b[index]) || std::abs(a[index] - b[index]) > 2e-4F)
            return ::testing::AssertionFailure() << "mismatch at " << index << ": " << a[index] << " != " << b[index];
    return ::testing::AssertionSuccess();
}
void check_rounds(model::Gemma4Impl& model, std::span<const std::int32_t> prompt, std::size_t vocabulary) {
    constexpr std::size_t MAXIMUM = 10;
    const inference::SearchOptions options{.vocabulary_size = vocabulary,
                                           .end_id = 1,
                                           .pad_id = 0,
                                           .maximum_steps = MAXIMUM,
                                           .compute_score = false,
                                           .stop_on_eos = false};
    auto reference_state = require(model.create_state(16));
    auto reference = require(inference::GreedyState::create(options));
    require(
        reference.accept({{}, inference::ScoreKind::LOGITS, require(model.forward_token(prompt, *reference_state))}));
    while (!reference.finished()) {
        const auto token = reference.token();
        require(reference.accept(
            {{}, inference::ScoreKind::LOGITS, require(model.forward_token(std::span(&token, 1), *reference_state))}));
    }
    const auto& expected = reference.result().token_ids;
    for (int mode = 0; mode < 3; ++mode) {
        SCOPED_TRACE(mode);
        auto state = require(model.create_state(16));
        tensor::Tensor hidden, verified_logits;
        const auto first_logits = require(model.forward(prompt, *state, false, &hidden));
        std::size_t verify_position = 0;
        inference::SpeculativeStats stats;
        std::size_t generated = 1;
        const auto generation = require(inference::Decoder::generate_speculative(
            {require(first_logits.data<float>())}, options, 3,
            [&](std::int32_t, std::size_t limit, const inference::GreedyState::StopPredicate&,
                std::vector<std::int32_t>& drafts) -> Result<void> {
                for (std::size_t index = 0; index < limit; ++index) {
                    const auto token = expected[generated + index];
                    drafts.push_back(mode == 0 || (mode == 2 && index == 0)
                                         ? token
                                         : static_cast<std::int32_t>((token + 1) % vocabulary));
                }
                return {};
            },
            [&](std::span<const std::int32_t> inputs) -> Result<inference::TokenScores> {
                verify_position = state->sequence_position();
                auto logits = model.forward(inputs, *state, true, &hidden);
                if (!logits) return std::unexpected(logits.error());
                verified_logits = std::move(*logits);
                return inference::TokenScores{require(verified_logits.data<float>())};
            },
            [&](std::size_t committed_inputs) -> Result<void> {
                generated += committed_inputs;
                return model.rewind(*state, verify_position + committed_inputs);
            },
            &stats));
        ASSERT_EQ(generation.token_ids, expected);
        ASSERT_TRUE(stats.verification_steps);
        if (mode == 0) {
            ASSERT_EQ(stats.accepted_draft_tokens, stats.draft_tokens);
        }
        if (mode == 1) {
            ASSERT_EQ(stats.accepted_draft_tokens, 0U);
        }
    }
}
void check(const std::filesystem::path& directory) {
    SCOPED_TRACE(directory.filename().string());
    const auto config = YAML::LoadFile((directory / "config.json").string());
    const auto checkpoint = require(checkpoint::Weights::load(directory / "model.safetensors"));
    const ModuleScope construction(tensor::DType::F32, false, tensor::Device::cpu());
    auto model = require(model::Gemma4Impl::create(config));
    require(model->set_checkpoint(checkpoint));
    const std::array<std::int32_t, 3> prompt{2, 5, 7};
    auto prefix_ptr = require(model->create_state(16));
    auto& prefix = static_cast<model::Gemma4State&>(*prefix_ptr);
    for (const auto& layer : prefix.layers)
        if (layer.key.dtype() == tensor::DType::I8) {
            ASSERT_EQ(layer.key_quantization.scales.size(), 1U);
            ASSERT_EQ(layer.value_quantization.scales.size(), 1U);
            ASSERT_GT(layer.key_quantization.scales[0], 0);
            ASSERT_GT(layer.value_quantization.scales[0], 0);
        }
    tensor::Tensor features_hidden;
    const auto features_logits = require(model->forward(prompt, prefix, true, &features_hidden));
    const auto feature_logits = snapshot(features_logits);
    ASSERT_EQ(features_hidden.dimensions(), 3U);
    ASSERT_EQ(features_hidden.size(1), prompt.size());
    ASSERT_EQ(features_hidden.size(2), config["hidden_size"].as<std::size_t>());
    ASSERT_EQ(features_logits.size(1), prompt.size());
    ASSERT_EQ(prefix.position, prompt.size());
    auto oracle = require(model->create_state(16));
    ASSERT_TRUE(close(feature_logits, require(model->forward(prompt, *oracle, true))));
    const std::array<std::int32_t, 4> proposed{3, 8, 6, 9};
    const std::array<std::int32_t, 1> correction{11};
    for (std::size_t committed = 1; committed <= proposed.size(); ++committed) {
        SCOPED_TRACE(committed);
        auto speculative_ptr = require(model->fork_state(prefix, prompt.size(), 16));
        auto serial_ptr = require(model->fork_state(prefix, prompt.size(), 16));
        auto& speculative = *speculative_ptr;
        auto& serial = *serial_ptr;
        tensor::Tensor verification_hidden;
        const auto verification_logits = require(model->forward(proposed, speculative, true, &verification_hidden));
        ASSERT_EQ(verification_hidden.size(1), proposed.size());
        ASSERT_EQ(verification_logits.size(1), proposed.size());
        ASSERT_EQ(speculative.sequence_position(), prompt.size() + proposed.size());
        require(model->rewind(speculative, prompt.size() + committed));
        for (const auto token : std::span(proposed).first(committed))
            require(model->forward(std::span(&token, 1), serial));
        const auto actual = snapshot(require(model->forward(correction, speculative)));
        const auto expected = require(model->forward(correction, serial));
        ASSERT_TRUE(close(actual, expected));
        ASSERT_EQ(speculative.sequence_position(), serial.sequence_position());
    }
    const auto position = prefix.position;
    ASSERT_FALSE(model->rewind(prefix, position + 1));
    ASSERT_EQ(prefix.position, position);
    require(model->rewind(prefix, 0));
    tensor::Tensor replayed_hidden;
    const auto replayed_logits = snapshot(require(model->forward(prompt, prefix, false, &replayed_hidden)));
    auto fresh = require(model->create_state(16));
    const auto expected = require(model->forward(prompt, *fresh));
    ASSERT_EQ(replayed_hidden.size(1), 1U);
    ASSERT_TRUE(close(replayed_logits, expected));
    tensor::Tensor one_hidden;
    const auto one_logits = require(model->forward(correction, prefix, true, &one_hidden));
    ASSERT_TRUE(one_hidden.defined());
    ASSERT_EQ(one_hidden.size(1), 1U);
    ASSERT_EQ(one_logits.size(1), 1U);
    model::Gemma4State invalid;
    ASSERT_FALSE(model->rewind(invalid, 0));
    ASSERT_FALSE(model->forward({}, prefix, true, &one_hidden));
    ASSERT_NO_FATAL_FAILURE(check_rounds(*model, prompt, config["vocab_size"].as<std::size_t>()));
    if (!config["quantization_config"]) {
        auto packed = require(model::Gemma4Impl::create(config));
        require(packed->set_checkpoint(checkpoint, 4, 4, false));
        SCOPED_TRACE("packed");
        ASSERT_NO_FATAL_FAILURE(check_rounds(*packed, prompt, config["vocab_size"].as<std::size_t>()));
    }
}

TEST(Gemma4MtpTest, FeaturesAndCacheRewind) { check(std::filesystem::path(KIDI_GEMMA4_FIXTURE)); }

TEST(Gemma4MtpTest, QatFeaturesAndCacheRewind) {
    check(std::filesystem::path(KIDI_GEMMA4_FIXTURE).parent_path() / "gemma4-qat");
}
} // namespace
