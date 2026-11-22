#include "kidi/inference/generator.h"
#include "kidi/model/gemma4_assistant.h"
#include "kidi/search/hnsw.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <gtest/gtest.h>

namespace {
using namespace kidi;
using ops::require;
auto close(const tensor::Tensor& actual, const tensor::Tensor& expected) -> ::testing::AssertionResult {
    if (!std::ranges::equal(actual.shape(), expected.shape())) return ::testing::AssertionFailure() << "shapes differ";
    const auto left = require(actual.data<float>()), right = require(expected.data<float>());
    for (std::size_t index = 0; index < left.size(); ++index)
        if (!std::isfinite(left[index]) || std::abs(left[index] - right[index]) > 2e-4F)
            return ::testing::AssertionFailure()
                   << "assistant activation mismatch at " << index << ": " << left[index] << " != " << right[index];
    return ::testing::AssertionSuccess();
}
auto row(const tensor::Tensor& value, std::size_t index) -> tensor::Tensor {
    const auto rows = require(value.select(0, 0));
    const auto selected = require(rows.select(0, index));
    return require(selected.reshape({1, 1, static_cast<std::int64_t>(selected.size(0))}));
}
void check_reference(const std::filesystem::path& target_directory, const std::filesystem::path& directory) {
    const auto config = YAML::LoadFile((target_directory / "config.json").string());
    const auto assistant_config = YAML::LoadFile((directory / "config.json").string());
    const auto reference = require(checkpoint::Weights::load(directory / "reference.safetensors"));
    const auto tokens = require(reference.tensor("tokens"));
    const auto input = require(tokens.data<std::int32_t>());
    const ModuleScope construction(tensor::DType::F32, false, tensor::Device::cpu());
    auto target = require(model::Gemma4Impl::create(config));
    require(target->set_checkpoint(require(checkpoint::Weights::load(target_directory / "model.safetensors"))));
    auto state_ptr = require(target->create_state(16));
    auto& state = static_cast<model::Gemma4State&>(*state_ptr);
    tensor::Tensor target_hidden;
    require(target->forward(input, state, true, &target_hidden));
    ASSERT_TRUE(close(target_hidden, require(reference.tensor("target_hidden"))));
    std::vector<std::vector<std::byte>> caches;
    for (const auto& layer : state.layers) {
        caches.push_back(require(layer.key.copy_to_host()));
        caches.push_back(require(layer.value.copy_to_host()));
    }
    auto assistant = require(model::Gemma4AssistantImpl::create(assistant_config, config));
    require(assistant->set_checkpoint(require(checkpoint::Weights::load(directory / "model.safetensors"))));
    for (const bool ordered : {true, false}) {
        SCOPED_TRACE(ordered ? "ordered" : "dense");
        auto current = assistant;
        if (!ordered) {
            auto dense = YAML::Clone(assistant_config);
            dense["use_ordered_embeddings"] = false;
            current = require(model::Gemma4AssistantImpl::create(dense, config));
            auto weights = assistant->state_dict();
            std::erase_if(weights, [](const auto& entry) { return entry.first.starts_with("masked_embedding."); });
            require(current->set_state(weights));
        }
        const auto suffix = ordered ? std::string{} : "_dense";
        const auto expected_tokens_tensor = require(reference.tensor("draft_tokens" + suffix));
        const auto expected_tokens = require(expected_tokens_tensor.data<std::int32_t>());
        const auto expected_hidden = require(reference.tensor("draft_hidden" + suffix));
        const auto first = require(reference.tensor("first_token"));
        auto token = require(first.data<std::int32_t>())[0];
        auto hidden = row(target_hidden, input.size() - 1);
        for (std::size_t step = 0; step < expected_tokens.size(); ++step) {
            auto output = require(current->forward(*target, hidden, token, state));
            ASSERT_EQ(output.token, expected_tokens[step])
                << "assistant token mismatch, ordered=" << ordered << ", step=" << step;
            ASSERT_TRUE(close(output.hidden, row(expected_hidden, step))) << "step " << step;
            ASSERT_EQ(state.position, input.size()) << "step " << step;
            token = output.token;
            hidden = std::move(output.hidden);
        }
    }
    std::size_t index = 0;
    for (const auto& layer : state.layers)
        for (const auto* tensor : {&layer.key, &layer.value})
            ASSERT_EQ(require(tensor->copy_to_host()), caches[index++]) << "target cache " << index - 1 << " changed";
    auto invalid = YAML::Clone(assistant_config);
    invalid["centroid_intermediate_top_k"] = 5;
    EXPECT_FALSE(model::Gemma4AssistantImpl::create(invalid, config)) << "centroid_intermediate_top_k";
    invalid = YAML::Clone(assistant_config);
    invalid["backbone_hidden_size"] = 9;
    EXPECT_FALSE(model::Gemma4AssistantImpl::create(invalid, config)) << "backbone_hidden_size";
    auto empty_ptr = require(target->create_state(16));
    auto& empty = static_cast<model::Gemma4State&>(*empty_ptr);
    EXPECT_FALSE(assistant->forward(*target, row(target_hidden, 0), 2, empty));
    EXPECT_FALSE(assistant->forward(*target, row(target_hidden, 0), 16, state));
}
void check_generation(const std::filesystem::path& fixture, const std::filesystem::path& assistant) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = std::filesystem::temp_directory_path() / ("kidi-mtp-" + std::to_string(stamp));
    ASSERT_TRUE(std::filesystem::create_directory(directory)) << "cannot create MTP test directory";
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code error;
            std::filesystem::remove_all(path, error);
            if (error) std::cerr << "MTP test cleanup failed: " << error.message() << '\n';
        }
    } cleanup{directory};
    std::filesystem::copy_file(fixture / "model.safetensors", directory / "model.safetensors");
    auto config = YAML::LoadFile((fixture / "config.json").string());
    config["decode"]["context_size"] = 16;
    config["decode"]["maximum_new_tokens"] = 10;
    std::ofstream(directory / "config.json") << config;
    std::ofstream(directory / "tokenizer.json") << R"({
      "version":"1.0", "pre_tokenizer":{"type":"WhitespaceSplit"},
      "decoder":{"type":"WordPiece","prefix":"##","cleanup":false},
      "model":{"type":"WordLevel","unk_token":"<unk>","vocab":{
        "<pad>":0,"<eos>":1,"<bos>":2,"<turn|>":3,"<|turn>":4,"<unk>":5,
        "alpha":6,"beta":7,"gamma":8,"delta":9,"theta":10,"zeta":11,"eta":12,"iota":13,"kappa":14,"lambda":15}}
    })";
    const auto qat = static_cast<bool>(config["quantization_config"]);
    for (const auto cache_precision : qat ? std::array{core::KVCachePrecision::FP32, core::KVCachePrecision::INT8}
                                          : std::array{core::KVCachePrecision::FP32, core::KVCachePrecision::FP32}) {
        SCOPED_TRACE(core::to_string(cache_precision));
        auto baseline = require(
            inference::Generator::load(directory, {.kv_cache_precision = cache_precision, .dense_lm_head = true}));
        auto speculative = require(inference::Generator::load(directory, {.kv_cache_precision = cache_precision,
                                                                          .dense_lm_head = true,
                                                                          .assistant_model = assistant,
                                                                          .draft_tokens = 3}));
        ASSERT_TRUE(speculative.speculative());
        ASSERT_FALSE(baseline.speculative());
        ASSERT_EQ(speculative.kv_cache_precision(), cache_precision);
        for (const auto prompt : {"alpha beta gamma", "<eos>"})
            for (const bool ignore_eos : {true, false})
                for (const std::size_t maximum : {1, 2, 10}) {
                    const inference::GenerationOptions options{.maximum_new_tokens = maximum,
                                                               .context_size = 16,
                                                               .prefill_chunk_size = 4,
                                                               .raw_prompt = true,
                                                               .ignore_eos = ignore_eos};
                    const auto expected = require(baseline.generate(prompt, options));
                    const auto actual = require(speculative.generate(prompt, options));
                    SCOPED_TRACE(::testing::Message() << "generation mismatch: " << fixture.filename() << ", cache="
                                                      << core::to_string(cache_precision) << ", prompt=" << prompt
                                                      << ", limit=" << maximum << ", ignore_eos=" << ignore_eos);
                    ASSERT_EQ(actual.generation.token_ids, expected.generation.token_ids);
                    ASSERT_EQ(actual.generation.decoder_steps, expected.generation.decoder_steps);
                    ASSERT_EQ(actual.generation.truncated, expected.generation.truncated);
                    ASSERT_EQ(actual.text, expected.text);
                    ASSERT_TRUE(actual.stats.speculative);
                }
    }
    ASSERT_FALSE(inference::Generator::load(
        directory,
        {.kv_cache_precision = core::KVCachePrecision::E4M3, .dense_lm_head = true, .assistant_model = assistant}));
    ASSERT_FALSE(inference::Generator::load(directory,
                                            {.dense_lm_head = true, .assistant_model = assistant, .draft_tokens = 17}));
    auto speculative = require(inference::Generator::load(
        directory, {.dense_lm_head = true, .assistant_model = assistant, .draft_tokens = 3}));
    ASSERT_FALSE(speculative.configure_serving({.maximum_active = 2}));
    ASSERT_FALSE(speculative.configure_serving({.maximum_active = 1, .deferred_tokens = true}));
    require(speculative.configure_serving(
        {.maximum_active = 1, .maximum_requests = 2, .cache_token_budget = 16, .prefill_tokens_per_step = 4}));
    const inference::GenerationOptions streaming{.maximum_new_tokens = 10,
                                                 .context_size = 16,
                                                 .prefill_chunk_size = 4,
                                                 .raw_prompt = true,
                                                 .ignore_eos = true,
                                                 .stream_text = true};
    require(speculative.enqueue("alpha beta gamma", streaming));
    std::string text;
    std::vector<std::int32_t> emitted;
    std::optional<inference::TextGeneration> completed;
    while (speculative.pending_requests())
        for (auto& event : require(speculative.step()).events) {
            text += event.text;
            if (event.token) emitted.push_back(*event.token);
            if (event.completed) completed = std::move(event.completed);
        }
    ASSERT_TRUE(completed);
    ASSERT_EQ(emitted, completed->generation.token_ids);
    ASSERT_EQ(text, completed->text);
    ASSERT_TRUE(completed->stats.verification_steps);
    ASSERT_TRUE(completed->stats.draft_tokens);
    require(speculative.release_workspaces());
    ASSERT_FALSE(speculative.generate_batch(std::array<std::string, 1>{"alpha"}));
    if (!qat) {
        const ModuleScope construction(tensor::DType::F32, false, tensor::Device::cpu());
        auto target = require(model::Gemma4Impl::create(config));
        require(target->set_checkpoint(require(checkpoint::Weights::load(directory / "model.safetensors"))));
        const auto weights = target->state_dict().at("embed_tokens.weight");
        auto index = require(search::HnswIndex::build(weights, {.m = 4,
                                                                .ef_construction = weights.size(0),
                                                                .seed = 42,
                                                                .candidates = weights.size(0),
                                                                .ef_search = weights.size(0)}));
        require(index.save(directory / "lm_head.hnsw"));
        auto dense = require(inference::Generator::load(directory, {.dense_lm_head = true}));
        auto indexed =
            require(inference::Generator::load(directory, {.assistant_model = assistant, .draft_tokens = 3}));
        const inference::GenerationOptions options{.maximum_new_tokens = 10,
                                                   .context_size = 16,
                                                   .prefill_chunk_size = 4,
                                                   .raw_prompt = true,
                                                   .ignore_eos = true};
        const auto expected = require(dense.generate("alpha beta gamma", options));
        const auto actual = require(indexed.generate("alpha beta gamma", options));
        ASSERT_EQ(actual.generation.token_ids, expected.generation.token_ids);
        ASSERT_EQ(actual.text, expected.text);
        ASSERT_TRUE(actual.stats.speculative);
    }
}

auto target_fixture() -> std::filesystem::path { return KIDI_GEMMA4_FIXTURE; }
auto assistant_fixture() -> std::filesystem::path { return target_fixture().parent_path() / "gemma4-assistant"; }

TEST(Gemma4DraftTest, SharesTargetCache) {
    check_reference(target_fixture(), assistant_fixture());
}

TEST(Gemma4DraftTest, MatchesGreedy) {
    check_generation(target_fixture(), assistant_fixture());
}

TEST(Gemma4DraftTest, QatMatchesGreedy) {
    check_generation(target_fixture().parent_path() / "gemma4-qat", assistant_fixture());
}
} // namespace
