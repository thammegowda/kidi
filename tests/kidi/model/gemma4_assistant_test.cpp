#include "kidi/inference/generator.h"
#include "kidi/search/hnsw.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
using namespace kidi;
using ops::require;
auto close(const tensor::Tensor& actual, const tensor::Tensor& expected) -> bool {
    if (!std::ranges::equal(actual.shape(), expected.shape())) return false;
    const auto left = require(actual.data<float>()), right = require(expected.data<float>());
    for (std::size_t index = 0; index < left.size(); ++index)
        if (!std::isfinite(left[index]) || std::abs(left[index] - right[index]) > 2e-4F) {
            std::cerr << "assistant activation mismatch at " << index << ": " << left[index] << " != " << right[index]
                      << '\n';
            return false;
        }
    return true;
}
auto row(const tensor::Tensor& value, std::size_t index) -> tensor::Tensor {
    const auto rows = require(value.select(0, 0));
    const auto selected = require(rows.select(0, index));
    return require(selected.reshape({1, 1, static_cast<std::int64_t>(selected.size(0))}));
}
auto check_reference(const std::filesystem::path& target_directory, const std::filesystem::path& directory) -> bool {
    const auto config = YAML::LoadFile((target_directory / "model.yaml").string())["model"];
    const auto assistant_config = YAML::LoadFile((directory / "config.json").string());
    const auto reference = require(checkpoint::Weights::load(directory / "reference.safetensors"));
    const auto tokens = require(reference.tensor("tokens"));
    const auto input = require(tokens.data<std::int32_t>());
    const ModuleScope construction(tensor::DType::F32, false, tensor::Device::cpu());
    auto target = require(model::Gemma4Impl::create(config));
    require(target->set_checkpoint(require(checkpoint::Weights::load(target_directory / "model.safetensors"))));
    auto state = require(target->create_state(16));
    const auto features = require(target->forward_features(input, state));
    if (!close(features.hidden, require(reference.tensor("target_hidden")))) return false;
    std::vector<std::vector<std::byte>> caches;
    for (const auto& layer : state.layers) {
        caches.push_back(require(layer.key.copy_to_host()));
        caches.push_back(require(layer.value.copy_to_host()));
    }
    auto assistant = require(model::Gemma4AssistantImpl::create(assistant_config, config));
    require(assistant->set_checkpoint(require(checkpoint::Weights::load(directory / "model.safetensors"))));
    for (const bool ordered : {true, false}) {
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
        auto hidden = row(features.hidden, input.size() - 1);
        for (std::size_t step = 0; step < expected_tokens.size(); ++step) {
            auto output = require(current->forward(*target, hidden, token, state));
            if (output.token != expected_tokens[step]) {
                std::cerr << "assistant token mismatch, ordered=" << ordered << ", step=" << step << ": "
                          << output.token << " != " << expected_tokens[step] << '\n';
                return false;
            }
            if (!close(output.hidden, row(expected_hidden, step)) || state.position != input.size()) return false;
            token = output.token;
            hidden = std::move(output.hidden);
        }
    }
    std::size_t index = 0;
    for (const auto& layer : state.layers)
        for (const auto* tensor : {&layer.key, &layer.value})
            if (require(tensor->copy_to_host()) != caches[index++]) return false;
    auto invalid = YAML::Clone(assistant_config);
    invalid["centroid_intermediate_top_k"] = 5;
    if (model::Gemma4AssistantImpl::create(invalid, config)) return false;
    invalid = YAML::Clone(assistant_config);
    invalid["backbone_hidden_size"] = 9;
    if (model::Gemma4AssistantImpl::create(invalid, config)) return false;
    auto empty = require(target->create_state(16));
    return !assistant->forward(*target, row(features.hidden, 0), 2, empty) &&
           !assistant->forward(*target, row(features.hidden, 0), 16, state);
}
auto check_generation(const std::filesystem::path& fixture, const std::filesystem::path& assistant) -> bool {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = std::filesystem::temp_directory_path() / ("kidi-mtp-" + std::to_string(stamp));
    if (!std::filesystem::create_directory(directory)) throw std::runtime_error("cannot create MTP test directory");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code error;
            std::filesystem::remove_all(path, error);
            if (error) std::cerr << "MTP test cleanup failed: " << error.message() << '\n';
        }
    } cleanup{directory};
    std::filesystem::copy_file(fixture / "model.safetensors", directory / "model.safetensors");
    auto config = YAML::LoadFile((fixture / "model.yaml").string());
    config["format_version"] = 1;
    config["weights_file"] = "model.safetensors";
    config["tokenizer_file"] = "tokenizer.json";
    config["decode"]["context_size"] = 16;
    config["decode"]["maximum_new_tokens"] = 10;
    std::ofstream(directory / "model.yaml") << config;
    std::ofstream(directory / "tokenizer.json") << R"({
      "version":"1.0", "pre_tokenizer":{"type":"WhitespaceSplit"},
      "decoder":{"type":"WordPiece","prefix":"##","cleanup":false},
      "model":{"type":"WordLevel","unk_token":"<unk>","vocab":{
        "<pad>":0,"<eos>":1,"<bos>":2,"<turn|>":3,"<|turn>":4,"<unk>":5,
        "alpha":6,"beta":7,"gamma":8,"delta":9,"theta":10,"zeta":11,"eta":12,"iota":13,"kappa":14,"lambda":15}}
    })";
    const auto qat = static_cast<bool>(config["model"]["quantization_config"]);
    for (const auto cache_precision : qat ? std::array{core::KVCachePrecision::FP32, core::KVCachePrecision::INT8}
                                          : std::array{core::KVCachePrecision::FP32, core::KVCachePrecision::FP32}) {
        auto baseline = require(
            inference::Generator::load(directory, {.kv_cache_precision = cache_precision, .dense_lm_head = true}));
        auto speculative = require(inference::Generator::load(directory, {.kv_cache_precision = cache_precision,
                                                                          .dense_lm_head = true,
                                                                          .assistant_model = assistant,
                                                                          .draft_tokens = 3}));
        if (!speculative.speculative() || baseline.speculative() || speculative.kv_cache_precision() != cache_precision)
            return false;
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
                    if (actual.generation.token_ids != expected.generation.token_ids ||
                        actual.generation.decoder_steps != expected.generation.decoder_steps ||
                        actual.generation.truncated != expected.generation.truncated || actual.text != expected.text ||
                        !actual.stats.speculative) {
                        std::cerr << "generation mismatch: " << fixture.filename()
                                  << ", cache=" << core::to_string(cache_precision) << ", prompt=" << prompt
                                  << ", limit=" << maximum << ", ignore_eos=" << ignore_eos << '\n';
                        std::cerr << "expected:";
                        for (const auto token : expected.generation.token_ids) std::cerr << ' ' << token;
                        std::cerr << "\nactual:";
                        for (const auto token : actual.generation.token_ids) std::cerr << ' ' << token;
                        std::cerr << '\n';
                        return false;
                    }
                }
    }
    if (inference::Generator::load(directory, {.kv_cache_precision = core::KVCachePrecision::E4M3,
                                               .dense_lm_head = true,
                                               .assistant_model = assistant}) ||
        inference::Generator::load(
            directory, {.dense_lm_head = true, .assistant_model = assistant, .draft_tokens = 17}))
        return false;
    auto speculative = require(inference::Generator::load(
        directory, {.dense_lm_head = true, .assistant_model = assistant, .draft_tokens = 3}));
    if (speculative.configure_serving({.maximum_active = 2}) ||
        speculative.configure_serving({.maximum_active = 1, .deferred_tokens = true}))
        return false;
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
    if (!completed || emitted != completed->generation.token_ids || text != completed->text ||
        !completed->stats.verification_steps || !completed->stats.draft_tokens)
        return false;
    require(speculative.release_workspaces());
    if (speculative.generate_batch(std::array<std::string, 1>{"alpha"})) return false;
    if (!qat) {
        const ModuleScope construction(tensor::DType::F32, false, tensor::Device::cpu());
        auto target = require(model::Gemma4Impl::create(config["model"]));
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
        if (actual.generation.token_ids != expected.generation.token_ids || actual.text != expected.text ||
            !actual.stats.speculative)
            return false;
    }
    return true;
}
} // namespace

auto main() -> int {
    try {
        const auto target = std::filesystem::path(KIDI_GEMMA4_FIXTURE);
        const auto assistant = target.parent_path() / "gemma4-assistant";
        if (!check_reference(target, assistant)) {
            std::cerr << "Gemma MTP assistant reference failed\n";
            return 1;
        }
        if (!check_generation(target, assistant) || !check_generation(target.parent_path() / "gemma4-qat", assistant)) {
            std::cerr << "Gemma MTP generation parity failed\n";
            return 1;
        }
        std::cout << "Gemma MTP reference, shared KV, streaming, and greedy parity passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
