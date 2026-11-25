#include "kidi/graph/compiler.h"
#include "kidi/model/gemma4.h"
#include "kidi/checkpoint/safetensors/mapped.h"
#include "kidi/inference/generator.h"
#include "kidi/runtime/operator.h"
#include "kidi/search/hnsw.h"
#include "kidi/tensor/backend.h"
#include "kidi/tensor/external.h"

#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <utility>
#include <string>

#include <gtest/gtest.h>

namespace {
/// A step compiler that declines every step, so accelerator-shaped steps (captured prefill chunks) replay on CPU.
class DecliningCompiler final : public kidi::graph::Compiler {
public:
    auto name() const -> std::string_view override { return "declining"; }
    auto requires_second_capture(std::string_view) const -> std::optional<bool> override { return false; }
    auto prefill_chunk_size(std::size_t requested) const -> std::size_t override {
        return std::max(requested, std::size_t{16});
    }
    auto compile(const kidi::graph::Graph&, std::string_view key) -> std::unique_ptr<kidi::graph::Executable> override {
        keys.emplace_back(key);
        return nullptr;
    }
    std::vector<std::string> keys;
};

/// Captured prefill chunks and decode steps under a step compiler match eager execution bit for bit.
void check_captured_prefill(const YAML::Node& config, const kidi::checkpoint::Weights& checkpoint,
                            std::span<const std::int32_t> tokens) {
    using kidi::ops::require;
    auto extended = YAML::Clone(config);
    extended["max_position_embeddings"] = 128;
    auto compiler = std::make_shared<DecliningCompiler>();
    std::optional<kidi::model::Gemma4> captured;
    {
        const kidi::ops::StepCompilerScope scope(compiler);
        captured = require(kidi::model::Gemma4Impl::create(extended));
    }
    auto eager = require(kidi::model::Gemma4Impl::create(extended));
    require((*captured)->set_checkpoint(checkpoint));
    require(eager->set_checkpoint(checkpoint));
    ASSERT_EQ((*captured)->accelerator(), "declining") << "step compiler was not attached to the captured model";
    ASSERT_TRUE(eager->accelerator().empty()) << "step compiler was not attached to the captured model";
    std::vector<std::int32_t> sequence(80);
    for (std::size_t index = 0; index < sequence.size(); ++index) sequence[index] = tokens[index % tokens.size()];
    auto captured_state = require((*captured)->create_state(128));
    auto eager_state = require(eager->create_state(128));
    for (std::size_t offset = 0; offset < 64; offset += 16) {
        require((*captured)->prefill(std::span(sequence).subspan(offset, 16), *captured_state));
        require(eager->prefill(std::span(sequence).subspan(offset, 16), *eager_state));
    }
    require((*captured)->prefill(std::span(sequence).subspan(64, 7), *captured_state));
    require(eager->prefill(std::span(sequence).subspan(64, 7), *eager_state));
    ASSERT_EQ(captured_state->sequence_position(), 71U) << "padded prefill advanced by its padded length";
    ASSERT_EQ(eager_state->sequence_position(), 71U) << "padded prefill advanced by its padded length";
    for (std::size_t position = 71; position < sequence.size(); ++position) {
        const auto token = std::span(sequence).subspan(position, 1);
        const auto actual = require((*captured)->forward(token, *captured_state));
        const auto expected = require(eager->forward(token, *eager_state));
        ASSERT_TRUE(std::ranges::equal(require(actual.data<float>()), require(expected.data<float>())))
            << "captured prefill or decode differs from eager at " << position;
    }
    const auto compiled = [&](std::string_view prefix) {
        return std::ranges::any_of(compiler->keys, [&](const std::string& key) { return key.starts_with(prefix); });
    };
    EXPECT_TRUE(compiled("gemma4_prefill:16:") && compiled("gemma4_decode:1:"))
        << "expected captured prefill and decode steps; compiled: " << ::testing::PrintToString(compiler->keys);
}

/// Image embeddings are step inputs, not graph constants: a captured image-prefill shape must rebind a later image.
void check_image_prefill_rebinding(const YAML::Node& config, const kidi::checkpoint::Weights& checkpoint,
                                   std::span<const std::int32_t> tokens) {
    using kidi::ops::require;
    using kidi::tensor::Tensor;
    auto extended = YAML::Clone(config);
    extended["max_position_embeddings"] = 128;
    auto compiler = std::make_shared<DecliningCompiler>();
    std::optional<kidi::model::Gemma4> captured;
    {
        const kidi::ops::StepCompilerScope scope(compiler);
        captured = require(kidi::model::Gemma4Impl::create(extended));
    }
    auto eager = require(kidi::model::Gemma4Impl::create(extended));
    require((*captured)->set_checkpoint(checkpoint));
    require(eager->set_checkpoint(checkpoint));
    const auto hidden = extended["hidden_size"].as<std::int64_t>();
    const auto embeddings = [&](float offset) {
        std::vector<float> values(static_cast<std::size_t>(4 * hidden));
        for (std::size_t index = 0; index < values.size(); ++index)
            values[index] = offset + static_cast<float>(index % static_cast<std::size_t>(hidden)) / hidden;
        return require(Tensor::from_host({1, 4, hidden}, std::span<const float>(values)));
    };
    std::array<std::int32_t, 16> sequence{};
    for (std::size_t index = 0; index < sequence.size(); ++index) sequence[index] = tokens[index % tokens.size()];
    const auto run = [&](kidi::model::Gemma4& model, float offset) {
        auto state = require(model->create_state(128));
        require(state->attach_embeddings({{3, embeddings(offset)}}));
        require(model->prefill(sequence, *state));
        return state;
    };
    const auto first_ptr = run(*captured, 0.25F);
    const auto rebound_ptr = run(*captured, 3.5F);
    const auto expected_ptr = run(eager, 3.5F);
    const auto& first = static_cast<const kidi::model::Gemma4State&>(*first_ptr);
    const auto& rebound = static_cast<const kidi::model::Gemma4State&>(*rebound_ptr);
    const auto& expected = static_cast<const kidi::model::Gemma4State&>(*expected_ptr);
    bool distinct = false;
    for (std::size_t layer = 0; layer < rebound.layers.size(); ++layer) {
        for (const auto pair : {std::pair{&first.layers[layer].key, &rebound.layers[layer].key},
                                std::pair{&first.layers[layer].value, &rebound.layers[layer].value}}) {
            distinct =
                distinct || !std::ranges::equal(require(pair.first->host_bytes()), require(pair.second->host_bytes()));
        }
        for (const auto pair : {std::pair{&rebound.layers[layer].key, &expected.layers[layer].key},
                                std::pair{&rebound.layers[layer].value, &expected.layers[layer].value}})
            ASSERT_TRUE(std::ranges::equal(require(pair.first->host_bytes()), require(pair.second->host_bytes())))
                << "captured image prefill did not rebind layer " << layer;
    }
    ASSERT_TRUE(distinct) << "distinct image embeddings produced identical caches";
    EXPECT_TRUE(std::ranges::any_of(
        compiler->keys,
        [](const std::string& key) { return key.starts_with("gemma4_prefill:16:") && key.ends_with(":images"); }))
        << "expected a captured image prefill step; compiled: " << ::testing::PrintToString(compiler->keys);
}

auto fixture_directory(std::string_view fixture) -> std::filesystem::path {
    const auto* configured = std::getenv("KIDI_GEMMA4_FIXTURE_DIR");
    const auto base =
        configured ? std::filesystem::path(configured) : std::filesystem::path(KIDI_GEMMA4_FIXTURE).parent_path();
    return base.filename() == std::filesystem::path(fixture) ? base : base / fixture;
}

auto write_tokenizer(const std::filesystem::path& path) -> void {
    std::ofstream(path) << R"({
      "version":"1.0", "pre_tokenizer":{"type":"WhitespaceSplit"},
      "decoder":{"type":"WordPiece","prefix":"##","cleanup":false},
      "model":{"type":"WordLevel","unk_token":"<unk>","vocab":{
        "<pad>":0,"<eos>":1,"<bos>":2,"<turn|>":3,"<|turn>":4,"<unk>":5,
        "alpha":6,"beta":7,"gamma":8,"delta":9,"theta":10,"zeta":11,"eta":12,"iota":13,"kappa":14,"lambda":15}}
    })";
}

/// Rows of a host table, counting gathers, standing in for a browser-held per-layer embedding table.
class CountingRows final : public kidi::tensor::RowSource {
public:
    explicit CountingRows(kidi::tensor::Tensor table) : table_(std::move(table)) {}
    auto gather(std::span<const std::int32_t> rows,
                std::span<std::byte> destination) const -> kidi::Result<void> override {
        gathered += rows.size();
        return kidi::tensor::gather_rows(table_, rows, destination);
    }
    mutable std::size_t gathered = 0;

private:
    kidi::tensor::Tensor table_;
};

/// A per-layer embedding table supplied outside the checkpoint generates the same tokens through eager prefill and
/// captured decode steps, and a declared external table without a source fails to load.
auto check_external_rows(const std::filesystem::path& fixture, const YAML::Node& config,
                         kidi::tensor::Device device) -> void {
    using kidi::ops::require;
    const std::string table_name = "model.language_model.embed_tokens_per_layer.weight";
    const auto state = require(require(kidi::checkpoint::Weights::load(fixture / "model.safetensors")).state_dict());
    if (!state.contains(table_name)) return;
    const auto directory = std::filesystem::temp_directory_path() /
                           ("kidi-external-rows-test-" + fixture.filename().string());
    std::filesystem::remove_all(directory);
    for (const auto* name : {"internal", "external"}) std::filesystem::create_directories(directory / name);
    auto reduced = state;
    reduced.erase(table_name);
    require(kidi::checkpoint::Weights::save(directory / "internal" / "model.safetensors", state));
    require(kidi::checkpoint::Weights::save(directory / "external" / "model.safetensors", reduced));
    auto manifest = YAML::Clone(config);
    manifest["decode"]["maximum_new_tokens"] = 4;
    manifest["decode"]["context_size"] = 16;
    std::ofstream(directory / "internal" / "config.json") << manifest;
    const auto& table = state.at(table_name);
    YAML::Node declared;
    declared["name"] = table_name;
    declared["dtype"] = std::string(kidi::tensor::to_string(table.dtype()));
    declared["shape"] = std::vector<std::int64_t>(table.shape().begin(), table.shape().end());
    manifest["external_tensors"].push_back(declared);
    std::ofstream(directory / "external" / "config.json") << manifest;
    for (const auto* name : {"internal", "external"}) write_tokenizer(directory / name / "tokenizer.json");
    ASSERT_FALSE(kidi::inference::Generator::load(directory / "external", device))
        << "declared external tensors loaded without a source";
    auto rows = std::make_shared<CountingRows>(table);
    auto external = require(kidi::inference::Generator::load(
        directory / "external", device, 0, 128, false,
        [&](std::string_view name, kidi::tensor::DType dtype, std::span<const std::int64_t> shape) {
            if (name != table_name) throw std::runtime_error("unexpected external tensor");
            return kidi::tensor::external_tensor({shape.begin(), shape.end()}, dtype, rows);
        }));
    auto internal = require(kidi::inference::Generator::load(directory / "internal", device));
    kidi::inference::GenerationOptions options;
    options.maximum_new_tokens = 4;
    options.context_size = 16;
    options.prefill_chunk_size = 2;
    options.raw_prompt = true;
    options.ignore_eos = true;
    const auto expected = require(internal.generate("alpha beta gamma delta eta", options));
    // A load-time warm-up only prepares operators; it must not leak state into the first request.
    require(external.warm_up());
    const auto actual = require(external.generate("alpha beta gamma delta eta", options));
    ASSERT_EQ(actual.generation.token_ids, expected.generation.token_ids)
        << "external per-layer embeddings changed tokens or skipped row gathers";
    ASSERT_GE(rows->gathered, 5U + 3U) << "external per-layer embeddings changed tokens or skipped row gathers";
    std::filesystem::remove_all(directory);
}

auto check_serving_cache(const std::filesystem::path& fixture, const YAML::Node& config,
                         kidi::tensor::Device device) -> void {
    using kidi::ops::require;
    const auto directory = std::filesystem::temp_directory_path() /
                           ("kidi-serving-cache-test-" + fixture.filename().string());
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    std::filesystem::copy_file(fixture / "model.safetensors", directory / "model.safetensors");
    auto manifest = YAML::Clone(config);
    manifest["decode"]["maximum_new_tokens"] = 3;
    manifest["decode"]["context_size"] = 16;
    std::ofstream(directory / "config.json") << manifest;
    write_tokenizer(directory / "tokenizer.json");
    auto cached = require(kidi::inference::Generator::load(directory, device));
    auto plain = require(kidi::inference::Generator::load(directory, device));
    require(cached.configure_serving({1, 4, 16, 2}));
    require(plain.configure_serving({1, 4, 16, 2}));
    kidi::inference::GenerationOptions options;
    options.maximum_new_tokens = 3;
    options.context_size = 16;
    options.prefill_chunk_size = 2;
    options.prefix_cache_bytes = 1024 * 1024;
    options.raw_prompt = true;
    options.ignore_eos = true;
    const auto run = [&](auto& generator, std::string_view prompt, auto settings) {
        require(generator.enqueue(prompt, settings));
        std::optional<kidi::inference::TextGeneration> completed;
        while (generator.pending_requests())
            for (auto& event : require(generator.step()).events)
                if (event.completed) completed = std::move(event.completed);
        if (!completed) throw std::runtime_error("missing serving completion");
        return std::move(*completed);
    };
    auto warmed = require(kidi::inference::Generator::load(directory, device));
    require(warmed.warm_up(2, 16));
    require(warmed.configure_serving({1, 4, 16, 2}));
    const auto warmed_result = run(warmed, "alpha beta gamma delta", options);
    auto warm_reference = options;
    warm_reference.prefix_cache_bytes = 0;
    auto exact = require(kidi::inference::Generator::load(directory, device));
    const auto expected_warm = require(exact.generate("alpha beta gamma delta", warm_reference));
    ASSERT_EQ(warmed_result.generation.token_ids, expected_warm.generation.token_ids)
        << "serving bucket warm-up changed output or left operators unprepared";
    if (device == kidi::tensor::Device::cpu()) {
        ASSERT_EQ(warmed_result.stats.preparation_ns, 0U)
            << "serving bucket warm-up changed output or left operators unprepared";
    }
    const auto check = [&](std::string_view prompt, auto settings, std::size_t reused) {
        const auto actual = run(cached, prompt, settings);
        settings.prefix_cache_bytes = 0;
        const auto expected = run(plain, prompt, settings);
        const auto message = "serving prefix reuse changed output or reused the wrong token count";
        EXPECT_EQ(actual.generation.token_ids, expected.generation.token_ids) << message << " for " << prompt;
        EXPECT_EQ(actual.text, expected.text) << message << " for " << prompt;
        EXPECT_EQ(actual.stats.reused_prompt_tokens, reused) << message << " for " << prompt;
        return actual;
    };
    const auto first = check("alpha beta gamma delta", options, 0);
    ASSERT_TRUE(first.stats.prefix_cache_bytes) << "serving cache did not retain bounded KV storage";
    ASSERT_LE(first.stats.prefix_reserved_bytes, options.prefix_cache_bytes)
        << "serving cache did not retain bounded KV storage";
    const std::array words{"<pad>", "<eos>", "<bos>", "<turn|>", "<|turn>", "<unk>", "alpha", "beta",
                           "gamma", "delta", "theta", "zeta",    "eta",     "iota",  "kappa", "lambda"};
    std::string continuation = "alpha beta gamma delta";
    for (const auto token : first.generation.token_ids) continuation += " " + std::string(words.at(token));
    continuation += " beta";
    check(continuation, options, 6);
    check("alpha beta theta delta", options, 2);
    check("zeta eta", options, 0);
    auto too_small = options;
    too_small.prefix_cache_bytes = 1;
    ASSERT_FALSE(check("zeta eta", too_small, 0).stats.prefix_reserved_bytes)
        << "serving cache exceeded its byte budget";
    check("alpha beta gamma delta", options, 0);
    require(cached.configure_serving({1, 4, 16, 2}));
    check("alpha beta gamma delta", options, 0);
    auto uncached = options;
    uncached.prefix_cache_bytes = 0;
    check("alpha beta gamma delta", uncached, 0);
    const auto cancelled = require(cached.enqueue("alpha beta gamma delta", options));
    require(cached.step());
    ASSERT_FALSE(cached.release_workspaces()) << "generator released workspaces with an active request";
    ASSERT_FALSE(cached.trim_workspaces()) << "generator trimmed workspaces with an active request";
    require(cached.cancel(cancelled));
    require(cached.trim_workspaces());
    check("alpha beta gamma delta", options, 2);
    require(cached.release_workspaces());
    auto smaller_context = options;
    smaller_context.context_size = 8;
    check("alpha beta gamma delta", smaller_context, 0);
    auto full_attention = options;
    full_attention.full_attention_cache = true;
    check("alpha beta gamma delta", full_attention, 0);
    check("alpha beta gamma delta", options, 0);
    auto different_chunk = options;
    different_chunk.prefill_chunk_size = 1;
    check("alpha beta gamma delta", different_chunk, 0);
    auto limited_prompt = uncached;
    limited_prompt.maximum_prompt_tokens = 3;
    ASSERT_FALSE(cached.enqueue("alpha beta gamma delta", limited_prompt))
        << "serving accepted a prompt above its token limit";
    check("zeta eta", limited_prompt, 0);
    require(cached.configure_serving({2, 4, 32, 2}));
    check("alpha beta gamma delta", options, 0);
    const std::array prompts{"alpha beta gamma delta", "zeta eta gamma"};
    const std::array ids{require(cached.enqueue(prompts[0], options)), require(cached.enqueue(prompts[1], options))};
    std::array<std::optional<kidi::inference::TextGeneration>, 2> results;
    while (cached.pending_requests())
        for (auto& event : require(cached.step()).events)
            if (event.completed) results[event.request_id == ids[0] ? 0 : 1] = std::move(event.completed);
    for (std::size_t index = 0; index < results.size(); ++index) {
        const auto expected = run(plain, prompts[index], uncached);
        ASSERT_TRUE(results[index]) << "queued requests shared or lost streaming KV state";
        ASSERT_EQ(results[index]->generation.token_ids, expected.generation.token_ids)
            << "queued requests shared or lost streaming KV state";
        ASSERT_EQ(results[index]->stats.reused_prompt_tokens, index == 0 ? 3U : 0U)
            << "queued requests shared or lost streaming KV state";
    }
    // Deferred tokens report each selection one step later, alone and batched, without changing the generation.
    auto deferred = require(kidi::inference::Generator::load(directory, device));
    for (const std::size_t active : {1, 2}) {
        require(deferred.configure_serving({.maximum_active = active,
                                            .maximum_requests = 4,
                                            .cache_token_budget = 32,
                                            .prefill_tokens_per_step = 2,
                                            .deferred_tokens = true}));
        const std::array deferred_ids{require(deferred.enqueue(prompts[0], uncached)),
                                      require(deferred.enqueue(prompts[1], uncached))};
        std::array<std::optional<kidi::inference::TextGeneration>, 2> deferred_results;
        while (deferred.pending_requests())
            for (auto& event : require(deferred.step()).events)
                if (event.completed)
                    deferred_results[event.request_id == deferred_ids[0] ? 0 : 1] = std::move(event.completed);
        for (std::size_t index = 0; index < deferred_results.size(); ++index) {
            ASSERT_TRUE(deferred_results[index]) << "deferred token serving changed generated tokens";
            ASSERT_EQ(deferred_results[index]->generation.token_ids,
                      run(plain, prompts[index], uncached).generation.token_ids)
                << "deferred token serving changed generated tokens";
        }
    }
    manifest["max_position_embeddings"] = 1024;
    std::ofstream(directory / "config.json") << manifest;
    auto compact = require(kidi::inference::Generator::load(directory, device));
    auto full = require(kidi::inference::Generator::load(directory, device));
    require(compact.configure_serving({1, 1, 1024, 2, true}));
    require(full.configure_serving({1, 1, 1024, 2}));
    auto bounded = uncached;
    bounded.context_size = 1024;
    require(compact.enqueue(prompts[0], bounded));
    const auto step = require(compact.step());
    const auto byte_limit = device.capabilities().attention.int8_cache_tokens;
    const auto expected_capacity =
        byte_limit < 1024 ? std::bit_ceil(std::max<std::size_t>(byte_limit + 1, 128)) : std::size_t{128};
    ASSERT_EQ(step.reserved_cache_tokens, expected_capacity)
        << "compact reservation changed the cache precision policy or exceeded its bucket";
    std::optional<kidi::inference::TextGeneration> compact_result;
    while (compact.pending_requests())
        for (auto& event : require(compact.step()).events)
            if (event.completed) compact_result = std::move(event.completed);
    const auto full_result = run(full, prompts[0], bounded);
    ASSERT_TRUE(compact_result) << "compact cache reservation changed generated tokens";
    ASSERT_EQ(compact_result->generation.token_ids, full_result.generation.token_ids)
        << "compact cache reservation changed generated tokens";
    // Generation is clamped to the remaining context; only a prompt that cannot fit is rejected.
    std::string overflow;
    for (int word = 0; word < 1100; ++word) overflow += "alpha ";
    ASSERT_FALSE(compact.enqueue(overflow, bounded)) << "compact cache accepted context overflow";
    std::filesystem::remove_all(directory);
}

// Largest absolute difference and where it occurs; non-finite values count as infinite.
auto difference(std::span<const float> left, std::span<const float> right) -> std::pair<float, std::size_t> {
    std::pair<float, std::size_t> worst{0.F, 0};
    for (std::size_t index = 0; index < left.size(); ++index) {
        const auto gap = std::isfinite(left[index]) && std::isfinite(right[index])
                             ? std::abs(left[index] - right[index])
                             : std::numeric_limits<float>::infinity();
        if (gap > worst.first) worst = {gap, index};
    }
    return worst;
}
auto executes(kidi::tensor::Device device) -> bool {
    const auto backend = kidi::tensor::BackendRegistry::instance().backend(device);
    return backend && (*backend)->is_available(device) && (*backend)->supports_execution();
}

void check_async_serving(const YAML::Node& config) {
    using namespace kidi;
    using ops::require;
    const auto device = tensor::Device::cuda();
    const auto directory = std::filesystem::temp_directory_path() / "kidi-async-decode-test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    auto model_config = YAML::Clone(config);
    model_config["max_position_embeddings"] = 64;
    StateDict state;
    {
        const ModuleScope construction(tensor::DType::F32, true, tensor::Device::cpu());
        auto skeleton = require(model::Gemma4Impl::create(model_config));
        ops::Context context;
        bool embedding = false, gate = false, projection = false;
        for (const auto& [name, parameter] : skeleton->state_dict()) {
            std::vector<float> values(parameter.numel(), parameter.dimensions() < 2 ? 1.F : 0.F);
            const auto width = parameter.dimensions() == 2 ? parameter.size(1) : 0;
            if (name.ends_with("embed_tokens.weight") || name.ends_with("lm_head.weight")) {
                values[6 * width] = values[7 * width + 1] = values[width + 2] = values[8 * width + 3] = 1.F;
                embedding = true;
            }
            if (name.ends_with("embed_tokens_per_layer.weight")) std::ranges::fill(values, 1.F);
            if (name.ends_with("layers.3.per_layer_input_gate.weight")) {
                values[0] = values[width + 1] = 1.F;
                gate = true;
            }
            if (name.ends_with("layers.3.per_layer_projection.weight")) {
                values[0] = values[width + 1] = -1.F;
                values[width] = values[2 * width + 1] = 1.F;
                projection = true;
            }
            auto value = require(tensor::Tensor::from_host({parameter.shape().begin(), parameter.shape().end()},
                                                           std::span<const float>(values)));
            if (parameter.dimensions() == 2) value = context.cast(value, tensor::DType::BF16);
            if (name == "lm_head.weight") continue;
            state.emplace("model.language_model." + name, std::move(value));
        }
        ASSERT_TRUE(embedding && gate && projection) << "async EOS fixture lacks transition weights";
    }
    require(checkpoint::Weights::save(directory / "model.safetensors", state));
    auto manifest = YAML::Clone(model_config);
    manifest["decode"]["maximum_new_tokens"] = 12;
    manifest["decode"]["context_size"] = 64;
    std::ofstream(directory / "config.json") << manifest;
    write_tokenizer(directory / "tokenizer.json");
    auto pipelined = require(inference::Generator::load(directory, device));
    auto serial = require(inference::Generator::load(directory, device));
    require(pipelined.warm_up(4, 64));
    const auto preallocated_bytes = pipelined.memory().cache_bytes;
    ASSERT_TRUE(preallocated_bytes) << "CUDA warm-up did not preallocate its KV cache";
    require(pipelined.configure_serving({.maximum_active = 1,
                                         .maximum_requests = 1,
                                         .cache_token_budget = 64,
                                         .prefill_tokens_per_step = 4,
                                         .decode_ahead_tokens = 8}));
    require(serial.configure_serving({.maximum_active = 1,
                                      .maximum_requests = 1,
                                      .cache_token_budget = 64,
                                      .prefill_tokens_per_step = 4,
                                      .decode_ahead_tokens = 1}));
    inference::GenerationOptions options;
    options.maximum_new_tokens = 12;
    options.context_size = 64;
    options.prefill_chunk_size = 4;
    options.raw_prompt = true;
    options.stream_text = true;
    options.prefix_cache_bytes = 1024 * 1024;
    const auto run = [&](auto& generator, std::string_view prompt, auto settings) {
        require(generator.enqueue(prompt, settings));
        std::optional<inference::TextGeneration> completed;
        std::vector<std::int32_t> streamed;
        while (generator.pending_requests())
            for (auto& event : require(generator.step()).events) {
                if (event.token) streamed.push_back(*event.token);
                if (event.completed) completed = std::move(event.completed);
            }
        if (!completed) throw std::runtime_error("asynchronous token events were lost or reordered");
        EXPECT_EQ(streamed, completed->generation.token_ids) << "asynchronous token events were lost or reordered";
        return std::move(*completed);
    };
    const auto eos = run(pipelined, "alpha", options);
    const auto expected = run(serial, "alpha", options);
    {
        const auto message = "asynchronous EOS handling published queued post-EOS tokens";
        ASSERT_EQ(eos.generation.token_ids, std::vector<std::int32_t>{7}) << message;
        ASSERT_EQ(eos.generation.decoder_steps, 2U) << message;
        ASSERT_EQ(eos.generation.token_ids, expected.generation.token_ids) << message;
        ASSERT_EQ(eos.stats.decode_queue_peak, 8U) << message;
        ASSERT_GE(eos.stats.decode_submitted_tokens, 8U) << message;
    }
    const auto continued = run(pipelined, "alpha beta gamma", options);
    ASSERT_EQ(continued.stats.reused_prompt_tokens, 2U)
        << "EOS rewind retained the speculative rather than logical KV prefix";
    options.prefix_cache_bytes = 0;
    options.ignore_eos = true;
    const auto maximum = run(pipelined, "gamma", options);
    const auto synchronous = run(serial, "gamma", options);
    {
        const auto message = "decode-ahead violated the maximum or changed synchronous tokens";
        ASSERT_EQ(maximum.generation.token_ids, synchronous.generation.token_ids) << message;
        ASSERT_EQ(maximum.generation.token_ids.size(), 12U) << message;
        ASSERT_EQ(maximum.stats.decode_submitted_tokens, 11U) << message;
        ASSERT_EQ(maximum.stats.decode_queue_peak, 8U) << message;
    }
    const auto generated = require(pipelined.generate("gamma", options));
    ASSERT_EQ(generated.generation.token_ids, synchronous.generation.token_ids)
        << "serial generation bypassed the device-feedback pipeline";
    ASSERT_EQ(generated.stats.decode_queue_peak, 8U) << "serial generation bypassed the device-feedback pipeline";
    const auto reusable_bytes = pipelined.memory().cache_bytes;
    ASSERT_EQ(reusable_bytes, preallocated_bytes)
        << "completed CUDA request discarded or duplicated its preallocated KV cache";
    for (const auto* prompt : {"alpha beta gamma", "alpha", "gamma"}) {
        require(serial.release_workspaces());
        const auto fresh = run(serial, prompt, options);
        const auto reused = require(pipelined.generate(prompt, options));
        const auto message = "reused CUDA state retained prompt data or duplicated the KV cache";
        ASSERT_EQ(reused.generation.token_ids, fresh.generation.token_ids) << message << " for " << prompt;
        ASSERT_FALSE(reused.stats.reused_prompt_tokens) << message << " for " << prompt;
        ASSERT_EQ(pipelined.memory().cache_bytes, reusable_bytes) << message << " for " << prompt;
    }
    const auto serial_bytes = serial.memory().cache_bytes;
    const auto eager_reused = require(serial.generate("gamma", options));
    ASSERT_EQ(eager_reused.generation.token_ids, synchronous.generation.token_ids)
        << "eager CUDA generation failed to reuse its request cache";
    ASSERT_EQ(serial.memory().cache_bytes, serial_bytes) << "eager CUDA generation failed to reuse its request cache";
    for (const std::size_t capacity : {32, 64}) {
        options.context_size = capacity;
        const auto resized = require(pipelined.generate("gamma", options));
        ASSERT_EQ(resized.generation.token_ids, synchronous.generation.token_ids)
            << "CUDA cache reuse ignored a change in request capacity";
        ASSERT_EQ(pipelined.memory().cache_bytes, preallocated_bytes * capacity / 64)
            << "CUDA cache reuse ignored a change in request capacity";
    }
    const auto cancelled = require(pipelined.enqueue("gamma", options));
    require(pipelined.step());
    require(pipelined.step());
    require(pipelined.cancel(cancelled));
    ASSERT_FALSE(pipelined.pending_requests()) << "cancelled decode request remained active";
    const auto cancelled_cache = pipelined.memory().cache_bytes;
    require(pipelined.enqueue("gamma", options));
    while (pipelined.pending_requests()) {
        const auto step = require(pipelined.step());
        ASSERT_LE(step.reserved_cache_tokens, 64U)
            << "cancelled in-flight decode released its cache budget before completion";
        ASSERT_LE(pipelined.memory().cache_bytes, cancelled_cache)
            << "cancelled in-flight decode released its cache budget before completion";
    }
    require(pipelined.release_workspaces());
    ASSERT_FALSE(pipelined.memory().cache_bytes) << "release_workspaces retained reusable CUDA request caches";
    std::filesystem::remove_all(directory);
}

void check_hnsw_head(kidi::model::Gemma4& network, const kidi::tensor::Tensor& weights,
                     std::span<const std::int32_t> tokens) {
    using namespace kidi;
    using ops::require;
    SCOPED_TRACE("HNSW head on " + std::string(tensor::to_string(network->device())));
    const auto path =
        std::filesystem::temp_directory_path() /
        ("kidi-hnsw-model-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    } cleanup{path};
    auto index =
        require(search::HnswIndex::build(weights, {.m = 2,
                                                   .ef_construction = std::max<std::size_t>(4, weights.size(0)),
                                                   .seed = 42,
                                                   .candidates = weights.size(0),
                                                   .ef_search = weights.size(0)}));
    require(index.save(path));
    if (!network->device().capabilities().compute.host()) {
        EXPECT_FALSE(network->configure_lm_head(path)) << "non-CPU inference accepted a host-synchronized HNSW head";
        return;
    }
    const auto run = [&] {
        std::vector<std::int32_t> output;
        auto serial = require(network->create_state(8));
        std::array<std::unique_ptr<model::LanguageModelState>, 2> batched;
        for (std::size_t row = 0; row < batched.size(); ++row) {
            batched[row] = require(network->create_state(8));
            require(network->prefill(tokens.first(row + 1), *batched[row]));
        }
        require(network->prefill(tokens.first(1), *serial));
        for (std::size_t step = 0; step < 3; ++step) {
            const std::array input{tokens[step], tokens[step + 1]};
            output.push_back(require(network->forward_token(std::span(input).first(1), *serial)));
            std::array<model::LanguageModelState*, 2> states{batched[0].get(), batched[1].get()};
            const auto selected = require(network->forward_batch_tokens(input, states));
            output.insert(output.end(), selected.begin(), selected.end());
        }
        return output;
    };
    const auto dense = run();
    require(network->configure_lm_head(path));
    ASSERT_EQ(run(), dense) << "all-candidate HNSW head differs from dense token selection";
    network->release_workspaces();
    ASSERT_EQ(run(), dense) << "head workspaces did not rebuild correctly";
    std::filesystem::remove(path);
    auto subset =
        require(search::HnswIndex::build(weights, {.m = 4,
                                                   .ef_construction = std::max<std::size_t>(4, weights.size(0)),
                                                   .seed = 42,
                                                   .candidates = 4,
                                                   .ef_search = weights.size(0)}));
    require(subset.save(path));
    require(network->configure_lm_head(path));
    ASSERT_EQ(run(), dense) << "candidate-subset HNSW head differs from dense fixture selection";
    require(network->set_state(network->state_dict(), true, true));
    auto stale = require(network->create_state(8));
    EXPECT_FALSE(network->forward_token(tokens.first(1), *stale))
        << "stale index binding was accepted after replacing output weights";
    EXPECT_FALSE(stale->sequence_position()) << "stale index binding was accepted after replacing output weights";
}

void check_cuda_cache_precision(const YAML::Node& config, const kidi::checkpoint::Weights& checkpoint,
                                std::span<const std::int32_t> tokens) {
    using namespace kidi;
    using ops::require;
    const auto device = tensor::Device::cuda();
    const ModuleScope construction(tensor::DType::F32, false, device);
    auto network = require(model::Gemma4Impl::create(config));
    require(network->set_checkpoint(checkpoint));
    network->set_kv_cache_precision(core::KVCachePrecision::FP32);
    auto reference = require(network->create_state(8));
    const auto first = tokens.first(2);
    const auto next = tokens.subspan(2, 1);
    const auto expected_prefill = require(require(network->forward(first, *reference, true)).to(tensor::Device::cpu()));
    const auto expected_decode = require(require(network->forward(next, *reference)).to(tensor::Device::cpu()));
    const auto close = [&](const tensor::Tensor& output, const tensor::Tensor& expected) {
        const auto host = require(output.to(tensor::Device::cpu()));
        ASSERT_EQ(host.dtype(), tensor::DType::F32) << "BF16 cache changed output dtype or geometry";
        ASSERT_TRUE(std::ranges::equal(host.shape(), expected.shape()))
            << "BF16 cache changed output dtype or geometry";
        const auto actual = require(host.data<float>()), values = require(expected.data<float>());
        for (std::size_t index = 0; index < actual.size(); ++index) {
            ASSERT_TRUE(std::isfinite(actual[index])) << "BF16 cache changed CUDA model logits beyond tolerance";
            ASSERT_LE(std::abs(actual[index] - values[index]), 0.03F + 0.02F * std::abs(values[index]))
                << "BF16 cache changed CUDA model logits beyond tolerance at " << index;
        }
    };
    for (const auto precision : {core::KVCachePrecision::AUTO, core::KVCachePrecision::BF16}) {
        SCOPED_TRACE(core::to_string(precision));
        network->set_kv_cache_precision(precision);
        auto state = require(network->create_state(8));
        for (const auto& layer : static_cast<const model::Gemma4State&>(*state).layers) {
            ASSERT_EQ(layer.key.dtype(), tensor::DType::BF16) << "CUDA cache did not use BF16 storage";
            ASSERT_EQ(layer.value.dtype(), tensor::DType::BF16) << "CUDA cache did not use BF16 storage";
        }
        ASSERT_NO_FATAL_FAILURE(close(require(network->forward(first, *state, true)), expected_prefill));
        auto fork = require(network->fork_state(*state, first.size(), 8));
        ASSERT_NO_FATAL_FAILURE(close(require(network->forward(next, *state)), expected_decode));
        ASSERT_NO_FATAL_FAILURE(close(require(network->forward(next, *fork)), expected_decode));
        require(network->rewind(*state, 0));
        ASSERT_NO_FATAL_FAILURE(close(require(network->forward(first, *state, true)), expected_prefill));
    }
}

void check_bfloat_model(const YAML::Node& config, const kidi::checkpoint::Weights& checkpoint,
                        std::span<const std::int32_t> tokens) {
    using namespace kidi;
    using ops::require;
    StateDict state;
    {
        const ModuleScope construction(tensor::DType::F32, false, tensor::Device::cpu());
        auto source = require(model::Gemma4Impl::create(config));
        require(source->set_checkpoint(checkpoint));
        ops::Context context(tensor::Device::cpu());
        for (const auto& [name, value] : source->state_dict())
            state.emplace(name, value.dimensions() == 2 ? context.cast(value, tensor::DType::BF16) : value);
    }
    model::Gemma4 reference{nullptr}, actual{nullptr};
    {
        const ModuleScope construction(tensor::DType::BF16, false, tensor::Device::cpu());
        reference = require(model::Gemma4Impl::create(config));
        require(reference->set_state(state));
    }
    {
        const ModuleScope construction(tensor::DType::BF16, false, tensor::Device::cuda());
        actual = require(model::Gemma4Impl::create(config));
        require(actual->set_state(state));
    }
    ASSERT_NO_FATAL_FAILURE(check_hnsw_head(actual, state.at("embed_tokens.weight"), tokens));
    auto reference_state = require(reference->create_state(8));
    auto actual_state = require(actual->create_state(8));
    ASSERT_EQ(static_cast<const model::Gemma4State&>(*actual_state).layers.front().key.dtype(), tensor::DType::BF16);
    const auto expected = require(reference->forward(tokens, *reference_state, true));
    const auto output = require(require(actual->forward(tokens, *actual_state, true)).to(tensor::Device::cpu()));
    ASSERT_EQ(output.dtype(), tensor::DType::F32);
    ASSERT_TRUE(std::ranges::equal(output.shape(), expected.shape()));
    const auto expected_values = require(expected.data<float>()), actual_values = require(output.data<float>());
    for (std::size_t index = 0; index < actual_values.size(); ++index) {
        ASSERT_TRUE(std::isfinite(actual_values[index])) << "at " << index;
        ASSERT_LE(std::abs(actual_values[index] - expected_values[index]),
                  0.03F + 0.02F * std::abs(expected_values[index]))
            << "at " << index;
    }
    auto extended_config = YAML::Clone(config);
    extended_config["max_position_embeddings"] = 384;
    model::Gemma4 extended{nullptr};
    {
        const ModuleScope construction(tensor::DType::BF16, false, tensor::Device::cuda());
        extended = require(model::Gemma4Impl::create(extended_config));
        require(extended->set_state(state));
    }
    std::vector<std::int32_t> sequence(258);
    for (std::size_t index = 0; index < sequence.size(); ++index) sequence[index] = tokens[index % tokens.size()];
    auto cropped = require(extended->create_state(384)), uncropped = require(extended->create_state(384));
    extended->configure_state(*uncropped, true, true);
    require(extended->prefill(sequence, *cropped));
    require(extended->prefill(sequence, *uncropped));
    for (int iteration = 0; iteration < 4; ++iteration) {
        const auto token = tokens[iteration % tokens.size()];
        const std::array input{token};
        const auto first = require(extended->forward(input, *cropped));
        const auto second = require(extended->forward(input, *uncropped));
        const auto first_values = require(require(first.to(tensor::Device::cpu())).copy_to_host());
        const auto second_values = require(require(second.to(tensor::Device::cpu())).copy_to_host());
        ASSERT_TRUE(first_values == second_values) << "cropped and uncropped caches differ at iteration " << iteration;
    }
}

struct Fixture {
    std::filesystem::path directory;
    YAML::Node config;
    std::optional<kidi::checkpoint::Weights> checkpoint, reference;
    std::vector<std::int32_t> token_values;
    std::vector<float> logit_values;
    std::span<const std::int32_t> tokens;
    std::span<const float> values;
};

template <typename T>
auto copy_tensor_values(const kidi::tensor::Tensor& tensor) -> std::vector<T> {
    using kidi::ops::require;
    auto bytes = require(tensor.copy_to_host());
    if (bytes.size() % sizeof(T) != 0) throw std::runtime_error("reference tensor has an invalid byte size");
    std::vector<T> values(bytes.size() / sizeof(T));
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
}

auto load_fixture(std::string_view name) -> std::unique_ptr<Fixture> {
    using kidi::ops::require;
    auto fixture = std::make_unique<Fixture>();
    fixture->directory = fixture_directory(name);
    fixture->config = YAML::LoadFile((fixture->directory / "config.json").string());
    fixture->checkpoint.emplace(require(kidi::checkpoint::Weights::load(fixture->directory / "model.safetensors")));
    fixture->reference.emplace(require(kidi::checkpoint::Weights::load(fixture->directory / "reference.safetensors")));
    fixture->token_values = copy_tensor_values<std::int32_t>(require(fixture->reference->tensor("tokens")));
    fixture->logit_values = copy_tensor_values<float>(require(fixture->reference->tensor("logits")));
    fixture->tokens = fixture->token_values;
    fixture->values = fixture->logit_values;
    return fixture;
}

auto create_model(const Fixture& fixture) -> kidi::model::Gemma4 {
    using kidi::ops::require;
    auto model = require(kidi::model::Gemma4Impl::create(fixture.config));
    require(model->set_checkpoint(*fixture.checkpoint));
    return model;
}

auto devices() -> std::vector<kidi::tensor::Device> {
    std::vector devices{kidi::tensor::Device::cpu()};
#if defined(__APPLE__)
    devices.push_back(kidi::tensor::Device::apple_gpu());
#endif
    if (executes(kidi::tensor::Device::vulkan())) devices.push_back(kidi::tensor::Device::vulkan());
    return devices;
}

// QAT checkpoints keep an INT8 KV cache unless KIDI_INT8_KV_CACHE=0 or the device cannot hold `capacity` tokens.
auto expected_cache_dtype(bool qat, kidi::tensor::Device device, std::size_t capacity) -> kidi::tensor::DType {
    const auto* cache_override = std::getenv("KIDI_INT8_KV_CACHE");
    const auto& capabilities = device.capabilities();
    return qat && (cache_override == nullptr || std::string_view(cache_override) != "0") &&
                   capabilities.compute.supports(kidi::tensor::TransformerSupport::QUANTIZED) &&
                   capabilities.attention.int8_cache_tokens && capacity <= capabilities.attention.int8_cache_tokens
               ? kidi::tensor::DType::I8
               : kidi::tensor::DType::F32;
}

class Gemma4RefTest : public ::testing::TestWithParam<const char*> {
protected:
    void SetUp() override { fixture = load_fixture(GetParam()); }
    auto qat() const -> bool { return GetParam() == std::string_view("gemma4-qat"); }

    std::unique_ptr<Fixture> fixture;
};

INSTANTIATE_TEST_SUITE_P(, Gemma4RefTest, ::testing::Values("gemma4", "gemma4-qat"),
                         [](const ::testing::TestParamInfo<const char*>& info) -> std::string {
                             return info.param == std::string_view("gemma4-qat") ? "qat" : "base";
                         });

TEST_P(Gemma4RefTest, MissingHeadDim) {
    auto invalid_config = YAML::Clone(fixture->config);
    invalid_config.remove("global_head_dim");
    EXPECT_FALSE(kidi::model::Gemma4Impl::create(invalid_config))
        << "Gemma accepted a configuration without global_head_dim";
}

TEST_P(Gemma4RefTest, CudaCachePrecision) {
    if (!executes(kidi::tensor::Device::cuda())) GTEST_SKIP() << "CUDA execution is unavailable";
    check_cuda_cache_precision(fixture->config, *fixture->checkpoint, fixture->tokens);
}

TEST_P(Gemma4RefTest, ServingCache) {
    for (const auto device : devices()) {
        SCOPED_TRACE(kidi::tensor::to_string(device));
        ASSERT_NO_FATAL_FAILURE(check_serving_cache(fixture->directory, fixture->config, device));
    }
}

TEST_P(Gemma4RefTest, ExternalRows) {
    for (const auto device : devices()) {
        SCOPED_TRACE(kidi::tensor::to_string(device));
        ASSERT_NO_FATAL_FAILURE(check_external_rows(fixture->directory, fixture->config, device));
    }
}

TEST_P(Gemma4RefTest, FusesProjections) {
    using kidi::ops::require;
    const auto& config = fixture->config;
    for (const auto device : devices()) {
        SCOPED_TRACE(kidi::tensor::to_string(device));
        const kidi::ModuleScope construction(kidi::tensor::DType::F32, false, device);
        auto model = create_model(*fixture);
        const auto parameters = model->state_dict();
        for (const auto& [name, parameter] : parameters)
            ASSERT_FALSE(name.find(".mlp.gate_proj.") != std::string::npos ||
                         name.find(".mlp.up_proj.") != std::string::npos ||
                         name.find(".self_attn.k_proj.") != std::string::npos ||
                         name.find(".self_attn.v_proj.") != std::string::npos)
                << "unfused projection " << name;
        const auto shared_begin = config["num_hidden_layers"].as<int>() - config["num_kv_shared_layers"].as<int>();
        for (int layer = 0; layer < config["num_hidden_layers"].as<int>(); ++layer) {
            SCOPED_TRACE(layer);
            const auto prefix = "layers." + std::to_string(layer) + ".mlp.";
            const auto gate =
                require(fixture->checkpoint->tensor("model.language_model." + prefix + "gate_proj.weight"));
            const auto& combined = parameters.at(prefix + "gate_up_proj.weight");
            ASSERT_EQ(combined.size(0), 2 * gate.size(0));
            ASSERT_EQ(combined.size(1), gate.size(1));
            if (layer < shared_begin) {
                const auto attn_prefix = "layers." + std::to_string(layer) + ".self_attn.";
                const auto key =
                    require(fixture->checkpoint->tensor("model.language_model." + attn_prefix + "k_proj.weight"));
                const auto& kv = parameters.at(attn_prefix + "kv_proj.weight");
                ASSERT_EQ(kv.size(0), 2 * key.size(0));
                ASSERT_EQ(kv.size(1), key.size(1));
            }
        }
    }
}

TEST_P(Gemma4RefTest, KvCachePrecision) {
    using namespace kidi;
    using ops::require;
    for (const auto device : devices()) {
        SCOPED_TRACE(tensor::to_string(device));
        const ModuleScope construction(tensor::DType::F32, false, device);
        auto model = create_model(*fixture);
        auto full = require(model->create_state(8));
        const auto cache_dtype = expected_cache_dtype(qat(), device, 8);
        for (const auto& cache : static_cast<const model::Gemma4State&>(*full).layers) {
            ASSERT_EQ(cache.key.dtype(), cache_dtype);
            ASSERT_EQ(cache.value.dtype(), cache_dtype);
        }
        model->set_kv_cache_precision(core::KVCachePrecision::FP32);
        const auto fp32_cache = require(model->create_state(8));
        for (const auto& cache : static_cast<const model::Gemma4State&>(*fp32_cache).layers) {
            ASSERT_EQ(cache.key.dtype(), tensor::DType::F32);
            ASSERT_EQ(cache.value.dtype(), tensor::DType::F32);
        }
        if (cache_dtype == tensor::DType::I8) {
            model->set_kv_cache_precision(core::KVCachePrecision::INT8);
            const auto int8_cache = require(model->create_state(8));
            for (const auto& cache : static_cast<const model::Gemma4State&>(*int8_cache).layers) {
                ASSERT_EQ(cache.key.dtype(), tensor::DType::I8);
                ASSERT_EQ(cache.value.dtype(), tensor::DType::I8);
            }
        }
        model->set_kv_cache_precision(core::KVCachePrecision::AUTO);
        model->set_kv_cache_precision(core::KVCachePrecision::BF16);
        const auto unsupported_cache = model->create_state(8);
        ASSERT_FALSE(unsupported_cache);
        ASSERT_EQ(unsupported_cache.error().code, ErrorCode::UNSUPPORTED);
    }
}

TEST_P(Gemma4RefTest, ReferenceLogits) {
    using namespace kidi;
    using ops::require;
    const auto tokens = fixture->tokens;
    const auto values = fixture->values;
    for (const auto device : devices()) {
        SCOPED_TRACE(tensor::to_string(device));
        const ModuleScope construction(tensor::DType::F32, false, device);
        auto model = create_model(*fixture);
        auto full = require(model->create_state(8));
        auto prefill = require(model->forward(tokens, *full, true));
        const auto prefill_values = require(prefill.data<float>());
        ASSERT_EQ(prefill_values.size(), values.size());
        for (std::size_t index = 0; index < values.size(); ++index) {
            ASSERT_TRUE(std::isfinite(prefill_values[index]))
                << "Gemma reference mismatch at " << index << ": " << prefill_values[index] << " != " << values[index];
            ASSERT_LE(std::abs(prefill_values[index] - values[index]), 2e-4F)
                << "Gemma reference mismatch at " << index << ": " << prefill_values[index] << " != " << values[index];
        }
        auto incremental = require(model->create_state(8));
        auto selected_state = require(model->create_state(8));
        for (std::size_t position = 0; position < tokens.size(); ++position) {
            auto output = require(model->forward(std::span(tokens).subspan(position, 1), *incremental));
            const auto actual = require(output.data<float>());
            const auto selected =
                require(model->forward_token(std::span(tokens).subspan(position, 1), *selected_state));
            ASSERT_EQ(selected, std::ranges::max_element(actual) - actual.begin()) << "at " << position;
            for (std::size_t token = 0; token < actual.size(); ++token) {
                ASSERT_TRUE(std::isfinite(actual[token]))
                    << "Gemma cached logits differ at " << position << ':' << token;
                ASSERT_LE(std::abs(actual[token] - values[position * actual.size() + token]), 2e-4F)
                    << "Gemma cached logits differ at " << position << ':' << token;
            }
        }
        ASSERT_EQ(incremental->sequence_position(), tokens.size());
        ASSERT_EQ(full->sequence_position(), tokens.size());
        const std::array<std::int32_t, 1> invalid{-1};
        EXPECT_FALSE(model->forward(invalid, *incremental));
        EXPECT_EQ(incremental->sequence_position(), tokens.size());
    }
}

TEST_P(Gemma4RefTest, ReplayedDecode) {
    using namespace kidi;
    using ops::require;
    const auto tokens = fixture->tokens;
    for (const auto device : devices()) {
        SCOPED_TRACE(tensor::to_string(device));
        const ModuleScope construction(tensor::DType::F32, false, device);
        auto model = create_model(*fixture);
        // Replayed decode steps match eagerly executed steps bit for bit.
        setenv("KIDI_REPLAY", "0", 1);
        auto eager_model = require(model::Gemma4Impl::create(fixture->config));
        unsetenv("KIDI_REPLAY");
        require(eager_model->set_checkpoint(*fixture->checkpoint));
        auto replayed = require(model->create_state(8));
        auto eager = require(eager_model->create_state(8));
        for (std::size_t position = 0; position < tokens.size(); ++position) {
            const auto token = std::span(tokens).subspan(position, 1);
            const auto actual = require(model->forward(token, *replayed));
            const auto expected = require(eager_model->forward(token, *eager));
            ASSERT_TRUE(std::ranges::equal(require(actual.data<float>()), require(expected.data<float>())))
                << "replayed Gemma decode differs from eager at " << position;
        }
        // Token tensors fed back from the previous step decode like host token IDs.
        auto fed = require(model->create_state(8));
        auto integer = require(model->create_state(8));
        auto token = require(tensor::Tensor::from_host({1}, tokens.first(1), device));
        auto next = tokens[0];
        for (std::size_t step = 0; step + 1 < fed->sequence_capacity(); ++step) {
            const auto selected = require(model->forward_device_token(token, *fed));
            next = require(model->forward_token(std::span(&next, 1), *integer));
            ASSERT_EQ(selected.device(), device) << "Gemma token tensor feedback differs at step " << step;
            ASSERT_EQ(require(selected.data<std::int32_t>())[0], next)
                << "Gemma token tensor feedback differs at step " << step;
            token = selected;
        }
        EXPECT_FALSE(
            model->forward_device_token(require(tensor::Tensor::from_host({2}, tokens.first(2), device)), *fed));
    }
}

TEST_P(Gemma4RefTest, CapturedPrefill) {
    const kidi::ModuleScope construction(kidi::tensor::DType::F32, false, kidi::tensor::Device::cpu());
    check_captured_prefill(fixture->config, *fixture->checkpoint, fixture->tokens);
}

TEST_P(Gemma4RefTest, ImagePrefill) {
    const kidi::ModuleScope construction(kidi::tensor::DType::F32, false, kidi::tensor::Device::cpu());
    check_image_prefill_rebinding(fixture->config, *fixture->checkpoint, fixture->tokens);
}

TEST_P(Gemma4RefTest, BatchedDecode) {
    using namespace kidi;
    using ops::require;
    const auto tokens = fixture->tokens;
    for (const auto device : devices()) {
        SCOPED_TRACE(tensor::to_string(device));
        const ModuleScope construction(tensor::DType::F32, false, device);
        auto model = create_model(*fixture);
        for (const std::size_t batch_size : {2, 4}) {
            std::vector<std::unique_ptr<model::LanguageModelState>> batch, serial;
            std::vector<std::size_t> order;
            for (std::size_t row = 0; row < batch_size; ++row) {
                auto source = require(model->create_state(8));
                require(model->prefill(tokens.first(row + 1), *source));
                batch.push_back(require(model->fork_state(*source, row + 1, 8)));
                serial.push_back(require(model->fork_state(*source, row + 1, 8)));
                order.push_back(row);
            }
            std::array<model::LanguageModelState*, 2> duplicate{batch[0].get(), batch[0].get()};
            ASSERT_FALSE(model->forward_batch(std::array{tokens[0], tokens[1]}, duplicate));
            ASSERT_EQ(batch[0]->sequence_position(), 1U);
            for (std::size_t step = 0; step < 3; ++step) {
                if (step == 1) std::ranges::reverse(order);
                if (step == 2) order.pop_back();
                std::vector<model::LanguageModelState*> states;
                std::vector<std::int32_t> input_ids;
                std::vector<float> expected;
                for (auto row : order) {
                    states.push_back(batch[row].get());
                    input_ids.push_back(tokens[(row + step) % tokens.size()]);
                    const auto output =
                        require(model->forward(std::span<const std::int32_t>(&input_ids.back(), 1), *serial[row]));
                    const auto row_values = require(output.data<float>());
                    expected.insert(expected.end(), row_values.begin(), row_values.end());
                }
                const auto output = require(model->forward_batch(input_ids, states));
                const auto actual = require(output.data<float>());
                ASSERT_EQ(actual.size(), expected.size());
                for (std::size_t index = 0; index < actual.size(); ++index) {
                    const auto where = ::testing::Message()
                                       << "batched Gemma mismatch " << GetParam() << ' ' << tensor::to_string(device)
                                       << " batch " << batch_size << " step " << step << " index " << index;
                    ASSERT_TRUE(std::isfinite(actual[index])) << where;
                    ASSERT_LE(std::abs(actual[index] - expected[index]), 2e-4F) << where;
                }
                for (auto row : order) ASSERT_EQ(batch[row]->sequence_position(), serial[row]->sequence_position());
            }
        }
    }
}

TEST_P(Gemma4RefTest, ChunkedPrefill) {
    using namespace kidi;
    using ops::require;
    const auto tokens = fixture->tokens;
    for (const auto device : devices()) {
        SCOPED_TRACE(tensor::to_string(device));
        const ModuleScope construction(tensor::DType::F32, false, device);
        auto model = create_model(*fixture);
        auto chunked = require(model->create_state(8));
        require(model->prefill(tokens.first(3), *chunked));
        auto complete_prefix = require(model->create_state(8));
        require(model->forward(tokens.first(3), *complete_prefix, true));
        ASSERT_EQ(chunked->sequence_position(), complete_prefix->sequence_position());
        const auto& chunked_gemma = static_cast<const model::Gemma4State&>(*chunked);
        const auto& prefix_gemma = static_cast<const model::Gemma4State&>(*complete_prefix);
        for (std::size_t producer = 0; producer < chunked_gemma.layers.size(); ++producer)
            ASSERT_TRUE(require(chunked_gemma.layers[producer].key.copy_to_host()) ==
                            require(prefix_gemma.layers[producer].key.copy_to_host()) &&
                        require(chunked_gemma.layers[producer].value.copy_to_host()) ==
                            require(prefix_gemma.layers[producer].value.copy_to_host()))
                << "cache-only prefill differs from full evaluation at producer " << producer;
        auto snapshot = require(model->fork_state(*chunked, 3, 3));
        const auto saved_key = require(static_cast<const model::Gemma4State&>(*snapshot).layers[0].key.copy_to_host());
        auto resumed = require(model->fork_state(*snapshot, 3, 8));
        const auto resumed_output = require(model->forward(tokens.subspan(3), *resumed, true));
        const auto tail = require(model->forward(tokens.subspan(3), *chunked, true));
        const auto tail_values = require(tail.data<float>());
        ASSERT_TRUE(std::ranges::equal(require(resumed_output.data<float>()), tail_values));
        ASSERT_TRUE(saved_key ==
                    require(static_cast<const model::Gemma4State&>(*snapshot).layers[0].key.copy_to_host()));
        ASSERT_FALSE(model->fork_state(*snapshot, 4, 8));
        ASSERT_FALSE(model->fork_state(*snapshot, 3, 2));
        const auto reference_tail = fixture->values.last(tail_values.size());
        for (std::size_t index = 0; index < tail_values.size(); ++index) {
            ASSERT_TRUE(std::isfinite(tail_values[index])) << "at " << index;
            ASSERT_LE(std::abs(tail_values[index] - reference_tail[index]), 2e-4F) << "at " << index;
        }
    }
}

TEST_P(Gemma4RefTest, Requantization) {
    using namespace kidi;
    using ops::require;
    const auto tokens = fixture->tokens;
    const auto values = fixture->values;
    const auto& checkpoint = *fixture->checkpoint;
    for (const auto device : devices()) {
        SCOPED_TRACE(tensor::to_string(device));
        const ModuleScope construction(tensor::DType::F32, false, device);
        auto model = create_model(*fixture);
        ASSERT_FALSE(model->set_checkpoint(checkpoint, 4, 3));
        require(model->set_checkpoint(checkpoint));
        if (fixture->config["quantization_config"]) {
            ASSERT_FALSE(model->set_checkpoint(checkpoint, 8, 4));
            require(model->set_checkpoint(checkpoint, 0, 128, true));
            auto packed_state = require(model->create_state(8));
            const auto packed_output = require(model->forward(tokens, *packed_state, true));
            const auto packed_values = require(packed_output.data<float>());
            for (std::size_t index = 0; index < values.size(); ++index) {
                ASSERT_TRUE(std::isfinite(packed_values[index])) << "at " << index;
                ASSERT_LE(std::abs(packed_values[index] - values[index]), 2e-4F) << "at " << index;
            }
        } else {
            require(model->set_checkpoint(checkpoint, 8, 4));
            auto quantized_state = require(model->create_state(8));
            const auto quantized = require(model->forward(tokens.first(1), *quantized_state));
            const auto quantized_values = require(quantized.data<float>());
            for (std::size_t index = 0; index < quantized_values.size(); ++index) {
                ASSERT_TRUE(std::isfinite(quantized_values[index])) << "at " << index;
                ASSERT_LE(std::abs(quantized_values[index] - values[index]), 0.01F) << "at " << index;
            }
            require(model->set_checkpoint(checkpoint));
            auto restored_state = require(model->create_state(8));
            const auto restored = require(model->forward(tokens.first(1), *restored_state));
            const auto restored_values = require(restored.data<float>());
            for (std::size_t index = 0; index < restored_values.size(); ++index) {
                ASSERT_TRUE(std::isfinite(restored_values[index])) << "at " << index;
                ASSERT_LE(std::abs(restored_values[index] - values[index]), 2e-4F) << "at " << index;
            }
        }
    }
}

TEST_P(Gemma4RefTest, LongPrefill) {
    using namespace kidi;
    using ops::require;
    const auto tokens = fixture->tokens;
    for (const auto device : devices()) {
        SCOPED_TRACE(tensor::to_string(device));
        const ModuleScope construction(tensor::DType::F32, false, device);
        for (const std::size_t length : {259, 1027}) {
            SCOPED_TRACE(::testing::Message() << "length " << length);
            const auto capacity = ((length + 127) / 128) * 128;
            auto extended_config = YAML::Clone(fixture->config);
            extended_config["max_position_embeddings"] = capacity;
            auto extended = require(model::Gemma4Impl::create(extended_config));
            require(extended->set_checkpoint(*fixture->checkpoint));
            std::vector<std::int32_t> sequence(length);
            for (std::size_t index = 0; index < sequence.size(); ++index)
                sequence[index] = tokens[index % tokens.size()];
            auto prefix_state = require(extended->create_state(capacity));
            const auto extended_cache_dtype = expected_cache_dtype(qat(), device, capacity);
            for (const auto& cache : static_cast<const model::Gemma4State&>(*prefix_state).layers) {
                ASSERT_EQ(cache.key.dtype(), extended_cache_dtype);
                ASSERT_EQ(cache.value.dtype(), extended_cache_dtype);
            }
            const auto prefix = require(extended->forward(sequence, *prefix_state));
            auto oracle_state = require(extended->create_state(capacity));
            const auto all_logits = require(extended->forward(sequence, *oracle_state, true));
            const auto last_values = require(prefix.data<float>());
            const auto oracle_values = require(all_logits.data<float>()).last(last_values.size());
            {
                const auto [gap, at] = difference(last_values, oracle_values);
                ASSERT_LE(gap, 2e-4F) << "incremental vs all-logits differs by " << gap << " at " << at;
            }
            auto history_state = require(extended->create_state(capacity));
            extended->configure_state(*history_state, true, true);
            require(extended->prefill(std::span(sequence).first(length - 1), *history_state));
            const auto history = require(extended->forward(std::span(sequence).last(1), *history_state));
            const auto history_values = require(history.data<float>());
            auto bucket_state = require(extended->create_state(capacity));
            require(extended->prefill(std::span(sequence).first(128), *bucket_state));
            require(extended->prefill(std::span(sequence).subspan(128, length - 129), *bucket_state));
            const auto bucket = require(extended->forward(std::span(sequence).last(1), *bucket_state));
            const auto prefix_values = require(prefix.data<float>()), bucket_values = require(bucket.data<float>());
            auto larger_state = require(extended->create_state(capacity));
            const std::size_t chunk = length == 259 ? 256 : 512;
            for (std::size_t offset = 0; offset < length - 3; offset += chunk)
                require(extended->prefill(std::span(sequence).subspan(offset, chunk), *larger_state));
            const auto larger = require(extended->forward(std::span(sequence).last(3), *larger_state));
            const auto larger_values = require(larger.data<float>());
            // CPU prefill is batch-invariant, so every chunking yields the same logits. Metal's QAT prefill
            // multiplies FP16-expanded weights, which can move a calibrated activation by one int8 step from
            // the exact decode kernel, so accelerators must produce the same tokens rather than logits.
            if (device == tensor::Device::cpu()) {
                for (const auto& [name, other] :
                     {std::pair{"full prefix", prefix_values}, std::pair{"uncropped history", history_values},
                      std::pair{"larger chunks", larger_values}}) {
                    const auto [gap, at] = difference(bucket_values, other);
                    ASSERT_LE(gap, 2e-4F) << "chunked prefill vs " << name << " differs by " << gap << " at " << at;
                }
            }
            // Greedy tokens after the prompt: the argmax of its last logits, then three decoded tokens. These
            // run last because decoding may reuse the buffers holding earlier logits.
            const auto continuation = [&](std::span<const float> logits, model::LanguageModelState& state) {
                std::vector<std::int32_t> generated{
                    static_cast<std::int32_t>(std::ranges::max_element(logits) - logits.begin())};
                for (int step = 0; step < 3; ++step) {
                    const auto next = require(extended->forward_token(std::span(generated).last(1), state));
                    generated.push_back(next);
                }
                return generated;
            };
            const std::array generated{
                continuation(bucket_values, *bucket_state), continuation(prefix_values, *prefix_state),
                continuation(history_values, *history_state), continuation(larger_values, *larger_state)};
            for (std::size_t index = 1; index < generated.size(); ++index)
                ASSERT_EQ(generated[index], generated[0])
                    << "chunking " << index << " generates different tokens from 128-token chunks";
        }
    }
}

TEST(Gemma4Test, QatRejectsZeroScales) {
    using namespace kidi;
    const auto fixture = load_fixture("gemma4-qat");
    const auto& directory = fixture->directory;
    const auto& config = fixture->config;
    const auto sentinel_path = std::filesystem::temp_directory_path() / "kidi-gemma4-zero-scale.safetensors";
    std::filesystem::copy_file(directory / "model.safetensors", sentinel_path,
                               std::filesystem::copy_options::overwrite_existing);
    std::array<std::streamoff, 4> offsets;
    std::streamoff weight_scale_offset;
    {
        const kidi::checkpoint::safetensors::MappedCheckpoint mapped(sentinel_path.string());
        const auto base = mapped.owner()->data();
        offsets = {
            mapped.at("lm_head.input_activation_scale").data - base,
            mapped.at("lm_head.output_activation_scale").data - base,
            mapped.at("model.language_model.layers.0.self_attn.k_cache_scale").data - base,
            mapped.at("model.language_model.layers.0.self_attn.v_cache_scale").data - base,
        };
        weight_scale_offset = mapped.at("lm_head.weight_scale").data - base;
    }
    {
        std::fstream output(sentinel_path, std::ios::binary | std::ios::in | std::ios::out);
        constexpr float ZERO = 0.F;
        for (const auto offset : offsets) {
            output.seekp(offset);
            output.write(reinterpret_cast<const char*>(&ZERO), sizeof(ZERO));
        }
    }
    auto sentinel_checkpoint = ops::require(checkpoint::Weights::load(sentinel_path));
    ops::require(sentinel_checkpoint.add(
        "model.language_model.layers.2.self_attn.kv_proj.weight",
        ops::require(sentinel_checkpoint.tensor("model.language_model.layers.0.self_attn.k_proj.weight"))));
    const ModuleScope construction(tensor::DType::F32, false, tensor::Device::cpu());
    auto sentinel_model = ops::require(model::Gemma4Impl::create(config));
    ops::require(sentinel_model->set_checkpoint(sentinel_checkpoint));
    auto state_ptr = ops::require(sentinel_model->create_state(config["max_position_embeddings"].as<std::size_t>()));
    const auto& state = static_cast<const model::Gemma4State&>(*state_ptr);
    ASSERT_FALSE(std::ranges::any_of(state.layers, [](const auto& layer) {
        return layer.key.dtype() != tensor::DType::F32 || layer.value.dtype() != tensor::DType::F32;
    }));
    const std::array<std::int32_t, 2> uncalibrated_tokens{2, 6};
    const auto uncalibrated_logits = ops::require(sentinel_model->forward(uncalibrated_tokens, *state_ptr));
    ASSERT_FALSE(std::ranges::any_of(ops::require(uncalibrated_logits.data<float>()),
                                     [](float value) { return !std::isfinite(value); }));
    {
        std::fstream output(sentinel_path, std::ios::binary | std::ios::in | std::ios::out);
        constexpr float ZERO = 0.F;
        output.seekp(weight_scale_offset);
        output.write(reinterpret_cast<const char*>(&ZERO), sizeof(ZERO));
    }
    auto invalid_checkpoint = ops::require(checkpoint::Weights::load(sentinel_path));
    ASSERT_FALSE(sentinel_model->set_checkpoint(invalid_checkpoint));
    std::filesystem::remove(sentinel_path);
}

TEST(Gemma4Test, BfloatCudaModel) {
    if (!executes(kidi::tensor::Device::cuda())) GTEST_SKIP() << "CUDA execution is unavailable";
    const auto fixture = load_fixture("gemma4");
    check_bfloat_model(fixture->config, *fixture->checkpoint, fixture->tokens);
}

TEST(Gemma4Test, AsyncCudaServing) {
    if (!executes(kidi::tensor::Device::cuda())) GTEST_SKIP() << "CUDA execution is unavailable";
    check_async_serving(load_fixture("gemma4")->config);
}

TEST(Gemma4Test, HnswHeadMatchesDense) {
    using namespace kidi;
    using ops::require;
    const auto fixture = load_fixture("gemma4");
    const ModuleScope construction(tensor::DType::F32, false, tensor::Device::cpu());
    auto model = create_model(*fixture);
    auto hnsw_model = create_model(*fixture);
    ASSERT_NO_FATAL_FAILURE(check_hnsw_head(
        hnsw_model, require(fixture->checkpoint->tensor("model.language_model.embed_tokens.weight")), fixture->tokens));
    StateDict bfloat;
    ops::Context context(tensor::Device::cpu());
    for (const auto& [name, value] : model->state_dict())
        bfloat.emplace(name, value.dimensions() == 2 ? context.cast(value, tensor::DType::BF16) : value);
    const ModuleScope bfloat_scope(tensor::DType::BF16, false, tensor::Device::cpu());
    auto bfloat_model = require(model::Gemma4Impl::create(fixture->config));
    require(bfloat_model->set_state(bfloat));
    SCOPED_TRACE("BF16");
    check_hnsw_head(bfloat_model, bfloat.at("embed_tokens.weight"), fixture->tokens);
}

TEST(Gemma4Test, QuantizedTokenPrefill) {
    using namespace kidi;
    using ops::require;
    const auto fixture = load_fixture("gemma4");
    const auto tokens = fixture->tokens;
    const ModuleScope construction(tensor::DType::F32, false, tensor::Device::cpu());
    for (const auto bits : {4, 8}) {
        SCOPED_TRACE(bits);
        auto quantized = require(model::Gemma4Impl::create(fixture->config));
        require(quantized->set_checkpoint(*fixture->checkpoint, bits, 2));
        ASSERT_TRUE(quantized->last_token_prefill());
        auto quantized_state = require(quantized->create_state(8));
        const auto first = require(quantized->forward_token(tokens.first(2), *quantized_state));
        const std::array next_input{first};
        const auto second = require(quantized->forward_token(next_input, *quantized_state));
        EXPECT_GE(first, 0);
        EXPECT_GE(second, 0);
    }
}

TEST(Gemma4Test, QatPrecisionsFinite) {
    using namespace kidi;
    using ops::require;
    const auto fixture = load_fixture("gemma4-qat");
    const ModuleScope construction(tensor::DType::F32, false, tensor::Device::cpu());
    auto model = create_model(*fixture);
    std::vector precisions{
        core::InferencePrecision::FP32,    core::InferencePrecision::BF16,    core::InferencePrecision::Q2A16,
        core::InferencePrecision::Q4A16,   core::InferencePrecision::Q8A16,   core::InferencePrecision::Q2AE4M3,
        core::InferencePrecision::Q4AE4M3, core::InferencePrecision::Q8AE4M3, core::InferencePrecision::Q2AE5M2,
        core::InferencePrecision::Q4AE5M2, core::InferencePrecision::Q8AE5M2, core::InferencePrecision::QAT_FP32,
    };
#if defined(__APPLE__)
    precisions.push_back(core::InferencePrecision::LOWBIT_PARITY);
#endif
    for (const auto precision : precisions) {
        SCOPED_TRACE(core::to_string(precision));
        model->set_precision(precision);
        auto precision_state = require(model->create_state(8));
        const auto precision_output = require(model->forward(fixture->tokens, *precision_state, true));
        for (const auto value : require(precision_output.data<float>())) ASSERT_TRUE(std::isfinite(value));
    }
    model->set_precision(core::InferencePrecision::CHECKPOINT);
}
} // namespace
