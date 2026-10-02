#include "kidi/model/gemma4.h"
#include "kidi/checkpoint/safetensors/mapped.h"
#include "kidi/inference/generator.h"
#include "kidi/runtime/operator.h"
#include "kidi/tensor/backend.h"
#include "kidi/tensor/external.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <utility>
#include <string>

namespace {
/// A step compiler that declines every step, so accelerator-shaped steps (captured prefill chunks) replay on CPU.
class DecliningCompiler final : public kidi::runtime::StepCompiler {
public:
    auto name() const -> std::string_view override { return "declining"; }
    auto compile(const kidi::graph::Graph&,
                 std::string_view key) -> std::unique_ptr<kidi::runtime::StepExecutable> override {
        keys.emplace_back(key);
        return nullptr;
    }
    std::vector<std::string> keys;
};

/// Captured prefill chunks and decode steps under a step compiler match eager execution bit for bit.
auto check_captured_prefill(const YAML::Node& config, const kidi::checkpoint::Weights& checkpoint,
                            std::span<const std::int32_t> tokens) -> bool {
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
    if ((*captured)->accelerator() != "declining" || !eager->accelerator().empty()) {
        std::cerr << "step compiler was not attached to the captured model\n";
        return false;
    }
    std::vector<std::int32_t> sequence(80);
    for (std::size_t index = 0; index < sequence.size(); ++index) sequence[index] = tokens[index % tokens.size()];
    auto captured_state = require((*captured)->create_state(128));
    auto eager_state = require(eager->create_state(128));
    for (std::size_t offset = 0; offset < 64; offset += 16) {
        require((*captured)->prefill(std::span(sequence).subspan(offset, 16), captured_state));
        require(eager->prefill(std::span(sequence).subspan(offset, 16), eager_state));
    }
    for (std::size_t position = 64; position < sequence.size(); ++position) {
        const auto token = std::span(sequence).subspan(position, 1);
        const auto actual = require((*captured)->forward(token, captured_state));
        const auto expected = require(eager->forward(token, eager_state));
        if (!std::ranges::equal(require(actual.data<float>()), require(expected.data<float>()))) {
            std::cerr << "captured prefill or decode differs from eager at " << position << '\n';
            return false;
        }
    }
    const auto compiled = [&](std::string_view prefix) {
        return std::ranges::any_of(compiler->keys, [&](const std::string& key) { return key.starts_with(prefix); });
    };
    if (compiled("gemma4_prefill:16:") && compiled("gemma4_decode:1:")) return true;
    std::cerr << "expected captured prefill and decode steps; compiled:";
    for (const auto& key : compiler->keys) std::cerr << ' ' << key;
    std::cerr << '\n';
    return false;
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
    auto gather(std::span<const std::int32_t> rows, std::span<std::byte> destination) const
        -> kidi::Result<void> override {
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
    const auto directory = std::filesystem::temp_directory_path() / "kidi-external-rows-test";
    std::filesystem::remove_all(directory);
    for (const auto* name : {"internal", "external"}) std::filesystem::create_directories(directory / name);
    auto reduced = state;
    reduced.erase(table_name);
    require(kidi::checkpoint::Weights::save(directory / "internal" / "model.safetensors", state));
    require(kidi::checkpoint::Weights::save(directory / "external" / "model.safetensors", reduced));
    YAML::Node manifest;
    manifest["format_version"] = 1;
    manifest["weights_file"] = "model.safetensors";
    manifest["tokenizer_file"] = "tokenizer.json";
    manifest["model"] = YAML::Clone(config);
    manifest["decode"]["maximum_new_tokens"] = 4;
    manifest["decode"]["context_size"] = 16;
    std::ofstream(directory / "internal" / "model.yaml") << manifest;
    const auto& table = state.at(table_name);
    YAML::Node declared;
    declared["name"] = table_name;
    declared["dtype"] = std::string(kidi::tensor::to_string(table.dtype()));
    declared["shape"] = std::vector<std::int64_t>(table.shape().begin(), table.shape().end());
    manifest["external_tensors"].push_back(declared);
    std::ofstream(directory / "external" / "model.yaml") << manifest;
    for (const auto* name : {"internal", "external"}) write_tokenizer(directory / name / "tokenizer.json");
    if (kidi::inference::Generator::load(directory / "external", device))
        throw std::runtime_error("declared external tensors loaded without a source");
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
    if (actual.generation.token_ids != expected.generation.token_ids || rows->gathered < 5 + 3)
        throw std::runtime_error("external per-layer embeddings changed tokens or skipped row gathers");
    std::filesystem::remove_all(directory);
}

auto check_serving_cache(const std::filesystem::path& fixture, const YAML::Node& config,
                         kidi::tensor::Device device) -> void {
    using kidi::ops::require;
    const auto directory = std::filesystem::temp_directory_path() / "kidi-serving-cache-test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    std::filesystem::copy_file(fixture / "model.safetensors", directory / "model.safetensors");
    YAML::Node manifest;
    manifest["format_version"] = 1;
    manifest["weights_file"] = "model.safetensors";
    manifest["tokenizer_file"] = "tokenizer.json";
    manifest["model"] = YAML::Clone(config);
    manifest["decode"]["maximum_new_tokens"] = 3;
    manifest["decode"]["context_size"] = 16;
    std::ofstream(directory / "model.yaml") << manifest;
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
    const auto check = [&](std::string_view prompt, auto settings, std::size_t reused) {
        const auto actual = run(cached, prompt, settings);
        settings.prefix_cache_bytes = 0;
        const auto expected = run(plain, prompt, settings);
        if (actual.generation.token_ids != expected.generation.token_ids || actual.text != expected.text ||
            actual.stats.reused_prompt_tokens != reused)
            throw std::runtime_error("serving prefix reuse changed output or reused the wrong token count");
        return actual;
    };
    const auto first = check("alpha beta gamma delta", options, 0);
    if (!first.stats.prefix_cache_bytes || first.stats.prefix_reserved_bytes > options.prefix_cache_bytes)
        throw std::runtime_error("serving cache did not retain bounded KV storage");
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
    if (check("zeta eta", too_small, 0).stats.prefix_reserved_bytes)
        throw std::runtime_error("serving cache exceeded its byte budget");
    check("alpha beta gamma delta", options, 0);
    require(cached.configure_serving({1, 4, 16, 2}));
    check("alpha beta gamma delta", options, 0);
    auto uncached = options;
    uncached.prefix_cache_bytes = 0;
    check("alpha beta gamma delta", uncached, 0);
    const auto cancelled = require(cached.enqueue("alpha beta gamma delta", options));
    require(cached.step());
    require(cached.cancel(cancelled));
    check("alpha beta gamma delta", options, 2);
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
        if (!results[index] || results[index]->generation.token_ids != expected.generation.token_ids ||
            results[index]->stats.reused_prompt_tokens != (index == 0 ? 3 : 0))
            throw std::runtime_error("queued requests shared or lost streaming KV state");
    }
    // Deferred tokens report each selection one step later, alone and batched, without changing the generation.
    auto deferred = require(kidi::inference::Generator::load(directory, device));
    for (const std::size_t active : {1, 2}) {
        require(deferred.configure_serving(
            {.maximum_active = active, .maximum_requests = 4, .cache_token_budget = 32, .prefill_tokens_per_step = 2,
             .deferred_tokens = true}));
        const std::array deferred_ids{require(deferred.enqueue(prompts[0], uncached)),
                                      require(deferred.enqueue(prompts[1], uncached))};
        std::array<std::optional<kidi::inference::TextGeneration>, 2> deferred_results;
        while (deferred.pending_requests())
            for (auto& event : require(deferred.step()).events)
                if (event.completed)
                    deferred_results[event.request_id == deferred_ids[0] ? 0 : 1] = std::move(event.completed);
        for (std::size_t index = 0; index < deferred_results.size(); ++index)
            if (!deferred_results[index] || deferred_results[index]->generation.token_ids !=
                                                run(plain, prompts[index], uncached).generation.token_ids)
                throw std::runtime_error("deferred token serving changed generated tokens");
    }
    manifest["model"]["max_position_embeddings"] = 1024;
    std::ofstream(directory / "model.yaml") << manifest;
    auto compact = require(kidi::inference::Generator::load(directory, device));
    auto full = require(kidi::inference::Generator::load(directory, device));
    require(compact.configure_serving({1, 1, 1024, 2, true}));
    require(full.configure_serving({1, 1, 1024, 2}));
    auto bounded = uncached;
    bounded.context_size = 1024;
    require(compact.enqueue(prompts[0], bounded));
    const auto step = require(compact.step());
    const auto byte_limit = kidi::tensor::DEVICE_CAPABILITIES[device.kind].blockwise_int8_attention_max_tokens;
    const auto expected_capacity = byte_limit < 1024 ? ((byte_limit + 128) / 128) * 128 : 128;
    if (step.reserved_cache_tokens != expected_capacity)
        throw std::runtime_error("compact reservation changed the cache precision policy or exceeded its bucket");
    std::optional<kidi::inference::TextGeneration> compact_result;
    while (compact.pending_requests())
        for (auto& event : require(compact.step()).events)
            if (event.completed) compact_result = std::move(event.completed);
    const auto full_result = run(full, prompts[0], bounded);
    if (!compact_result || compact_result->generation.token_ids != full_result.generation.token_ids)
        throw std::runtime_error("compact cache reservation changed generated tokens");
    bounded.maximum_new_tokens = 1024;
    if (compact.enqueue(prompts[0], bounded)) throw std::runtime_error("compact cache accepted context overflow");
    std::filesystem::remove_all(directory);
}
} // namespace

// Failing checks report where they failed: several compare exactly, so results can differ between machines.
std::string current_fixture = "setup", current_device = "none";
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
auto failed(int line) -> int {
    std::cerr << "gemma4_test check at line " << line << " failed for " << current_fixture << " on " << current_device
              << '\n';
    return 1;
}

auto main() -> int {
    using namespace kidi;
    try {
        for (const auto* fixture : {"gemma4", "gemma4-qat"}) {
            current_fixture = fixture;
            current_device = "cpu";
            const auto directory = fixture_directory(fixture);
            const auto config = YAML::LoadFile((directory / "model.yaml").string())["model"];
            auto checkpoint = ops::require(checkpoint::Weights::load(directory / "model.safetensors"));
            auto reference = ops::require(checkpoint::Weights::load(directory / "reference.safetensors"));
            if (fixture == std::string_view("gemma4-qat")) {
                const auto sentinel_path =
                    std::filesystem::temp_directory_path() / "kidi-gemma4-zero-scale.safetensors";
                std::filesystem::copy_file(directory / "model.safetensors", sentinel_path,
                                           std::filesystem::copy_options::overwrite_existing);
                std::array<std::streamoff, 2> offsets;
                {
                    const kidi::checkpoint::safetensors::MappedCheckpoint mapped(sentinel_path.string());
                    const auto base = mapped.owner()->data();
                    offsets = {
                        mapped.at("lm_head.input_activation_scale").data - base,
                        mapped.at("lm_head.output_activation_scale").data - base,
                    };
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
                const ModuleScope construction(tensor::DType::F32, false, tensor::Device::cpu());
                auto sentinel_model = ops::require(model::Gemma4Impl::create(config));
                ops::require(sentinel_model->set_checkpoint(sentinel_checkpoint));
                std::filesystem::remove(sentinel_path);
            }
            const auto token_tensor = ops::require(reference.tensor("tokens"));
            const auto expected = ops::require(reference.tensor("logits"));
            const auto tokens = ops::require(token_tensor.data<std::int32_t>());
            const auto values = ops::require(expected.data<float>());
            auto invalid_config = YAML::Clone(config);
            invalid_config.remove("global_head_dim");
            if (model::Gemma4Impl::create(invalid_config)) {
                std::cerr << "Gemma accepted a configuration without global_head_dim\n";
                return failed(__LINE__);
            }
            std::vector devices{tensor::Device::cpu()};
#if defined(__APPLE__)
            devices.push_back(tensor::Device::apple_gpu());
#endif
            const auto vulkan = tensor::BackendRegistry::instance().backend(tensor::Device::vulkan());
            if (vulkan && (*vulkan)->is_available(tensor::Device::vulkan()) && (*vulkan)->supports_execution())
                devices.push_back(tensor::Device::vulkan());
            for (auto device : devices) {
                current_device = tensor::to_string(device);
                check_serving_cache(directory, config, device);
                check_external_rows(directory, config, device);
                const ModuleScope construction(tensor::DType::F32, false, device);
                auto model = ops::require(model::Gemma4Impl::create(config));
                ops::require(model->set_checkpoint(checkpoint));
                const auto parameters = model->state_dict();
                for (const auto& [name, parameter] : parameters)
                    if (name.find(".mlp.gate_proj.") != std::string::npos ||
                        name.find(".mlp.up_proj.") != std::string::npos)
                        return failed(__LINE__);
                for (int layer = 0; layer < config["num_hidden_layers"].as<int>(); ++layer) {
                    const auto prefix = "layers." + std::to_string(layer) + ".mlp.";
                    const auto gate =
                        ops::require(checkpoint.tensor("model.language_model." + prefix + "gate_proj.weight"));
                    const auto& combined = parameters.at(prefix + "gate_up_proj.weight");
                    if (combined.size(0) != 2 * gate.size(0) || combined.size(1) != gate.size(1)) return failed(__LINE__);
                }
                auto full = ops::require(model->create_state(8));
                const auto* cache_override = std::getenv("KIDI_INT8_KV_CACHE");
                const auto& capabilities = tensor::DEVICE_CAPABILITIES[device.kind];
                const auto cache_dtype =
                    fixture == std::string_view("gemma4-qat") &&
                            (cache_override == nullptr || std::string_view(cache_override) != "0") &&
                            capabilities.calibrated_int8_cast && capabilities.blockwise_int8_attention &&
                            8 <= capabilities.blockwise_int8_attention_max_tokens
                        ? tensor::DType::I8
                        : tensor::DType::F32;
                for (const auto& cache : full.layers)
                    if (cache.key.dtype() != cache_dtype || cache.value.dtype() != cache_dtype) return failed(__LINE__);
                auto prefill = ops::require(model->forward(tokens, full, true));
                const auto prefill_values = ops::require(prefill.data<float>());
                if (prefill_values.size() != values.size()) return failed(__LINE__);
                for (std::size_t index = 0; index < values.size(); ++index)
                    if (!std::isfinite(prefill_values[index]) ||
                        std::abs(prefill_values[index] - values[index]) > 2e-4F) {
                        std::cerr << "Gemma reference mismatch at " << index << ": " << prefill_values[index]
                                  << " != " << values[index] << '\n';
                        return failed(__LINE__);
                    }
                auto incremental = ops::require(model->create_state(8));
                auto selected_state = ops::require(model->create_state(8));
                for (std::size_t position = 0; position < tokens.size(); ++position) {
                    auto output = ops::require(model->forward(std::span(tokens).subspan(position, 1), incremental));
                    const auto actual = ops::require(output.data<float>());
                    const auto selected =
                        ops::require(model->forward_token(std::span(tokens).subspan(position, 1), selected_state));
                    if (selected != std::ranges::max_element(actual) - actual.begin()) return failed(__LINE__);
                    for (std::size_t token = 0; token < actual.size(); ++token)
                        if (!std::isfinite(actual[token]) ||
                            std::abs(actual[token] - values[position * actual.size() + token]) > 2e-4F) {
                            std::cerr << "Gemma cached logits differ at " << position << ':' << token << '\n';
                            return failed(__LINE__);
                        }
                }
                if (incremental.position != tokens.size() || full.position != tokens.size()) return failed(__LINE__);
                {
                    // Replayed decode steps match eagerly executed steps bit for bit.
                    setenv("KIDI_REPLAY", "0", 1);
                    auto eager_model = ops::require(model::Gemma4Impl::create(config));
                    unsetenv("KIDI_REPLAY");
                    ops::require(eager_model->set_checkpoint(checkpoint));
                    auto replayed = ops::require(model->create_state(8));
                    auto eager = ops::require(eager_model->create_state(8));
                    for (std::size_t position = 0; position < tokens.size(); ++position) {
                        const auto token = std::span(tokens).subspan(position, 1);
                        const auto actual = ops::require(model->forward(token, replayed));
                        const auto expected = ops::require(eager_model->forward(token, eager));
                        if (!std::ranges::equal(ops::require(actual.data<float>()),
                                                ops::require(expected.data<float>()))) {
                            std::cerr << "replayed Gemma decode differs from eager at " << position << '\n';
                            return failed(__LINE__);
                        }
                    }
                    // Token tensors fed back from the previous step decode like host token IDs.
                    auto fed = ops::require(model->create_state(8));
                    auto integer = ops::require(model->create_state(8));
                    auto token = ops::require(tensor::Tensor::from_host({1}, tokens.first(1), device));
                    auto next = tokens[0];
                    for (std::size_t step = 0; step + 1 < fed.capacity; ++step) {
                        const auto selected = ops::require(model->forward_token(token, fed));
                        next = ops::require(model->forward_token(std::span(&next, 1), integer));
                        if (ops::require(selected.data<std::int32_t>())[0] != next) {
                            std::cerr << "Gemma token tensor feedback differs at step " << step << '\n';
                            return failed(__LINE__);
                        }
                        token = selected;
                    }
                    if (model->forward_token(ops::require(tensor::Tensor::from_host({2}, tokens.first(2), device)),
                                             fed))
                        return failed(__LINE__);
                    if (device == tensor::Device::cpu() && !check_captured_prefill(config, checkpoint, tokens))
                        return failed(__LINE__);
                }
                for (const std::size_t batch_size : {2, 4}) {
                    std::vector<model::Gemma4State> batch, serial;
                    std::vector<std::size_t> order;
                    for (std::size_t row = 0; row < batch_size; ++row) {
                        auto source = ops::require(model->create_state(8));
                        ops::require(model->prefill(tokens.first(row + 1), source));
                        batch.push_back(ops::require(model->fork_state(source, row + 1, 8)));
                        serial.push_back(ops::require(model->fork_state(source, row + 1, 8)));
                        order.push_back(row);
                    }
                    std::array duplicate{&batch[0], &batch[0]};
                    if (model->forward_batch(std::array{tokens[0], tokens[1]}, duplicate) || batch[0].position != 1)
                        return failed(__LINE__);
                    for (std::size_t step = 0; step < 3; ++step) {
                        if (step == 1) std::ranges::reverse(order);
                        if (step == 2) order.pop_back();
                        std::vector<model::Gemma4State*> states;
                        std::vector<std::int32_t> input_ids;
                        std::vector<float> expected;
                        for (auto row : order) {
                            states.push_back(&batch[row]);
                            input_ids.push_back(tokens[(row + step) % tokens.size()]);
                            const auto output = ops::require(
                                model->forward(std::span<const std::int32_t>(&input_ids.back(), 1), serial[row]));
                            const auto row_values = ops::require(output.data<float>());
                            expected.insert(expected.end(), row_values.begin(), row_values.end());
                        }
                        const auto output = ops::require(model->forward_batch(input_ids, states));
                        const auto actual = ops::require(output.data<float>());
                        if (actual.size() != expected.size()) return failed(__LINE__);
                        for (std::size_t index = 0; index < actual.size(); ++index)
                            if (!std::isfinite(actual[index]) || std::abs(actual[index] - expected[index]) > 2e-4F) {
                                std::cerr << "batched Gemma mismatch " << fixture << ' ' << tensor::to_string(device)
                                          << " batch " << batch_size << " step " << step << " index " << index << '\n';
                                return failed(__LINE__);
                            }
                        for (auto row : order)
                            if (batch[row].position != serial[row].position) return failed(__LINE__);
                    }
                }
                auto chunked = ops::require(model->create_state(8));
                ops::require(model->prefill(tokens.first(3), chunked));
                auto complete_prefix = ops::require(model->create_state(8));
                ops::require(model->forward(tokens.first(3), complete_prefix, true));
                if (chunked.position != complete_prefix.position) return failed(__LINE__);
                for (std::size_t producer = 0; producer < chunked.layers.size(); ++producer)
                    if (ops::require(chunked.layers[producer].key.copy_to_host()) !=
                            ops::require(complete_prefix.layers[producer].key.copy_to_host()) ||
                        ops::require(chunked.layers[producer].value.copy_to_host()) !=
                            ops::require(complete_prefix.layers[producer].value.copy_to_host())) {
                        std::cerr << "cache-only prefill differs from full evaluation at producer " << producer << '\n';
                        return failed(__LINE__);
                    }
                auto snapshot = ops::require(model->fork_state(chunked, 3, 3));
                const auto saved_key = ops::require(snapshot.layers[0].key.copy_to_host());
                auto resumed = ops::require(model->fork_state(snapshot, 3, 8));
                const auto resumed_output = ops::require(model->forward(tokens.subspan(3), resumed, true));
                const auto tail = ops::require(model->forward(tokens.subspan(3), chunked, true));
                const auto tail_values = ops::require(tail.data<float>());
                if (!std::ranges::equal(ops::require(resumed_output.data<float>()), tail_values) ||
                    saved_key != ops::require(snapshot.layers[0].key.copy_to_host()) ||
                    model->fork_state(snapshot, 4, 8) || model->fork_state(snapshot, 3, 2))
                    return failed(__LINE__);
                const auto reference_tail = values.last(tail_values.size());
                for (std::size_t index = 0; index < tail_values.size(); ++index)
                    if (!std::isfinite(tail_values[index]) ||
                        std::abs(tail_values[index] - reference_tail[index]) > 2e-4F)
                        return failed(__LINE__);
                const std::array<std::int32_t, 1> invalid{-1};
                if (model->forward(invalid, incremental) || incremental.position != tokens.size()) return failed(__LINE__);
                if (model->set_checkpoint(checkpoint, 4, 3)) return failed(__LINE__);
                ops::require(model->set_checkpoint(checkpoint));
                if (config["quantization_config"]) {
                    if (model->set_checkpoint(checkpoint, 8, 4)) return failed(__LINE__);
                    ops::require(model->set_checkpoint(checkpoint, 0, 128, true));
                    auto packed_state = ops::require(model->create_state(8));
                    const auto packed_output = ops::require(model->forward(tokens, packed_state, true));
                    const auto packed_values = ops::require(packed_output.data<float>());
                    for (std::size_t index = 0; index < values.size(); ++index)
                        if (!std::isfinite(packed_values[index]) ||
                            std::abs(packed_values[index] - values[index]) > 2e-4F)
                            return failed(__LINE__);
                } else {
                    ops::require(model->set_checkpoint(checkpoint, 8, 4));
                    auto quantized_state = ops::require(model->create_state(8));
                    const auto quantized = ops::require(model->forward(tokens.first(1), quantized_state));
                    const auto quantized_values = ops::require(quantized.data<float>());
                    for (std::size_t index = 0; index < quantized_values.size(); ++index)
                        if (!std::isfinite(quantized_values[index]) ||
                            std::abs(quantized_values[index] - values[index]) > 0.01F)
                            return failed(__LINE__);
                    ops::require(model->set_checkpoint(checkpoint));
                    auto restored_state = ops::require(model->create_state(8));
                    const auto restored = ops::require(model->forward(tokens.first(1), restored_state));
                    const auto restored_values = ops::require(restored.data<float>());
                    for (std::size_t index = 0; index < restored_values.size(); ++index)
                        if (!std::isfinite(restored_values[index]) ||
                            std::abs(restored_values[index] - values[index]) > 2e-4F)
                            return failed(__LINE__);
                }
                for (const std::size_t length : {259, 1027}) {
                    const auto capacity = ((length + 127) / 128) * 128;
                    auto extended_config = YAML::Clone(config);
                    extended_config["max_position_embeddings"] = capacity;
                    auto extended = ops::require(model::Gemma4Impl::create(extended_config));
                    ops::require(extended->set_checkpoint(checkpoint));
                    std::vector<std::int32_t> sequence(length);
                    for (std::size_t index = 0; index < sequence.size(); ++index)
                        sequence[index] = tokens[index % tokens.size()];
                    auto prefix_state = ops::require(extended->create_state(capacity));
                    const auto extended_cache_dtype =
                        fixture == std::string_view("gemma4-qat") &&
                                (cache_override == nullptr || std::string_view(cache_override) != "0") &&
                                capabilities.calibrated_int8_cast && capabilities.blockwise_int8_attention &&
                                capacity <= capabilities.blockwise_int8_attention_max_tokens
                            ? tensor::DType::I8
                            : tensor::DType::F32;
                    for (const auto& cache : prefix_state.layers)
                        if (cache.key.dtype() != extended_cache_dtype || cache.value.dtype() != extended_cache_dtype)
                            return failed(__LINE__);
                    const auto prefix = ops::require(extended->forward(sequence, prefix_state));
                    auto oracle_state = ops::require(extended->create_state(capacity));
                    const auto all_logits = ops::require(extended->forward(sequence, oracle_state, true));
                    const auto last_values = ops::require(prefix.data<float>());
                    const auto oracle_values = ops::require(all_logits.data<float>()).last(last_values.size());
                    if (const auto [gap, at] = difference(last_values, oracle_values); gap > 2e-4F) {
                        std::cerr << "length " << length << ": incremental vs all-logits differs by " << gap << " at "
                                  << at << '\n';
                        return failed(__LINE__);
                    }
                    auto history_state = ops::require(extended->create_state(capacity));
                    history_state.crop_local_attention = false;
                    ops::require(extended->prefill(std::span(sequence).first(length - 1), history_state));
                    const auto history = ops::require(extended->forward(std::span(sequence).last(1), history_state));
                    const auto history_values = ops::require(history.data<float>());
                    auto bucket_state = ops::require(extended->create_state(capacity));
                    ops::require(extended->prefill(std::span(sequence).first(128), bucket_state));
                    ops::require(extended->prefill(std::span(sequence).subspan(128, length - 129), bucket_state));
                    const auto bucket = ops::require(extended->forward(std::span(sequence).last(1), bucket_state));
                    const auto prefix_values = ops::require(prefix.data<float>()),
                               bucket_values = ops::require(bucket.data<float>());
                    auto larger_state = ops::require(extended->create_state(capacity));
                    const std::size_t chunk = length == 259 ? 256 : 512;
                    for (std::size_t offset = 0; offset < length - 3; offset += chunk)
                        ops::require(extended->prefill(std::span(sequence).subspan(offset, chunk), larger_state));
                    const auto larger = ops::require(extended->forward(std::span(sequence).last(3), larger_state));
                    const auto larger_values = ops::require(larger.data<float>());
                    // CPU prefill is batch-invariant, so every chunking yields the same logits. Metal's QAT prefill
                    // multiplies FP16-expanded weights, which can move a calibrated activation by one int8 step from
                    // the exact decode kernel, so accelerators must produce the same tokens rather than logits.
                    if (device == tensor::Device::cpu()) {
                        for (const auto& [name, other] :
                             {std::pair{"full prefix", prefix_values}, std::pair{"uncropped history", history_values},
                              std::pair{"larger chunks", larger_values}})
                            if (const auto [gap, at] = difference(bucket_values, other); gap > 2e-4F) {
                                std::cerr << "length " << length << ": chunked prefill vs " << name << " differs by "
                                          << gap << " at " << at << '\n';
                                return failed(__LINE__);
                            }
                    }
                    // Greedy tokens after the prompt: the argmax of its last logits, then three decoded tokens. These
                    // run last because decoding may reuse the buffers holding earlier logits.
                    const auto continuation = [&](std::span<const float> logits, model::Gemma4State& state) {
                        std::vector<std::int32_t> generated{
                            static_cast<std::int32_t>(std::ranges::max_element(logits) - logits.begin())};
                        for (int step = 0; step < 3; ++step) {
                            const auto next = ops::require(extended->forward_token(std::span(generated).last(1), state));
                            generated.push_back(next);
                        }
                        return generated;
                    };
                    const std::array generated{continuation(bucket_values, bucket_state),
                                               continuation(prefix_values, prefix_state),
                                               continuation(history_values, history_state),
                                               continuation(larger_values, larger_state)};
                    for (std::size_t index = 1; index < generated.size(); ++index)
                        if (generated[index] != generated[0]) {
                            std::cerr << "length " << length << ": chunking " << index
                                      << " generates different tokens from 128-token chunks\n";
                            return failed(__LINE__);
                        }
                }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << current_fixture << " on " << current_device << ": " << error.what() << '\n';
        return 1;
    }
}