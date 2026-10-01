#include "kidi/model/gemma4.h"
#include "kidi/checkpoint/safetensors/mapped.h"
#include "kidi/inference/generator.h"
#include "kidi/runtime/operator.h"
#include "kidi/tensor/backend.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

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
    std::ofstream(directory / "tokenizer.json") << R"({
      "version":"1.0", "pre_tokenizer":{"type":"WhitespaceSplit"},
      "decoder":{"type":"WordPiece","prefix":"##","cleanup":false},
      "model":{"type":"WordLevel","unk_token":"<unk>","vocab":{
        "<pad>":0,"<eos>":1,"<bos>":2,"<turn|>":3,"<|turn>":4,"<unk>":5,
        "alpha":6,"beta":7,"gamma":8,"delta":9,"theta":10,"zeta":11,"eta":12,"iota":13,"kappa":14,"lambda":15}}
    })";
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
    std::filesystem::remove_all(directory);
}
} // namespace

auto main() -> int {
    using namespace kidi;
    try {
        for (const auto* fixture : {"gemma4", "gemma4-qat"}) {
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
                return 1;
            }
            std::vector devices{tensor::Device::cpu()};
#if defined(__APPLE__)
            devices.push_back(tensor::Device::apple_gpu());
#endif
            const auto vulkan = tensor::BackendRegistry::instance().backend(tensor::Device::vulkan());
            if (vulkan && (*vulkan)->is_available(tensor::Device::vulkan()) && (*vulkan)->supports_execution())
                devices.push_back(tensor::Device::vulkan());
            for (auto device : devices) {
                check_serving_cache(directory, config, device);
                const ModuleScope construction(tensor::DType::F32, false, device);
                auto model = ops::require(model::Gemma4Impl::create(config));
                ops::require(model->set_checkpoint(checkpoint));
                const auto parameters = model->state_dict();
                for (const auto& [name, parameter] : parameters)
                    if (name.find(".mlp.gate_proj.") != std::string::npos ||
                        name.find(".mlp.up_proj.") != std::string::npos)
                        return 1;
                for (int layer = 0; layer < config["num_hidden_layers"].as<int>(); ++layer) {
                    const auto prefix = "layers." + std::to_string(layer) + ".mlp.";
                    const auto gate =
                        ops::require(checkpoint.tensor("model.language_model." + prefix + "gate_proj.weight"));
                    const auto& combined = parameters.at(prefix + "gate_up_proj.weight");
                    if (combined.size(0) != 2 * gate.size(0) || combined.size(1) != gate.size(1)) return 1;
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
                    if (cache.key.dtype() != cache_dtype || cache.value.dtype() != cache_dtype) return 1;
                auto prefill = ops::require(model->forward(tokens, full, true));
                const auto prefill_values = ops::require(prefill.data<float>());
                if (prefill_values.size() != values.size()) return 1;
                for (std::size_t index = 0; index < values.size(); ++index)
                    if (!std::isfinite(prefill_values[index]) ||
                        std::abs(prefill_values[index] - values[index]) > 2e-4F) {
                        std::cerr << "Gemma reference mismatch at " << index << ": " << prefill_values[index]
                                  << " != " << values[index] << '\n';
                        return 1;
                    }
                auto incremental = ops::require(model->create_state(8));
                auto selected_state = ops::require(model->create_state(8));
                for (std::size_t position = 0; position < tokens.size(); ++position) {
                    auto output = ops::require(model->forward(std::span(tokens).subspan(position, 1), incremental));
                    const auto actual = ops::require(output.data<float>());
                    const auto selected =
                        ops::require(model->forward_token(std::span(tokens).subspan(position, 1), selected_state));
                    if (selected != std::ranges::max_element(actual) - actual.begin()) return 1;
                    for (std::size_t token = 0; token < actual.size(); ++token)
                        if (!std::isfinite(actual[token]) ||
                            std::abs(actual[token] - values[position * actual.size() + token]) > 2e-4F) {
                            std::cerr << "Gemma cached logits differ at " << position << ':' << token << '\n';
                            return 1;
                        }
                }
                if (incremental.position != tokens.size() || full.position != tokens.size()) return 1;
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
                            return 1;
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
                            return 1;
                        }
                        token = selected;
                    }
                    if (model->forward_token(ops::require(tensor::Tensor::from_host({2}, tokens.first(2), device)),
                                             fed))
                        return 1;
                    if (device == tensor::Device::cpu() && !check_captured_prefill(config, checkpoint, tokens))
                        return 1;
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
                        return 1;
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
                        if (actual.size() != expected.size()) return 1;
                        for (std::size_t index = 0; index < actual.size(); ++index)
                            if (!std::isfinite(actual[index]) || std::abs(actual[index] - expected[index]) > 2e-4F) {
                                std::cerr << "batched Gemma mismatch " << fixture << ' ' << tensor::to_string(device)
                                          << " batch " << batch_size << " step " << step << " index " << index << '\n';
                                return 1;
                            }
                        for (auto row : order)
                            if (batch[row].position != serial[row].position) return 1;
                    }
                }
                auto chunked = ops::require(model->create_state(8));
                ops::require(model->prefill(tokens.first(3), chunked));
                auto complete_prefix = ops::require(model->create_state(8));
                ops::require(model->forward(tokens.first(3), complete_prefix, true));
                if (chunked.position != complete_prefix.position) return 1;
                for (std::size_t producer = 0; producer < chunked.layers.size(); ++producer)
                    if (ops::require(chunked.layers[producer].key.copy_to_host()) !=
                            ops::require(complete_prefix.layers[producer].key.copy_to_host()) ||
                        ops::require(chunked.layers[producer].value.copy_to_host()) !=
                            ops::require(complete_prefix.layers[producer].value.copy_to_host())) {
                        std::cerr << "cache-only prefill differs from full evaluation at producer " << producer << '\n';
                        return 1;
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
                    return 1;
                const auto reference_tail = values.last(tail_values.size());
                for (std::size_t index = 0; index < tail_values.size(); ++index)
                    if (!std::isfinite(tail_values[index]) ||
                        std::abs(tail_values[index] - reference_tail[index]) > 2e-4F)
                        return 1;
                const std::array<std::int32_t, 1> invalid{-1};
                if (model->forward(invalid, incremental) || incremental.position != tokens.size()) return 1;
                if (model->set_checkpoint(checkpoint, 4, 3)) return 1;
                ops::require(model->set_checkpoint(checkpoint));
                if (config["quantization_config"]) {
                    if (model->set_checkpoint(checkpoint, 8, 4)) return 1;
                    ops::require(model->set_checkpoint(checkpoint, 0, 128, true));
                    auto packed_state = ops::require(model->create_state(8));
                    const auto packed_output = ops::require(model->forward(tokens, packed_state, true));
                    const auto packed_values = ops::require(packed_output.data<float>());
                    for (std::size_t index = 0; index < values.size(); ++index)
                        if (!std::isfinite(packed_values[index]) ||
                            std::abs(packed_values[index] - values[index]) > 2e-4F)
                            return 1;
                } else {
                    ops::require(model->set_checkpoint(checkpoint, 8, 4));
                    auto quantized_state = ops::require(model->create_state(8));
                    const auto quantized = ops::require(model->forward(tokens.first(1), quantized_state));
                    const auto quantized_values = ops::require(quantized.data<float>());
                    for (std::size_t index = 0; index < quantized_values.size(); ++index)
                        if (!std::isfinite(quantized_values[index]) ||
                            std::abs(quantized_values[index] - values[index]) > 0.01F)
                            return 1;
                    ops::require(model->set_checkpoint(checkpoint));
                    auto restored_state = ops::require(model->create_state(8));
                    const auto restored = ops::require(model->forward(tokens.first(1), restored_state));
                    const auto restored_values = ops::require(restored.data<float>());
                    for (std::size_t index = 0; index < restored_values.size(); ++index)
                        if (!std::isfinite(restored_values[index]) ||
                            std::abs(restored_values[index] - values[index]) > 2e-4F)
                            return 1;
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
                            return 1;
                    const auto prefix = ops::require(extended->forward(sequence, prefix_state));
                    auto oracle_state = ops::require(extended->create_state(capacity));
                    const auto all_logits = ops::require(extended->forward(sequence, oracle_state, true));
                    const auto last_values = ops::require(prefix.data<float>());
                    const auto oracle_values = ops::require(all_logits.data<float>()).last(last_values.size());
                    for (std::size_t index = 0; index < last_values.size(); ++index)
                        if (!std::isfinite(last_values[index]) ||
                            std::abs(last_values[index] - oracle_values[index]) > 2e-4F)
                            return 1;
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
                    for (std::size_t index = 0; index < bucket_values.size(); ++index)
                        if (!std::isfinite(bucket_values[index]) ||
                            std::abs(bucket_values[index] - prefix_values[index]) > 2e-4F ||
                            std::abs(bucket_values[index] - history_values[index]) > 2e-4F)
                            return 1;
                    auto larger_state = ops::require(extended->create_state(capacity));
                    const std::size_t chunk = length == 259 ? 256 : 512;
                    for (std::size_t offset = 0; offset < length - 3; offset += chunk)
                        ops::require(extended->prefill(std::span(sequence).subspan(offset, chunk), larger_state));
                    const auto larger = ops::require(extended->forward(std::span(sequence).last(3), larger_state));
                    const auto larger_values = ops::require(larger.data<float>());
                    for (std::size_t index = 0; index < larger_values.size(); ++index)
                        if (!std::isfinite(larger_values[index]) ||
                            std::abs(larger_values[index] - prefix_values[index]) > 2e-4F)
                            return 1;
                }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}