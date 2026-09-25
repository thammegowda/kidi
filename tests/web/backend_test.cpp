#include "kidi/tensor/web_gpu.h"
#include "kidi/ops/context.h"
#include "kidi/model/gemma4.h"
#include "kidi/model/config.h"
#include "kidi/text/tokenizer.h"
#include <emscripten.h>
#include <iostream>
#include <cmath>

extern "C" EMSCRIPTEN_KEEPALIVE auto kidi_gpu_test() -> int {
    try {
        using namespace kidi;
        const std::array values{1.F, -2.F, 3.F, 4.F};
        auto tensor =
            ops::require(tensor::Tensor::from_host({4}, std::span<const float>(values), tensor::Device::web_gpu()));
        if (tensor.is_host_accessible() || tensor.host_bytes()) return 1;
        auto view = ops::require(tensor.narrow(0, 1, 2));
        auto result = ops::require(view.to(tensor::Device::cpu()));
        const auto actual = ops::require(result.data<float>());
        if (actual[0] != -2.F || actual[1] != 3.F) return 2;
        ops::Context context(tensor::Device::web_gpu());
        auto doubled = context.add(tensor, tensor);
        context.multiply_(doubled, tensor);
        context.synchronize();
        auto squared = ops::require(doubled.to(tensor::Device::cpu()));
        if (!std::ranges::equal(ops::require(squared.data<float>()), std::array{2.F, 8.F, 18.F, 32.F})) return 4;
        ops::Context cpu;
        for (const int bits : {2, 4, 8})
            for (const int width : {8, 64, 256}) {
                const int columns = 12;
                std::vector<std::uint8_t> bytes(columns * width / (8 / bits));
                for (std::size_t index = 0; index < bytes.size(); ++index)
                    bytes[index] = static_cast<std::uint8_t>(index * 71 + 37);
                auto weight = ops::require(
                    tensor::Tensor::from_host({columns, width / (8 / bits)}, std::span<const std::uint8_t>(bytes)));
                const std::vector<float> scale_values(columns, 0.125F);
                auto scales =
                    ops::require(tensor::Tensor::from_host({columns, 1}, std::span<const float>(scale_values)));
                std::vector<float> dense_values(columns * width);
                for (std::size_t index = 0; index < dense_values.size(); ++index) {
                    const auto raw = (bytes[index / (8 / bits)] >> ((index % (8 / bits)) * bits)) & ((1 << bits) - 1);
                    dense_values[index] = static_cast<float>((raw ^ (1 << (bits - 1))) - (1 << (bits - 1))) * 0.125F;
                }
                auto dense =
                    ops::require(tensor::Tensor::from_host({columns, width}, std::span<const float>(dense_values)));
                for (const int rows : {1, 4}) {
                    std::vector<float> values(rows * width);
                    for (std::size_t index = 0; index < values.size(); ++index)
                        values[index] = (static_cast<int>(index % 41) - 20) * 0.13F;
                    auto input = ops::require(tensor::Tensor::from_host({rows, width}, std::span<const float>(values)));
                    auto expected = cpu.packed_linear(input, weight, scales, bits, width, 0.25F, 0.125F);
                    auto actual = context.packed_linear(ops::require(input.to(tensor::Device::web_gpu())), weight,
                                                        scales, bits, width, 0.25F, 0.125F);
                    auto host = ops::require(actual.to(tensor::Device::cpu()));
                    if (!std::ranges::equal(ops::require(host.data<float>()), ops::require(expected.data<float>()))) {
                        std::cerr << "packed Q" << bits << " rows=" << rows << " width=" << width << " mismatch\n";
                        return 5;
                    }
                    auto float_expected = cpu.linear(input, dense, {}, true);
                    auto float_actual = context.packed_linear(ops::require(input.to(tensor::Device::web_gpu())), weight,
                                                              scales, bits, width);
                    auto float_host = ops::require(float_actual.to(tensor::Device::cpu()));
                    const auto left = ops::require(float_expected.data<float>()),
                               right = ops::require(float_host.data<float>());
                    for (std::size_t index = 0; index < left.size(); ++index)
                        if (!std::isfinite(right[index]) || std::abs(left[index] - right[index]) > 1e-3F)
                            throw std::runtime_error("uncalibrated packed projection mismatch at width " +
                                                     std::to_string(width) + " bits " + std::to_string(bits));
                }
                const std::array<std::int32_t, 3> rows_to_read{0, columns / 2, columns - 1};
                auto indices = ops::require(tensor::Tensor::from_host({3}, std::span<const std::int32_t>(rows_to_read),
                                                                      tensor::Device::web_gpu()));
                auto gathered = ops::require(
                    context.embedding(indices, weight, scales, width, bits, 2.F).to(tensor::Device::cpu()));
                const auto gathered_values = ops::require(gathered.data<float>());
                for (std::size_t row = 0; row < rows_to_read.size(); ++row)
                    for (int channel = 0; channel < width; ++channel)
                        if (std::abs(gathered_values[row * width + channel] -
                                     dense_values[rows_to_read[row] * width + channel] * 2.F) > 1e-3F)
                            throw std::runtime_error("packed embedding mismatch at width " + std::to_string(width) +
                                                     " bits " + std::to_string(bits));
            }
        std::cout << "Q2/Q4/Q8 calibrated one/four-row projection parity passed\n";
        auto compare = [&](const tensor::Tensor& expected, const tensor::Tensor& actual, float tolerance) {
            auto host = ops::require(actual.to(tensor::Device::cpu()));
            auto left = ops::require(expected.data<float>());
            auto right = ops::require(host.data<float>());
            if (left.size() != right.size()) throw std::runtime_error("output size mismatch");
            for (std::size_t index = 0; index < left.size(); ++index)
                if (!std::isfinite(right[index]) || std::abs(left[index] - right[index]) > tolerance)
                    throw std::runtime_error("numerical mismatch at " + std::to_string(index) + ": " +
                                             std::to_string(left[index]) + " vs " + std::to_string(right[index]));
        };
        auto scale = ops::require(tensor::Tensor::from_host({4}, std::span<const float>(values)));
        auto host_tensor = ops::require(tensor.to(tensor::Device::cpu()));
        const std::array host_pieces{host_tensor, host_tensor, host_tensor};
        const std::array device_pieces{tensor, tensor, tensor};
        compare(cpu.concat(host_pieces, 0), context.concat(device_pieces, 0), 0.F);
        compare(cpu.rms_norm(host_tensor, scale, 1e-6F), context.rms_norm(tensor, scale, 1e-6F), 1e-5F);
        compare(cpu.softmax(host_tensor), context.softmax(tensor), 1e-6F);
        const std::array large_values{-1e10F, -100.F, -10.F, -3.F, 0.F, 3.F, 10.F, 100.F, 1e10F};
        auto large = ops::require(tensor::Tensor::from_host({9}, std::span<const float>(large_values)));
        auto device_large = ops::require(large.to(tensor::Device::web_gpu()));
        compare(cpu.gelu(large, true), context.gelu(device_large, true), 1e-5F);
        compare(cpu.tanh(large), context.tanh(device_large), 1e-6F);
        for (const int queries : {1, 3}) {
            std::vector<float> query_values(queries * 16), key_values(7 * 8), value_values(7 * 8),
                mask_values(queries * 7);
            for (std::size_t index = 0; index < query_values.size(); ++index)
                query_values[index] = (static_cast<int>(index % 11) - 5) * 0.1F;
            for (std::size_t index = 0; index < key_values.size(); ++index) {
                key_values[index] = (static_cast<int>(index % 7) - 3) * 0.07F;
                value_values[index] = (static_cast<int>(index % 13) - 6) * 0.2F;
            }
            mask_values[6] = -1e9F;
            auto query =
                ops::require(tensor::Tensor::from_host({1, queries, 16}, std::span<const float>(query_values)));
            auto key = ops::require(tensor::Tensor::from_host({1, 7, 8}, std::span<const float>(key_values)));
            auto value = ops::require(tensor::Tensor::from_host({1, 7, 8}, std::span<const float>(value_values)));
            auto mask =
                ops::require(tensor::Tensor::from_host({1, 1, queries, 7}, std::span<const float>(mask_values)));
            compare(cpu.grouped_query_attention(query, key, value, 4, 2, mask),
                    context.grouped_query_attention(ops::require(query.to(tensor::Device::web_gpu())),
                                                    ops::require(key.to(tensor::Device::web_gpu())),
                                                    ops::require(value.to(tensor::Device::web_gpu())), 4, 2,
                                                    ops::require(mask.to(tensor::Device::web_gpu()))),
                    1e-5F);
        }
        {
            // Long key ranges are split across workgroups and recombined, which short cases never reach.
            const int keys = 2200, heads = 4, key_heads = 2, head_width = 16;
            std::vector<float> query_values(heads * head_width), key_values(keys * key_heads * head_width),
                value_values(key_values.size()), mask_values(keys, 0.F);
            for (std::size_t index = 0; index < query_values.size(); ++index)
                query_values[index] = std::sin(static_cast<float>(index) * 0.37F);
            for (std::size_t index = 0; index < key_values.size(); ++index) {
                key_values[index] = std::sin(static_cast<float>(index) * 0.011F) * 0.8F;
                value_values[index] = std::cos(static_cast<float>(index) * 0.017F);
            }
            for (int key = keys - 40; key < keys; ++key) mask_values[key] = -1e9F;
            auto query = ops::require(
                tensor::Tensor::from_host({1, 1, heads * head_width}, std::span<const float>(query_values)));
            auto key = ops::require(
                tensor::Tensor::from_host({1, keys, key_heads * head_width}, std::span<const float>(key_values)));
            auto value = ops::require(
                tensor::Tensor::from_host({1, keys, key_heads * head_width}, std::span<const float>(value_values)));
            auto mask = ops::require(tensor::Tensor::from_host({1, 1, 1, keys}, std::span<const float>(mask_values)));
            compare(cpu.grouped_query_attention(query, key, value, heads, key_heads, mask),
                    context.grouped_query_attention(ops::require(query.to(tensor::Device::web_gpu())),
                                                    ops::require(key.to(tensor::Device::web_gpu())),
                                                    ops::require(value.to(tensor::Device::web_gpu())), heads, key_heads,
                                                    ops::require(mask.to(tensor::Device::web_gpu()))),
                    1e-5F);
            // An INT8 cache must match an FP32 cache holding the same rounded values exactly.
            const float key_scale = 0.05F, value_scale = 0.02F;
            auto rounded_key = cpu.static_round(key, key_scale), rounded_value = cpu.static_round(value, value_scale);
            auto byte_key =
                context.cast(ops::require(rounded_key.to(tensor::Device::web_gpu())), tensor::DType::I8, key_scale);
            auto byte_value =
                context.cast(ops::require(rounded_value.to(tensor::Device::web_gpu())), tensor::DType::I8, value_scale);
            compare(cpu.grouped_query_attention(query, rounded_key, rounded_value, heads, key_heads, mask),
                    context.grouped_query_attention(
                        ops::require(query.to(tensor::Device::web_gpu())), byte_key, byte_value, heads, key_heads,
                        ops::require(mask.to(tensor::Device::web_gpu())), 1.F, 0, key_scale, value_scale),
                    1e-4F);
        }
        auto token = context.greedy_token(tensor);
        context.synchronize();
        if (ops::require(token.data<std::int32_t>())[0] != 3) return 6;
        std::cout << "RMSNorm, softmax, grouped attention and device selection passed\n";
        for (const auto* fixture : {"gemma4", "gemma4-qat"}) {
            const auto directory = std::filesystem::path("/fixtures") / fixture;
            const auto config = YAML::LoadFile((directory / "model.yaml").string())["model"];
            auto weights = ops::require(model::Weights::load(directory / "model.safetensors"));
            auto reference = ops::require(model::Weights::load(directory / "reference.safetensors"));
            const auto token_storage = ops::require(reference.tensor("tokens"));
            const auto tokens = ops::require(token_storage.data<std::int32_t>());
            const auto expected = ops::require(reference.tensor("logits"));
            const ModuleScope scope(tensor::DType::F32, false, tensor::Device::web_gpu());
            auto model = ops::require(model::Gemma4Impl::create(config));
            ops::require(model->set_checkpoint(weights));
            auto state = ops::require(model->create_state(8));
            compare(expected, ops::require(model->forward(tokens, state, true)), 2e-4F);
            auto incremental = ops::require(model->create_state(8));
            const auto reference_values = ops::require(expected.data<float>());
            const auto vocabulary = expected.size(-1);
            for (std::size_t position = 0; position < tokens.size(); ++position) {
                const auto selected = ops::require(model->forward_token(tokens.subspan(position, 1), incremental));
                const auto row = reference_values.subspan(position * vocabulary, vocabulary);
                if (selected != std::ranges::max_element(row) - row.begin())
                    throw std::runtime_error("Gemma selected token mismatch");
            }
            std::cout << fixture << " full logits and incremental tokens passed\n";
        }
        std::cout << "WebGPU C++ tensor view and JSPI readback passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 3;
    }
}

extern "C" EMSCRIPTEN_KEEPALIVE auto kidi_gpu_diagnose() -> int {
    try {
        using namespace kidi;
        const auto config = ops::require(model::load_config("/model/model.yaml"));
        auto weights = ops::require(model::Weights::load("/model/model.safetensors"));
        auto tokenizer = ops::require(text::Tokenizer::load("/model/tokenizer.json"));
        const std::array messages{text::ChatMessage{"user", "Name three practical uses of binary search."}};
        auto tokens = ops::require(tokenizer.encode(ops::require(tokenizer.format_chat(messages))));
        const auto dtype = ops::require(weights.tensor("model.language_model.norm.weight")).dtype();
        const ModuleScope scope(dtype, false, tensor::Device::cpu());
        auto model = ops::require(model::Gemma4Impl::create(config["model"]));
        ops::require(model->set_checkpoint(weights));
        const auto state = model->state_dict();
        ops::Context cpu, gpu(tensor::Device::web_gpu());
        const auto compare = [&](std::string_view label, const tensor::Tensor& expected, const tensor::Tensor& actual) {
            auto host = ops::require(actual.to(tensor::Device::cpu()));
            const auto left = ops::require(expected.data<float>());
            const auto right = ops::require(host.data<float>());
            float maximum = 0;
            std::size_t index = 0;
            for (std::size_t offset = 0; offset < left.size(); ++offset)
                if (!std::isfinite(right[offset]) || std::abs(left[offset] - right[offset]) > maximum) {
                    maximum = std::abs(left[offset] - right[offset]);
                    index = offset;
                }
            std::cout << label << " max_error=" << maximum << " index=" << index << " cpu=" << left[index]
                      << " gpu=" << right[index] << " count=" << left.size() << '\n';
        };
        const auto hidden = config["model"]["hidden_size"].as<int>();
        const auto layer_width = config["model"]["hidden_size_per_layer_input"].as<int>();
        const auto layers = config["model"]["num_hidden_layers"].as<int>();
        layers::TokenEmbedding embedding(config["model"]["vocab_size"].as<int>(), hidden,
                                         std::sqrt(static_cast<float>(hidden)), 2);
        ops::require(
            embedding->set_state(StateDict{{"embedding_quantized", state.at("embed_tokens.embedding_quantized")},
                                           {"embedding_scale", state.at("embed_tokens.embedding_scale")}}));
        auto host_hidden = embedding->forward(cpu, tokens), device_hidden = embedding->forward(gpu, tokens);
        compare("token_embedding", host_hidden, device_hidden);
        layers::TokenEmbedding per_layer(config["model"]["vocab_size_per_layer_input"].as<int>(), layers * layer_width,
                                         std::sqrt(static_cast<float>(layer_width)), 4, layers);
        ops::require(per_layer->set_state(
            StateDict{{"embedding_quantized", state.at("embed_tokens_per_layer.embedding_quantized")},
                      {"embedding_scale", state.at("embed_tokens_per_layer.embedding_scale")}}));
        compare("per_layer_embedding", per_layer->forward(cpu, tokens), per_layer->forward(gpu, tokens));
        layers::Linear projection(hidden, layers * layer_width, true, false);
        ops::require(projection->set_state(StateDict{{"weight", state.at("per_layer_model_projection.weight")}}));
        compare("per_layer_projection", projection->forward(cpu, host_hidden), projection->forward(gpu, device_hidden));
        const auto first = std::string("layers.0.");
        const auto epsilon = config["model"]["rms_norm_eps"].as<float>();
        const auto norm_weight = state.at(first + "input_layernorm.weight");
        auto host_normalized = cpu.rms_norm(host_hidden, norm_weight, epsilon);
        auto gpu_normalized = gpu.rms_norm(device_hidden, norm_weight, epsilon);
        compare("input_norm", host_normalized, gpu_normalized);
        for (const auto* name : {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}) {
            const auto prefix = first + name + '.';
            const auto weight = state.at(prefix + "weight");
            const auto scales = state.at(prefix + "weight_scale");
            const auto input_scale = ops::require(state.at(prefix + "input_activation_scale").data<float>())[0];
            const auto output_scale = ops::require(state.at(prefix + "output_activation_scale").data<float>())[0];
            compare(name, cpu.packed_linear(host_normalized, weight, scales, 4, hidden, input_scale, output_scale),
                    gpu.packed_linear(gpu_normalized, weight, scales, 4, hidden, input_scale, output_scale));
        }
        const auto length = static_cast<std::int64_t>(tokens.size());
        const std::vector<std::int64_t> per_shape{1, length, layers, layer_width};
        const auto host_scale = ops::require(tensor::Tensor::from_host(
            {}, std::span<const float>(std::array{1.F / std::sqrt(static_cast<float>(hidden))})));
        const auto host_combination =
            ops::require(tensor::Tensor::from_host({}, std::span<const float>(std::array{std::sqrt(0.5F)})));
        auto host_per = cpu.reshape(cpu.multiply(projection->forward(cpu, host_hidden), host_scale), per_shape);
        auto gpu_per = gpu.reshape(gpu.multiply(projection->forward(gpu, device_hidden),
                                                ops::require(host_scale.to(tensor::Device::web_gpu()))),
                                   per_shape);
        host_per = cpu.multiply(cpu.add(cpu.rms_norm(host_per, state.at("per_layer_projection_norm.weight"), epsilon),
                                        cpu.reshape(per_layer->forward(cpu, tokens), per_shape)),
                                host_combination);
        gpu_per = gpu.multiply(gpu.add(gpu.rms_norm(gpu_per, state.at("per_layer_projection_norm.weight"), epsilon),
                                       gpu.reshape(per_layer->forward(gpu, tokens), per_shape)),
                               ops::require(host_combination.to(tensor::Device::web_gpu())));
        compare("per_layer_combined", host_per, gpu_per);
        const auto head_width = config["model"]["head_dim"].as<int>();
        const auto heads = config["model"]["num_attention_heads"].as<int>();
        const auto key_heads = config["model"]["num_key_value_heads"].as<int>();
        const auto intermediate = config["model"]["intermediate_size"].as<int>();
        StateDict first_state;
        for (const auto& [name, value] : state)
            if (name.starts_with(first)) first_state.emplace(name.substr(first.size()), value);
        auto make_block = [&](tensor::Device device) {
            const ModuleScope construction(dtype, false, device);
            layers::Gemma4Block block(hidden, intermediate, heads, key_heads, head_width, layer_width, epsilon, false,
                                      4, 4, 8);
            ops::require(block->set_state(first_state));
            return block;
        };
        auto host_block = make_block(tensor::Device::cpu()), gpu_block = make_block(tensor::Device::web_gpu());
        std::vector<float> cosine(length * head_width / 2), sine(cosine.size()), mask_values(length * 128, -1e9F);
        const auto theta = config["model"]["rope_parameters"]["sliding_attention"]["rope_theta"].as<float>();
        for (std::int64_t row = 0; row < length; ++row) {
            for (int channel = 0; channel < head_width / 2; ++channel) {
                const float angle = row * std::pow(theta, -2.F * channel / head_width);
                cosine[row * head_width / 2 + channel] = std::cos(angle);
                sine[row * head_width / 2 + channel] = std::sin(angle);
            }
            for (std::int64_t column = 0; column <= row; ++column) mask_values[row * 128 + column] = 0.F;
        }
        const auto angles_shape = std::vector<std::int64_t>{1, length, 1, head_width / 2};
        auto cosines = ops::require(tensor::Tensor::from_host(angles_shape, std::span<const float>(cosine)));
        auto sines = ops::require(tensor::Tensor::from_host(angles_shape, std::span<const float>(sine)));
        auto mask = ops::require(tensor::Tensor::from_host({1, 1, length, 128}, std::span<const float>(mask_values)));
        auto make_cache = [&](tensor::Device device) {
            const std::vector<std::int64_t> shape{1, 128, key_heads * head_width};
            return layers::KeyValue{ops::require(tensor::Tensor::zeros(shape, tensor::DType::F32, device)),
                                    ops::require(tensor::Tensor::zeros(shape, tensor::DType::F32, device))};
        };
        auto host_cache = make_cache(tensor::Device::cpu()), gpu_cache = make_cache(tensor::Device::web_gpu());
        auto host_input = cpu.reshape(cpu.slice(host_per, 2, 0, 1), {1, length, layer_width});
        auto gpu_input = gpu.reshape(gpu.slice(gpu_per, 2, 0, 1), {1, length, layer_width});
        auto host_output = host_block->forward(cpu, host_hidden, host_input, host_cache, 0, mask, cosines, sines);
        auto gpu_output = gpu_block->forward(
            gpu, device_hidden, gpu_input, gpu_cache, 0, ops::require(mask.to(tensor::Device::web_gpu())),
            ops::require(cosines.to(tensor::Device::web_gpu())), ops::require(sines.to(tensor::Device::web_gpu())));
        compare("first_block", host_output, gpu_output);
        compare("first_key_cache", host_cache.key, gpu_cache.key);
        compare("first_value_cache", host_cache.value, gpu_cache.value);
        const auto linear = [&](ops::Context& context, const tensor::Tensor& input, const std::string& name, int bits) {
            const auto prefix = first + name + '.';
            return context.packed_linear(input, state.at(prefix + "weight"), state.at(prefix + "weight_scale"), bits,
                                         input.size(-1),
                                         ops::require(state.at(prefix + "input_activation_scale").data<float>())[0],
                                         ops::require(state.at(prefix + "output_activation_scale").data<float>())[0]);
        };
        const auto query_shape = std::vector<std::int64_t>{1, length, heads, head_width};
        auto host_query = cpu.rms_rotary(cpu.reshape(linear(cpu, host_normalized, "self_attn.q_proj", 4), query_shape),
                                         state.at(first + "self_attn.q_norm.weight"), cosines, sines, epsilon);
        auto gpu_query =
            gpu.rms_rotary(gpu.reshape(linear(gpu, gpu_normalized, "self_attn.q_proj", 4), query_shape),
                           ops::require(state.at(first + "self_attn.q_norm.weight").to(tensor::Device::web_gpu())),
                           ops::require(cosines.to(tensor::Device::web_gpu())),
                           ops::require(sines.to(tensor::Device::web_gpu())), epsilon);
        compare("query_rotary", host_query, gpu_query);
        const auto attention_shape = std::vector<std::int64_t>{1, length, heads * head_width};
        auto host_attention = cpu.grouped_query_attention(cpu.reshape(host_query, attention_shape), host_cache.key,
                                                          host_cache.value, heads, key_heads, mask);
        auto gpu_attention =
            gpu.grouped_query_attention(gpu.reshape(gpu_query, attention_shape), gpu_cache.key, gpu_cache.value, heads,
                                        key_heads, ops::require(mask.to(tensor::Device::web_gpu())));
        compare("attention", host_attention, gpu_attention);
        auto host_attended = linear(cpu, host_attention, "self_attn.o_proj", 4),
             gpu_attended = linear(gpu, gpu_attention, "self_attn.o_proj", 4);
        compare("attention_projection", host_attended, gpu_attended);
        const auto residual = [&](ops::Context& context, const tensor::Tensor& input, const tensor::Tensor& residual,
                                  const std::string& name) {
            return context.rms_norm_residual(
                input, ops::require(state.at(first + name + ".weight").to(context.device())), residual, epsilon);
        };
        auto host_after = residual(cpu, host_attended, host_hidden, "post_attention_layernorm"),
             gpu_after = residual(gpu, gpu_attended, device_hidden, "post_attention_layernorm");
        compare("attention_residual", host_after, gpu_after);
        auto host_ff = cpu.rms_norm(host_after, state.at(first + "pre_feedforward_layernorm.weight"), epsilon);
        auto gpu_ff = gpu.rms_norm(gpu_after, state.at(first + "pre_feedforward_layernorm.weight"), epsilon);
        compare("mlp_norm", host_ff, gpu_ff);
        auto host_gate = linear(cpu, host_ff, "mlp.gate_proj", 4), gpu_gate = linear(gpu, gpu_ff, "mlp.gate_proj", 4);
        auto host_up = linear(cpu, host_ff, "mlp.up_proj", 4), gpu_up = linear(gpu, gpu_ff, "mlp.up_proj", 4);
        compare("mlp_gate", host_gate, gpu_gate);
        compare("mlp_up", host_up, gpu_up);
        auto host_activation = cpu.gelu_multiply(host_gate, host_up),
             gpu_activation = gpu.gelu_multiply(gpu_gate, gpu_up);
        compare("mlp_activation", host_activation, gpu_activation);
        compare("mlp_down", linear(cpu, host_activation, "mlp.down_proj", 4),
                linear(gpu, gpu_activation, "mlp.down_proj", 4));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}