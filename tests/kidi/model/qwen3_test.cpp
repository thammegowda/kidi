#include "kidi/model/qwen3.h"

#include <cmath>
#include <iostream>

int main() {
    const auto fail = [](std::string_view message) {
        std::cerr << message << '\n';
        return 1;
    };
    YAML::Node config;
    config["model_type"] = "qwen3";
    config["hidden_act"] = "silu";
    config["vocab_size"] = 8;
    config["hidden_size"] = 4;
    config["intermediate_size"] = 6;
    config["num_hidden_layers"] = 1;
    config["num_attention_heads"] = 2;
    config["num_key_value_heads"] = 1;
    config["head_dim"] = 2;
    config["max_position_embeddings"] = 16;
    config["rms_norm_eps"] = 1e-6F;
    config["rope_parameters"]["rope_theta"] = 10000.F;
    config["layer_types"].push_back("full_attention");

    const kidi::ModuleScope scope(kidi::tensor::DType::I8, true, kidi::tensor::Device::cpu());
    kidi::model::Qwen3 model(config);
    auto state = model->state_dict();
    for (const auto* name : {"layers.0.self_attn.qkv_proj.weight", "layers.0.self_attn.qkv_proj.scale",
                             "layers.0.mlp.gate_up_proj.weight", "layers.0.mlp.gate_up_proj.scale"})
        if (!state.contains(name)) return fail("Qwen3 fused projection state is missing");
    for (const auto* name :
         {"layers.0.self_attn.q_proj.weight", "layers.0.self_attn.k_proj.weight", "layers.0.self_attn.v_proj.weight",
          "layers.0.mlp.gate_proj.weight", "layers.0.mlp.up_proj.weight"})
        if (state.contains(name)) return fail("Qwen3 retained an unfused projection");
    if (!std::ranges::equal(state.at("layers.0.self_attn.qkv_proj.weight").shape(),
                            std::array<std::int64_t, 2>{8, 4}) ||
        !std::ranges::equal(state.at("layers.0.mlp.gate_up_proj.weight").shape(), std::array<std::int64_t, 2>{12, 4}))
        return fail("Qwen3 fused projection shape is invalid");

    for (auto& [name, tensor] : state) {
        if (tensor.dtype() == kidi::tensor::DType::I8) {
            auto values = tensor.data<std::int8_t>();
            if (!values) return fail("cannot initialize Qwen3 INT8 state");
            std::ranges::fill(*values, name.ends_with("norm.weight") ? 127 : 1);
        } else {
            auto values = tensor.data<float>();
            if (!values) return fail("cannot initialize Qwen3 scale state");
            std::ranges::fill(*values, name.ends_with("norm.scale") ? 1.F / 127.F : 0.01F);
        }
    }

    kidi::ops::Context context(kidi::tensor::Device::cpu());
    const std::array<std::int32_t, 2> tokens{1, 2};
    const auto embeddings = model->embed(context, tokens);
    const auto output = model->forward(context, embeddings);
    context.synchronize();
    if (output.dimensions() != 3 || output.size(0) != 1 || output.size(1) != 2 || output.size(2) != 4)
        return fail("Qwen3 output shape is invalid");
    const auto values = output.data<float>();
    if (!values || !std::ranges::all_of(*values, [](float value) { return std::isfinite(value); }))
        return fail("Qwen3 output contains a non-finite value");
    const std::vector expected(values->begin(), values->end());
    const auto mask = kidi::ops::require(kidi::tensor::Tensor::zeros({1, 1, 2, 2}, kidi::tensor::DType::F32));
    const auto explicit_mask = model->forward(context, embeddings, mask);
    const auto replayed = model->forward(context, embeddings);
    context.synchronize();
    const auto explicit_values = kidi::ops::require(explicit_mask.data<float>());
    const auto replayed_values = kidi::ops::require(replayed.data<float>());
    if (!std::ranges::equal(expected, explicit_values) || !std::ranges::equal(expected, replayed_values))
        return fail("Qwen3 masked replay changed the output");

    const std::array<std::int32_t, 1> one_token{2};
    const auto single_embedding = model->embed(context, one_token);
    const auto single_output = model->forward(context, single_embedding);
    const std::array packed_inputs{embeddings, single_embedding};
    const auto packed_embedding = context.concat(packed_inputs, 1);
    constexpr std::array<float, 9> MASK{0.F, 0.F, -1.0e9F, 0.F, 0.F, -1.0e9F, -1.0e9F, -1.0e9F, 0.F};
    const auto packed_mask =
        kidi::ops::require(kidi::tensor::Tensor::from_host({1, 1, 3, 3}, std::span<const float>(MASK)));
    const auto packed_output = model->forward(context, packed_embedding, packed_mask);
    const std::array<std::size_t, 2> segment_lengths{2, 1};
    const auto segmented_output = model->forward_segmented(context, packed_embedding, segment_lengths);
    context.synchronize();
    const auto single_values = kidi::ops::require(single_output.data<float>());
    const auto packed_values = kidi::ops::require(packed_output.data<float>());
    const auto segmented_values = kidi::ops::require(segmented_output.data<float>());
    for (std::size_t index = 0; index < expected.size(); ++index)
        if (std::abs(expected[index] - packed_values[index]) > 1e-5F) return fail("Qwen3 packed first segment differs");
    for (std::size_t index = 0; index < single_values.size(); ++index)
        if (std::abs(single_values[index] - packed_values[expected.size() + index]) > 1e-5F)
            return fail("Qwen3 packed second segment differs");
    for (std::size_t index = 0; index < expected.size(); ++index)
        if (std::abs(expected[index] - segmented_values[index]) > 1e-5F)
            return fail("Qwen3 segmented first segment differs");
    for (std::size_t index = 0; index < single_values.size(); ++index)
        if (std::abs(single_values[index] - segmented_values[expected.size() + index]) > 1e-5F)
            return fail("Qwen3 segmented second segment differs");

    const std::array batch_inputs{embeddings, embeddings};
    const auto batch_embedding = context.concat(batch_inputs, 0);
    const auto batch_mask = kidi::ops::require(kidi::tensor::Tensor::zeros({2, 1, 2, 2}, kidi::tensor::DType::F32));
    const auto batch_output = model->forward(context, batch_embedding, batch_mask);
    context.synchronize();
    const auto batch_values = kidi::ops::require(batch_output.data<float>());
    for (std::size_t index = 0; index < batch_values.size(); ++index)
        if (std::abs(expected[index % expected.size()] - batch_values[index]) > 1e-5F)
            return fail("Qwen3 batched guidance differs");

#if defined(__APPLE__)
    const std::array<std::int32_t, 4> metal_tokens{1, 2, 1, 2};
    const auto cpu_metal_reference = model->forward(context, model->embed(context, metal_tokens));
    context.synchronize();
    const auto reference_bytes = kidi::ops::require(cpu_metal_reference.copy_to_host());
    {
        const kidi::ModuleScope metal_scope(kidi::tensor::DType::I8, false, kidi::tensor::Device::apple_gpu());
        kidi::model::Qwen3 metal(config);
        kidi::ops::require(metal->set_state(state));
        kidi::ops::Context metal_context(kidi::tensor::Device::apple_gpu());
        const auto metal_output = metal->forward(metal_context, metal->embed(metal_context, metal_tokens));
        metal_context.synchronize();
        const auto metal_bytes = kidi::ops::require(metal_output.copy_to_host());
        const auto reference =
            std::span(reinterpret_cast<const float*>(reference_bytes.data()), reference_bytes.size() / sizeof(float));
        const auto actual =
            std::span(reinterpret_cast<const float*>(metal_bytes.data()), metal_bytes.size() / sizeof(float));
        if (reference.size() != actual.size()) return fail("Qwen3 Metal output shape differs");
        for (std::size_t index = 0; index < actual.size(); ++index)
            if (!std::isfinite(actual[index]) || std::abs(reference[index] - actual[index]) > 0.02F)
                return fail("Qwen3 Metal numerical mismatch");
    }
#endif
    return 0;
}
