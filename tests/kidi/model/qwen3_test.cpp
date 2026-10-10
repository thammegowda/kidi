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
    return 0;
}
