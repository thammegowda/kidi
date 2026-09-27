#include <iostream>

#include "kidi/model/whisper.h"

auto main() -> int {
    auto config = YAML::Load(R"({
        "type":"whisper", "model_type":"whisper", "architectures":["WhisperForConditionalGeneration"],
        "activation_function":"gelu", "activation_dropout":0, "attention_dropout":0,
        "decoder_layerdrop":0, "dropout":0, "encoder_layerdrop":0, "scale_embedding":false,
        "is_encoder_decoder":true, "use_cache":true, "d_model":384, "encoder_attention_heads":6,
        "encoder_ffn_dim":1536, "encoder_layers":4, "decoder_attention_heads":6, "decoder_ffn_dim":1536,
        "decoder_layers":4, "max_source_positions":1500, "max_target_positions":448,
        "num_mel_bins":80, "vocab_size":51865
    })");
    const kidi::ModuleScope scope(kidi::tensor::DType::F32, false, kidi::tensor::Device::cpu());
    auto model = kidi::model::WhisperImpl::create(config);
    if (!model) {
        std::cerr << model.error().message << '\n';
        return 1;
    }
    const auto state = (*model)->state_dict();
    for (const auto* name :
         {"encoder.conv1.weight", "encoder.layers.3.self_attn.k_proj.weight", "decoder.embed_tokens.weight",
          "decoder.layers.0.encoder_attn.q_proj.bias", "proj_out.weight"})
        if (!state.contains(name)) return 1;
    config["d_model"] = 512;
    config["encoder_layers"] = 6;
    config["decoder_layers"] = 6;
    config["encoder_attention_heads"] = 8;
    config["decoder_attention_heads"] = 8;
    config["encoder_ffn_dim"] = 2048;
    config["decoder_ffn_dim"] = 2048;
    if (!kidi::model::WhisperImpl::validate_config(config)) return 1;
    config["d_model"] = 1024;
    if (kidi::model::WhisperImpl::validate_config(config)) return 1;
    return 0;
}