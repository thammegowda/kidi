#include "kidi/model/gemma.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <numeric>
#include <regex>

namespace kidi::model {
using ops::require;
using tensor::DType;
using tensor::Tensor;
namespace {
auto scalar(float value, tensor::Device device) -> Tensor {
    return require(Tensor::from_host({}, std::span<const float>(&value, 1), device));
}
auto quant_bits(const YAML::Node& config, const std::string& name) -> std::int32_t {
    const auto quantization = config["quantization_config"];
    if (!quantization) return 0;
    for (const auto& item : quantization["modules_to_not_convert"])
        if (name.find(item.as<std::string>()) != std::string::npos) return 0;
    auto bits = quantization["num_bits"].as<int>();
    for (const auto& item : quantization["module_quant_configs"])
        if (std::regex_search(name, std::regex(item.first.as<std::string>()))) {
            bits = item.second["num_bits"].as<int>();
            break;
        }
    if (bits != 2 && bits != 4 && bits != 8) throw ops::Failure({ErrorCode::UNSUPPORTED, "unsupported QAT bit width"});
    return bits;
}
auto embedding(std::int32_t vocabulary, std::int32_t width, float scale, std::int32_t bits = 0, std::int32_t groups = 1)
    -> layers::TokenEmbedding {
    const ModuleScope host(tensor::Device::cpu());
    return layers::TokenEmbedding(vocabulary, width, scale, bits, groups);
}
auto output_projection(std::int32_t hidden, std::int32_t vocabulary, std::int32_t bits = 0) -> layers::Linear {
    const ModuleScope host(tensor::Device::cpu());
    return layers::Linear(hidden, vocabulary, true, false, bits);
}
auto float_parameter(const Tensor& input) -> Tensor {
    if (input.dtype() == DType::F32) return input;
    if (input.dtype() != DType::BF16)
        throw ops::Failure({ErrorCode::UNSUPPORTED, "Gemma parameters must be BF16 or FP32"});
    auto output = require(Tensor::empty({input.shape().begin(), input.shape().end()}, DType::F32));
    const auto bytes = require(input.host_bytes());
    const auto source = reinterpret_cast<const std::uint16_t*>(bytes.data());
    auto values = require(output.data<float>());
    for (std::size_t index = 0; index < values.size(); ++index)
        values[index] = std::bit_cast<float>(static_cast<std::uint32_t>(source[index]) << 16);
    return output;
}
} // namespace

struct Gemma4Impl::State {
    ops::Context context;
    std::int32_t hidden, per_layer_width, layer_count, shared_begin, maximum_position, window;
    std::int32_t heads, key_heads;
    std::array<std::int32_t, 2> head_width;
    std::array<float, 2> theta, rotary_fraction;
    float cap;
    YAML::Node construction_config;
    bool qat;
    std::vector<std::size_t> cache_layer;
    std::vector<int> kind;
    layers::TokenEmbedding tokens, per_layer_tokens;
    ModuleList<layers::GemmaBlockImpl> layers;
    layers::Linear per_layer_projection, lm_head;
    layers::RmsNorm per_layer_norm, norm;
    Tensor projection_scale, combination_scale, logit_scale, inverse_logit_scale;

    explicit State(const YAML::Node& config)
        : context(module_device),
          hidden(config["hidden_size"].as<int>()),
          per_layer_width(config["hidden_size_per_layer_input"].as<int>()),
          layer_count(config["num_hidden_layers"].as<int>()),
          shared_begin(layer_count - config["num_kv_shared_layers"].as<int>()),
          maximum_position(config["max_position_embeddings"].as<int>()),
          window(config["sliding_window"].as<int>()),
          heads(config["num_attention_heads"].as<int>()),
          key_heads(config["num_key_value_heads"].as<int>()),
          head_width{config["head_dim"].as<int>(), config["global_head_dim"].as<int>()},
          cap(config["final_logit_softcapping"].as<float>()),
          construction_config(YAML::Clone(config)),
          qat(static_cast<bool>(config["quantization_config"])),
          tokens(embedding(config["vocab_size"].as<int>(), hidden, std::sqrt(static_cast<float>(hidden)),
                           quant_bits(config, "model.language_model.embed_tokens"))),
          per_layer_tokens(embedding(config["vocab_size_per_layer_input"].as<int>(), layer_count * per_layer_width,
                                     std::sqrt(static_cast<float>(per_layer_width)),
                                     quant_bits(config, "model.language_model.embed_tokens_per_layer"), layer_count)),
          per_layer_projection(hidden, layer_count * per_layer_width, true, false),
          lm_head(output_projection(hidden, config["vocab_size"].as<int>(), quant_bits(config, "lm_head"))),
          per_layer_norm(per_layer_width, config["rms_norm_eps"].as<float>()),
          norm(hidden, config["rms_norm_eps"].as<float>()),
          projection_scale(scalar(1.F / std::sqrt(static_cast<float>(hidden)), module_device)),
          combination_scale(scalar(std::sqrt(0.5F), module_device)),
          logit_scale(scalar(cap, module_device)),
          inverse_logit_scale(scalar(1.F / cap, module_device)) {
        for (std::size_t type = 0; type < 2; ++type) {
            const auto rope = config["rope_parameters"][type == 0 ? "sliding_attention" : "full_attention"];
            theta[type] = rope["rope_theta"].as<float>();
            rotary_fraction[type] = rope["partial_rotary_factor"].as<float>(1.F);
        }
        std::array<std::size_t, 2> previous{};
        for (std::int32_t index = 0; index < layer_count; ++index) {
            const int type = config["layer_types"][index].as<std::string>() == "sliding_attention" ? 0 : 1;
            const bool shared = index >= shared_begin;
            if (!shared) previous[type] = index;
            cache_layer.push_back(previous[type]);
            kind.push_back(type);
            const auto intermediate =
                config["intermediate_size"].as<int>() * (shared && config["use_double_wide_mlp"].as<bool>() ? 2 : 1);
            layers->push_back(layers::GemmaBlock(
                hidden, intermediate, heads, key_heads, head_width[type], per_layer_width,
                config["rms_norm_eps"].as<float>(), shared,
                quant_bits(config, "model.language_model.layers." + std::to_string(index) + ".mlp.gate_proj"),
                quant_bits(config, "model.language_model.layers." + std::to_string(index) + ".self_attn.q_proj"),
                quant_bits(config, "model.language_model.layers." + std::to_string(index) + ".per_layer_input_gate")));
        }
    }
    auto positions(int type, std::size_t start, std::size_t length) -> std::array<Tensor, 2> {
        const auto half = head_width[type] / 2;
        const auto rotated = static_cast<int>(rotary_fraction[type] * half);
        std::vector<float> cosine(length * half), sine(length * half);
        for (std::size_t row = 0; row < length; ++row)
            for (int channel = 0; channel < half; ++channel) {
                const auto frequency =
                    channel < rotated ? std::pow(theta[type], -2.F * channel / head_width[type]) : 0.F;
                const auto angle = static_cast<float>(start + row) * frequency;
                cosine[row * half + channel] = std::cos(angle);
                sine[row * half + channel] = std::sin(angle);
            }
        const std::vector<std::int64_t> shape{1, static_cast<std::int64_t>(length), 1, half};
        return {require(Tensor::from_host(shape, std::span<const float>(cosine), context.device())),
                require(Tensor::from_host(shape, std::span<const float>(sine), context.device()))};
    }
};

auto Gemma4Impl::validate_config(const YAML::Node& config) -> Result<void> {
    try {
        if (config["quantization_config"] &&
            (config["quantization_config"]["quant_method"].as<std::string>() != "gemma" ||
             !config["quantization_config"]["quantize_embeddings"].as<bool>()))
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, "unsupported QAT checkpoint format"});
        if (config["type"].as<std::string>() != "gemma4_text" || config["enable_moe_block"].as<bool>() ||
            config["attention_bias"].as<bool>() || config["attention_k_eq_v"].as<bool>() ||
            config["hidden_activation"].as<std::string>() != "gelu_pytorch_tanh" ||
            (!config["tie_word_embeddings"].as<bool>() && !config["quantization_config"]))
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, "unsupported Gemma text architecture"});
        for (const auto* name :
             {"hidden_size", "hidden_size_per_layer_input", "num_hidden_layers", "intermediate_size",
              "max_position_embeddings", "sliding_window", "num_attention_heads", "num_key_value_heads", "head_dim",
              "global_head_dim", "vocab_size", "vocab_size_per_layer_input"})
            if (config[name].as<int>() <= 0)
                return std::unexpected(
                    Error{ErrorCode::INVALID_MANIFEST, std::string("invalid Gemma dimension: ") + name});
        const auto count = config["num_hidden_layers"].as<int>();
        const auto shared = config["num_kv_shared_layers"].as<int>();
        if (shared < 0 || shared >= count || config["layer_types"].size() != static_cast<std::size_t>(count) ||
            config["num_attention_heads"].as<int>() % config["num_key_value_heads"].as<int>() ||
            config["head_dim"].as<int>() % 2 || config["global_head_dim"].as<int>() % 2 ||
            (!config["num_global_key_value_heads"].IsNull() &&
             config["num_global_key_value_heads"].as<int>() != config["num_key_value_heads"].as<int>()))
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "invalid Gemma attention or sharing dimensions"});
        std::array<bool, 2> seen{};
        for (int index = 0; index < count; ++index) {
            const auto name = config["layer_types"][index].as<std::string>();
            if (name != "sliding_attention" && name != "full_attention")
                return std::unexpected(Error{ErrorCode::UNSUPPORTED, "unknown Gemma layer type"});
            const auto type = name == "sliding_attention" ? 0 : 1;
            if (index < count - shared)
                seen[type] = true;
            else if (!seen[type])
                return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "missing shared KV source"});
        }
        for (const auto* name : {"rms_norm_eps", "final_logit_softcapping"}) {
            const auto value = config[name].as<float>();
            if (!std::isfinite(value) || value <= 0)
                return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "invalid Gemma normalization or logit cap"});
        }
        for (const auto* name : {"sliding_attention", "full_attention"}) {
            const auto rope = config["rope_parameters"][name];
            const auto theta = rope["rope_theta"].as<float>();
            const auto fraction = rope["partial_rotary_factor"].as<float>(1.F);
            const auto type = rope["rope_type"].as<std::string>();
            if ((type != "default" && type != "proportional") || !std::isfinite(theta) || theta <= 0 ||
                !std::isfinite(fraction) || fraction <= 0 || fraction > 1 || rope["factor"].as<float>(1.F) != 1.F)
                return std::unexpected(Error{ErrorCode::UNSUPPORTED, "unsupported Gemma rotary configuration"});
        }
        return {};
    } catch (const YAML::Exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, error.what()});
    }
}
Gemma4Impl::Gemma4Impl(const YAML::Node& config) {
    require(validate_config(config));
    impl_ = std::make_unique<State>(config);
    register_module("embed_tokens", impl_->tokens);
    register_module("embed_tokens_per_layer", impl_->per_layer_tokens);
    register_module("layers", impl_->layers);
    register_module("per_layer_model_projection", impl_->per_layer_projection);
    register_module("per_layer_projection_norm", impl_->per_layer_norm);
    register_module("norm", impl_->norm);
    register_module("lm_head", impl_->lm_head);
    if (!impl_->qat) tie_parameter("lm_head.weight", "embed_tokens.weight");
}
Gemma4Impl::~Gemma4Impl() = default;
auto Gemma4Impl::create(const YAML::Node& config) -> Result<Gemma4> {
    try {
        return Gemma4(config);
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}
auto Gemma4Impl::set_checkpoint(const Weights& weights, std::int32_t weight_bits, std::int32_t group_size,
                                bool packed_prefill) -> Result<void> {
    try {
        if ((weight_bits != 0 && weight_bits != 4 && weight_bits != 8) || group_size <= 0)
            throw ops::Failure(
                {ErrorCode::INVALID_ARGUMENT, "weight bits must be 0, 4 or 8 and group size must be positive"});
        if (impl_->qat && weight_bits)
            throw ops::Failure(
                {ErrorCode::INVALID_ARGUMENT, "QAT checkpoint precision is trained; do not override weight bits"});
        const auto checkpoint = require(weights.state_dict());
        const auto declared = state_dict();
        StateDict state;
        for (const auto& [key, value] : checkpoint) {
            constexpr std::string_view PREFIX = "model.language_model.";
            if (!key.starts_with(PREFIX) && !(impl_->qat && key.starts_with("lm_head."))) continue;
            const auto name = key.starts_with(PREFIX) ? key.substr(PREFIX.size()) : key;
            if (!declared.contains(name)) {
                if (name.starts_with("layers.") &&
                    (name.find(".self_attn.k_proj.") != std::string::npos ||
                     name.find(".self_attn.v_proj.") != std::string::npos ||
                     name.find(".self_attn.k_norm.") != std::string::npos || name.ends_with(".k_cache_scale") ||
                     name.ends_with(".v_cache_scale")))
                    continue;
                return std::unexpected(
                    Error{ErrorCode::INVALID_ARGUMENT, "unknown Gemma checkpoint parameter: " + name});
            }
            if (impl_->qat && (value.dtype() == DType::U8 || value.dtype() == DType::I8)) {
                const auto suffix = name.ends_with(".embedding_quantized") ? std::string(".embedding_quantized")
                                                                           : std::string(".weight");
                const auto module = name.substr(0, name.size() - suffix.size());
                const auto bits = quant_bits(impl_->construction_config,
                                             module == "lm_head" ? module : "model.language_model." + module);
                if (!bits || (bits < 8 && value.dtype() != DType::U8) || (bits == 8 && value.dtype() != DType::I8))
                    throw ops::Failure(
                        {ErrorCode::INVALID_ARGUMENT, "QAT weight storage does not match configured precision"});
                auto converted = require(Tensor::empty({value.shape().begin(), value.shape().end()}, DType::U8));
                const auto source = require(value.host_bytes());
                auto destination = require(converted.data<std::uint8_t>());
                const std::uint8_t sign_mask = bits == 2 ? 0xaa : bits == 4 ? 0x88 : 0;
                for (std::size_t offset = 0; offset < destination.size(); ++offset)
                    destination[offset] = std::to_integer<std::uint8_t>(source[offset]) ^ sign_mask;
                state.emplace(name, std::move(converted));
            } else {
                if (impl_->qat && name.ends_with("scale")) {
                    const auto scales = require(value.data<float>());
                    const bool positive = name.ends_with("weight_scale") || name.ends_with("embedding_scale");
                    if (std::ranges::any_of(scales, [&](float scale) {
                            return !std::isfinite(scale) || (positive ? scale <= 0 : scale < 0);
                        }))
                        throw ops::Failure(
                            {ErrorCode::INVALID_ARGUMENT, "invalid trained quantization scale: " + name});
                }
                state.emplace(name, name.ends_with("norm.weight") || name.ends_with("layer_scalar")
                                        ? float_parameter(value)
                                        : value);
            }
        }
        if (weight_bits)
            for (const auto& [name, value] : state)
                if (value.dimensions() == 2 && name != "embed_tokens.weight" &&
                    name != "embed_tokens_per_layer.weight" && weight_bits == 4 &&
                    name.find(".mlp.") != std::string::npos && (group_size % 2 || value.size(1) % group_size))
                    throw ops::Failure(
                        {ErrorCode::INVALID_ARGUMENT, "quantization group must divide projection width"});
        impl_->context.synchronize();
        require(set_state(state));
        impl_->context = ops::Context(device(), packed_prefill);
        if (weight_bits)
            for (const auto& [name, weight] : state_dict())
                if (weight.dimensions() == 2 && name != "embed_tokens.weight" &&
                    name != "embed_tokens_per_layer.weight")
                    impl_->context.prepare_linear_weights(
                        weight, weight_bits == 4 && name.find(".mlp.") == std::string::npos ? 8 : weight_bits,
                        weight_bits == 8 || name.find(".mlp.") == std::string::npos
                            ? static_cast<std::int32_t>(weight.size(1))
                            : group_size,
                        packed_prefill);
        return {};
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}
auto Gemma4Impl::create_state(std::size_t capacity) -> Result<GemmaState> {
    try {
        if (!capacity || capacity > static_cast<std::size_t>(impl_->maximum_position))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Gemma cache capacity"});
        GemmaState result;
        result.capacity = capacity;
        for (int index = 0; index < impl_->shared_begin; ++index) {
            const std::vector<std::int64_t> shape{1, static_cast<std::int64_t>(capacity),
                                                  impl_->key_heads * impl_->head_width[impl_->kind[index]]};
            result.layers.push_back({require(Tensor::zeros(shape, DType::F32, device())),
                                     require(Tensor::zeros(shape, DType::F32, device()))});
        }
        return result;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}
auto Gemma4Impl::forward(std::span<const std::int32_t> tokens, GemmaState& state, bool all_logits) -> Result<Tensor> {
    return run(tokens, state, all_logits, true);
}
auto Gemma4Impl::prefill(std::span<const std::int32_t> tokens, GemmaState& state) -> Result<void> {
    auto result = run(tokens, state, false, false);
    if (!result) return std::unexpected(std::move(result.error()));
    return {};
}
auto Gemma4Impl::run(std::span<const std::int32_t> tokens, GemmaState& state, bool all_logits, bool project)
    -> Result<Tensor> {
    try {
        if (tokens.empty() || state.position > state.capacity || tokens.size() > state.capacity - state.position ||
            state.layers.size() != static_cast<std::size_t>(impl_->shared_begin))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Gemma input or cache state"});
        auto& context = impl_->context;
        const auto length = static_cast<std::int64_t>(tokens.size());
        auto hidden = impl_->tokens->forward(context, tokens);
        auto token_inputs = impl_->per_layer_tokens->forward(context, tokens);
        auto projection =
            context.multiply(impl_->per_layer_projection->forward(context, hidden), impl_->projection_scale);
        projection = context.reshape(projection, {1, length, impl_->layer_count, impl_->per_layer_width});
        auto per_layer = context.multiply(
            context.add(impl_->per_layer_norm->forward(context, projection),
                        context.reshape(token_inputs, {1, length, impl_->layer_count, impl_->per_layer_width})),
            impl_->combination_scale);
        std::vector<std::int32_t> indices(tokens.size());
        std::iota(indices.begin(), indices.end(), state.position);
        auto index = require(Tensor::from_host({length}, std::span<const std::int32_t>(indices), device()));
        std::array<Tensor, 2> masks;
        std::array<std::array<Tensor, 2>, 2> positions;
        const auto extent = std::min(state.capacity, ((state.position + tokens.size() + 127) / 128) * 128);
        const std::array<std::size_t, 2> key_starts{state.crop_local_attention && device() == tensor::Device::cpu() &&
                                                            state.position + 1 > static_cast<std::size_t>(impl_->window)
                                                        ? ((state.position + 1 - impl_->window) / 128) * 128
                                                        : 0,
                                                    0};
        for (int type = 0; type < 2; ++type) {
            const auto start = key_starts[type], count = extent - start;
            std::vector<float> values(tokens.size() * count, -1e9F);
            for (std::size_t row = 0; row < tokens.size(); ++row)
                for (std::size_t column = start; column <= state.position + row; ++column)
                    if (type == 1 || state.position + row - column < static_cast<std::size_t>(impl_->window))
                        values[row * count + column - start] = 0.F;
            masks[type] = require(Tensor::from_host({1, 1, length, static_cast<std::int64_t>(count)},
                                                    std::span<const float>(values), device()));
            positions[type] = impl_->positions(type, state.position, tokens.size());
        }
        context.profile_phase("gemma_decoder");
        for (int layer = 0; layer < impl_->layer_count; ++layer) {
            auto input = context.reshape(context.slice(per_layer, 2, layer, 1), {1, length, impl_->per_layer_width});
            const auto type = impl_->kind[layer];
            hidden = impl_->layers->at(layer)->forward(context, hidden, input, state.layers[impl_->cache_layer[layer]],
                                                       index, masks[type], positions[type][0], positions[type][1],
                                                       key_starts[type]);
        }
        Tensor logits;
        if (project) {
            if (!all_logits) hidden = context.slice(hidden, 1, length - 1, 1);
            hidden = impl_->norm->forward(context, hidden);
            context.profile_phase("gemma_generator");
            logits = impl_->lm_head->forward(context, hidden);
            logits = context.multiply(context.tanh(context.multiply(logits, impl_->inverse_logit_scale)),
                                      impl_->logit_scale);
        }
        context.synchronize();
        state.position += tokens.size();
        return logits;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}
auto Gemma4Impl::preparation_ns() const -> std::uint64_t { return impl_->context.preparation_ns(); }
} // namespace kidi::model