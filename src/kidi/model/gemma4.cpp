#include "kidi/model/gemma4.h"

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
auto selected_token(Result<Tensor> output) -> Result<std::int32_t> {
    if (!output) return std::unexpected(std::move(output.error()));
    auto selected = output->data<std::int32_t>();
    if (!selected) return std::unexpected(std::move(selected.error()));
    if ((*selected)[0] < 0) return std::unexpected(Error{ErrorCode::RUNTIME, "Gemma 4 returned invalid token scores"});
    return (*selected)[0];
}
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
    const ModuleScope storage(tensor::Device::cpu());
    return layers::TokenEmbedding(vocabulary, width, scale, bits, groups);
}
auto output_projection(std::int32_t hidden, std::int32_t vocabulary, std::int32_t bits = 0) -> layers::Linear {
    const ModuleScope host(tensor::Device::cpu());
    return layers::Linear(hidden, vocabulary, true, false, bits);
}
auto float_parameter(const Tensor& input) -> Tensor {
    if (input.dtype() == DType::F32) return input;
    if (input.dtype() != DType::BF16)
        throw ops::Failure({ErrorCode::UNSUPPORTED, "Gemma 4 parameters must be BF16 or FP32"});
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
    bool packed_prefill = false;
    std::int32_t vocabulary, per_layer_vocabulary;
    std::vector<std::size_t> cache_layer;
    std::vector<int> kind;
    layers::TokenEmbedding tokens, per_layer_tokens;
    ModuleList<layers::Gemma4BlockImpl> layers;
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
          vocabulary(config["vocab_size"].as<int>()),
          per_layer_vocabulary(config["vocab_size_per_layer_input"].as<int>()),
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
            layers->push_back(layers::Gemma4Block(
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
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, "unsupported Gemma 4 text architecture"});
        for (const auto* name :
             {"hidden_size", "hidden_size_per_layer_input", "num_hidden_layers", "intermediate_size",
              "max_position_embeddings", "sliding_window", "num_attention_heads", "num_key_value_heads", "head_dim",
              "global_head_dim", "vocab_size", "vocab_size_per_layer_input"})
            if (config[name].as<int>() <= 0)
                return std::unexpected(
                    Error{ErrorCode::INVALID_MANIFEST, std::string("invalid Gemma 4 dimension: ") + name});
        const auto count = config["num_hidden_layers"].as<int>();
        const auto shared = config["num_kv_shared_layers"].as<int>();
        if (shared < 0 || shared >= count || config["layer_types"].size() != static_cast<std::size_t>(count) ||
            config["num_attention_heads"].as<int>() % config["num_key_value_heads"].as<int>() ||
            config["head_dim"].as<int>() % 2 || config["global_head_dim"].as<int>() % 2 ||
            (!config["num_global_key_value_heads"].IsNull() &&
             config["num_global_key_value_heads"].as<int>() != config["num_key_value_heads"].as<int>()))
            return std::unexpected(
                Error{ErrorCode::INVALID_MANIFEST, "invalid Gemma 4 attention or sharing dimensions"});
        std::array<bool, 2> seen{};
        for (int index = 0; index < count; ++index) {
            const auto name = config["layer_types"][index].as<std::string>();
            if (name != "sliding_attention" && name != "full_attention")
                return std::unexpected(Error{ErrorCode::UNSUPPORTED, "unknown Gemma 4 layer type"});
            const auto type = name == "sliding_attention" ? 0 : 1;
            if (index < count - shared)
                seen[type] = true;
            else if (!seen[type])
                return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "missing shared KV source"});
        }
        for (const auto* name : {"rms_norm_eps", "final_logit_softcapping"}) {
            const auto value = config[name].as<float>();
            if (!std::isfinite(value) || value <= 0)
                return std::unexpected(
                    Error{ErrorCode::INVALID_MANIFEST, "invalid Gemma 4 normalization or logit cap"});
        }
        for (const auto* name : {"sliding_attention", "full_attention"}) {
            const auto rope = config["rope_parameters"][name];
            const auto theta = rope["rope_theta"].as<float>();
            const auto fraction = rope["partial_rotary_factor"].as<float>(1.F);
            const auto type = rope["rope_type"].as<std::string>();
            if ((type != "default" && type != "proportional") || !std::isfinite(theta) || theta <= 0 ||
                !std::isfinite(fraction) || fraction <= 0 || fraction > 1 || rope["factor"].as<float>(1.F) != 1.F)
                return std::unexpected(Error{ErrorCode::UNSUPPORTED, "unsupported Gemma 4 rotary configuration"});
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
                    Error{ErrorCode::INVALID_ARGUMENT, "unknown Gemma 4 checkpoint parameter: " + name});
            }
            if (impl_->qat && (value.dtype() == DType::U8 || value.dtype() == DType::I8)) {
                const auto suffix = name.ends_with(".embedding_quantized") ? std::string(".embedding_quantized")
                                                                           : std::string(".weight");
                const auto module = name.substr(0, name.size() - suffix.size());
                const auto bits = quant_bits(impl_->construction_config,
                                             module == "lm_head" ? module : "model.language_model." + module);
                if (impl_->construction_config["packed_weights_signed"].as<bool>(false)) {
                    if (!bits || value.dtype() != DType::U8)
                        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "signed packed weights must use U8 storage"});
                    state.emplace(name, value);
                    continue;
                }
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
        impl_->packed_prefill = packed_prefill;
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
auto Gemma4Impl::create_state(std::size_t capacity) -> Result<Gemma4State> {
    try {
        if (!capacity || capacity > static_cast<std::size_t>(impl_->maximum_position))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Gemma 4 cache capacity"});
        Gemma4State result;
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
auto Gemma4Impl::forward(std::span<const std::int32_t> tokens, Gemma4State& state, bool all_logits) -> Result<Tensor> {
    return project(tokens, state, all_logits, false);
}
auto Gemma4Impl::fork_state(const Gemma4State& source, std::size_t prefix_length, std::size_t capacity)
    -> Result<Gemma4State> {
    try {
        if (prefix_length > source.position || source.position > source.capacity || prefix_length > capacity ||
            source.layers.size() != static_cast<std::size_t>(impl_->shared_begin))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Gemma 4 prefix snapshot"});
        for (std::size_t layer = 0; layer < source.layers.size(); ++layer)
            for (const auto* tensor : {&source.layers[layer].key, &source.layers[layer].value})
                if (!tensor->defined() || tensor->device() != device() || tensor->dtype() != DType::F32 ||
                    tensor->dimensions() != 3 || tensor->size(0) != 1 || tensor->size(1) != source.capacity ||
                    tensor->size(2) != impl_->key_heads * impl_->head_width[impl_->kind[layer]])
                    throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Gemma 4 snapshot cache geometry"});
        auto result = require(create_state(capacity));
        impl_->context.synchronize();
        if (prefix_length) {
            for (std::size_t layer = 0; layer < source.layers.size(); ++layer) {
                const auto key = impl_->context.slice(source.layers[layer].key, 1, 0, prefix_length);
                const auto value = impl_->context.slice(source.layers[layer].value, 1, 0, prefix_length);
                impl_->context.copy_slice_(result.layers[layer].key, key, 1, 0);
                impl_->context.copy_slice_(result.layers[layer].value, value, 1, 0);
            }
            impl_->context.synchronize();
        }
        result.position = prefix_length;
        result.crop_local_attention = source.crop_local_attention;
        return result;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}
auto Gemma4Impl::forward_token(std::span<const std::int32_t> tokens, Gemma4State& state) -> Result<std::int32_t> {
    return selected_token(project(tokens, state, false, true));
}
auto Gemma4Impl::forward_batch(std::span<const std::int32_t> tokens, std::span<Gemma4State*> states) -> Result<Tensor> {
    return run_batch(tokens, states, false);
}
auto Gemma4Impl::forward_batch_tokens(std::span<const std::int32_t> tokens, std::span<Gemma4State*> states)
    -> Result<std::vector<std::int32_t>> {
    auto output = run_batch(tokens, states, true);
    if (!output) return std::unexpected(std::move(output.error()));
    auto selected = output->data<std::int32_t>();
    if (!selected) return std::unexpected(std::move(selected.error()));
    if (std::ranges::any_of(*selected, [](auto token) { return token < 0; }))
        return std::unexpected(Error{ErrorCode::RUNTIME, "batched Gemma 4 returned invalid token scores"});
    return std::vector<std::int32_t>(selected->begin(), selected->end());
}
auto Gemma4Impl::run_batch(std::span<const std::int32_t> tokens, std::span<Gemma4State*> states, bool select)
    -> Result<Tensor> {
    try {
        if (tokens.empty() || tokens.size() != states.size())
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "batched decode requires one token per state"});
        for (std::size_t row = 0; row < states.size(); ++row) {
            const auto* state = states[row];
            if (!state || state->position >= state->capacity ||
                state->layers.size() != static_cast<std::size_t>(impl_->shared_begin))
                throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid batched decode state"});
            for (std::size_t producer = 0; producer < state->layers.size(); ++producer) {
                for (const auto* tensor : {&state->layers[producer].key, &state->layers[producer].value})
                    if (!tensor->defined() || tensor->device() != device() || tensor->dtype() != DType::F32 ||
                        tensor->dimensions() != 3 || tensor->size(0) != 1 || tensor->size(1) != state->capacity ||
                        tensor->size(2) != impl_->key_heads * impl_->head_width[impl_->kind[producer]])
                        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid batched cache geometry"});
                for (std::size_t previous = 0; previous < row; ++previous)
                    if (require(state->layers[producer].key.host_bytes()).data() ==
                            require(states[previous]->layers[producer].key.host_bytes()).data() ||
                        require(state->layers[producer].value.host_bytes()).data() ==
                            require(states[previous]->layers[producer].value.host_bytes()).data())
                        throw ops::Failure(
                            {ErrorCode::INVALID_ARGUMENT, "batched requests must not alias mutable caches"});
            }
        }
        return project(tokens, *states.front(), true, select, states);
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}
struct Gemma4Impl::Attention {
    std::vector<std::array<Tensor, 2>> masks;
    std::vector<std::array<std::size_t, 2>> key_starts;
    std::array<std::array<Tensor, 2>, 2> angles;
};

auto Gemma4Impl::embed(std::span<const std::int32_t> tokens) -> std::array<Tensor, 2> {
    auto& context = impl_->context;
    const auto length = static_cast<std::int64_t>(tokens.size());
    auto hidden = impl_->tokens->forward(context, tokens);
    auto token_inputs = impl_->per_layer_tokens->forward(context, tokens);
    auto projection = context.multiply(impl_->per_layer_projection->forward(context, hidden), impl_->projection_scale);
    projection = context.reshape(projection, {1, length, impl_->layer_count, impl_->per_layer_width});
    auto per_layer = context.multiply(
        context.add(impl_->per_layer_norm->forward(context, projection),
                    context.reshape(token_inputs, {1, length, impl_->layer_count, impl_->per_layer_width})),
        impl_->combination_scale);
    return {std::move(hidden), std::move(per_layer)};
}

auto Gemma4Impl::attention_inputs(Gemma4State& state, std::span<Gemma4State*> batch_states, std::size_t step_count)
    -> Attention {
    const auto requests = batch_states.empty() ? 1 : batch_states.size();
    Attention result;
    result.masks.resize(requests);
    result.key_starts.resize(requests);
    std::vector<std::array<std::array<Tensor, 2>, 2>> row_angles(requests);
    for (std::size_t row = 0; row < requests; ++row) {
        const auto& request = batch_states.empty() ? state : *batch_states[row];
        const auto extent = std::min(request.capacity, ((request.position + step_count + 127) / 128) * 128);
        result.key_starts[row] = {request.crop_local_attention && device() == tensor::Device::cpu() &&
                                          request.position + 1 > static_cast<std::size_t>(impl_->window)
                                      ? ((request.position + 1 - impl_->window) / 128) * 128
                                      : 0,
                                  0};
        for (int type = 0; type < 2; ++type) {
            const auto start = result.key_starts[row][type], count = extent - start;
            std::vector<float> values(step_count * count, -1e9F);
            for (std::size_t query = 0; query < step_count; ++query)
                for (std::size_t column = start; column <= request.position + query; ++column)
                    if (type == 1 || request.position + query - column < static_cast<std::size_t>(impl_->window))
                        values[query * count + column - start] = 0.F;
            result.masks[row][type] = require(
                Tensor::from_host({1, 1, static_cast<std::int64_t>(step_count), static_cast<std::int64_t>(count)},
                                  std::span<const float>(values), device()));
            row_angles[row][type] = impl_->positions(type, request.position, step_count);
        }
    }
    for (int type = 0; type < 2; ++type)
        for (std::size_t component = 0; component < 2; ++component) {
            if (requests == 1) {
                result.angles[type][component] = row_angles[0][type][component];
            } else {
                std::vector<Tensor> pieces;
                for (const auto& row : row_angles) pieces.push_back(row[type][component]);
                result.angles[type][component] = impl_->context.concat(pieces, 1);
            }
        }
    return result;
}

auto Gemma4Impl::per_layer_input(const Tensor& per_layer, int layer, std::int64_t length) -> Tensor {
    auto& context = impl_->context;
    if (length == 1)
        return context.reshape(
            require(context.reshape(per_layer, {impl_->layer_count, impl_->per_layer_width}).select(0, layer)),
            {1, 1, impl_->per_layer_width});
    return context.reshape(context.slice(per_layer, 2, layer, 1), {1, length, impl_->per_layer_width});
}

auto Gemma4Impl::head(const Tensor& hidden, bool select) -> Tensor {
    auto& context = impl_->context;
    auto logits = impl_->lm_head->forward(context, impl_->norm->forward(context, hidden));
    logits = context.multiply(context.tanh(context.multiply(logits, impl_->inverse_logit_scale)), impl_->logit_scale);
    return select ? context.greedy_token(logits) : logits;
}

auto Gemma4Impl::prefill(std::span<const std::int32_t> tokens, Gemma4State& state) -> Result<void> {
    try {
        if (tokens.empty() || state.position > state.capacity || tokens.size() > state.capacity - state.position ||
            state.layers.size() != static_cast<std::size_t>(impl_->shared_begin))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Gemma 4 input or cache state"});
        auto& context = impl_->context;
        if (!state.profile_started) {
            context.profile_phase("gemma4_request");
            state.profile_started = true;
        }
        context.profile_phase(state.prefilling ? "prefill_embedding" : "decode_embedding");
        const auto length = static_cast<std::int64_t>(tokens.size());
        auto [hidden, per_layer] = embed(tokens);
        auto attention = attention_inputs(state, {}, tokens.size());
        context.profile_phase(state.prefilling ? "prefill_body" : "decode_body");
        std::array<layers::Gemma4AttentionSegment, 1> segments;
        for (int layer = 0; layer < impl_->shared_begin; ++layer) {
            // The last producer layer only has to write its K/V slice; its output is never read.
            const bool cache_only = layer + 1 == impl_->shared_begin;
            const auto input = cache_only ? Tensor{} : per_layer_input(per_layer, layer, length);
            const auto type = impl_->kind[layer];
            segments[0] = {&state.layers[impl_->cache_layer[layer]], state.position, tokens.size(),
                           &attention.masks[0][type], static_cast<std::int64_t>(attention.key_starts[0][type])};
            hidden = impl_->layers->at(layer)->forward_segments(
                context, hidden, input, segments, attention.angles[type][0], attention.angles[type][1], cache_only);
        }
        context.synchronize();
        state.position += tokens.size();
        return {};
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}

auto Gemma4Impl::project(std::span<const std::int32_t> tokens, Gemma4State& state, bool all_logits, bool select,
                         std::span<Gemma4State*> batch_states) -> Result<Tensor> {
    const ops::DecodeScope decode_scope(!batch_states.empty());
    try {
        const std::size_t token_count = tokens.size();
        const std::size_t step_count = batch_states.empty() ? token_count : 1;
        if (!token_count || state.position > state.capacity || step_count > state.capacity - state.position ||
            state.layers.size() != static_cast<std::size_t>(impl_->shared_begin))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Gemma 4 input or cache state"});
        if (!all_logits && token_count > 1 && batch_states.empty() && last_token_prefill()) {
            for (auto token : tokens)
                if (token < 0 || token >= impl_->vocabulary || token >= impl_->per_layer_vocabulary)
                    throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "embedding token outside vocabulary"});
            require(prefill(tokens.first(token_count - 1), state));
            return project(tokens.last(1), state, false, select);
        }
        auto& context = impl_->context;
        if (!state.profile_started) {
            context.profile_phase("gemma4_request");
            state.profile_started = true;
        }
        context.profile_phase(state.prefilling ? "prefill_embedding" : "decode_embedding");
        const auto length = static_cast<std::int64_t>(token_count);
        auto [hidden, per_layer] = embed(tokens);
        auto attention = attention_inputs(state, batch_states, step_count);
        context.profile_phase(state.prefilling ? "prefill_body" : "decode_body");
        const auto requests = batch_states.empty() ? 1 : batch_states.size();
        std::vector<layers::Gemma4AttentionSegment> segments(requests);
        // Only the final queries reach the shared-K/V layers when just the last logit is needed.
        const bool shared_tail = !all_logits && batch_states.empty() && length >= 64 &&
                                 impl_->shared_begin < impl_->layer_count && shared_prefill_tail();
        std::size_t query_offset = 0;
        auto active_length = length;
        for (int layer = 0; layer < impl_->layer_count; ++layer) {
            if (shared_tail && layer == impl_->shared_begin) {
                active_length = 4;
                query_offset = length - active_length;
                hidden = context.slice(hidden, 1, query_offset, active_length);
                per_layer = context.slice(per_layer, 1, query_offset, active_length);
                for (int type = 0; type < 2; ++type) {
                    attention.masks[0][type] = context.slice(attention.masks[0][type], 2, query_offset, active_length);
                    for (auto& component : attention.angles[type])
                        component = context.slice(component, 1, query_offset, active_length);
                }
            }
            const auto input = per_layer_input(per_layer, layer, active_length);
            const auto type = impl_->kind[layer];
            for (std::size_t row = 0; row < requests; ++row) {
                auto& request = batch_states.empty() ? state : *batch_states[row];
                segments[row] = {&request.layers[impl_->cache_layer[layer]], request.position + query_offset,
                                 batch_states.empty() ? static_cast<std::size_t>(active_length) : step_count,
                                 &attention.masks[row][type],
                                 static_cast<std::int64_t>(attention.key_starts[row][type])};
            }
            hidden = impl_->layers->at(layer)->forward_segments(context, hidden, input, segments,
                                                                attention.angles[type][0], attention.angles[type][1]);
        }
        context.profile_phase(state.prefilling ? "prefill_head" : "decode_head");
        if (!all_logits) hidden = context.slice(hidden, 1, active_length - 1, 1);
        auto logits = head(hidden, select);
        context.synchronize();
        for (std::size_t row = 0; row < requests; ++row) {
            auto& request = batch_states.empty() ? state : *batch_states[row];
            request.position += step_count;
            request.prefilling = false;
        }
        return logits;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}
auto Gemma4Impl::preparation_ns() const -> std::uint64_t { return impl_->context.preparation_ns(); }
auto Gemma4Impl::last_token_prefill() const -> bool { return impl_->qat && device() == tensor::Device::cpu(); }
auto Gemma4Impl::shared_prefill_tail() const -> bool {
    return impl_->qat && device() == tensor::Device::apple_gpu() && !impl_->packed_prefill;
}
} // namespace kidi::model