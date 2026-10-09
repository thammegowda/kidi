#include "kidi/model/gemma4.h"
#include "kidi/runtime/operator.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
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
/// Reuses `tensor` as a step input when its layout matches, reallocating only when the shape changes.
auto step_input(Tensor& tensor, std::vector<std::int64_t> shape, DType dtype, tensor::Device device) -> Tensor& {
    if (device == tensor::Device::web_gpu()) device = tensor::Device::cpu();
    if (!tensor.defined() || tensor.dtype() != dtype || tensor.device() != device ||
        !std::ranges::equal(tensor.shape(), shape))
        tensor = require(Tensor::empty(std::move(shape), dtype, device));
    return tensor;
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
auto embedding(std::int32_t vocabulary, std::int32_t width, float scale, std::int32_t bits = 0,
               std::int32_t groups = 1) -> layers::TokenEmbedding {
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
auto concatenate_projection_rows(const Tensor& first, const Tensor& second) -> Tensor {
    if (first.dimensions() != 2 || second.dimensions() != 2 || first.dtype() != second.dtype() ||
        !std::ranges::equal(first.shape(), second.shape()))
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "gate/up projection shape or dtype mismatch"});
    auto shape = std::vector<std::int64_t>(first.shape().begin(), first.shape().end());
    shape[0] *= 2;
    if (first.device() == tensor::Device::cpu() && second.device() == tensor::Device::cpu()) {
        const auto first_view = first.host_bytes(), second_view = second.host_bytes();
        // Checkpoints laid out with each up projection directly after its gate projection need no copy.
        if (first_view && second_view && first_view->data() + first_view->size() == second_view->data())
            return require(Tensor::from_blob(shape, first.dtype(), {first_view->data(), first_view->size() * 2},
                                             std::make_shared<std::array<Tensor, 2>>(std::array{first, second})));
    }
    auto output = require(Tensor::empty(shape, first.dtype()));
    auto destination = require(output.host_bytes());
    const auto first_host = require(first.to(tensor::Device::cpu()));
    const auto second_host = require(second.to(tensor::Device::cpu()));
    const auto first_bytes = require(first_host.host_bytes());
    const auto second_bytes = require(second_host.host_bytes());
    std::ranges::copy(first_bytes, destination.begin());
    std::ranges::copy(second_bytes, destination.begin() + first_bytes.size());
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
    core::KVCachePrecision kv_cache_precision = core::KVCachePrecision::AUTO;
    bool packed_prefill = false;
    std::int32_t vocabulary, per_layer_vocabulary;
    std::vector<std::size_t> cache_layer;
    std::vector<int> kind;
    layers::TokenEmbedding tokens, per_layer_tokens;
    ModuleList<layers::Gemma4BlockImpl> layers;
    layers::Linear per_layer_projection, lm_head;
    layers::RmsNorm per_layer_norm, norm;
    Tensor projection_scale, combination_scale, logit_scale, inverse_logit_scale;
    std::vector<Tensor> step_inputs;
    std::string npu_prefill_key, npu_decode_key;

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
        std::vector<float> cosine(length * half), sine(length * half);
        rotary_angles(type, start, cosine, sine);
        const std::vector<std::int64_t> shape{1, static_cast<std::int64_t>(length), 1, half};
        return {require(Tensor::from_host(shape, std::span<const float>(cosine), context.device())),
                require(Tensor::from_host(shape, std::span<const float>(sine), context.device()))};
    }
    /// Rotary cosine and sine rows for positions starting at `start`, `head_width / 2` values per row.
    auto rotary_angles(int type, std::size_t start, std::span<float> cosine, std::span<float> sine) const -> void {
        const auto half = static_cast<std::size_t>(head_width[type] / 2);
        const auto rotated = static_cast<std::size_t>(rotary_fraction[type] * static_cast<float>(half));
        for (std::size_t row = 0; row < cosine.size() / half; ++row)
            for (std::size_t channel = 0; channel < half; ++channel) {
                const auto frequency =
                    channel < rotated ? std::pow(theta[type], -2.F * channel / head_width[type]) : 0.F;
                const auto angle = static_cast<float>(start + row) * frequency;
                cosine[row * half + channel] = std::cos(angle);
                sine[row * half + channel] = std::sin(angle);
            }
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
auto Gemma4Impl::set_checkpoint(const checkpoint::Weights& weights, std::int32_t weight_bits, std::int32_t group_size,
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
            auto declaration = name;
            for (const auto marker : {std::string_view(".mlp.gate_proj."), std::string_view(".mlp.up_proj.")})
                if (const auto position = name.find(marker); position != std::string::npos)
                    declaration.replace(position, marker.size(), ".mlp.gate_up_proj.");
            if (!declared.contains(declaration)) {
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
                const auto bits = quant_bits(impl_->construction_config, key.substr(0, key.size() - suffix.size()));
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
                const bool quantization_scale = name.ends_with("weight_scale") || name.ends_with("embedding_scale") ||
                                                name.ends_with("input_activation_scale") ||
                                                name.ends_with("output_activation_scale") ||
                                                name.ends_with("k_cache_scale") || name.ends_with("v_cache_scale");
                if (impl_->qat && quantization_scale) {
                    const bool activation_scale =
                        name.ends_with("input_activation_scale") || name.ends_with("output_activation_scale");
                    const auto scales = require(value.data<float>());
                    if (std::ranges::any_of(scales, [&](float scale) {
                            return !std::isfinite(scale) || scale < 0 || (!activation_scale && scale == 0);
                        }))
                        throw ops::Failure(
                            {ErrorCode::INVALID_ARGUMENT, "invalid trained quantization scale: " + name});
                }
                state.emplace(name, name.ends_with("norm.weight") || name.ends_with("layer_scalar")
                                        ? float_parameter(value)
                                        : value);
            }
        }
        for (std::int32_t layer = 0; layer < impl_->layer_count; ++layer) {
            const auto prefix = "layers." + std::to_string(layer) + ".mlp.";
            for (const auto suffix :
                 {std::string_view("weight"), std::string_view("weight_scale"),
                  std::string_view("input_activation_scale"), std::string_view("output_activation_scale")}) {
                if (!impl_->qat && suffix != "weight") continue;
                // Loaders that place weights on a device may supply the pair already fused.
                if (state.contains(prefix + "gate_up_proj." + std::string(suffix))) continue;
                const auto gate_name = prefix + "gate_proj." + std::string(suffix);
                const auto up_name = prefix + "up_proj." + std::string(suffix);
                const auto gate = state.find(gate_name), up = state.find(up_name);
                if (gate == state.end() || up == state.end())
                    throw ops::Failure(
                        {ErrorCode::INVALID_ARGUMENT, "missing gate/up checkpoint parameter: " + prefix});
                Tensor combined;
                if (suffix.ends_with("activation_scale")) {
                    const auto& gate_scale = gate->second;
                    const auto& up_scale = up->second;
                    if (gate_scale.numel() != 1 || up_scale.numel() != 1 ||
                        require(gate_scale.data<float>())[0] != require(up_scale.data<float>())[0])
                        throw ops::Failure({ErrorCode::UNSUPPORTED,
                                            "gate/up activation scales must match for a fused linear: " + prefix});
                    combined = gate->second;
                } else {
                    combined = concatenate_projection_rows(gate->second, up->second);
                }
                state.emplace(prefix + "gate_up_proj." + std::string(suffix), std::move(combined));
                state.erase(gate);
                state.erase(up);
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
        const ops::StepCompilerScope compiler(impl_->context.step_compiler());
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
auto Gemma4Impl::set_precision(core::InferencePrecision precision) noexcept -> void {
    impl_->context.set_precision(precision);
}
auto Gemma4Impl::set_kv_cache_precision(core::KVCachePrecision precision) noexcept -> void {
    impl_->kv_cache_precision = precision;
}
auto Gemma4Impl::create_state(std::size_t capacity) -> Result<Gemma4State> {
    try {
        if (!capacity || capacity > static_cast<std::size_t>(impl_->maximum_position))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Gemma 4 cache capacity"});
        Gemma4State result;
        result.capacity = capacity;
        if (impl_->kv_cache_precision == core::KVCachePrecision::BF16 ||
            impl_->kv_cache_precision == core::KVCachePrecision::E4M3 ||
            impl_->kv_cache_precision == core::KVCachePrecision::E5M2)
            throw ops::Failure({ErrorCode::UNSUPPORTED, "selected KV-cache precision is not implemented"});
        // The QAT path already rounds cache rows onto an INT8 grid, so bytes cost no accuracy and a quarter the reads.
        const auto* cache_override = std::getenv("KIDI_INT8_KV_CACHE");
        const auto& capabilities = tensor::DEVICE_CAPABILITIES[device().kind];
        auto byte_cache_supported =
            impl_->qat && capabilities.calibrated_int8_cast && capabilities.blockwise_int8_attention &&
            capacity <= capabilities.blockwise_int8_attention_max_tokens && impl_->shared_begin > 0;
        for (int index = 0; byte_cache_supported && index < impl_->shared_begin; ++index)
            byte_cache_supported = impl_->layers->at(static_cast<std::size_t>(index))->has_cache_scales();
        const auto request_byte_cache = impl_->kv_cache_precision == core::KVCachePrecision::INT8;
        if (request_byte_cache && !byte_cache_supported)
            throw ops::Failure({ErrorCode::UNSUPPORTED, "INT8 KV cache is unsupported by this model or backend"});
        const auto byte_cache =
            byte_cache_supported &&
            (request_byte_cache || (impl_->kv_cache_precision == core::KVCachePrecision::AUTO &&
                                    (cache_override == nullptr || std::string_view(cache_override) != "0")));
        const auto cache_dtype = byte_cache ? DType::I8 : DType::F32;
        for (int index = 0; index < impl_->shared_begin; ++index) {
            const std::vector<std::int64_t> shape{1, static_cast<std::int64_t>(capacity),
                                                  impl_->key_heads * impl_->head_width[impl_->kind[index]]};
            result.layers.push_back({require(Tensor::zeros(shape, cache_dtype, device())),
                                     require(Tensor::zeros(shape, cache_dtype, device()))});
        }
        return result;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}
auto Gemma4Impl::forward(std::span<const std::int32_t> tokens, Gemma4State& state, bool all_logits) -> Result<Tensor> {
    return project(tokens, state, all_logits, false);
}
auto Gemma4Impl::fork_state(const Gemma4State& source, std::size_t prefix_length,
                            std::size_t capacity) -> Result<Gemma4State> {
    try {
        if (prefix_length > source.position || source.position > source.capacity || prefix_length > capacity ||
            source.layers.size() != static_cast<std::size_t>(impl_->shared_begin))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Gemma 4 prefix snapshot"});
        for (std::size_t layer = 0; layer < source.layers.size(); ++layer)
            for (const auto* tensor : {&source.layers[layer].key, &source.layers[layer].value})
                if (!tensor->defined() || tensor->device() != device() ||
                    (tensor->dtype() != DType::F32 && tensor->dtype() != DType::I8) || tensor->dimensions() != 3 ||
                    tensor->size(0) != 1 || tensor->size(1) != source.capacity ||
                    tensor->size(2) != impl_->key_heads * impl_->head_width[impl_->kind[layer]])
                    throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Gemma 4 snapshot cache geometry"});
        auto result = require(create_state(capacity));
        for (std::size_t layer = 0; layer < source.layers.size(); ++layer) {
            result.layers[layer].key_quantization = source.layers[layer].key_quantization;
            result.layers[layer].value_quantization = source.layers[layer].value_quantization;
        }
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
        result.images = source.images;
        result.crop_local_attention = source.crop_local_attention;
        return result;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}
auto Gemma4Impl::forward_token(std::span<const std::int32_t> tokens, Gemma4State& state) -> Result<std::int32_t> {
    return selected_token(project(tokens, state, false, true));
}
auto Gemma4Impl::select_token(std::span<const std::int32_t> tokens, Gemma4State& state) -> Result<Tensor> {
    return project(tokens, state, false, true);
}
auto Gemma4Impl::select_batch_tokens(std::span<const std::int32_t> tokens, std::span<Gemma4State*> states)
    -> Result<Tensor> {
    return run_batch(tokens, states, true);
}
auto Gemma4Impl::forward_token(const Tensor& token, Gemma4State& state) -> Result<Tensor> {
    try {
        if (token.dtype() != DType::I32 || token.dimensions() != 1 || token.size(0) != 1 ||
            token.device() != device() || state.position >= state.capacity ||
            state.layers.size() != static_cast<std::size_t>(impl_->shared_begin))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Gemma 4 token tensor or cache state"});
        if (!can_decode_step(state))
            throw ops::Failure({ErrorCode::UNSUPPORTED, "token tensor decoding needs host-writable step inputs"});
        auto& context = impl_->context;
        if (!state.profile_started) {
            context.profile_phase("gemma4_request");
            state.profile_started = true;
        }
        context.profile_phase("decode_body");
        auto output = decode_step(token, state, true);
        context.synchronize();
        ++state.position;
        state.prefilling = false;
        return output;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}
auto Gemma4Impl::forward_batch(std::span<const std::int32_t> tokens, std::span<Gemma4State*> states) -> Result<Tensor> {
    return run_batch(tokens, states, false);
}
auto Gemma4Impl::forward_batch_tokens(std::span<const std::int32_t> tokens,
                                      std::span<Gemma4State*> states) -> Result<std::vector<std::int32_t>> {
    auto output = run_batch(tokens, states, true);
    if (!output) return std::unexpected(std::move(output.error()));
    auto selected = output->data<std::int32_t>();
    if (!selected) return std::unexpected(std::move(selected.error()));
    if (std::ranges::any_of(*selected, [](auto token) { return token < 0; }))
        return std::unexpected(Error{ErrorCode::RUNTIME, "batched Gemma 4 returned invalid token scores"});
    return std::vector<std::int32_t>(selected->begin(), selected->end());
}
auto Gemma4Impl::run_batch(std::span<const std::int32_t> tokens, std::span<Gemma4State*> states,
                           bool select) -> Result<Tensor> {
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
                    if (!tensor->defined() || tensor->device() != device() ||
                        (tensor->dtype() != DType::F32 && tensor->dtype() != DType::I8) || tensor->dimensions() != 3 ||
                        tensor->size(0) != 1 || tensor->size(1) != state->capacity ||
                        tensor->size(2) != impl_->key_heads * impl_->head_width[impl_->kind[producer]])
                        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid batched cache geometry"});
                for (std::size_t previous = 0; previous < row; ++previous)
                    if (state->layers[producer].key.storage_identity() ==
                            states[previous]->layers[producer].key.storage_identity() ||
                        state->layers[producer].value.storage_identity() ==
                            states[previous]->layers[producer].value.storage_identity())
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

auto Gemma4Impl::embed(std::span<const std::int32_t> tokens, std::span<const Gemma4ImageTokens> images,
                       std::size_t position) -> std::array<Tensor, 2> {
    auto& context = impl_->context;
    const auto length = static_cast<std::int64_t>(tokens.size());
    std::vector<std::int32_t> text_tokens;
    if (!images.empty()) {
        text_tokens.assign(tokens.begin(), tokens.end());
        for (const auto& image : images) {
            if (!image.embeddings.defined() || image.embeddings.dimensions() != 3 || image.embeddings.size(0) != 1 ||
                image.embeddings.size(2) != impl_->hidden || image.embeddings.dtype() != DType::F32 ||
                image.embeddings.device() != device())
                throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid image token embeddings"});
            const auto start = std::max(position, image.position);
            const auto end = std::min(position + tokens.size(), image.position + image.embeddings.size(1));
            for (auto index = start; index < end; ++index) text_tokens[index - position] = 0;
        }
        tokens = text_tokens;
    }
    auto hidden = impl_->tokens->forward(context, tokens);
    auto token_inputs = impl_->per_layer_tokens->forward(context, tokens);
    for (const auto& image : images) {
        const auto start = std::max(position, image.position);
        const auto end = std::min(position + tokens.size(), image.position + image.embeddings.size(1));
        if (start < end)
            context.copy_slice_(hidden, context.slice(image.embeddings, 1, start - image.position, end - start), 1,
                                start - position);
    }
    auto projection = context.multiply(impl_->per_layer_projection->forward(context, hidden), impl_->projection_scale);
    projection = context.reshape(projection, {1, length, impl_->layer_count, impl_->per_layer_width});
    auto per_layer = context.multiply(
        context.add(impl_->per_layer_norm->forward(context, projection),
                    context.reshape(token_inputs, {1, length, impl_->layer_count, impl_->per_layer_width})),
        impl_->combination_scale);
    return {std::move(hidden), std::move(per_layer)};
}

auto Gemma4Impl::attention_inputs(Gemma4State& state, std::span<Gemma4State*> batch_states,
                                  std::size_t step_count) -> Attention {
    const auto requests = batch_states.empty() ? 1 : batch_states.size();
    Attention result;
    result.masks.resize(requests);
    result.key_starts.resize(requests);
    std::vector<std::array<std::array<Tensor, 2>, 2>> row_angles(requests);
    for (std::size_t row = 0; row < requests; ++row) {
        const auto& request = batch_states.empty() ? state : *batch_states[row];
        const auto extent = std::min(request.capacity, ((request.position + step_count + 127) / 128) * 128);
        const auto crop_device = device() == tensor::Device::cpu() || device() == tensor::Device::vulkan() ||
                                 device() == tensor::Device::web_gpu();
        result.key_starts[row] = {request.crop_local_attention && crop_device &&
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
    if (select) return context.greedy_token(logits);
    logits = context.multiply(context.tanh(context.multiply(logits, impl_->inverse_logit_scale)), impl_->logit_scale);
    return logits;
}

auto Gemma4Impl::can_decode_step(const Gemma4State& state) const -> bool {
    return std::ranges::none_of(state.images, [&](const Gemma4ImageTokens& image) {
        return image.position <= state.position && state.position < image.position + image.embeddings.size(1);
    });
}

auto Gemma4Impl::decode_step(const Tensor& token, Gemma4State& state, bool select) -> Tensor {
    if (state.prefilling && state.images.empty() && !impl_->context.accelerator().empty())
        impl_->context.clear_replays("gemma4_prefill:");
    return captured_step(token, state, state.step, false, select);
}

auto Gemma4Impl::captured_step(const Tensor& tokens, Gemma4State& state, Gemma4StepInputs& inputs, bool prefill,
                               bool select) -> Tensor {
    auto& context = impl_->context;
    const auto device = context.device();
    const auto position = state.position, length = tokens.numel();
    auto index =
        require(step_input(inputs.index, {static_cast<std::int64_t>(length)}, DType::I32, device).data<std::int32_t>());
    for (std::size_t query = 0; query < length; ++query) index[query] = static_cast<std::int32_t>(position + query);
    // Key extents follow the context's step policy: 128-position buckets and local-attention crops on CPU, Vulkan and
    // WebGPU, coarser fixed shapes when an accelerator compiles the step.
    const auto extent = context.step_extent(position + length, state.capacity);
    const bool crop = state.crop_local_attention &&
                      (device == tensor::Device::cpu() || device == tensor::Device::vulkan() ||
                       device == tensor::Device::web_gpu()) &&
                      context.crop_local_attention();
    const std::array<std::size_t, 2> key_starts{crop && position + 1 > static_cast<std::size_t>(impl_->window)
                                                    ? ((position + 1 - impl_->window) / 128) * 128
                                                    : 0,
                                                0};
    for (int type = 0; type < 2; ++type) {
        const auto start = key_starts[type], count = extent - start;
        auto mask = require(step_input(inputs.masks[type],
                                       {1, 1, static_cast<std::int64_t>(length), static_cast<std::int64_t>(count)},
                                       DType::F32, device)
                                .data<float>());
        for (std::size_t query = 0; query < length; ++query)
            for (std::size_t column = 0; column < count; ++column) {
                const auto key = start + column, current = position + query;
                mask[query * count + column] =
                    key <= current && (type == 1 || current - key < static_cast<std::size_t>(impl_->window)) ? 0.F
                                                                                                             : -1e9F;
            }
        const std::vector<std::int64_t> shape{1, static_cast<std::int64_t>(length), 1, impl_->head_width[type] / 2};
        impl_->rotary_angles(type, position,
                             require(step_input(inputs.angles[type][0], shape, DType::F32, device).data<float>()),
                             require(step_input(inputs.angles[type][1], shape, DType::F32, device).data<float>()));
    }
    // Step inputs: tokens, positions, masks, angles, then key/value caches per producer layer.
    auto& step = impl_->step_inputs;
    step.assign({tokens, inputs.index, inputs.masks[0], inputs.masks[1], inputs.angles[0][0], inputs.angles[0][1],
                 inputs.angles[1][0], inputs.angles[1][1]});
    for (auto& input : step)
        if (input.device() != device) input = require(input.to(device));
    for (const auto& cache : state.layers) step.insert(step.end(), {cache.key, cache.value});
    // Image embeddings are request data, not graph constants. Prepare the multimodal embedding on the host and bind
    // it as step inputs so the same compiled transformer body works for every image of this shape.
    const bool image_rows = prefill && !state.images.empty();
    if (image_rows) {
        const auto ids = require(tokens.data<std::int32_t>());
        const auto embedded = embed(ids, state.images, position);
        step.insert(step.end(), embedded.begin(), embedded.end());
    }
    // An external per-layer table is read on the host, which a replay would skip, so its rows enter as a step input.
    const bool external_rows = !image_rows && impl_->per_layer_tokens->external();
    if (external_rows) {
        const auto ids = require(tokens.copy_to_host());
        step.push_back(impl_->per_layer_tokens->forward(
            context, std::span(reinterpret_cast<const std::int32_t*>(ids.data()), tokens.numel())));
    }
    const auto key = std::string(prefill ? "gemma4_prefill:" : "gemma4_decode:") + std::to_string(length) + ':' +
                     std::to_string(state.capacity) + ':' + std::to_string(extent) + ':' +
                     std::to_string(key_starts[0]) +
                     (image_rows ? ":images" : "") +
                     (prefill  ? ""
                      : select ? ":token"
                               : ":logits");
    if (context.accelerator() == "qnn-htp") {
        // Bound NPU prefill and decode shape caches to one executable each so attention buckets do not accumulate
        // duplicate HTP weights.
        auto& previous = prefill ? impl_->npu_prefill_key : impl_->npu_decode_key;
        if (previous != key) {
            context.clear_replays(prefill ? "gemma4_prefill:" : "gemma4_decode:");
            previous = key;
        }
    }
    const auto outputs = context.replay(key, step, [&](std::span<const Tensor> operands) {
        std::vector<layers::KeyValue> caches;
        for (std::size_t producer = 0; producer < state.layers.size(); ++producer)
            caches.push_back({operands[8 + 2 * producer], operands[9 + 2 * producer],
                              state.layers[producer].key_quantization, state.layers[producer].value_quantization});
        const auto layers = impl_->layer_count, width = impl_->per_layer_width;
        const auto rows = static_cast<std::int64_t>(length);
        auto hidden = image_rows ? operands[8 + 2 * state.layers.size()]
                                 : impl_->tokens->forward(context, operands[0]);
        const auto per_layer = [&] {
            if (image_rows) return operands[9 + 2 * state.layers.size()];
            const auto token_inputs = external_rows ? operands[8 + 2 * state.layers.size()]
                                                    : impl_->per_layer_tokens->forward(context, operands[0]);
            auto projection =
                context.multiply(impl_->per_layer_projection->forward(context, hidden), impl_->projection_scale);
            projection = context.reshape(projection, {1, rows, layers, width});
            return context.multiply(context.add(impl_->per_layer_norm->forward(context, projection),
                                                 context.reshape(token_inputs, {1, rows, layers, width})),
                                    impl_->combination_scale);
        }();
        std::array<layers::Gemma4AttentionSegment, 1> segments;
        // Prefill only has to write the producer layers' K/V; the last producer's output is never read.
        const auto count = prefill ? impl_->shared_begin : impl_->layer_count;
        for (int layer = 0; layer < count; ++layer) {
            const auto type = impl_->kind[layer];
            const bool cache_only = prefill && layer + 1 == count;
            segments[0] = {&caches[impl_->cache_layer[layer]],          position,    length, &operands[2 + type],
                           static_cast<std::int64_t>(key_starts[type]), &operands[1]};
            hidden = impl_->layers->at(layer)->forward_segments(
                context, hidden, cache_only ? Tensor{} : per_layer_input(per_layer, layer, rows), segments,
                operands[4 + 2 * type], operands[5 + 2 * type], cache_only);
        }
        for (std::size_t producer = 0; producer < state.layers.size(); ++producer) {
            state.layers[producer].key_quantization = caches[producer].key_quantization;
            state.layers[producer].value_quantization = caches[producer].value_quantization;
        }
        if (prefill) return std::vector<Tensor>{};
        return std::vector{head(hidden, select)};
    });
    return prefill ? Tensor{} : outputs[0];
}

auto Gemma4Impl::prefill(std::span<const std::int32_t> tokens, Gemma4State& state) -> Result<void> {
    return prefill_impl(tokens, state);
}
auto Gemma4Impl::prefill_impl(std::span<const std::int32_t> tokens, Gemma4State& state) -> Result<void> {
    try {
        if (tokens.empty() || state.position > state.capacity || tokens.size() > state.capacity - state.position ||
            state.layers.size() != static_cast<std::size_t>(impl_->shared_begin))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Gemma 4 input or cache state"});
        auto& context = impl_->context;
        if (!state.profile_started) {
            context.profile_phase("gemma4_request");
            state.profile_started = true;
        }
        // With an accelerator, power-of-two chunks run as captured steps; other lengths (prompt tails) stay eager
        // so each accelerator compiles only a few shapes.
        if (state.capture_prefill && !context.accelerator().empty() && context.step_compiler()->captures_prefill()) {
            // A short prompt tail reuses the compiled chunk. Padded rows follow every real row, so causal attention
            // never reads them, and later steps overwrite those cache positions.
            const auto chunk = context.prefill_chunk_size(1);
            const bool exact = std::has_single_bit(tokens.size()) && tokens.size() >= 16;
            const bool padded = !exact && tokens.size() < chunk && state.position + chunk <= state.capacity;
            if (exact || padded) {
                context.profile_phase("prefill_body");
                const auto length = padded ? chunk : tokens.size();
                auto& ids = step_input(state.prefill_step.token, {static_cast<std::int64_t>(length)}, DType::I32,
                                       device());
                const auto destination = require(ids.data<std::int32_t>());
                std::ranges::copy(tokens, destination.begin());
                std::fill(destination.begin() + static_cast<std::ptrdiff_t>(tokens.size()), destination.end(), 0);
                captured_step(ids, state, state.prefill_step, true, false);
                context.synchronize();
                state.position += tokens.size();
                return {};
            }
        }
        context.profile_phase(state.prefilling ? "prefill_embedding" : "decode_embedding");
        const auto length = static_cast<std::int64_t>(tokens.size());
        auto [hidden, per_layer] = embed(tokens, state.images, state.position);
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
            require(prefill_impl(tokens.first(token_count - 1), state));
            return project(tokens.last(1), state, false, select);
        }
        auto& context = impl_->context;
        if (!state.profile_started) {
            context.profile_phase("gemma4_request");
            state.profile_started = true;
        }
        if (token_count == 1 && batch_states.empty() && can_decode_step(state)) {
            context.profile_phase(state.prefilling ? "prefill_body" : "decode_body");
            auto& token = step_input(state.step.token, {1}, DType::I32, device());
            require(token.data<std::int32_t>())[0] = tokens[0];
            auto output = decode_step(token, state, select);
            context.synchronize();
            ++state.position;
            state.prefilling = false;
            return output;
        }
        context.profile_phase(state.prefilling ? "prefill_embedding" : "decode_embedding");
        const auto length = static_cast<std::int64_t>(token_count);
        auto [hidden, per_layer] = embed(tokens,
                                         batch_states.empty() ? std::span<const Gemma4ImageTokens>(state.images)
                                                              : std::span<const Gemma4ImageTokens>{},
                                         state.position);
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
auto Gemma4Impl::release_workspaces() -> void { impl_->context.release_workspaces(); }
auto Gemma4Impl::accelerator() const -> std::string_view { return impl_->context.accelerator(); }
auto Gemma4Impl::prefill_chunk_size(std::size_t requested) const -> std::size_t {
    return impl_->context.prefill_chunk_size(requested);
}
auto Gemma4Impl::last_token_prefill() const -> bool {
    return impl_->qat && (device() == tensor::Device::cpu() || device() == tensor::Device::vulkan());
}
auto Gemma4Impl::shared_prefill_tail() const -> bool {
    return impl_->qat && device() == tensor::Device::apple_gpu() && !impl_->packed_prefill;
}
} // namespace kidi::model