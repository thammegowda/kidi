#include "kidi/model/gemma4_vision.h"
#include "kidi/layers/gemma4.h"

#include <cmath>
#include <utility>

namespace kidi::model {
using ops::require;
using tensor::DType;
using tensor::Tensor;

namespace {
KIDI_MODULE(VisionAttention);
class VisionAttentionImpl : public Module {
public:
    VisionAttentionImpl(const YAML::Node& config, bool quantized)
        : width_(config["head_dim"].as<int>()),
          heads_(config["num_attention_heads"].as<int>()),
          query_(config["hidden_size"].as<int>(), heads_ * width_, true, false, quantized ? 8 : 0),
          key_(config["hidden_size"].as<int>(), heads_ * width_, true, false, quantized ? 8 : 0),
          value_(config["hidden_size"].as<int>(), heads_ * width_, true, false, quantized ? 8 : 0),
          output_(heads_ * width_, config["hidden_size"].as<int>(), true, false, quantized ? 8 : 0),
          query_norm_(width_, config["rms_norm_eps"].as<float>()),
          key_norm_(width_, config["rms_norm_eps"].as<float>()),
          value_norm_(width_, config["rms_norm_eps"].as<float>(), false) {
        register_module("q_proj", query_);
        register_module("k_proj", key_);
        register_module("v_proj", value_);
        register_module("o_proj", output_);
        register_module("q_norm", query_norm_);
        register_module("k_norm", key_norm_);
    }

    auto forward(ops::Context& context, const Tensor& input, const Tensor& cosine, const Tensor& sine,
                 const Tensor& mask) -> Tensor {
        const auto length = static_cast<std::int64_t>(input.size(1));
        const auto rotary = [&](const Tensor& projected, layers::RmsNorm& norm) {
            auto normalized = norm->forward(context, context.reshape(projected, {1, length, heads_, width_}));
            std::array<Tensor, 2> axes;
            for (int axis = 0; axis < 2; ++axis)
                axes[axis] = context.rotary(context.slice(normalized, 3, axis * width_ / 2, width_ / 2),
                                            context.slice(cosine, 0, axis, 1), context.slice(sine, 0, axis, 1));
            return context.reshape(context.concat(axes, 3), {1, length, heads_ * width_});
        };
        const auto query = rotary(query_->forward(context, input), query_norm_);
        const auto key = rotary(key_->forward(context, input), key_norm_);
        const auto value =
            context.reshape(value_norm_->forward(
                                context, context.reshape(value_->forward(context, input), {1, length, heads_, width_})),
                            {1, length, heads_ * width_});
        return output_->forward(context, context.grouped_query_attention(query, key, value, heads_, heads_, mask, 1.F));
    }

private:
    std::int32_t width_, heads_;
    layers::Linear query_, key_, value_, output_;
    layers::RmsNorm query_norm_, key_norm_, value_norm_;
};

KIDI_MODULE(VisionMlp);
class VisionMlpImpl : public Module {
public:
    VisionMlpImpl(const YAML::Node& config, bool quantized)
        : gate_(config["hidden_size"].as<int>(), config["intermediate_size"].as<int>(), true, false, quantized ? 8 : 0),
          up_(config["hidden_size"].as<int>(), config["intermediate_size"].as<int>(), true, false, quantized ? 8 : 0),
          down_(config["intermediate_size"].as<int>(), config["hidden_size"].as<int>(), true, false,
                quantized ? 8 : 0) {
        register_module("gate_proj", gate_);
        register_module("up_proj", up_);
        register_module("down_proj", down_);
    }
    auto forward(ops::Context& context, const Tensor& input) -> Tensor {
        return down_->forward(context,
                              context.gelu_multiply(gate_->forward(context, input), up_->forward(context, input)));
    }

private:
    layers::Linear gate_, up_, down_;
};

KIDI_MODULE(VisionBlock);
class VisionBlockImpl : public Module {
public:
    VisionBlockImpl(const YAML::Node& config, bool quantized)
        : attention_(config, quantized),
          mlp_(config, quantized),
          input_norm_(config["hidden_size"].as<int>(), config["rms_norm_eps"].as<float>()),
          attention_norm_(config["hidden_size"].as<int>(), config["rms_norm_eps"].as<float>()),
          feed_forward_norm_(config["hidden_size"].as<int>(), config["rms_norm_eps"].as<float>()),
          output_norm_(config["hidden_size"].as<int>(), config["rms_norm_eps"].as<float>()) {
        register_module("self_attn", attention_);
        register_module("mlp", mlp_);
        register_module("input_layernorm", input_norm_);
        register_module("post_attention_layernorm", attention_norm_);
        register_module("pre_feedforward_layernorm", feed_forward_norm_);
        register_module("post_feedforward_layernorm", output_norm_);
    }
    auto forward(ops::Context& context, const Tensor& input, const Tensor& cosine, const Tensor& sine,
                 const Tensor& mask) -> Tensor {
        const auto hidden = context.add(
            input, attention_norm_->forward(context, attention_->forward(context, input_norm_->forward(context, input),
                                                                         cosine, sine, mask)));
        return context.add(hidden, output_norm_->forward(
                                       context, mlp_->forward(context, feed_forward_norm_->forward(context, hidden))));
    }

private:
    VisionAttention attention_;
    VisionMlp mlp_;
    layers::RmsNorm input_norm_, attention_norm_, feed_forward_norm_, output_norm_;
};

KIDI_MODULE(VisionEncoder);
class VisionEncoderImpl : public Module {
public:
    ModuleList<VisionBlockImpl> layers;
    VisionEncoderImpl(const YAML::Node& config, bool quantized) {
        register_module("layers", layers);
        for (int index = 0; index < config["num_hidden_layers"].as<int>(); ++index)
            layers->push_back(VisionBlock(config, quantized));
    }
};

KIDI_MODULE(VisionPatch);
class VisionPatchImpl : public Module {
public:
    layers::Linear projection;
    Tensor positions;
    explicit VisionPatchImpl(const YAML::Node& config) : projection(768, config["hidden_size"].as<int>(), true, false) {
        register_module("input_proj", projection);
        register_parameter("position_embedding_table", positions,
                           {2, config["position_embedding_size"].as<int>(), config["hidden_size"].as<int>()});
    }
};

KIDI_MODULE(VisionTower);
class VisionTowerImpl : public Module {
public:
    VisionPatch patch;
    VisionEncoder encoder;
    VisionTowerImpl(const YAML::Node& config, bool quantized) : patch(config), encoder(config, quantized) {
        register_module("patch_embedder", patch);
        register_module("encoder", encoder);
    }
};

KIDI_MODULE(VisionProjection);
class VisionProjectionImpl : public Module {
public:
    layers::Linear projection;
    layers::RmsNorm norm;
    VisionProjectionImpl(int hidden, int text_width, float epsilon)
        : projection(hidden, text_width, true, false), norm(hidden, epsilon, false) {
        register_module("embedding_projection", projection);
    }
};
} // namespace

struct Gemma4VisionImpl::State {
    ops::Context context{tensor::Device::cpu(), true};
    int hidden, heads, head_width, positions;
    float theta;
    VisionTower tower;
    VisionProjection projection;
    State(const YAML::Node& config, int text_width, bool quantized)
        : hidden(config["hidden_size"].as<int>()),
          heads(config["num_attention_heads"].as<int>()),
          head_width(config["head_dim"].as<int>()),
          positions(config["position_embedding_size"].as<int>()),
          theta(config["rope_parameters"]["rope_theta"].as<float>()),
          tower(config, quantized),
          projection(hidden, text_width, config["rms_norm_eps"].as<float>()) {}
};

Gemma4VisionImpl::Gemma4VisionImpl(const YAML::Node& config, std::int32_t text_width, bool quantized) {
    if (module_device != tensor::Device::cpu() || config["patch_size"].as<int>() != 16 ||
        config["pooling_kernel_size"].as<int>() != 3 || config["standardize"].as<bool>(false) ||
        config["use_clipped_linears"].as<bool>(false) || config["head_dim"].as<int>() % 4 ||
        config["num_key_value_heads"].as<int>() != config["num_attention_heads"].as<int>())
        throw ops::Failure({ErrorCode::UNSUPPORTED, "unsupported Gemma vision architecture"});
    impl_ = std::make_unique<State>(config, text_width, quantized);
    register_module("vision_tower", impl_->tower);
    register_module("embed_vision", impl_->projection);
}
Gemma4VisionImpl::~Gemma4VisionImpl() = default;

auto Gemma4VisionImpl::set_checkpoint(const Weights& weights) -> Result<void> {
    try {
        StateDict state;
        for (const auto& [name, declaration] : state_dict()) {
            auto key = "model." + name;
            if (!weights.contains(key)) {
                for (const auto suffix :
                     {".weight", ".weight_scale", ".input_activation_scale", ".output_activation_scale"}) {
                    if (key.ends_with(suffix)) {
                        key.insert(key.size() - std::string_view(suffix).size(), ".linear");
                        break;
                    }
                }
            }
            auto value = require(weights.tensor(key));
            if (value.dtype() == DType::I8) {
                const auto bytes = require(std::as_const(value).host_bytes());
                value = require(Tensor::from_host(
                    std::vector<std::int64_t>(value.shape().begin(), value.shape().end()),
                    std::span(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()), device()));
            } else if (value.dtype() != DType::F32) {
                value = impl_->context.cast(require(value.to(device())), DType::F32);
            }
            state.emplace(name, std::move(value));
        }
        return set_state(state);
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}

auto Gemma4VisionImpl::forward(const image::Gemma4Image& image) -> Result<Tensor> {
    try {
        auto& state = *impl_;
        auto& context = state.context;
        const std::int64_t length = static_cast<std::int64_t>(image.patch_rows) * image.patch_columns;
        if (image.patch_rows <= 0 || image.patch_columns <= 0 || image.patch_rows % 3 || image.patch_columns % 3 ||
            image.patch_rows > state.positions || image.patch_columns > state.positions || length > 10080 ||
            image.patches.size() != length * 768)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Gemma image patches"});
        std::vector<float> pixels(image.patches.size()), position(length * state.hidden);
        std::ranges::transform(image.patches, pixels.begin(), [](float value) { return 2.F * (value - 0.5F); });
        const auto table = require(std::as_const(state.tower->patch->positions).data<float>());
        for (std::int64_t patch = 0; patch < length; ++patch)
            for (int channel = 0; channel < state.hidden; ++channel)
                position[patch * state.hidden + channel] =
                    table[(patch % image.patch_columns) * state.hidden + channel] +
                    table[(state.positions + patch / image.patch_columns) * state.hidden + channel];
        auto hidden =
            context.add(state.tower->patch->projection->forward(
                            context, require(Tensor::from_host({1, length, 768}, std::span<const float>(pixels)))),
                        require(Tensor::from_host({1, length, state.hidden}, std::span<const float>(position))));
        std::vector<float> cosine(2 * length * (state.head_width / 4)), sine(cosine.size());
        for (std::int64_t patch = 0; patch < length; ++patch)
            for (int axis = 0; axis < 2; ++axis)
                for (int channel = 0; channel < state.head_width / 4; ++channel) {
                    const auto coordinate = axis ? patch / image.patch_columns : patch % image.patch_columns;
                    const auto frequency = std::pow(state.theta, -static_cast<float>(channel) * 4 / state.head_width);
                    const auto offset = (axis * length + patch) * (state.head_width / 4) + channel;
                    cosine[offset] = std::cos(coordinate * frequency);
                    sine[offset] = std::sin(coordinate * frequency);
                }
        const auto cos =
            require(Tensor::from_host({2, length, 1, state.head_width / 4}, std::span<const float>(cosine)));
        const auto sin = require(Tensor::from_host({2, length, 1, state.head_width / 4}, std::span<const float>(sine)));
        const auto mask = require(Tensor::zeros({1, 1, length, length}, DType::F32));
        for (const auto& layer : *state.tower->encoder->layers)
            hidden = layer->forward(context, hidden, cos, sin, mask);
        const auto values = require(hidden.data<float>());
        std::vector<float> pooled(length / 9 * state.hidden, 0.F);
        for (std::int64_t patch = 0; patch < length; ++patch) {
            const auto group =
                (patch / image.patch_columns / 3) * (image.patch_columns / 3) + (patch % image.patch_columns / 3);
            for (int channel = 0; channel < state.hidden; ++channel)
                pooled[group * state.hidden + channel] += values[patch * state.hidden + channel] / 9.F;
        }
        for (auto& value : pooled) value *= std::sqrt(static_cast<float>(state.hidden));
        auto output = state.projection->projection->forward(
            context,
            state.projection->norm->forward(
                context, require(Tensor::from_host({1, length / 9, state.hidden}, std::span<const float>(pooled)))));
        context.synchronize();
        return output;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}
} // namespace kidi::model