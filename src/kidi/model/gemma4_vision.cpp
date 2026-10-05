#include "kidi/model/gemma4_vision.h"
#include "kidi/layers/gemma4.h"
#include "kidi/runtime/parity_math.h"

#include <bit>
#include <cmath>
#include <limits>
#include <cstdint>
#include <utility>

namespace kidi::model {
using ops::require;
using tensor::DType;
using tensor::Tensor;

namespace {
using StageCallback = std::function<void(std::string_view, const Tensor&)>;

// CPU slices bound workspace memory. On a GPU each slice costs fixed dispatch overhead, so slices are large: whole
// projections, and attention query blocks whose score buffers stay small.
#if defined(__ANDROID__)
constexpr std::int64_t VISION_ROWS = 256;
#else
constexpr std::int64_t VISION_ROWS = 32;
#endif
auto slice_rows(const ops::Context& context, bool attention) -> std::int64_t {
    if (context.device() == tensor::Device::cpu()) {
        const auto precision = context.precision();
        if (precision == core::InferencePrecision::LOWBIT_PARITY ||
            (attention && precision == core::InferencePrecision::QAT_FP32))
            return std::numeric_limits<std::int64_t>::max();
    }
    if (context.device() == tensor::Device::cpu()) return VISION_ROWS;
    return attention ? 512 : std::numeric_limits<std::int64_t>::max();
}

auto project_rows(ops::Context& context, const layers::Linear& layer, const Tensor& input) -> Tensor {
    const auto length = static_cast<std::int64_t>(input.size(1));
    const auto rows = slice_rows(context, false);
    if (length <= rows) return layer->forward(context, input);
    Tensor output;
    for (std::int64_t start = 0; start < length; start += rows) {
        const auto count = std::min(rows, length - start);
        const auto part = layer->forward(context, context.slice(input, 1, start, count));
        if (!output.defined())
            output = require(Tensor::empty({1, length, static_cast<std::int64_t>(part.size(2))}, part.dtype(),
                                            context.device()));
        context.copy_slice_(output, part, 1, start);
    }
    return output;
}

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
                 const Tensor& mask, const StageCallback& observer) -> Tensor {
        const auto length = static_cast<std::int64_t>(input.size(1));
        const auto rotary = [&](const Tensor& projected, layers::RmsNorm& norm, std::string_view norm_stage,
                                std::string_view rope_stage) {
            auto normalized = norm->forward(context, context.reshape(projected, {1, length, heads_, width_}));
            if (observer) observer(norm_stage, normalized);
            std::array<Tensor, 2> axes;
            for (int axis = 0; axis < 2; ++axis)
                axes[axis] = context.rotary(context.slice(normalized, 3, axis * width_ / 2, width_ / 2),
                                            context.slice(cosine, 0, axis, 1), context.slice(sine, 0, axis, 1));
            auto rotated = context.concat(axes, 3);
            if (observer) observer(rope_stage, rotated);
            return context.reshape(rotated, {1, length, heads_ * width_});
        };
        const auto query_projected = project_rows(context, query_, input);
        if (observer) observer("q_proj", query_projected);
        const auto query = rotary(query_projected, query_norm_, "q_norm", "q_rope");
        const auto key_projected = project_rows(context, key_, input);
        if (observer) observer("k_proj", key_projected);
        const auto key = rotary(key_projected, key_norm_, "k_norm", "k_rope");
        const auto value_projected = project_rows(context, value_, input);
        if (observer) observer("v_proj", value_projected);
        const auto value_normalized =
            value_norm_->forward(context, context.reshape(value_projected, {1, length, heads_, width_}));
        if (observer) observer("v_norm", value_normalized);
        const auto value = context.reshape(value_normalized, {1, length, heads_ * width_});
        auto attended = require(Tensor::empty({1, length, heads_ * width_}, DType::F32, context.device()));
        const auto rows = slice_rows(context, true);
        for (std::int64_t start = 0; start < length; start += rows) {
            const auto count = std::min(rows, length - start);
            const auto part = context.grouped_query_attention(context.slice(query, 1, start, count), key, value,
                                                              heads_, heads_, mask, 1.F);
            context.copy_slice_(attended, part, 1, start);
        }
        if (observer) observer("attended", attended);
        auto output = project_rows(context, output_, attended);
        if (observer) observer("o_proj", output);
        return output;
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
    auto forward(ops::Context& context, const Tensor& input, const StageCallback& observer) -> Tensor {
        const auto length = static_cast<std::int64_t>(input.size(1));
        auto output = require(Tensor::empty({1, length, static_cast<std::int64_t>(input.size(2))}, input.dtype(),
                                            context.device()));
        Tensor gate_output, up_output, hidden_output;
        const auto chunk = slice_rows(context, false);
        for (std::int64_t start = 0; start < length; start += chunk) {
            const auto count = std::min(chunk, length - start);
            const auto rows = context.slice(input, 1, start, count);
            const auto gate = gate_->forward(context, rows);
            const auto up = up_->forward(context, rows);
            const auto hidden = context.gelu_multiply(gate, up);
            const auto part = down_->forward(context, hidden);
            if (observer) {
                if (!gate_output.defined()) {
                    const auto intermediate = static_cast<std::int64_t>(gate.size(2));
                    gate_output = require(Tensor::empty({1, length, intermediate}, DType::F32, context.device()));
                    up_output = require(Tensor::empty({1, length, intermediate}, DType::F32, context.device()));
                    hidden_output = require(Tensor::empty({1, length, intermediate}, DType::F32, context.device()));
                }
                context.copy_slice_(gate_output, gate, 1, start);
                context.copy_slice_(up_output, up, 1, start);
                context.copy_slice_(hidden_output, hidden, 1, start);
            }
            context.copy_slice_(output, part, 1, start);
        }
        if (observer) {
            observer("gate_proj", gate_output);
            observer("up_proj", up_output);
            observer("mlp_hidden", hidden_output);
            observer("down_proj", output);
        }
        return output;
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
                 const Tensor& mask, const StageCallback& observer) -> Tensor {
        if (observer) observer("input", input);
        const auto attention_input = input_norm_->forward(context, input);
        if (observer) observer("input_norm", attention_input);
        const auto attention_output = attention_->forward(context, attention_input, cosine, sine, mask, observer);
        Tensor hidden;
        if (observer) {
            const auto normalized = attention_norm_->forward(context, attention_output);
            observer("post_attention_norm", normalized);
            hidden = context.add(input, normalized);
        } else {
            hidden = attention_norm_->forward_residual(context, attention_output, input);
        }
        if (observer) observer("attention_residual", hidden);
        const auto feed_forward_input = feed_forward_norm_->forward(context, hidden);
        if (observer) observer("pre_ffn_norm", feed_forward_input);
        const auto mlp_output = mlp_->forward(context, feed_forward_input, observer);
        if (observer) observer("mlp", mlp_output);
        if (!observer) return output_norm_->forward_residual(context, mlp_output, hidden);
        const auto normalized = output_norm_->forward(context, mlp_output);
        observer("post_ffn_norm", normalized);
        return context.add(hidden, normalized);
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
        // Image positions are assembled on the host, so the table never needs a device read-back.
        register_parameter("position_embedding_table", positions,
                           {2, config["position_embedding_size"].as<int>(), config["hidden_size"].as<int>()},
                           DType::F32, allocate_parameters, tensor::Device::cpu());
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
    ops::Context context{module_device, true};
    int hidden, heads, head_width, positions;
    float theta;
    core::InferencePrecision precision = core::InferencePrecision::CHECKPOINT;
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
    if (config["patch_size"].as<int>() != 16 ||
        config["pooling_kernel_size"].as<int>() != 3 || config["standardize"].as<bool>(false) ||
        config["use_clipped_linears"].as<bool>(false) || config["head_dim"].as<int>() % 4 ||
        config["num_key_value_heads"].as<int>() != config["num_attention_heads"].as<int>())
        throw ops::Failure({ErrorCode::UNSUPPORTED, "unsupported Gemma vision architecture"});
    impl_ = std::make_unique<State>(config, text_width, quantized);
    register_module("vision_tower", impl_->tower);
    register_module("embed_vision", impl_->projection);
}
Gemma4VisionImpl::~Gemma4VisionImpl() = default;
auto Gemma4VisionImpl::set_precision(core::InferencePrecision precision) noexcept -> void {
    impl_->precision = precision;
    impl_->context.set_precision(precision);
}
auto Gemma4VisionImpl::release_workspaces() -> void { impl_->context.release_workspaces(); }

auto Gemma4VisionImpl::set_checkpoint(const checkpoint::Weights& weights) -> Result<void> {
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
            } else if (value.dtype() == DType::BF16 && name.ends_with("position_embedding_table")) {
                const auto bytes = require(std::as_const(value).host_bytes());
                const auto source = reinterpret_cast<const std::uint16_t*>(bytes.data());
                std::vector<float> values(value.numel());
                for (std::size_t index = 0; index < values.size(); ++index)
                    values[index] = std::bit_cast<float>(static_cast<std::uint32_t>(source[index]) << 16);
                value = require(Tensor::from_host(std::vector<std::int64_t>(value.shape().begin(), value.shape().end()),
                                                  std::span<const float>(values)));
            } else if (value.dtype() != DType::F32 && value.dtype() != DType::U8) {
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
    return forward(image, {});
}

auto Gemma4VisionImpl::forward(const image::Gemma4Image& image, const BlockObserver& observer) -> Result<Tensor> {
    return forward(image, observer, {}, 0);
}

auto Gemma4VisionImpl::forward(const image::Gemma4Image& image, const BlockObserver& block_observer,
                               const StageObserver& stage_observer, std::size_t stage_layers) -> Result<Tensor> {
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
        const auto table_storage = require(state.tower->patch->positions.to(tensor::Device::cpu()));
        const auto table = require(table_storage.data<float>());
        for (std::int64_t patch = 0; patch < length; ++patch)
            for (int channel = 0; channel < state.hidden; ++channel)
                position[patch * state.hidden + channel] =
                    table[(patch % image.patch_columns) * state.hidden + channel] +
                    table[(state.positions + patch / image.patch_columns) * state.hidden + channel];
        auto hidden =
            context.add(project_rows(context, state.tower->patch->projection,
                            require(Tensor::from_host({1, length, 768}, std::span<const float>(pixels), device()))),
                        require(Tensor::from_host({1, length, state.hidden}, std::span<const float>(position), device())));
        std::vector<float> cosine(2 * length * (state.head_width / 4)), sine(cosine.size());
        for (std::int64_t patch = 0; patch < length; ++patch)
            for (int axis = 0; axis < 2; ++axis)
                for (int channel = 0; channel < state.head_width / 4; ++channel) {
                    const auto coordinate = axis ? patch / image.patch_columns : patch % image.patch_columns;
                    const auto exponent = static_cast<float>(channel) * 4 / state.head_width;
                    const auto parity_math = state.precision == core::InferencePrecision::QAT_FP32 ||
                                             state.precision == core::InferencePrecision::LOWBIT_PARITY;
                    const auto frequency = parity_math
                                               ? 1.F / std::pow(state.theta, exponent)
                                               : std::pow(state.theta, -exponent);
                    const auto offset = (axis * length + patch) * (state.head_width / 4) + channel;
                    const auto angle = static_cast<float>(coordinate) * frequency;
                    if (parity_math) {
                        if (std::abs(angle) >= 125.F)
                            throw ops::Failure(
                                {ErrorCode::UNSUPPORTED, "qat-fp32 vision parity supports RoPE angles below 125"});
                        cosine[offset] = runtime::parity::cosine(angle);
                        sine[offset] = runtime::parity::sine(angle);
                    } else {
                        cosine[offset] = std::cos(angle);
                        sine[offset] = std::sin(angle);
                    }
                }
        const auto cos =
            require(Tensor::from_host({2, length, 1, state.head_width / 4}, std::span<const float>(cosine), device()));
        const auto sin = require(Tensor::from_host({2, length, 1, state.head_width / 4}, std::span<const float>(sine), device()));
        const auto mask = require(Tensor::zeros({1, 1, 1, length}, DType::F32, device()));
        if (stage_observer && stage_layers) {
            stage_observer(0, "rope_cos", cos);
            stage_observer(0, "rope_sin", sin);
        }
        std::size_t layer_index = 0;
        for (const auto& layer : *state.tower->encoder->layers) {
            StageCallback stage_callback;
            if (stage_observer && layer_index < stage_layers)
                stage_callback = [&](std::string_view stage, const Tensor& tensor) {
                    context.synchronize();
                    stage_observer(layer_index, stage, tensor);
                };
            hidden = layer->forward(context, hidden, cos, sin, mask, stage_callback);
            if (block_observer) {
                context.synchronize();
                block_observer(layer_index, hidden);
            }
            ++layer_index;
        }
        // Average each 3x3 patch neighbourhood on the device, viewing patches as [rows/3, 3, columns/3, 3, hidden].
        const auto grid =
            context.reshape(hidden, {image.patch_rows / 3, 3, image.patch_columns / 3, 3, state.hidden});
        Tensor pooled;
        for (std::int64_t row = 0; row < 3; ++row)
            for (std::int64_t column = 0; column < 3; ++column) {
                const auto part = context.slice(context.slice(grid, 1, row, 1), 3, column, 1);
                pooled = pooled.defined() ? context.add(pooled, part) : part;
            }
        const auto scale = std::sqrt(static_cast<float>(state.hidden)) / 9.F;
        pooled = context.multiply(context.reshape(pooled, {1, length / 9, state.hidden}),
                                  require(Tensor::from_host({}, std::span<const float>(&scale, 1), device())));
        auto output = state.projection->projection->forward(context, state.projection->norm->forward(context, pooled));
        context.synchronize();
        return output;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}
} // namespace kidi::model