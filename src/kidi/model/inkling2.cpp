#include "kidi/model/inkling2.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <numeric>
#include <thread>

namespace kidi::model {
using ops::require;
using tensor::DType;
using tensor::Tensor;

namespace {
constexpr std::string_view PREFIX = "model.llm.";
constexpr std::array STREAMS{"attn", "k", "v", "mlp"};
constexpr float EPSILON = 1e-6F;
constexpr std::int32_t CONV_WIDTH = 4;
constexpr std::array<float, 16> E2M1{0.F,  0.5F,  1.F,  1.5F,  2.F,  3.F,  4.F,  6.F,
                                     -0.F, -0.5F, -1.F, -1.5F, -2.F, -3.F, -4.F, -6.F};

auto e4m3(std::uint8_t bits) -> float {
    const auto exponent = (bits >> 3) & 15, mantissa = bits & 7;
    const float magnitude = exponent ? std::ldexp(1.F + static_cast<float>(mantissa) / 8.F, exponent - 7)
                                     : std::ldexp(static_cast<float>(mantissa) / 8.F, -6);
    if (exponent == 15 && mantissa == 7) return NAN;
    return bits & 0x80 ? -magnitude : magnitude;
}

auto bfloat(float value) -> std::uint16_t {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7FFF + ((bits >> 16) & 1);
    return static_cast<std::uint16_t>(bits >> 16);
}

auto host_floats(const checkpoint::Weights& weights, const std::string& name) -> std::vector<float> {
    const auto value = require(weights.float32_tensor(std::string(PREFIX) + name));
    const auto data = require(value.data<float>());
    return {data.begin(), data.end()};
}

auto bytes_of(const Tensor& tensor) -> std::span<const std::byte> { return require(tensor.host_bytes()); }

/// Dequantizes one expert projection [rows, columns] to BF16. NVFP4 packs two E2M1 values per byte (low nibble
/// first) with E4M3 scales per 16 columns and a per-expert global scale; MXFP8 stores E4M3 values with UE8M0
/// exponents per 32 columns. `interleaved` moves gate/up row pairs into [gate; up] halves.
struct ExpertFormat {
    bool nvfp4;
    std::int64_t rows, columns;
    Tensor values, scales;
    std::vector<float> global;
};

auto dequantize(const ExpertFormat& format, std::int64_t expert, bool interleaved) -> std::vector<std::uint16_t> {
    const auto rows = format.rows, columns = format.columns;
    const auto block = format.nvfp4 ? 16 : 32;
    std::vector<std::uint16_t> output(static_cast<std::size_t>(rows * columns));
    const auto* values = reinterpret_cast<const std::uint8_t*>(bytes_of(format.values).data());
    const auto* scales = reinterpret_cast<const std::uint8_t*>(bytes_of(format.scales).data());
    const auto value_row = format.nvfp4 ? columns / 2 : columns, scale_row = columns / block;
    for (std::int64_t row = 0; row < rows; ++row) {
        const auto target = interleaved ? (row % 2 ? rows / 2 + row / 2 : row / 2) : row;
        const auto* source = values + (expert * rows + row) * value_row;
        const auto* scale = scales + (expert * rows + row) * scale_row;
        auto* destination = output.data() + target * columns;
        for (std::int64_t column = 0; column < columns; ++column) {
            float value;
            if (format.nvfp4) {
                const auto byte = source[column / 2];
                value = E2M1[column % 2 ? byte >> 4 : byte & 15] * e4m3(scale[column / 16]) * format.global[expert];
            } else {
                value = e4m3(source[column]) * std::ldexp(1.F, static_cast<int>(scale[column / 32]) - 127);
            }
            destination[column] = bfloat(value);
        }
    }
    return output;
}
} // namespace

auto Inkling2State::cache_bytes() const noexcept -> std::size_t {
    std::size_t total = 0;
    for (const auto& visit : visits)
        for (const auto* tensor : {&visit.key, &visit.value, &visit.key_conv, &visit.value_conv,
                                   &visit.attention_conv, &visit.mlp_conv})
            total += tensor->nbytes();
    return total;
}

auto Inkling2Impl::validate_config(const YAML::Node& config) -> Result<void> {
    try {
        if (!config.IsMap() || config["model_type"].as<std::string>() != "inkling2_loop_text")
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "expected an inkling2_loop_text configuration"});
        const auto loop = config["loop_config"];
        if (config["sconv_kernel_size"].as<std::int32_t>() != CONV_WIDTH ||
            config["n_shared_experts"].as<std::int32_t>(0) != 0 || !config["use_value_norm_no_gamma"].as<bool>() ||
            !config["attn_res_normalize_values"].as<bool>() || config["q_bias"].as<bool>() ||
            config["o_bias"].as<bool>() || (loop && loop["is_varloop"].as<bool>(false)))
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, "unsupported Inkling-2 variant"});
        const auto block = config["n_layers_per_attn_res_block"].as<std::int32_t>();
        if (loop && (loop["core_start"].as<std::int32_t>() % block ||
                     loop["core_end"].as<std::int32_t>() <= loop["core_start"].as<std::int32_t>()))
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "Inkling-2 loop must start on a block boundary"});
        if (config["head_dim"].as<std::int32_t>() % 32 || config["head_dim"].as<std::int32_t>() > 256)
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, "Inkling-2 head width must be a multiple of 32"});
        return {};
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, error.what()});
    }
}

Inkling2Impl::Inkling2Impl(const YAML::Node& config)
    : hidden_(config["hidden_size"].as<std::int32_t>()),
      layers_(config["num_hidden_layers"].as<std::int32_t>()),
      heads_(config["num_attention_heads"].as<std::int32_t>()),
      key_heads_(config["num_key_value_heads"].as<std::int32_t>()),
      head_width_(config["head_dim"].as<std::int32_t>()),
      vocabulary_(config["unpadded_vocab_size"].as<std::int32_t>(config["vocab_size"].as<std::int32_t>())),
      padded_vocabulary_(config["vocab_size"].as<std::int32_t>()),
      experts_(config["n_routed_experts"].as<std::int32_t>()),
      experts_per_token_(config["num_experts_per_tok"].as<std::int32_t>()),
      intermediate_(config["intermediate_size"].as<std::int32_t>()),
      block_(config["n_layers_per_attn_res_block"].as<std::int32_t>()),
      window_(config["sliding_window_size"].as<std::int32_t>()),
      relative_extent_(config["rel_extent"].as<std::int32_t>()),
      core_start_(config["loop_config"] ? config["loop_config"]["core_start"].as<std::int32_t>() : layers_),
      core_end_(config["loop_config"] ? config["loop_config"]["core_end"].as<std::int32_t>() : layers_),
      passes_(config["loop_config"] ? config["loop_config"]["num_passes"].as<std::int32_t>() : 0),
      max_positions_(config["model_max_length"].as<std::size_t>(1U << 20)),
      route_scale_(config["route_scale"].as<float>()),
      gate_cap_(config["mlp_gate_softcap"].as<float>()),
      up_cap_(config["mlp_up_softcap"].as<float>()),
      logit_cap_(config["final_logit_softcapping"].as<float>()),
      qk_clip_(config["qk_l2_clip_threshold"].as<float>()),
      relative_clip_(config["rel_l2_clip_threshold"].as<float>()),
      log_alpha_(config["log_scaling_alpha"].as<float>()),
      log_floor_(config["log_scaling_n_floor"].as<float>()),
      local_layers_(config["local_layer_ids"].as<std::vector<std::int32_t>>()) {
    if (device().kind != tensor::DeviceKind::CUDA)
        throw ops::Failure({ErrorCode::UNSUPPORTED, "Inkling-2 currently requires the CUDA backend"});
    context_ = std::make_unique<ops::Context>(device(), false, false, DType::BF16);
}

auto Inkling2Impl::create(const YAML::Node& config) -> Result<Inkling2> {
    auto valid = validate_config(config);
    if (!valid) return std::unexpected(std::move(valid.error()));
    try {
        return Inkling2(config);
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

auto Inkling2Impl::local(std::int32_t layer) const -> bool { return std::ranges::contains(local_layers_, layer); }

auto Inkling2Impl::schedule() const -> std::vector<std::int32_t> {
    std::vector<std::int32_t> layers;
    for (std::int32_t layer = 0; layer < core_start_; ++layer) layers.push_back(layer);
    for (std::int32_t pass = 0; pass < passes_; ++pass)
        for (std::int32_t layer = core_start_; layer < core_end_; ++layer) layers.push_back(layer);
    for (std::int32_t layer = core_end_; layer < layers_; ++layer) layers.push_back(layer);
    return layers;
}

auto Inkling2Impl::weight(const std::string& name) const -> const Tensor& {
    const auto found = weights_.find(name);
    if (found == weights_.end()) throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "missing Inkling-2 weight: " + name});
    return found->second;
}

auto Inkling2Impl::set_checkpoint(const checkpoint::Weights& weights) -> Result<void> {
    try {
        const auto upload = [&](std::vector<std::int64_t> shape, const std::vector<float>& values) {
            return require(Tensor::from_host(std::move(shape), std::span<const float>(values), device()));
        };
        const auto D = static_cast<std::int64_t>(hidden_);
        // Small tensors that kernels read as FP32, with load-time layout changes.
        const auto convert = [&](const std::string& name, const std::string& stored) {
            auto values = host_floats(weights, name);
            const auto shape = require(weights.tensor(std::string(PREFIX) + name)).shape();
            std::vector<std::int64_t> dims(shape.begin(), shape.end());
            if (name.ends_with("forget_weight"))
                for (auto& value : values) value = -value; // blend uses dest + tanh(-f) * dest
            if (name.ends_with("sconv.weight")) {         // [C, 1, W] (last tap = current) -> lag-first [W, C]
                const auto channels = dims[0], width = dims[2];
                std::vector<float> lagged(values.size());
                for (std::int64_t c = 0; c < channels; ++c)
                    for (std::int64_t lag = 0; lag < width; ++lag)
                        lagged[lag * channels + c] = values[c * width + width - 1 - lag];
                values = std::move(lagged);
                dims = {width, channels};
            }
            if (name.starts_with("attn_res.")) dims = {static_cast<std::int64_t>(values.size()) / D, D};
            weights_[stored] = upload(std::move(dims), values);
        };
        for (const auto& full : weights.names()) {
            if (!full.starts_with(PREFIX) || full.find(".mlp.experts.") != std::string::npos) continue;
            const auto name = full.substr(PREFIX.size());
            const auto value = require(weights.tensor(full));
            const bool small = name.ends_with("norm.weight") || name.ends_with("clip.scale") ||
                               name.ends_with("rel_logits_proj.proj") || name.ends_with("mlp.gate.weight") ||
                               name.ends_with("gain_weight") || name.ends_with("forget_weight") ||
                               name.ends_with("sconv.weight") || name.ends_with("static_w") ||
                               name.starts_with("attn_res.") || name.starts_with("middle_cycle");
            if (small || value.dtype() == DType::F32)
                convert(name, name);
            else
                weights_[name] = require(value.to(device()));
        }
        for (auto layer = core_start_; layer < core_end_; ++layer) {
            const auto queries = host_floats(weights, "middle_cycle_cross_pass_queries");
            const auto kv = host_floats(weights, "middle_cycle_cross_pass_kv_queries");
            const auto j = static_cast<std::size_t>(layer - core_start_);
            std::vector<float> stacked;
            for (const auto* row : {&queries[2 * j * D], &kv[2 * j * D], &kv[(2 * j + 1) * D], &queries[(2 * j + 1) * D]})
                stacked.insert(stacked.end(), row, row + D);
            weights_["loop_queries." + std::to_string(layer)] = upload({4, D}, stacked);
        }
        weights_["ones.hidden"] = upload({D}, std::vector<float>(static_cast<std::size_t>(D), 1.F));
        weights_["ones.head"] = upload({head_width_}, std::vector<float>(static_cast<std::size_t>(head_width_), 1.F));
        weights_["logit_cap"] = upload({1}, {logit_cap_});
        weights_["inverse_logit_cap"] = upload({1}, {1.F / logit_cap_});

        experts_weights_.assign(static_cast<std::size_t>(layers_), {});
        const auto workers = std::clamp<unsigned>(std::thread::hardware_concurrency(), 1, 48);
        for (std::int32_t layer = 0; layer < layers_; ++layer) {
            const auto prefix = "layers." + std::to_string(layer) + ".mlp.experts.";
            std::array<ExpertFormat, 2> formats;
            for (std::size_t index = 0; index < 2; ++index) {
                const std::string projection = index ? "w2" : "w13";
                const auto tensor = require(weights.tensor(std::string(PREFIX) + prefix + projection + "_weight"));
                auto& format = formats[index];
                format.nvfp4 = tensor.dtype() == DType::U8;
                format.rows = static_cast<std::int64_t>(tensor.size(1));
                format.columns = static_cast<std::int64_t>(tensor.size(2)) * (format.nvfp4 ? 2 : 1);
                format.values = tensor;
                format.scales = require(weights.tensor(std::string(PREFIX) + prefix + projection + "_weight_scale"));
                if (format.nvfp4) {
                    format.global = host_floats(weights, prefix + projection + "_weight_amax");
                    const auto denominator = host_floats(weights, prefix + projection + "_weight_outer_scale_denominator");
                    for (auto& value : format.global) value /= denominator.front();
                }
            }
            std::vector<std::array<std::vector<std::uint16_t>, 2>> host(static_cast<std::size_t>(experts_));
            std::vector<std::thread> threads;
            for (unsigned worker = 0; worker < workers; ++worker)
                threads.emplace_back([&, worker] {
                    for (auto expert = static_cast<std::int64_t>(worker); expert < experts_; expert += workers)
                        for (std::size_t index = 0; index < 2; ++index)
                            host[expert][index] = dequantize(formats[index], expert, index == 0);
                });
            for (auto& thread : threads) thread.join();
            auto& stored = experts_weights_[static_cast<std::size_t>(layer)];
            for (std::int64_t expert = 0; expert < experts_; ++expert) {
                std::array<Tensor, 2> pair;
                for (std::size_t index = 0; index < 2; ++index) {
                    auto& values = host[expert][index];
                    pair[index] = require(Tensor::from_bytes({formats[index].rows, formats[index].columns}, DType::BF16,
                                                             std::as_bytes(std::span(values)), device()));
                    std::vector<std::uint16_t>().swap(values);
                }
                stored.push_back(std::move(pair));
            }
        }
        return {};
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

auto Inkling2Impl::create_state(std::size_t capacity) -> Result<std::unique_ptr<LanguageModelState>> {
    try {
        if (!capacity || capacity > max_positions_)
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Inkling-2 state capacity"});
        auto state = std::make_unique<Inkling2State>();
        state->capacity = capacity;
        const auto kv = static_cast<std::int64_t>(key_heads_) * head_width_, rows = std::int64_t{CONV_WIDTH - 1};
        const auto zeros = [&](std::int64_t first, std::int64_t second) {
            return require(Tensor::zeros({first, second}, DType::BF16, device()));
        };
        for ([[maybe_unused]] const auto layer : schedule())
            state->visits.push_back({zeros(static_cast<std::int64_t>(capacity), kv),
                                     zeros(static_cast<std::int64_t>(capacity), kv), zeros(rows, kv), zeros(rows, kv),
                                     zeros(rows, hidden_), zeros(rows, hidden_)});
        return state;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}

auto Inkling2Impl::rewind(LanguageModelState& base, std::size_t position) -> Result<void> {
    auto* state = dynamic_cast<Inkling2State*>(&base);
    if (!state || !state->capacity || state->position > state->capacity || position > state->position)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Inkling-2 state rewind"});
    if (position == state->position) return {};
    if (position != 0)
        return std::unexpected(Error{ErrorCode::UNSUPPORTED, "Inkling-2 supports rewinding only to the initial state"});
    try {
        context_->synchronize();
        for (auto& visit : state->visits)
            for (auto* cache : {&visit.key_conv, &visit.value_conv, &visit.attention_conv, &visit.mlp_conv})
                *cache = require(Tensor::zeros(std::vector(cache->shape().begin(), cache->shape().end()),
                                               cache->dtype(), device()));
        state->position = 0;
        return {};
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

auto Inkling2Impl::convolve(const Tensor& input, Tensor& state, const Tensor& weight, const Tensor& taps, float scale)
    -> Tensor {
    auto& context = *context_;
    const std::array parts{state, input};
    const auto padded = context.concat(parts, 0);
    auto output = context.short_convolution(padded, weight, taps, scale);
    state = context.slice(padded, 0, static_cast<std::int64_t>(input.size(0)), CONV_WIDTH - 1);
    return output;
}

auto Inkling2Impl::attention(std::int32_t layer, const std::array<Tensor, 3>& inputs, Inkling2Visit& cache,
                             std::size_t start) -> Tensor {
    auto& context = *context_;
    const auto p = "layers." + std::to_string(layer) + ".attn.";
    const auto linear = [&](const Tensor& input, const std::string& name) {
        return context.linear(input, weight(p + name), {}, true);
    };
    const bool is_local = local(layer);
    const auto T = static_cast<std::int64_t>(inputs[0].size(0));
    auto q = linear(inputs[0], "wq_du.weight");
    auto r = linear(inputs[0], "wr_du.weight");
    const auto gate = linear(inputs[0], "gate_attn.weight");
    const auto sink = linear(inputs[0], "attn_sink_proj.weight");
    auto k = linear(inputs[1], "wk_dv.weight");
    auto v = linear(inputs[2], "wv_dv.weight");
    const auto plain = 1.F / std::sqrt(static_cast<float>(CONV_WIDTH + 1));
    const auto dynamic = 1.F / std::sqrt(static_cast<float>(2 * CONV_WIDTH + 1));
    if (is_local) {
        k = convolve(k, cache.key_conv, weight(p + "k_sconv.weight"), {}, plain);
        v = convolve(v, cache.value_conv, weight(p + "v_sconv.weight"), {}, plain);
    } else {
        k = convolve(k, cache.key_conv, weight(p + "k_sconv.static_w"), linear(inputs[1], "k_sconv.weight_proj"),
                     dynamic);
        v = convolve(v, cache.value_conv, weight(p + "v_sconv.static_w"), linear(inputs[2], "v_sconv.weight_proj"),
                     dynamic);
    }
    q = context.l2_clip(context.reshape(q, {T, heads_, head_width_}), weight(p + "q_l2_clip.scale"), qk_clip_);
    k = context.l2_clip(context.reshape(k, {T, key_heads_, head_width_}), {}, qk_clip_);
    v = context.cast(context.rms_norm(context.reshape(v, {T, key_heads_, head_width_}), weight("ones.head"), EPSILON),
                     DType::BF16);
    const auto rel_proj = weight(p + "rel_logits_proj.proj");
    const auto width = static_cast<std::int64_t>(rel_proj.size(0));
    const auto extent = static_cast<std::int64_t>(rel_proj.size(1));
    r = context.l2_clip(context.reshape(r, {T, heads_, width}), weight(p + "rel_l2_clip.scale"), relative_clip_);
    if (!is_local && static_cast<float>(start + T) > log_floor_) {
        std::vector<float> tau(static_cast<std::size_t>(T));
        for (std::int64_t t = 0; t < T; ++t)
            tau[t] = 1.F + log_alpha_ * std::log(std::max(1.F, static_cast<float>(start + t + 1) / log_floor_));
        const auto scale = require(Tensor::from_host({T, 1, 1}, std::span<const float>(tau), device()));
        q = context.multiply(q, scale);
        r = context.multiply(r, scale);
    }
    const auto relative = context.reshape(
        context.matmul(context.cast(context.reshape(r, {T * heads_, width}), DType::F32), rel_proj), {T, heads_, extent});
    const auto kv = static_cast<std::int64_t>(key_heads_) * head_width_;
    context.copy_slice_(cache.key, context.reshape(k, {T, kv}), 0, static_cast<std::int64_t>(start));
    context.copy_slice_(cache.value, context.reshape(v, {T, kv}), 0, static_cast<std::int64_t>(start));
    const auto output = context.relative_attention(
        context.reshape(q, {T, heads_ * head_width_}), cache.key, cache.value, relative, sink, gate,
        {.heads = heads_,
         .key_heads = key_heads_,
         .start = static_cast<std::int64_t>(start),
         .window = is_local ? window_ : 0,
         .extent = extent});
    return linear(output, "wo_ud.weight");
}

auto Inkling2Impl::blend(std::int32_t layer, const std::string& stream, const Tensor& destination,
                         const Tensor& message, const Tensor& norm) -> Tensor {
    auto& context = *context_;
    const auto p = "layers." + std::to_string(layer) + ".controllers." + stream + ".";
    const auto& ones = weight("ones.hidden");
    const std::array pair{context.rms_norm(destination, ones, EPSILON), context.rms_norm(message, ones, EPSILON)};
    const auto hidden =
        context.cast(context.linear(context.cast(context.concat(pair, -1), DType::BF16), weight(p + "down_weight"), {}, true),
                     DType::F32);
    const auto gain = context.tanh(context.linear(hidden, weight(p + "gain_weight"), {}, true));
    const auto forget = context.tanh(context.linear(hidden, weight(p + "forget_weight"), {}, true)); // -tanh(f)
    const auto mixed = context.add(context.add(destination, context.multiply(forget, destination)),
                                   context.multiply(gain, message));
    return context.cast(context.rms_norm(mixed, norm, EPSILON), DType::BF16);
}

auto Inkling2Impl::moe(std::int32_t layer, const Tensor& input) -> Tensor {
    auto& context = *context_;
    const auto p = "layers." + std::to_string(layer) + ".mlp.";
    const auto T = static_cast<std::int64_t>(input.size(0));
    const auto logits = require(
        context.linear(context.cast(input, DType::F32), weight(p + "gate.weight"), {}, true).to(tensor::Device::cpu()));
    const auto bias = require(weight(p + "gate.bias").to(tensor::Device::cpu()));
    const auto scores_data = require(logits.data<float>());
    const auto bias_data = require(bias.data<float>());
    std::vector<std::vector<std::pair<std::int32_t, float>>> routes(static_cast<std::size_t>(experts_));
    std::vector<float> scores(static_cast<std::size_t>(experts_));
    std::vector<std::int32_t> order(static_cast<std::size_t>(experts_));
    for (std::int64_t t = 0; t < T; ++t) {
        for (std::int32_t e = 0; e < experts_; ++e) scores[e] = 1.F / (1.F + std::exp(-scores_data[t * experts_ + e]));
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + experts_per_token_, order.end(),
                          [&](auto a, auto b) { return scores[a] + bias_data[a] > scores[b] + bias_data[b]; });
        float total = 0;
        for (std::int32_t slot = 0; slot < experts_per_token_; ++slot) total += scores[order[slot]];
        for (std::int32_t slot = 0; slot < experts_per_token_; ++slot)
            routes[order[slot]].emplace_back(static_cast<std::int32_t>(t),
                                              scores[order[slot]] / (total + 1e-20F) * route_scale_);
    }
    auto output = require(Tensor::zeros({1, T, hidden_}, DType::F32, device()));
    const auto& experts = experts_weights_[static_cast<std::size_t>(layer)];
    for (std::int32_t e = 0; e < experts_; ++e) {
        const auto& route = routes[e];
        if (route.empty()) continue;
        std::vector<std::int32_t> tokens;
        std::vector<float> scales;
        for (const auto& [token, scale] : route) {
            tokens.push_back(token);
            scales.push_back(scale);
        }
        const auto n = static_cast<std::int64_t>(tokens.size());
        const auto index = require(Tensor::from_host({n}, std::span<const std::int32_t>(tokens), device()));
        const auto rows = T == 1 ? input : context.gather(input, index, 0);
        const auto gate_up = context.linear(rows, experts[e][0], {}, true);
        const auto activated = context.silu_gate(
            gate_up, gate_cap_, up_cap_, require(Tensor::from_host({n}, std::span<const float>(scales), device())));
        const auto projected = context.cast(context.linear(activated, experts[e][1], {}, true), DType::F32);
        if (T == 1) {
            context.add_(output, context.reshape(projected, {1, 1, hidden_}));
            continue;
        }
        const auto sum = context.add(context.gather(output, index, 1), context.reshape(projected, {1, n, hidden_}));
        context.scatter_unique_(output, sum, index);
    }
    return context.cast(context.reshape(output, {T, hidden_}), DType::BF16);
}

auto Inkling2Impl::visit(std::int32_t layer, Stream stream, Inkling2Visit& cache, std::size_t start,
                         const std::vector<Tensor>* bank) -> Stream {
    auto& context = *context_;
    const auto p = "layers." + std::to_string(layer) + ".";
    const auto T = static_cast<std::int64_t>(stream.partial.size(0));
    const bool is_local = local(layer);
    const std::array<Tensor, 3> norms =
        is_local ? std::array{weight(p + "attn_norm.weight"), weight(p + "attn_norm.weight"), weight(p + "attn_norm.weight")}
                 : std::array{weight(p + "q_input_norm.weight"), weight(p + "k_input_norm.weight"),
                              weight(p + "v_input_norm.weight")};
    const auto row = [&](const Tensor& stacked, std::int64_t index) {
        return context.reshape(context.slice(stacked, 0, index, 1), {T, hidden_});
    };
    Tensor messages;
    if (bank) messages = context.residual_mix(*bank, weight("loop_queries." + std::to_string(layer)), EPSILON);
    const auto prepare = [&](const Tensor& destination, std::size_t stream_index, const Tensor& norm) {
        if (messages.defined())
            return blend(layer, STREAMS[stream_index], destination, row(messages, static_cast<std::int64_t>(stream_index)),
                         norm);
        return context.cast(context.rms_norm(destination, norm, EPSILON), DType::BF16);
    };
    std::array<Tensor, 3> inputs;
    if (layer == 0) {
        inputs.fill(context.rms_norm(stream.partial, norms[0], EPSILON));
    } else {
        auto sources = stream.blocks;
        sources.push_back(stream.partial);
        const auto destinations =
            context.residual_mix(sources, weight("attn_res.attn_queries." + std::to_string(layer - 1)), EPSILON);
        const auto count = static_cast<std::int64_t>(destinations.size(0));
        for (std::size_t s = 0; s < 3; ++s)
            // Local layers share one input unless loop messages make the q/k/v streams differ.
            inputs[s] = is_local && s && !messages.defined() ? inputs[0]
                                      : prepare(row(destinations, count == 1 ? 0 : static_cast<std::int64_t>(s)), s, norms[s]);
    }
    const auto attended = attention(layer, inputs, cache, start);
    const auto plain = 1.F / std::sqrt(static_cast<float>(CONV_WIDTH + 1));
    const auto attention_output =
        context.cast(convolve(attended, cache.attention_conv, weight(p + "attn_sconv.weight"), {}, plain), DType::F32);
    Tensor residual;
    if (layer % block_ == 0) {
        stream.blocks.push_back(stream.partial);
        residual = attention_output;
    } else {
        residual = context.add(context.cast(stream.partial, DType::F32), attention_output);
    }
    auto sources = stream.blocks;
    sources.push_back(residual);
    const auto mixed = row(context.residual_mix(sources, weight("attn_res.mlp_queries." + std::to_string(layer)), EPSILON), 0);
    const auto mlp_input = prepare(mixed, 3, weight(p + "mlp_norm.weight"));
    const auto mlp_output = convolve(moe(layer, mlp_input), cache.mlp_conv, weight(p + "mlp_sconv.weight"), {}, plain);
    stream.partial = context.cast(context.add(residual, context.cast(mlp_output, DType::F32)), DType::BF16);
    return stream;
}

auto Inkling2Impl::prefill(std::span<const std::int32_t> tokens, LanguageModelState& state) -> Result<void> {
    auto logits = forward(tokens, state);
    if (!logits) return std::unexpected(std::move(logits.error()));
    return {};
}

auto Inkling2Impl::forward(std::span<const std::int32_t> tokens, LanguageModelState& base, bool all_logits,
                           tensor::Tensor* hidden) -> Result<Tensor> {
    try {
        auto& state = dynamic_cast<Inkling2State&>(base);
        if (tokens.empty() || state.position + tokens.size() > state.capacity)
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "Inkling-2 input exceeds state capacity"});
        auto& context = *context_;
        const auto T = static_cast<std::int64_t>(tokens.size());
        const auto start = state.position;
        const auto ids = require(Tensor::from_host({T}, tokens, device()));
        Stream stream{{}, context.rms_norm(context.gather(weight("embed.weight"), ids, 0), weight("embed_norm.weight"),
                                           EPSILON)};
        std::size_t visit_index = 0;
        for (std::int32_t layer = 0; layer < core_start_; ++layer)
            stream = visit(layer, std::move(stream), state.visits[visit_index++], start, nullptr);
        const auto prelude = stream;
        for (std::int32_t pass = 0; pass < passes_; ++pass) {
            std::vector<Tensor> bank;
            if (pass) {
                bank.assign(stream.blocks.begin() + static_cast<std::ptrdiff_t>(prelude.blocks.size() + 1),
                            stream.blocks.end());
                bank.push_back(stream.partial);
                stream = prelude;
            }
            for (auto layer = core_start_; layer < core_end_; ++layer)
                stream = visit(layer, std::move(stream), state.visits[visit_index++], start, pass ? &bank : nullptr);
        }
        for (auto layer = core_end_; layer < layers_; ++layer)
            stream = visit(layer, std::move(stream), state.visits[visit_index++], start, nullptr);

        auto sources = stream.blocks;
        sources.push_back(stream.partial);
        const auto rows = all_logits ? T : 1;
        if (!all_logits)
            for (auto& source : sources) source = context.slice(source, 0, T - 1, 1);
        const auto mixed = context.reshape(context.residual_mix(sources, weight("attn_res.output_query"), EPSILON),
                                           {rows, hidden_});
        const auto normalized = context.cast(context.rms_norm(mixed, weight("norm.weight"), EPSILON), DType::BF16);
        if (hidden) *hidden = normalized;
        auto logits = context.cast(context.linear(normalized, weight("unembed.weight"), {}, true), DType::F32);
        logits = context.multiply(context.tanh(context.multiply(logits, weight("inverse_logit_cap"))),
                                  weight("logit_cap"));
        state.position += tokens.size();
        return context.slice(logits, 1, 0, vocabulary_);
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

} // namespace kidi::model
