#include "kidi/model/inkling2.h"
#include "kidi/ops/cpu_kernels.h"
#include "kidi/ops/quantization.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <numeric>

namespace kidi::model {
using ops::require;
using tensor::DType;
using tensor::Tensor;

namespace {
constexpr std::string_view PREFIX = "model.llm.";
constexpr std::string_view PACKED_PREFIX = "kidi.cpu.packed.";
constexpr std::array STREAMS{"attn", "k", "v", "mlp"};
constexpr float EPSILON = 1e-6F;
constexpr std::int32_t CONV_WIDTH = 4;
// Longer inputs run as consecutive steps of at most this many tokens. Step activations grow with its length (about
// 2.6 MB per token on CUDA), so a file-sized prompt in one step would exhaust device memory.
constexpr std::size_t MAX_STEP_TOKENS = 512;
// Ring caches keep this many rows beyond the attention window, so a request may generate this many tokens after its
// prompt while the next turn can still rewind to that prompt and reuse it. A step must also fit in these rows.
constexpr std::size_t RING_SPARE_ROWS = 4096;
static_assert(RING_SPARE_ROWS >= MAX_STEP_TOKENS);

auto cpu_projection(std::string_view name) -> bool {
    constexpr std::array<std::string_view, 9> SUFFIXES{
        "qrg_sink.weight", "wkv_dv.weight", "wk_taps.weight", "wv_taps.weight",
        "wo_ud.weight", "rel_logits_proj.proj_t", "down_weight", "gain_forget_weight", "gate.weight"};
    return name == "unembed.weight" ||
           std::ranges::any_of(SUFFIXES, [&](std::string_view suffix) { return name.ends_with(suffix); });
}

auto narrow_packed(const ops::PackedWeight& packed, std::int64_t offset, std::int64_t rows) -> ops::PackedWeight {
    return {require(packed.values.narrow(0, offset, rows)),
            require(packed.scales.narrow(0, offset, rows)),
            packed.zeros.defined() ? require(packed.zeros.narrow(0, offset, rows)) : Tensor{},
            packed.bits,
            packed.group_size};
}

auto host_floats(const checkpoint::Weights& weights, const std::string& name) -> std::vector<float> {
    const auto value = require(weights.float32_tensor(std::string(PREFIX) + name));
    const auto data = require(value.data<float>());
    return {data.begin(), data.end()};
}

auto expert_global_scale(const checkpoint::Weights& weights, const std::string& name, const Tensor& values) -> Tensor {
    if (values.dtype() != DType::U8) return {};
    auto factors = host_floats(weights, name + "_weight_amax");
    const auto denominator = host_floats(weights, name + "_weight_outer_scale_denominator");
    if (denominator.size() != 1 || !std::isfinite(denominator.front()) || denominator.front() <= 0)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Inkling-2 expert scale denominator"});
    for (auto& value : factors) value /= denominator.front();
    if (!std::ranges::all_of(factors, [](float value) { return std::isfinite(value); }))
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "non-finite Inkling-2 global expert scale"});
    return require(Tensor::from_host({static_cast<std::int64_t>(values.size(0))}, std::span<const float>(factors)));
}

auto step_input(Tensor& tensor, std::vector<std::int64_t> shape, DType dtype) -> Tensor& {
    if (!tensor.defined() || tensor.dtype() != dtype || !std::ranges::equal(tensor.shape(), shape))
        tensor = require(Tensor::empty(std::move(shape), dtype, tensor::Device::cpu()));
    return tensor;
}

auto retains_local_history(const Inkling2State& state, std::size_t rewind, std::int32_t window) -> bool {
    return rewind == 0 || state.local_capacity >= state.capacity || state.position <= state.local_capacity ||
           state.position - rewind <= state.local_capacity - static_cast<std::size_t>(window) + 1;
}

/// Metal and CUDA keep sliding-window layers in ring caches instead of full-context caches.
auto ring_caches(tensor::Device device) -> bool {
    return device.kind == tensor::DeviceKind::A_GPU || device.kind == tensor::DeviceKind::CUDA;
}
} // namespace

auto Inkling2State::cache_bytes() const noexcept -> std::size_t {
    std::size_t total = 0;
    for (const auto& visit : visits) {
        for (const auto* tensor : {&visit.key, &visit.value, &visit.key_conv, &visit.value_conv,
                                   &visit.attention_conv, &visit.mlp_conv, &visit.key_scales, &visit.value_scales})
            total += tensor->nbytes();
        for (const auto& tensor : visit.rewind_inputs)
            if (tensor.defined()) total += tensor.nbytes();
    }
    for (const auto& tensor : rewind_convs) total += tensor.nbytes();
    for (const auto& tensor : verify_convs) total += tensor.nbytes();
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
    if (device().kind != tensor::DeviceKind::CUDA && device().kind != tensor::DeviceKind::CPU &&
        device().kind != tensor::DeviceKind::A_GPU)
        throw ops::Failure({ErrorCode::UNSUPPORTED, "Inkling-2 requires a CPU, CUDA, or Metal backend"});
    activation_ = device().kind == tensor::DeviceKind::A_GPU ? DType::F32 : DType::BF16;
    context_ = std::make_unique<ops::Context>(device(), false, std::nullopt, activation_);
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

auto Inkling2Impl::project_bf16(const Tensor& input, const std::string& name) -> Tensor {
    if (const auto found = packed_.find(name); found != packed_.end()) {
        const auto& packed = found->second;
        const auto tokens = input.dimensions() == 2 ? static_cast<std::int64_t>(input.size(0)) : 0;
        const auto k = static_cast<std::int64_t>(packed.values.size(1));
        // Small projections skip the cast/quantize/cast op chain, including flattened prefill head rows.
        const auto n = static_cast<std::int64_t>(packed.values.size(0));
        if (device().kind == tensor::DeviceKind::CPU && tokens > 0 && n * k < (1 << 20) && input.size(1) == k &&
            packed.group_size == k && input.is_contiguous() && input.is_host_accessible() &&
            (input.dtype() == DType::BF16 || input.dtype() == DType::F32)) {
            thread_local std::vector<float> floats;
            const float* values = nullptr;
            if (input.dtype() == DType::F32) {
                values = require(input.data<float>()).data();
            } else {
                floats.resize(static_cast<std::size_t>(tokens * k));
                ops::cpu::bf16_to_f32(reinterpret_cast<const std::uint16_t*>(require(input.host_bytes()).data()),
                                      tokens * k, floats.data());
                values = floats.data();
            }
            auto output = require(Tensor::empty({tokens, n}, DType::BF16, device()));
            const auto* weight =
                reinterpret_cast<const std::int8_t*>(require(packed.values.data<std::uint8_t>()).data());
            require(ops::cpu::int8_linear(values, tokens, k, weight, require(packed.scales.data<float>()).data(), n,
                                          reinterpret_cast<std::uint16_t*>(require(output.host_bytes()).data())));
            return output;
        }
        const auto x = input.dtype() == DType::F32 ? input : context_->cast(input, DType::F32);
        const auto y =
            context_->packed_linear(x, found->second.values, found->second.scales, 8, found->second.group_size);
        return activation_ == DType::F32 ? y : context_->cast(y, DType::BF16);
    }
    const auto out = context_->linear(input, weight(name), {}, true);
    return activation_ == DType::F32 ? out : context_->cast(out, DType::BF16);
}

auto Inkling2Impl::set_checkpoint(const checkpoint::Weights& weights) -> Result<void> {
    return set_checkpoint(weights, {});
}

auto Inkling2Impl::int8_preparation(const YAML::Node& config) -> Result<checkpoint::Preparation> {
    if (const auto valid = validate_config(config["model"]); !valid) return std::unexpected(valid.error());
    return checkpoint::Preparation{
        .source = config["weights_file"].as<std::string>(),
        .cache_directory = "kidi-inkling2-cpu-int8-v1",
        .format = "kidi-inkling2-cpu-int8-v1",
        .metadata = {{"model_config", YAML::Dump(config["model"])}},
        .convert = int8_checkpoint,
    };
}

auto Inkling2Impl::vq_preparation(const YAML::Node& config) -> Result<checkpoint::Preparation> {
    if (const auto valid = validate_config(config["model"]); !valid) return std::unexpected(valid.error());
    const std::string tag(core::to_string(core::InferencePrecision::VQ3D_HYBRID));
    return checkpoint::Preparation{
        .source = config["weights_file"].as<std::string>(),
        .cache_directory = "kidi-inkling2-" + tag + "-v2",
        .format = "kidi-inkling2-" + tag + "-v2",
        .metadata = {{"model_config", YAML::Dump(config["model"])}, {"precision", tag}},
        .convert = vq_checkpoint,
    };
}

auto Inkling2Impl::vq_checkpoint(const YAML::Node& config, const checkpoint::Weights& weights) -> Result<StateDict> {
    try {
        const ModuleScope construction(DType::BF16, false, tensor::Device::cpu());
        auto model = require(create(config));
        require(model->set_checkpoint(weights, {.precision = core::InferencePrecision::VQ3D_HYBRID}));
        StateDict result;
        const std::array<std::int32_t, 1> version{2};
        result.emplace("kidi.vq.version", require(Tensor::from_host({1}, std::span<const std::int32_t>(version))));
        // Store dense projections directly as compact row-scaled INT8 (halves dense RAM & skips re-packing on load).
        for (const auto& [name, value] : model->weights_) {
            if (model->packed_.contains(name) && !name.ends_with("rel_logits_proj.proj_t")) continue;
            result.emplace(std::string(PREFIX) + name, value);
        }
        for (const auto& [name, packed] : model->packed_) {
            if (name.ends_with("wk_dv.weight") || name.ends_with("wv_dv.weight")) continue;
            const auto prefix = std::string(PACKED_PREFIX) + name;
            result.emplace(prefix + ".values", packed.values);
            result.emplace(prefix + ".scales", packed.scales);
        }
        for (std::size_t layer = 0; layer < model->stacked_experts_.size(); ++layer)
            for (std::size_t proj = 0; proj < 2; ++proj) {
                const auto prefix = "kidi.vq.expert." + std::to_string(layer) + "." + std::to_string(proj);
                const auto& w = model->stacked_experts_[layer][proj];
                const std::array<std::int32_t, 1> fmt_code{static_cast<std::int32_t>(w.format)};
                result.emplace(prefix + ".format",
                               require(Tensor::from_host({1}, std::span<const std::int32_t>(fmt_code))));
                result.emplace(prefix + ".values", w.values);
                result.emplace(prefix + ".scales", w.scales);
                result.emplace(prefix + ".global", w.global);
            }
        return result;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}

auto Inkling2Impl::load_prepared_vq_checkpoint(const checkpoint::Weights& weights) -> Result<void> {
    try {
        const auto version = require(weights.tensor("kidi.vq.version"));
        if (version.dtype() != DType::I32 || version.numel() != 1 || require(version.data<std::int32_t>()).front() != 2)
            throw ops::Failure({ErrorCode::INVALID_MANIFEST, "invalid prepared Inkling-2 VQ checkpoint version"});
        const bool metal = device().kind == tensor::DeviceKind::A_GPU;
        decltype(weights_) staged_weights;
        for (const auto& name : weights.names()) {
            if (!name.starts_with(PREFIX)) continue;
            const auto local_name = name.substr(PREFIX.size());
            auto t = require(weights.tensor(name));
            // Stage only the requested rows instead of keeping the full vocabulary resident on Metal.
            if (metal && local_name == "embed.weight") {
                if (t.dimensions() != 2 || t.size(0) < static_cast<std::size_t>(vocabulary_) ||
                    t.size(1) != static_cast<std::size_t>(hidden_) || !t.is_host_accessible())
                    throw ops::Failure({ErrorCode::INVALID_MANIFEST, "invalid Inkling-2 host embedding matrix"});
                staged_weights.emplace(local_name, std::move(t));
                continue;
            }
            if (metal && t.dtype() == DType::BF16 && !cpu_projection(local_name)) {
                const auto raw = require(std::as_const(t).host_bytes());
                const auto* src = reinterpret_cast<const std::uint16_t*>(raw.data());
                std::vector<float> f32(t.numel());
                for (std::size_t i = 0; i < f32.size(); ++i)
                    f32[i] = std::bit_cast<float>(static_cast<std::uint32_t>(src[i]) << 16);
                std::vector<std::int64_t> sh(t.shape().begin(), t.shape().end());
                staged_weights.emplace(
                    local_name, require(Tensor::from_host(std::move(sh), std::span<const float>(f32), device())));
            } else {
                staged_weights.emplace(local_name, require(t.to(device())));
            }
            t.release_pages(false);
        }
        decltype(packed_) staged_packed;
        const auto load_packed = [&](std::string_view name) {
            if (metal) return require(weights.tensor(name, device()));
            const auto source = require(weights.tensor(name));
            auto result = require(source.to(device()));
            source.release_pages(false);
            return result;
        };
        for (const auto& name : weights.names()) {
            if (!name.starts_with(PACKED_PREFIX) || !name.ends_with(".values")) continue;
            const auto local_name = name.substr(
                PACKED_PREFIX.size(), name.size() - PACKED_PREFIX.size() - std::string_view(".values").size());
            auto vals = load_packed(name);
            auto scs = load_packed(std::string(PACKED_PREFIX) + local_name + ".scales");
            const auto width = static_cast<std::int32_t>(vals.size(1));
            staged_packed[local_name] = {std::move(vals), std::move(scs), {}, 8, width};
        }
        for (std::int32_t layer = 0; layer < layers_; ++layer) {
            if (!local(layer)) continue;
            const auto p = "layers." + std::to_string(layer) + ".attn.";
            const auto found = staged_packed.find(p + "wkv_dv.weight");
            if (found != staged_packed.end()) {
                const auto half_rows = static_cast<std::int64_t>(found->second.values.size(0) / 2);
                staged_packed.insert_or_assign(p + "wk_dv.weight", narrow_packed(found->second, 0, half_rows));
                staged_packed.insert_or_assign(p + "wv_dv.weight", narrow_packed(found->second, half_rows, half_rows));
            }
        }
        // Copy expert banks to preserve decode bandwidth when routes change between tokens.
        const auto load_expert = [&](const std::string& name) {
            const auto source = require(weights.tensor(name));
            auto result = require(source.to(device()));
            if (metal) source.release_pages(false);
            return result;
        };
        decltype(stacked_experts_) staged_stacks(static_cast<std::size_t>(layers_));
        for (std::int32_t layer = 0; layer < layers_; ++layer)
            for (std::size_t proj = 0; proj < 2; ++proj) {
                const auto prefix = "kidi.vq.expert." + std::to_string(layer) + "." + std::to_string(proj);
                const auto type = require(weights.tensor(prefix + ".format"));
                const auto fmt = static_cast<ops::BlockScaledFormat>(require(type.data<std::int32_t>()).front());
                if (fmt != ops::BlockScaledFormat::VQ3D && fmt != ops::BlockScaledFormat::VQ4D)
                    throw ops::Failure({ErrorCode::INVALID_MANIFEST, "invalid prepared Inkling-2 expert format"});
                const auto group = fmt == ops::BlockScaledFormat::VQ3D ? 48 : 64;
                staged_stacks[static_cast<std::size_t>(layer)][proj] = {load_expert(prefix + ".values"),
                                                                        load_expert(prefix + ".scales"),
                                                                        load_expert(prefix + ".global"), fmt, group};
            }
        activation_ = metal ? DType::F32 : DType::BF16;
        auto context = std::make_unique<ops::Context>(device(), false, std::nullopt, activation_);
        context_->synchronize();
        context_ = std::move(context);
        weights_ = std::move(staged_weights);
        packed_ = std::move(staged_packed);
        experts_weights_.clear();
        stacked_experts_ = std::move(staged_stacks);
        return {};
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}

auto Inkling2Impl::int8_checkpoint(const YAML::Node& config, const checkpoint::Weights& weights) -> Result<StateDict> {
    try {
        const ModuleScope construction(DType::BF16, false, tensor::Device::cpu());
        auto model = require(create(config));
        require(model->set_checkpoint(weights));
        StateDict result;
        result.emplace("kidi.cpu.version", require(Tensor::from_host({1}, std::span<const std::int32_t>(
                                                                            std::array<std::int32_t, 1>{1}))));
        for (const auto& [name, value] : model->weights_)
            result.emplace(std::string(PREFIX) + name, value);
        for (const auto& [name, packed] : model->packed_) {
            const auto prefix = std::string(PACKED_PREFIX) + name;
            result.emplace(prefix + ".values", packed.values);
            result.emplace(prefix + ".scales", packed.scales);
        }
        for (std::size_t layer = 0; layer < model->experts_weights_.size(); ++layer)
            for (std::size_t expert = 0; expert < model->experts_weights_[layer].size(); ++expert)
                for (std::size_t projection = 0; projection < 2; ++projection) {
                    const auto prefix = "kidi.cpu.expert." + std::to_string(layer) + "." + std::to_string(expert) +
                                        "." + std::to_string(projection);
                    const auto& weight = model->experts_weights_[layer][expert][projection];
                    result.emplace(prefix + ".values", weight.values);
                    result.emplace(prefix + ".scales", weight.scales);
                }
        return result;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}

auto Inkling2Impl::load_prepared_checkpoint(const checkpoint::Weights& weights) -> Result<void> {
    try {
        const auto version = require(weights.tensor("kidi.cpu.version"));
        if (version.dtype() != DType::I32 || version.numel() != 1 ||
            require(version.data<std::int32_t>()).front() != 1)
            throw ops::Failure({ErrorCode::INVALID_MANIFEST, "invalid prepared Inkling-2 CPU checkpoint version"});
        decltype(weights_) staged_weights;
        decltype(packed_) staged_packed;
        for (const auto& name : weights.names())
            if (name.starts_with(PREFIX)) staged_weights.emplace(name.substr(PREFIX.size()), require(weights.tensor(name)));
        const auto load_packed = [&](const std::string& prefix, std::int64_t rows, std::int64_t width) {
            auto values = require(weights.tensor(prefix + ".values"));
            auto scales = require(weights.tensor(prefix + ".scales"));
            if (values.dtype() != DType::U8 || scales.dtype() != DType::F32 ||
                values.dimensions() != 2 || scales.dimensions() != 2 ||
                values.size(0) != static_cast<std::size_t>(rows) ||
                values.size(1) != static_cast<std::size_t>(width) ||
                scales.size(0) != static_cast<std::size_t>(rows) || scales.size(1) != 1)
                throw ops::Failure({ErrorCode::INVALID_MANIFEST, "invalid prepared Inkling-2 INT8 weight: " + prefix});
            return ops::PackedWeight{std::move(values), std::move(scales), {}, 8, static_cast<std::int32_t>(width)};
        };
        for (const auto& [name, value] : staged_weights) {
            const auto prefix = std::string(PACKED_PREFIX) + name;
            if (cpu_projection(name)) {
                if (value.dimensions() != 2)
                    throw ops::Failure({ErrorCode::INVALID_MANIFEST, "invalid prepared Inkling-2 dense matrix"});
                staged_packed.emplace(name, load_packed(prefix, value.size(0), value.size(1)));
            }
        }
        bind_local_kv_packed(staged_weights, staged_packed);
        if (!staged_weights.contains("embed.weight") || !staged_packed.contains("unembed.weight"))
            throw ops::Failure({ErrorCode::INVALID_MANIFEST, "incomplete prepared Inkling-2 CPU checkpoint"});
        decltype(experts_weights_) staged_experts(static_cast<std::size_t>(layers_),
                                                   std::vector<std::array<ExpertWeight, 2>>(experts_));
        for (std::int32_t layer = 0; layer < layers_; ++layer)
            for (std::int32_t expert = 0; expert < experts_; ++expert)
                for (std::size_t projection = 0; projection < 2; ++projection) {
                    const auto prefix = "kidi.cpu.expert." + std::to_string(layer) + "." + std::to_string(expert) +
                                        "." + std::to_string(projection);
                    auto packed = load_packed(prefix, projection ? hidden_ : 2 * intermediate_,
                                              projection ? intermediate_ : hidden_);
                    staged_experts[layer][expert][projection] = {
                        std::move(packed.values), std::move(packed.scales), {}};
                }
        auto context = std::make_unique<ops::Context>(device(), false, std::nullopt, DType::BF16);
        context_->synchronize();
        context_ = std::move(context);
        weights_ = std::move(staged_weights);
        packed_ = std::move(staged_packed);
        experts_weights_ = std::move(staged_experts);
        stacked_experts_.clear();
        return {};
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}

auto Inkling2Impl::set_checkpoint(const checkpoint::Weights& weights, RuntimeQuantization quantization)
    -> Result<void> {
    try {
        const bool vq = quantization.precision == core::InferencePrecision::VQ3D_HYBRID;
        if ((quantization.weight_bits != 0 && quantization.weight_bits != 8) ||
            (quantization.precision != core::InferencePrecision::CHECKPOINT &&
             quantization.precision != core::InferencePrecision::Q8 &&
             quantization.precision != core::InferencePrecision::BF16 && !vq))
            throw ops::Failure({ErrorCode::UNSUPPORTED,
                                "Inkling-2 supports native checkpoint, INT8, BF16, or VQ3D/VQ4D expert weights"});
        if (quantization.weight_bits && (quantization.precision == core::InferencePrecision::BF16 || vq))
            throw ops::Failure(
                {ErrorCode::INVALID_ARGUMENT, "BF16 or VQ precision cannot be combined with weight packing"});
        const bool cpu = device().kind == tensor::DeviceKind::CPU;
        const bool metal = device().kind == tensor::DeviceKind::A_GPU;
        if (cpu && quantization.precision == core::InferencePrecision::BF16)
            throw ops::Failure({ErrorCode::UNSUPPORTED, "Inkling-2 CPU inference requires INT8 or VQ expert weights"});
        const bool packed =
            !vq && (cpu || quantization.weight_bits == 8 || quantization.precision == core::InferencePrecision::Q8);
        const bool native = !vq && quantization.precision == core::InferencePrecision::CHECKPOINT && !packed;
        if (vq && weights.contains("kidi.vq.version")) return load_prepared_vq_checkpoint(weights);
        if (cpu && !vq && weights.contains("kidi.cpu.version")) return load_prepared_checkpoint(weights);
        decltype(weights_) staged_weights;
        std::vector<Tensor> host_sources;
        std::vector<std::pair<std::string, std::size_t>> weight_uploads;
        std::vector<Tensor> mmap_tensors;
        const auto queue_host = [&](Tensor host) -> std::size_t {
            const auto index = host_sources.size();
            host_sources.push_back(std::move(host));
            return index;
        };
        const auto queue_floats = [&](std::vector<std::int64_t> shape, const std::vector<float>& values) -> std::size_t {
            return queue_host(require(Tensor::from_host(std::move(shape), std::span<const float>(values),
                                                        tensor::Device::cpu())));
        };
        const auto D = static_cast<std::int64_t>(hidden_);
        // Small tensors that kernels read as FP32, with load-time layout changes.
        const auto convert = [&](const std::string& name, const std::string& stored) {
            auto values = host_floats(weights, name);
            const auto shape = require(weights.tensor(std::string(PREFIX) + name)).shape();
            std::vector<std::int64_t> dims(shape.begin(), shape.end());
            if (name.starts_with("attn_res.")) dims = {static_cast<std::int64_t>(values.size()) / D, D};
            weight_uploads.emplace_back(stored, queue_floats(std::move(dims), values));
        };
        const auto skip_merged = [](std::string_view name) {
            return name.ends_with("wq_du.weight") || name.ends_with("wr_du.weight") ||
                   name.ends_with("gate_attn.weight") || name.ends_with("attn_sink_proj.weight") ||
                   name.ends_with("wk_dv.weight") || name.ends_with("wv_dv.weight") ||
                   name.ends_with("k_sconv.weight_proj") || name.ends_with("v_sconv.weight_proj") ||
                   name.ends_with("gain_weight") || name.ends_with("forget_weight") ||
                   name.ends_with("rel_logits_proj.proj");
        };
        for (const auto& full : weights.names()) {
            if (!full.starts_with(PREFIX) || full.find(".mlp.experts.") != std::string::npos) continue;
            const auto name = full.substr(PREFIX.size());
            if (skip_merged(name)) continue;
            const auto value = require(weights.tensor(full));
            if (name.ends_with("mlp.gate.bias")) {
                staged_weights[name] = require(weights.float32_tensor(full));
                weight_uploads.emplace_back(name + ".device", queue_host(staged_weights[name]));
                continue;
            }
            if (name.ends_with("sconv.weight") && value.dtype() == DType::BF16) {
                const auto channels = static_cast<std::int64_t>(value.size(0));
                const auto width = static_cast<std::int64_t>(value.size(2));
                const auto src_bytes = require(value.host_bytes());
                const auto* src = reinterpret_cast<const std::uint16_t*>(src_bytes.data());
                if (metal) {
                    std::vector<float> lagged(static_cast<std::size_t>(width * channels));
                    for (std::int64_t c = 0; c < channels; ++c)
                        for (std::int64_t lag = 0; lag < width; ++lag)
                            lagged[static_cast<std::size_t>(lag * channels + c)] = std::bit_cast<float>(
                                static_cast<std::uint32_t>(src[c * width + width - 1 - lag]) << 16);
                    weight_uploads.emplace_back(name, queue_floats({width, channels}, lagged));
                } else {
                    auto lagged = require(Tensor::empty({width, channels}, DType::BF16, tensor::Device::cpu()));
                    const auto dst_bytes = require(lagged.host_bytes());
                    auto* dst = reinterpret_cast<std::uint16_t*>(dst_bytes.data());
                    for (std::int64_t c = 0; c < channels; ++c)
                        for (std::int64_t lag = 0; lag < width; ++lag)
                            dst[lag * channels + c] = src[c * width + width - 1 - lag];
                    weight_uploads.emplace_back(name, queue_host(std::move(lagged)));
                }
                mmap_tensors.push_back(value);
                continue;
            }
            const bool small = name.ends_with("norm.weight") || name.ends_with("sconv.weight") ||
                               name.starts_with("attn_res.") || name.starts_with("middle_cycle") ||
                               (metal && !cpu_projection(name) && name != "embed.weight");
            if (small || value.dtype() == DType::F32)
                convert(name, name);
            else {
                weight_uploads.emplace_back(name, queue_host(value));
                mmap_tensors.push_back(value);
            }
        }
        const auto host_bf16 = [&](const std::string& name) {
            const auto tensor = require(weights.tensor(std::string(PREFIX) + name));
            if (tensor.dtype() != DType::BF16)
                throw ops::Failure({ErrorCode::UNSUPPORTED, "expected BF16 Inkling-2 weight: " + name});
            return tensor;
        };
        const auto concat_bf16 = [&](std::span<const Tensor> parts, std::int64_t cols) {
            std::int64_t rows = 0;
            for (const auto& part : parts) rows += static_cast<std::int64_t>(part.size(0));
            auto combined = require(Tensor::empty({rows, cols}, DType::BF16, tensor::Device::cpu()));
            auto dst = require(combined.host_bytes());
            std::size_t offset = 0;
            for (const auto& part : parts) {
                const auto src = require(part.host_bytes());
                std::memcpy(dst.data() + offset, src.data(), src.size());
                offset += src.size();
            }
            return combined;
        };
        std::vector<std::tuple<std::string, std::size_t, std::int64_t, std::int64_t>> narrow_views;
        for (std::int32_t layer = 0; layer < layers_; ++layer) {
            const auto p = "layers." + std::to_string(layer) + ".attn.";
            const auto wq = host_bf16(p + "wq_du.weight"), wr = host_bf16(p + "wr_du.weight");
            const auto gate = host_bf16(p + "gate_attn.weight"), sink = host_bf16(p + "attn_sink_proj.weight");
            const std::array qrg_parts{wq, wr, gate, sink};
            weight_uploads.emplace_back(p + "qrg_sink.weight", queue_host(concat_bf16(qrg_parts, D)));
            for (const auto& part : qrg_parts) mmap_tensors.push_back(part);

            const auto wk = host_bf16(p + "wk_dv.weight"), wv = host_bf16(p + "wv_dv.weight");
            mmap_tensors.push_back(wk);
            mmap_tensors.push_back(wv);
            if (local(layer)) {
                const std::array kv_parts{wk, wv};
                const auto slot = queue_host(concat_bf16(kv_parts, D));
                weight_uploads.emplace_back(p + "wkv_dv.weight", slot);
                narrow_views.emplace_back(p + "wk_dv.weight", slot, 0, static_cast<std::int64_t>(wk.size(0)));
                narrow_views.emplace_back(p + "wv_dv.weight", slot, static_cast<std::int64_t>(wk.size(0)),
                                          static_cast<std::int64_t>(wv.size(0)));
            } else {
                const auto kt = host_bf16(p + "k_sconv.weight_proj"), vt = host_bf16(p + "v_sconv.weight_proj");
                mmap_tensors.push_back(kt);
                mmap_tensors.push_back(vt);
                const std::array k_parts{wk, kt}, v_parts{wv, vt};
                weight_uploads.emplace_back(p + "wk_taps.weight", queue_host(concat_bf16(k_parts, D)));
                weight_uploads.emplace_back(p + "wv_taps.weight", queue_host(concat_bf16(v_parts, D)));
            }

            const auto rel_proj = host_bf16(p + "rel_logits_proj.proj");
            mmap_tensors.push_back(rel_proj);
            const auto width = static_cast<std::int64_t>(rel_proj.size(0));
            const auto extent = static_cast<std::int64_t>(rel_proj.size(1));
            auto rel_t = require(Tensor::empty({extent, width}, DType::BF16, tensor::Device::cpu()));
            const auto rel_src_bytes = require(rel_proj.host_bytes());
            const auto rel_dst_bytes = require(rel_t.host_bytes());
            const auto* rel_src = reinterpret_cast<const std::uint16_t*>(rel_src_bytes.data());
            auto* rel_dst = reinterpret_cast<std::uint16_t*>(rel_dst_bytes.data());
            for (std::int64_t r = 0; r < width; ++r)
                for (std::int64_t c = 0; c < extent; ++c)
                    rel_dst[c * width + r] = rel_src[r * extent + c];
            weight_uploads.emplace_back(p + "rel_logits_proj.proj_t", queue_host(std::move(rel_t)));

            if (layer >= core_start_ && layer < core_end_) {
                for (const auto* stream : STREAMS) {
                    const auto cp = "layers." + std::to_string(layer) + ".controllers." + stream + ".";
                    const auto gain = host_bf16(cp + "gain_weight"), forget = host_bf16(cp + "forget_weight");
                    mmap_tensors.push_back(gain);
                    mmap_tensors.push_back(forget);
                    const auto rows = static_cast<std::int64_t>(gain.size(0));
                    const auto rank = static_cast<std::int64_t>(gain.size(1));
                    auto combined = require(Tensor::empty({rows * 2, rank}, DType::BF16, tensor::Device::cpu()));
                    const auto gain_bytes = require(gain.host_bytes());
                    const auto forget_bytes = require(forget.host_bytes());
                    const auto dst_bytes = require(combined.host_bytes());
                    std::memcpy(dst_bytes.data(), gain_bytes.data(), gain_bytes.size());
                    const auto* forget_src = reinterpret_cast<const std::uint16_t*>(forget_bytes.data());
                    auto* forget_dst = reinterpret_cast<std::uint16_t*>(dst_bytes.data() + gain_bytes.size());
                    const auto count = static_cast<std::size_t>(rows * rank);
                    for (std::size_t idx = 0; idx < count; ++idx)
                        forget_dst[idx] = forget_src[idx] ^ std::uint16_t{0x8000U};
                    weight_uploads.emplace_back(cp + "gain_forget_weight", queue_host(std::move(combined)));
                }
            }
        }
        {
            const auto queries = host_floats(weights, "middle_cycle_cross_pass_queries");
            const auto kv = host_floats(weights, "middle_cycle_cross_pass_kv_queries");
            for (auto layer = core_start_; layer < core_end_; ++layer) {
                const auto j = static_cast<std::size_t>(layer - core_start_);
                std::vector<float> stacked;
                for (const auto* row :
                     {&queries[2 * j * D], &kv[2 * j * D], &kv[(2 * j + 1) * D], &queries[(2 * j + 1) * D]})
                    stacked.insert(stacked.end(), row, row + D);
                weight_uploads.emplace_back("loop_queries." + std::to_string(layer), queue_floats({4, D}, stacked));
            }
        }
        weight_uploads.emplace_back("ones.hidden",
                                    queue_floats({D}, std::vector<float>(static_cast<std::size_t>(D), 1.F)));
        weight_uploads.emplace_back("ones.head",
                                    queue_floats({head_width_}, std::vector<float>(static_cast<std::size_t>(head_width_), 1.F)));
        weight_uploads.emplace_back("logit_cap", queue_floats({1}, {logit_cap_}));
        weight_uploads.emplace_back("inverse_logit_cap", queue_floats({1}, {1.F / logit_cap_}));

        struct ExpertUpload {
            std::int32_t layer;
            std::size_t index, values, scales, global;
            Tensor host_values, host_scales, host_global;
        };
        std::vector<ExpertUpload> expert_uploads;
        expert_uploads.reserve(static_cast<std::size_t>(layers_) * 2);
        for (std::int32_t layer = 0; layer < layers_; ++layer) {
            const auto prefix = "layers." + std::to_string(layer) + ".mlp.experts.";
            for (std::size_t index = 0; index < 2; ++index) {
                const std::string projection = index ? "w2" : "w13";
                const auto values = require(weights.tensor(std::string(PREFIX) + prefix + projection + "_weight"));
                const auto scales =
                    require(weights.tensor(std::string(PREFIX) + prefix + projection + "_weight_scale"));
                std::size_t global_index = SIZE_MAX;
                auto host_global = expert_global_scale(weights, prefix + projection, values);
                if (!vq && host_global.defined()) global_index = queue_host(host_global);
                const auto width = index ? intermediate_ : hidden_;
                const auto rows = index ? hidden_ : 2 * intermediate_;
                const bool nvfp4 = values.dtype() == DType::U8;
                if (values.dimensions() != 3 || scales.dimensions() != 3 || (!nvfp4 && values.dtype() != DType::E4M3) ||
                    values.size(0) != static_cast<std::size_t>(experts_) ||
                    values.size(1) != static_cast<std::size_t>(rows) ||
                    values.size(2) != static_cast<std::size_t>(width / (nvfp4 ? 2 : 1)) ||
                    (scales.dtype() != DType::U8 && !(nvfp4 && scales.dtype() == DType::E4M3)) ||
                    scales.size(0) != values.size(0) || scales.size(1) != values.size(1) ||
                    scales.size(2) != static_cast<std::size_t>(width / (nvfp4 ? 16 : 32)))
                    throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Inkling-2 expert geometry or format mismatch"});
                if (vq) {
                    expert_uploads.push_back({layer, index, SIZE_MAX, SIZE_MAX, SIZE_MAX, values, scales, host_global});
                } else {
                    expert_uploads.push_back(
                        {layer, index, queue_host(values), queue_host(scales), global_index, {}, {}, {}});
                    mmap_tensors.push_back(values);
                    mmap_tensors.push_back(scales);
                }
            }
        }
        const auto uploaded = require(Tensor::to_batch(host_sources, device()));
        host_sources.clear();
        host_sources.shrink_to_fit();
        for (const auto& mapped : mmap_tensors) mapped.release_pages(false);
        for (const auto& [name, slot] : weight_uploads) staged_weights[name] = uploaded[slot];
        for (const auto& [name, slot, offset, length] : narrow_views)
            staged_weights[name] = require(uploaded[slot].narrow(0, offset, length));

        const bool stacked = native || vq;
        decltype(experts_weights_) staged_experts(
            stacked ? 0 : static_cast<std::size_t>(layers_),
            std::vector<std::array<ExpertWeight, 2>>(stacked ? 0 : static_cast<std::size_t>(experts_)));
        decltype(stacked_experts_) staged_stacks(stacked ? static_cast<std::size_t>(layers_) : 0);
        ops::Context loading(vq ? tensor::Device::cpu() : device(), false, false, DType::BF16);
        for (const auto& entry : expert_uploads) {
            if (vq) {
                // Decode each NVFP4/MXFP8 bank on the CPU in 64-row chunks, then release its mapped pages.
                // Down projections and the tail after the loop stay VQ3D; the other gate/up banks use smaller VQ4D.
                const bool vq3d = entry.index == 1 || entry.layer >= core_end_;
                const auto fmt = vq3d ? ops::BlockScaledFormat::VQ3D : ops::BlockScaledFormat::VQ4D;
                auto packed_vq = require(ops::pack_vq_expert_weights(entry.host_values, entry.host_scales,
                                                                     entry.host_global, entry.index == 0, fmt));
                entry.host_values.release_pages(false);
                entry.host_scales.release_pages(false);
                staged_stacks[static_cast<std::size_t>(entry.layer)][entry.index] = {
                    require(packed_vq.values.to(device())), require(packed_vq.scales.to(device())),
                    require(packed_vq.global.to(device())), fmt, packed_vq.group_size};
                continue;
            }
            const auto& device_values = uploaded[entry.values];
            const auto& device_scales = uploaded[entry.scales];
            const Tensor global = entry.global == SIZE_MAX ? Tensor{} : uploaded[entry.global];
            if (native) {
                const auto fmt =
                    device_values.dtype() == DType::U8 ? ops::BlockScaledFormat::NVFP4 : ops::BlockScaledFormat::MXFP8;
                staged_stacks[static_cast<std::size_t>(entry.layer)][entry.index] = {device_values, device_scales,
                                                                                     global, fmt, 0};
            } else if (packed) {
                const auto packed_weight =
                    loading.pack_experts(device_values, device_scales, global, entry.index == 0);
                for (std::int64_t expert = 0; expert < experts_; ++expert)
                    staged_experts[static_cast<std::size_t>(entry.layer)][expert][entry.index] = {
                        require(packed_weight.values.select(0, expert)),
                        require(packed_weight.scales.select(0, expert))};
            } else {
                const auto expanded =
                    loading.dequantize_experts(device_values, device_scales, global, entry.index == 0);
                for (std::int64_t expert = 0; expert < experts_; ++expert)
                    staged_experts[static_cast<std::size_t>(entry.layer)][expert][entry.index] = {
                        require(expanded.select(0, expert)), {}};
            }
            if (cpu) {
                device_values.release_pages(false);
                device_scales.release_pages(false);
            }
        }
        if (!native) loading.synchronize();
        activation_ = metal ? DType::F32 : DType::BF16;
        auto context = std::make_unique<ops::Context>(device(), false, std::nullopt, activation_);
        context_->synchronize();
        context_ = std::move(context);
        // CPU and Metal run dense projections as row-scaled INT8 (halves RAM and avoids MPSGraph constant copies).
        decltype(packed_) staged_packed;
        if (cpu || metal) {
            for (const auto& [name, tensor] : staged_weights)
                if (tensor.dimensions() == 2 && tensor.dtype() == DType::BF16 && cpu_projection(name)) {
                    auto host_t = tensor.is_host_accessible() ? tensor : require(tensor.to(tensor::Device::cpu()));
                    auto p = require(ops::pack_rows_int8(host_t));
                    staged_packed[name] = {
                        require(p.values.to(device())), require(p.scales.to(device())), {}, 8, p.group_size};
                }
            bind_local_kv_packed(staged_weights, staged_packed);
            if (metal) {
                for (auto it = staged_weights.begin(); it != staged_weights.end();) {
                    if (staged_packed.contains(it->first) && !it->first.ends_with("rel_logits_proj.proj_t"))
                        it = staged_weights.erase(it);
                    else
                        ++it;
                }
            }
        }
        weights_ = std::move(staged_weights);
        packed_ = std::move(staged_packed);
        experts_weights_ = std::move(staged_experts);
        stacked_experts_ = std::move(staged_stacks);
        return {};
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

auto Inkling2Impl::preparation_ns() const noexcept -> std::uint64_t { return context_->preparation_ns(); }
auto Inkling2Impl::release_workspaces() -> void { context_->release_workspaces(); }
auto Inkling2Impl::trim_workspaces() -> void {
    context_->synchronize();
    context_->clear_replays();
    context_->trim_workspaces();
}
auto Inkling2Impl::accelerator() const -> std::string_view { return context_->accelerator(); }
auto Inkling2Impl::supports_device_tokens() const -> bool {
    return device().capabilities().compute.embedding &&
           device().capabilities().sampling.supports(tensor::TokenSelection::CHAINED_DEVICE);
}

auto Inkling2Impl::cache_capacities(std::size_t capacity) const -> std::pair<std::size_t, std::size_t> {
    const auto global = std::min(max_positions_, ((capacity + 255) / 256) * 256);
    const auto local_cap = ring_caches(device()) && window_ > 0
                               ? std::min(global, static_cast<std::size_t>(window_) + RING_SPARE_ROWS)
                               : global;
    return {global, local_cap};
}

auto Inkling2Impl::create_state(std::size_t capacity) -> Result<std::unique_ptr<LanguageModelState>> {
    try {
        if (!capacity || capacity > max_positions_)
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Inkling-2 state capacity"});
        auto state = std::make_unique<Inkling2State>();
        state->capacity = capacity;
        const auto [global_cap, local_cap] = cache_capacities(capacity);
        state->local_capacity = local_cap;
        const auto kv = static_cast<std::int64_t>(key_heads_) * head_width_;
        const auto zeros = [&](std::vector<std::int64_t> shape) {
            return require(Tensor::zeros(std::move(shape), activation_, device()));
        };
        const auto cache_dtype = require(kv_cache_dtype(capacity));
        const auto cache = [&](std::size_t cap) {
            return require(Tensor::zeros({1, static_cast<std::int64_t>(cap), kv}, cache_dtype, device()));
        };
        const auto scales = [&](std::size_t cap) {
            return cache_dtype == DType::I8
                       ? require(Tensor::zeros({static_cast<std::int64_t>(cap), kv / std::gcd(32, head_width_)},
                                               DType::F32, device()))
                       : Tensor{};
        };
        for (const auto layer : schedule()) {
            const auto cap = local(layer) ? local_cap : global_cap;
            state->visits.push_back({cache(cap), cache(cap), zeros({1, CONV_WIDTH, kv}), zeros({1, CONV_WIDTH, kv}),
                                     zeros({1, CONV_WIDTH, hidden_}), zeros({1, CONV_WIDTH, hidden_}), scales(cap),
                                     scales(cap)});
        }
        return state;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}

auto Inkling2Impl::grow_state(LanguageModelState& base, std::size_t capacity) -> Result<void> {
    if (device().kind != tensor::DeviceKind::A_GPU) return CausalLanguageModel::grow_state(base, capacity);
    auto* state = dynamic_cast<Inkling2State*>(&base);
    if (!state || !state->capacity || state->position > state->capacity || capacity < state->capacity ||
        capacity > max_positions_)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Inkling-2 cache growth"});
    if (capacity == state->capacity) return {};
    try {
        const auto [global_cap, local_cap] = cache_capacities(capacity);
        const auto layers = schedule();
        const auto enlarge = [&](const Tensor& cache, std::int64_t axis, std::size_t rows) {
            if (!cache.defined()) return Tensor{};
            auto shape = std::vector(cache.shape().begin(), cache.shape().end());
            shape[axis] = static_cast<std::int64_t>(rows);
            auto expanded = require(Tensor::zeros(std::move(shape), cache.dtype(), device()));
            const auto old_rows = static_cast<std::int64_t>(cache.size(axis));
            const auto valid = static_cast<std::int64_t>(std::min(state->position, cache.size(axis)));
            const auto width = static_cast<std::int64_t>(cache.size(-1));
            if (valid) {
                const auto flat = require(cache.reshape({old_rows, width}));
                auto prefix = require(flat.narrow(0, 0, valid));
                if (axis == 1) prefix = require(prefix.reshape({1, valid, width}));
                context_->copy_slice_(expanded, prefix, axis, 0);
            }
            return expanded;
        };
        for (std::size_t index = 0; index < layers.size(); ++index) {
            auto& visit = state->visits[index];
            const auto cap = local(layers[index]) ? local_cap : global_cap;
            if (visit.key.size(1) >= cap) continue;
            auto key = enlarge(visit.key, 1, cap);
            auto value = enlarge(visit.value, 1, cap);
            auto key_scales = enlarge(visit.key_scales, 0, cap);
            auto value_scales = enlarge(visit.value_scales, 0, cap);
            // Once a local cache wraps, its size is fixed. Release old storage one visit at a time.
            context_->synchronize();
            visit.key = std::move(key);
            visit.value = std::move(value);
            visit.key_scales = std::move(key_scales);
            visit.value_scales = std::move(value_scales);
        }
        state->local_capacity = local_cap;
        state->capacity = capacity;
        return {};
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}

auto Inkling2Impl::rewind(LanguageModelState& base, std::size_t position) -> Result<void> {
    auto* state = dynamic_cast<Inkling2State*>(&base);
    if (!state || !state->capacity || state->position > state->capacity || position > state->position)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Inkling-2 state rewind"});
    if (position == state->position) return {};
    if (!retains_local_history(*state, position, window_))
        return std::unexpected(Error{ErrorCode::UNSUPPORTED, "Inkling-2 rewind would need overwritten local KV"});
    const bool marked = position != 0 && position == state->rewind_position && !state->rewind_convs.empty();
    const bool verified = position != 0 && !marked && !state->verify_convs.empty() &&
                          position >= state->verify_position && position <= state->verify_end;
    if (position != 0 && !marked && !verified)
        return std::unexpected(Error{ErrorCode::UNSUPPORTED,
                                     "Inkling-2 rewinds only to the initial state, its marked rewind point or "
                                     "inside the last CPU verification block"});
    try {
        // Required KV rows are still resident; only the short-convolution rings need restoring.
        std::size_t index = 0;
        for (auto& visit : state->visits) {
            std::size_t slot = 0;
            for (auto* cache : {&visit.key_conv, &visit.value_conv, &visit.attention_conv, &visit.mlp_conv}) {
                if (position == 0) {
                    require(cache->zero_());
                } else if (marked) {
                    context_->copy_slice_(*cache, state->rewind_convs[index], 1, 0);
                } else {
                    // Rings hold raw convolution inputs at slot position % W: restore the block-start ring, then
                    // write back the last accepted inputs recorded during verification.
                    const auto snapshot = require(state->verify_convs[index].host_bytes());
                    auto destination = require(cache->host_bytes());
                    std::memcpy(destination.data(), snapshot.data(), destination.size());
                    const auto row_bytes = destination.size() / CONV_WIDTH;
                    const auto count = std::min<std::size_t>(CONV_WIDTH, position - state->verify_position);
                    const auto source = require(visit.rewind_inputs[slot].host_bytes());
                    for (auto absolute = position - count; absolute < position; ++absolute)
                        std::memcpy(destination.data() + (absolute % CONV_WIDTH) * row_bytes,
                                    source.data() + (absolute - state->verify_position) * row_bytes, row_bytes);
                }
                ++index;
                ++slot;
            }
        }
        if (position == 0) {
            state->rewind_convs.clear();
            state->rewind_position = 0;
        }
        // Later steps overwrite rows after `position`; only the accepted part of the block stays valid.
        if (position < state->verify_position) state->verify_convs.clear();
        state->verify_end = std::min(state->verify_end, position);
        state->position = position;
        return {};
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

auto Inkling2Impl::mark_rewind_point(LanguageModelState& base) -> Result<void> {
    auto* state = dynamic_cast<Inkling2State*>(&base);
    if (!state) return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Inkling-2 state"});
    try {
        const bool allocate = state->rewind_convs.empty();
        std::size_t index = 0;
        for (auto& visit : state->visits)
            for (auto* cache : {&visit.key_conv, &visit.value_conv, &visit.attention_conv, &visit.mlp_conv}) {
                if (allocate)
                    state->rewind_convs.push_back(require(Tensor::zeros(
                        std::vector<std::int64_t>(cache->shape().begin(), cache->shape().end()), cache->dtype(),
                        device())));
                context_->copy_slice_(state->rewind_convs[index++], *cache, 1, 0);
            }
        state->rewind_position = state->position;
        return {};
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}

auto Inkling2Impl::rewind_floor(const LanguageModelState& base, std::size_t position) const -> std::size_t {
    const auto* state = dynamic_cast<const Inkling2State*>(&base);
    if (!state || position >= state->position) return position;
    if (!state->verify_convs.empty() && position >= state->verify_position && position <= state->verify_end)
        return position;
    return state->rewind_position <= position && retains_local_history(*state, state->rewind_position, window_)
               ? state->rewind_position
               : 0;
}

auto Inkling2Impl::snapshot_verification(Inkling2State& state) -> void {
    const bool allocate = state.verify_convs.empty();
    std::size_t index = 0;
    for (auto& visit : state.visits)
        for (auto* cache : {&visit.key_conv, &visit.value_conv, &visit.attention_conv, &visit.mlp_conv}) {
            if (allocate)
                state.verify_convs.push_back(require(Tensor::empty(
                    std::vector<std::int64_t>(cache->shape().begin(), cache->shape().end()), cache->dtype(),
                    device())));
            const auto source = require(cache->host_bytes());
            std::memcpy(require(state.verify_convs[index++].host_bytes()).data(), source.data(), source.size());
        }
    state.verify_position = state.verify_end = state.position;
}

auto Inkling2Impl::save_convolution_input(Tensor& saved, const Tensor& input) -> void {
    if (!saved.defined() || !std::ranges::equal(saved.shape(), input.shape()) || saved.dtype() != input.dtype())
        saved = require(
            Tensor::empty(std::vector(input.shape().begin(), input.shape().end()), input.dtype(), device()));
    context_->copy_slice_(saved, input, 0, 0);
}

auto Inkling2Impl::bind_local_kv_packed(const std::map<std::string, Tensor, std::less<>>& weights,
                                        std::map<std::string, ops::PackedWeight, std::less<>>& packed) const -> void {
    for (std::int32_t layer = 0; layer < layers_; ++layer) {
        if (!local(layer)) continue;
        const auto p = "layers." + std::to_string(layer) + ".attn.";
        const auto combined = packed.find(p + "wkv_dv.weight");
        const auto wk = weights.find(p + "wk_dv.weight");
        const auto wv = weights.find(p + "wv_dv.weight");
        if (combined == packed.end() || wk == weights.end() || wv == weights.end() || wk->second.dimensions() != 2 ||
            wv->second.dimensions() != 2 ||
            wk->second.size(0) + wv->second.size(0) != combined->second.values.size(0))
            throw ops::Failure({ErrorCode::INVALID_MANIFEST, "incomplete Inkling-2 local K/V packed weights"});
        const auto k_rows = static_cast<std::int64_t>(wk->second.size(0));
        const auto v_rows = static_cast<std::int64_t>(wv->second.size(0));
        packed.insert_or_assign(p + "wk_dv.weight", narrow_packed(combined->second, 0, k_rows));
        packed.insert_or_assign(p + "wv_dv.weight", narrow_packed(combined->second, k_rows, v_rows));
    }
}

auto Inkling2Impl::convolve(const Tensor& input, Tensor& state, const Tensor& weight, const Tensor& taps,
                            const Tensor& conv_slot, const Tensor& tail_slots, float scale, const Tensor& residual,
                            std::optional<DType> output_dtype, Tensor* saved_input,
                            bool fuse_single_update) -> Tensor {
    if (saved_input) save_convolution_input(*saved_input, input);
    auto& context = *context_;
    const auto T = static_cast<std::int64_t>(input.size(0));
    const auto C = static_cast<std::int64_t>(input.size(1));
    const bool fused_update = fuse_single_update && T == 1;
    auto output = context.short_convolution(input, require(state.reshape({CONV_WIDTH, C})), weight, taps, conv_slot,
                                            scale, residual, output_dtype, fused_update);
    if (!fused_update) {
        if (T == 1) {
            context.scatter_unique_(state, require(input.reshape({1, 1, C})), conv_slot);
        } else {
            const auto K = std::min<std::int64_t>(T, CONV_WIDTH);
            const auto tail = K == T ? input : require(input.narrow(0, T - K, K));
            context.scatter_unique_(state, require(tail.reshape({1, K, C})), tail_slots);
        }
    }
    return output;
}

auto Inkling2Impl::attention(std::int32_t layer, const std::array<Tensor, 3>& inputs, Inkling2Visit& cache,
                             const Tensor& positions, const Tensor& local_positions, const Tensor& conv_slot,
                             const Tensor& tail_slots) -> Tensor {
    auto& context = *context_;
    const auto p = "layers." + std::to_string(layer) + ".attn.";
    const auto linear = [&](const Tensor& input, const std::string& name) {
        return project_bf16(input, p + name);
    };
    const bool is_local = local(layer);
    const auto T = static_cast<std::int64_t>(inputs[0].size(0));
    const auto kv = static_cast<std::int64_t>(key_heads_) * head_width_;
    const auto q_width = static_cast<std::int64_t>(heads_) * head_width_;
    const auto& rel_proj = weight(p + "rel_logits_proj.proj_t");
    const auto extent = static_cast<std::int64_t>(rel_proj.size(0));
    const auto width = static_cast<std::int64_t>(rel_proj.size(1));
    const auto r_width = static_cast<std::int64_t>(heads_) * width;
    const auto split_cols = [&](const Tensor& tensor, std::int64_t offset, std::int64_t count) {
        if (T == 1)
            return require(require(require(tensor.reshape({static_cast<std::int64_t>(tensor.size(1))}))
                                       .narrow(0, offset, count))
                               .reshape({1, count}));
        return context.slice(tensor, 1, offset, count);
    };
    const auto qrg_sink = linear(inputs[0], "qrg_sink.weight");
    auto q = split_cols(qrg_sink, 0, q_width);
    auto r = split_cols(qrg_sink, q_width, r_width);
    const auto gate = split_cols(qrg_sink, q_width + r_width, q_width);
    const auto sink = split_cols(qrg_sink, q_width * 2 + r_width, heads_);
    Tensor k, v, k_taps, v_taps;
    if (is_local) {
        if (inputs[1].storage_identity() == inputs[2].storage_identity() &&
            inputs[1].storage_offset() == inputs[2].storage_offset()) {
            const auto kv_proj = linear(inputs[1], "wkv_dv.weight");
            k = split_cols(kv_proj, 0, kv);
            v = split_cols(kv_proj, kv, kv);
        } else {
            k = linear(inputs[1], "wk_dv.weight");
            v = linear(inputs[2], "wv_dv.weight");
        }
    } else {
        const auto k_proj = linear(inputs[1], "wk_taps.weight");
        const auto v_proj = linear(inputs[2], "wv_taps.weight");
        k = split_cols(k_proj, 0, kv);
        k_taps = split_cols(k_proj, kv, static_cast<std::int64_t>(k_proj.size(1)) - kv);
        v = split_cols(v_proj, 0, kv);
        v_taps = split_cols(v_proj, kv, static_cast<std::int64_t>(v_proj.size(1)) - kv);
    }
    const auto plain = 1.F / std::sqrt(static_cast<float>(CONV_WIDTH + 1));
    const auto dynamic = 1.F / std::sqrt(static_cast<float>(2 * CONV_WIDTH + 1));
    const auto& k_weight = weight(p + (is_local ? "k_sconv.weight" : "k_sconv.static_w"));
    const auto& v_weight = weight(p + (is_local ? "v_sconv.weight" : "v_sconv.static_w"));
    const auto conv_scale = is_local ? plain : dynamic;
    k = convolve(k, cache.key_conv, k_weight, k_taps, conv_slot, tail_slots, conv_scale, {}, {},
                 cache.record_rewind ? &cache.rewind_inputs[0] : nullptr, false);
    v = convolve(v, cache.value_conv, v_weight, v_taps, conv_slot, tail_slots, conv_scale, {}, {},
                 cache.record_rewind ? &cache.rewind_inputs[1] : nullptr, false);
    q = context.l2_clip(context.reshape(q, {T, heads_, head_width_}), weight(p + "q_l2_clip.scale"), qk_clip_);
    k = context.l2_clip(context.reshape(k, {T, key_heads_, head_width_}), {}, qk_clip_);
    v = context.rms_norm(context.reshape(v, {T, key_heads_, head_width_}), weight("ones.head"), EPSILON);
    r = context.l2_clip(context.reshape(r, {T, heads_, width}), weight(p + "rel_l2_clip.scale"), relative_clip_);
    const auto relative =
        context.reshape(project_bf16(context.reshape(r, {T * heads_, width}), p + "rel_logits_proj.proj_t"),
                        {T, heads_, extent});
    const auto& write_pos = (is_local && local_positions.defined()) ? local_positions : positions;
    if (cache.key.dtype() == DType::I8) {
        context.quantize_cache_(cache.key, cache.key_scales, k, write_pos);
        context.quantize_cache_(cache.value, cache.value_scales, v, write_pos);
    } else {
        auto k_store = require(k.reshape({1, T, kv}));
        auto v_store = require(v.reshape({1, T, kv}));
        if (k_store.dtype() != cache.key.dtype()) k_store = context.cast(k_store, cache.key.dtype());
        if (v_store.dtype() != cache.value.dtype()) v_store = context.cast(v_store, cache.value.dtype());
        context.scatter_unique_(cache.key, k_store, write_pos);
        context.scatter_unique_(cache.value, v_store, write_pos);
    }
    const auto capacity = static_cast<std::int64_t>(cache.key.size(1));
    const auto output = context.relative_attention(context.reshape(q, {T, heads_ * head_width_}),
                                                   require(cache.key.reshape({capacity, kv})),
                                                   require(cache.value.reshape({capacity, kv})), relative, sink, gate,
                                                   {.heads = heads_,
                                                    .key_heads = key_heads_,
                                                    .start = 0,
                                                    .window = is_local ? window_ : 0,
                                                    .extent = extent,
                                                    .log_alpha = is_local ? 0.F : log_alpha_,
                                                    .log_floor = is_local ? 0.F : log_floor_,
                                                    .ring_mode = is_local && local_positions.defined()},
                                                   positions, cache.key_scales, cache.value_scales);
    return linear(output, "wo_ud.weight");
}

auto Inkling2Impl::blend(std::int32_t layer, const std::string& stream, const Tensor& destination,
                         const Tensor& message, const Tensor& norm) -> Tensor {
    auto& context = *context_;
    const auto p = "layers." + std::to_string(layer) + ".controllers." + stream + ".";
    const auto pair = context.blend_prologue(destination, message, EPSILON);
    const auto hidden = project_bf16(pair, p + "down_weight");
    const auto gain_forget = project_bf16(hidden, p + "gain_forget_weight");
    return context.blend_epilogue(destination, message, gain_forget, norm, EPSILON);
}

auto Inkling2Impl::moe(std::int32_t layer, const Tensor& input) -> Tensor {
    auto& context = *context_;
    const auto p = "layers." + std::to_string(layer) + ".mlp.";
    const auto T = static_cast<std::int64_t>(input.size(0));
    const auto compact = [](const ExpertWeight& weight) {
        return ops::BlockScaledWeight{weight.format, weight.values, weight.scales, weight.global, weight.group_size};
    };
    const auto router_raw = project_bf16(input, p + "gate.weight");
    // Routing stays on the GPU: no host sync, uploads or per-expert launches. Decode reads each routed expert's
    // rows once per token; larger batches group rows by expert inside the projections.
    if (!stacked_experts_.empty()) {
        const auto routes =
            context.route_experts(router_raw, weight(p + "gate.bias.device"), experts_per_token_, route_scale_);
        const auto& [gate_up, down] = stacked_experts_[static_cast<std::size_t>(layer)];
        const auto activated = context.expert_silu(input, compact(gate_up), routes, gate_cap_, up_cap_);
        return context.expert_linear(activated, compact(down), routes, activation_);
    }
    const auto router = context.cast(router_raw, DType::F32);
    const auto logits = require(router.to(tensor::Device::cpu()));
    const auto scores_data = require(logits.data<float>());
    const auto bias_data = require(weight(p + "gate.bias").data<float>());
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
    std::vector<std::int32_t> tokens;
    std::vector<float> scales;
    tokens.reserve(static_cast<std::size_t>(T * experts_per_token_));
    scales.reserve(tokens.capacity());
    for (const auto& route : routes)
        for (const auto& [token, scale] : route) {
            tokens.push_back(token);
            scales.push_back(scale);
        }
    const auto& experts = experts_weights_[static_cast<std::size_t>(layer)];
    const auto row_int8 = [](const ExpertWeight& weight) {
        return weight.values.dtype() == DType::U8 && weight.values.dimensions() == 2 &&
               weight.values.is_contiguous() && weight.values.is_host_accessible() && weight.scales.defined() &&
               weight.scales.dtype() == DType::F32 && weight.scales.numel() == weight.values.size(0) &&
               !weight.global.defined();
    };
    if (device().kind == tensor::DeviceKind::CPU) {
        std::vector<ops::cpu::ExpertRows> chosen;
        std::vector<ops::cpu::RoutedExpert> batches;
        std::int64_t intermediate = 0;
        std::size_t batch_offset = 0;
        bool fused = true;
        for (std::int32_t e = 0; e < experts_ && fused; ++e) {
            if (routes[e].empty()) continue;
            const auto& gate_up = experts[e][0];
            const auto& down = experts[e][1];
            fused = row_int8(gate_up) && row_int8(down) && gate_up.values.size(1) == hidden_ &&
                    down.values.size(0) == hidden_ && gate_up.values.size(0) == 2 * down.values.size(1);
            if (!fused) break;
            intermediate = static_cast<std::int64_t>(down.values.size(1));
            chosen.push_back({reinterpret_cast<const std::int8_t*>(require(gate_up.values.data<std::uint8_t>()).data()),
                              require(gate_up.scales.data<float>()).data(),
                              reinterpret_cast<const std::int8_t*>(require(down.values.data<std::uint8_t>()).data()),
                              require(down.scales.data<float>()).data(), routes[e].front().second});
            batches.push_back({chosen.back(), std::span(tokens).subspan(batch_offset, routes[e].size()),
                               std::span(scales).subspan(batch_offset, routes[e].size())});
            batch_offset += routes[e].size();
        }
        if (fused) {
            const auto values = require(context.cast(input, DType::F32).to(tensor::Device::cpu()));
            auto result = require(Tensor::empty({T, hidden_}, DType::F32, device()));
            if (T == 1)
                require(ops::cpu::expert_decode(require(values.data<float>()).data(), chosen, hidden_, intermediate,
                                                gate_cap_, up_cap_, require(result.data<float>()).data()));
            else {
                require(ops::cpu::expert_prefill(
                    require(values.data<float>()).data(), T, batches, hidden_, intermediate, gate_cap_, up_cap_,
                    require(result.data<float>()).data()));
            }
            return context.cast(result, DType::BF16);
        }
    }
    const auto count = static_cast<std::int64_t>(scales.size());
    const auto row_weights = require(Tensor::from_host({count}, std::span<const float>(scales), device()));
    const auto indices =
        T == 1 ? Tensor{} : require(Tensor::from_host({count}, std::span<const std::int32_t>(tokens), device()));
    auto output = T == 1 ? Tensor{} : require(Tensor::zeros({1, T, hidden_}, DType::F32, device()));
    const auto native = [](const ExpertWeight& weight) {
        return weight.values.dtype() == DType::E4M3 || weight.scales.dtype() == DType::E4M3 ||
               weight.scales.dtype() == DType::U8;
    };
    const auto project = [&](const Tensor& rows, const ExpertWeight& weight) {
        if (native(weight)) return context.block_scaled_linear(rows, compact(weight));
        if (!weight.scales.defined()) return context.linear(rows, weight.values, {}, true);
        const auto activation = device().kind == tensor::DeviceKind::CPU ? context.cast(rows, DType::F32) : rows;
        return context.cast(context.packed_linear(activation, weight.values, weight.scales, 8,
                                                   static_cast<std::int32_t>(weight.values.size(1))), DType::BF16);
    };
    std::int64_t offset = 0;
    for (std::int32_t e = 0; e < experts_; ++e) {
        const auto& route = routes[e];
        if (route.empty()) continue;
        const auto n = static_cast<std::int64_t>(route.size());
        const auto index = T == 1 ? Tensor{} : require(indices.narrow(0, offset, n));
        const auto rows = T == 1 ? input : context.gather(input, index, 0);
        const auto routing = require(row_weights.narrow(0, offset, n));
        const auto& gate_up = experts[e][0];
        const auto activated = native(gate_up)
                                   ? context.block_scaled_silu(rows, compact(gate_up), routing, gate_cap_, up_cap_)
                                   : context.silu_gate(project(rows, gate_up), gate_cap_, up_cap_, routing);
        const auto projected = context.cast(project(activated, experts[e][1]), DType::F32);
        offset += n;
        if (T == 1) {
            if (!output.defined())
                output = context.reshape(projected, {1, 1, hidden_});
            else
                context.add_(output, context.reshape(projected, {1, 1, hidden_}));
            continue;
        }
        const auto sum = context.add(context.gather(output, index, 1), context.reshape(projected, {1, n, hidden_}));
        context.scatter_unique_(output, sum, index);
    }
    return context.cast(context.reshape(output, {T, hidden_}), DType::BF16);
}

auto Inkling2Impl::visit(std::int32_t layer, Stream stream, Inkling2Visit& cache, const Tensor& positions,
                         const Tensor& local_positions, const Tensor& conv_slot, const Tensor& tail_slots,
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
        return require(stacked.select(0, index));
    };
    Tensor messages;
    if (bank) messages = context.residual_mix(*bank, weight("loop_queries." + std::to_string(layer)), EPSILON);
    const auto prepare = [&](const Tensor& destination, std::size_t stream_index, const Tensor& norm) {
        if (messages.defined())
            return blend(layer, STREAMS[stream_index], destination, row(messages, static_cast<std::int64_t>(stream_index)),
                         norm);
        return context.rms_norm(destination, norm, EPSILON, activation_);
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
    const auto attended = attention(layer, inputs, cache, positions, local_positions, conv_slot, tail_slots);
    const auto plain = 1.F / std::sqrt(static_cast<float>(CONV_WIDTH + 1));
    if (layer % block_ == 0) stream.blocks.push_back(stream.partial);
    const auto residual = convolve(attended, cache.attention_conv, weight(p + "attn_sconv.weight"), {}, conv_slot,
                                   tail_slots, plain, layer % block_ == 0 ? Tensor{} : stream.partial, DType::F32,
                                   cache.record_rewind ? &cache.rewind_inputs[2] : nullptr);
    auto sources = stream.blocks;
    sources.push_back(residual);
    const auto mixed = row(context.residual_mix(sources, weight("attn_res.mlp_queries." + std::to_string(layer)), EPSILON), 0);
    const auto mlp_input = prepare(mixed, 3, weight(p + "mlp_norm.weight"));
    stream.partial =
        convolve(moe(layer, mlp_input), cache.mlp_conv, weight(p + "mlp_sconv.weight"), {}, conv_slot, tail_slots,
                 plain, residual, activation_, cache.record_rewind ? &cache.rewind_inputs[3] : nullptr);
    return stream;
}

auto Inkling2Impl::execute_step(const Tensor& ids, const Tensor& positions, const Tensor& local_positions,
                                const Tensor& conv_slot, const Tensor& tail_slots, Inkling2State& state, std::int64_t T,
                                bool all_logits, std::optional<ops::TokenOutput> select, Tensor* hidden) -> Tensor {
    auto& context = *context_;
    const auto& embedding = weight("embed.weight");
    auto embedded = [&] {
        if (embedding.device() == device()) return context.gather(embedding, ids, 0);
        auto rows = require(Tensor::empty({T, hidden_}, embedding.dtype(), device()));
        const auto source = require(embedding.host_bytes());
        const auto target = require(rows.host_bytes());
        const auto stride = embedding.nbytes() / embedding.size(0);
        const auto indices = require(ids.data<std::int32_t>());
        for (std::size_t i = 0; i < indices.size(); ++i) {
            const auto id = indices[i];
            if (id < 0 || static_cast<std::size_t>(id) >= embedding.size(0))
                throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Inkling-2 embedding index outside the vocabulary"});
            std::memcpy(target.data() + i * stride, source.data() + id * stride, stride);
        }
        return rows;
    }();
    if (embedded.dtype() != activation_) embedded = context.cast(embedded, activation_);
    Stream stream{{}, context.rms_norm(embedded, weight("embed_norm.weight"), EPSILON)};
    std::size_t visit_index = 0;
    for (std::int32_t layer = 0; layer < core_start_; ++layer)
        stream = visit(layer, std::move(stream), state.visits[visit_index++], positions, local_positions, conv_slot,
                       tail_slots, nullptr);
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
            stream = visit(layer, std::move(stream), state.visits[visit_index++], positions, local_positions, conv_slot,
                           tail_slots, pass ? &bank : nullptr);
    }
    for (auto layer = core_end_; layer < layers_; ++layer)
        stream = visit(layer, std::move(stream), state.visits[visit_index++], positions, local_positions, conv_slot,
                       tail_slots, nullptr);

    auto sources = stream.blocks;
    sources.push_back(stream.partial);
    const auto rows = all_logits ? T : 1;
    if (!all_logits && T > 1)
        for (auto& source : sources) source = context.slice(source, 0, T - 1, 1);
    const auto mixed =
        context.reshape(context.residual_mix(sources, weight("attn_res.output_query"), EPSILON), {rows, hidden_});
    const auto normalized = context.rms_norm(mixed, weight("norm.weight"), EPSILON, activation_);
    if (hidden) *hidden = normalized;
    auto logits = project_bf16(normalized, "unembed.weight");
    if (logits.dtype() != DType::F32) logits = context.cast(logits, DType::F32);
    if (T > 1) {
        context.multiply_(logits, weight("inverse_logit_cap"));
        logits = context.tanh(logits);
        context.multiply_(logits, weight("logit_cap"));
    } else {
        logits = context.multiply(context.tanh(context.multiply(logits, weight("inverse_logit_cap"))),
                                  weight("logit_cap"));
    }
    Tensor trimmed;
    if (rows == 1) {
        trimmed = require(require(require(logits.reshape({padded_vocabulary_})).narrow(0, 0, vocabulary_))
                              .reshape({1, vocabulary_}));
    } else {
        trimmed = context.slice(logits, 1, 0, vocabulary_);
    }
    if (select) return context.greedy_token(trimmed, *select);
    return trimmed;
}

auto Inkling2Impl::run_forward(const Tensor& ids, const Tensor& positions, const Tensor& local_positions,
                               const Tensor& conv_slot, const Tensor& tail_slots, Inkling2State& state, std::int64_t T,
                               bool all_logits, std::optional<ops::TokenOutput> select, Tensor* hidden) -> Tensor {
    auto& context = *context_;
    // T == 1 native decode runs inside a single captured CUDA graph across all sequence positions.
    if (T == 1 && !stacked_experts_.empty() && context.replay_enabled()) {
        // Ring caches add the local write slot as a step input, after ids, positions and conv_slot. Each visit
        // passes its caches, plus their scales when INT8, so every request's state binds to the captured step.
        const bool ring = local_positions.defined();
        const bool scaled = state.visits.front().key_scales.defined();
        const std::size_t first_visit = ring ? 4 : 3, per_visit = scaled ? 8 : 6;
        auto& step = state.replay_inputs;
        step.clear();
        step.reserve(first_visit + state.visits.size() * per_visit);
        step.push_back(ids);
        step.push_back(positions);
        step.push_back(conv_slot);
        if (ring) step.push_back(local_positions);
        for (const auto& visit : state.visits) {
            step.push_back(visit.key);
            step.push_back(visit.value);
            step.push_back(visit.key_conv);
            step.push_back(visit.value_conv);
            step.push_back(visit.attention_conv);
            step.push_back(visit.mlp_conv);
            if (scaled) {
                step.push_back(visit.key_scales);
                step.push_back(visit.value_scales);
            }
        }
        std::string key = "inkling2_decode:" + std::to_string(state.capacity) + (ring ? ":ring" : "") +
                          (scaled ? ":int8" : "");
        if (!select)
            key += ":logits";
        else if (*select == ops::TokenOutput::DEVICE)
            key += ":token_device";
        else
            key += ":token_host";
        if (hidden) key += ":hidden";
        const auto outputs = context.replay(key, step, [&](std::span<const Tensor> operands) {
            for (std::size_t index = 0; index < state.visits.size(); ++index) {
                const auto base = first_visit + index * per_visit;
                auto& visit = state.visits[index];
                visit.key = operands[base];
                visit.value = operands[base + 1];
                visit.key_conv = operands[base + 2];
                visit.value_conv = operands[base + 3];
                visit.attention_conv = operands[base + 4];
                visit.mlp_conv = operands[base + 5];
                if (scaled) {
                    visit.key_scales = operands[base + 6];
                    visit.value_scales = operands[base + 7];
                }
            }
            Tensor captured_hidden;
            auto result = execute_step(operands[0], operands[1], ring ? operands[3] : Tensor{}, operands[2], {}, state,
                                       1, false, select, hidden ? &captured_hidden : nullptr);
            if (hidden) return std::vector{std::move(result), std::move(captured_hidden)};
            return std::vector{std::move(result)};
        });
        if (hidden) *hidden = outputs[1];
        return outputs[0];
    }
    return execute_step(ids, positions, local_positions, conv_slot, tail_slots, state, T, all_logits, select, hidden);
}

auto Inkling2Impl::prefill(std::span<const std::int32_t> tokens, LanguageModelState& state) -> Result<void> {
    auto logits = forward(tokens, state);
    if (!logits) return std::unexpected(std::move(logits.error()));
    return {};
}

auto Inkling2Impl::stage_host_step(std::span<const std::int32_t> tokens, Inkling2State& state, bool all_logits,
                                   std::optional<ops::TokenOutput> select, Tensor* hidden) -> Result<Tensor> {
    try {
        if (tokens.empty() || state.position + tokens.size() > state.capacity)
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "Inkling-2 input exceeds state capacity"});
        const bool ring_local =
            ring_caches(device()) && state.local_capacity > 0 && state.local_capacity < state.capacity;
        auto max_step = MAX_STEP_TOKENS;
        if (ring_local && state.local_capacity > static_cast<std::size_t>(window_))
            max_step = std::min(max_step, state.local_capacity - static_cast<std::size_t>(window_));
        // Prepared CUDA operators keep output buffers for every step length they see, so arbitrary prompt tails would
        // accumulate memory. CUDA steps use power-of-two lengths of at least 16, plus one shorter tail.
        const auto step_length = [&](std::size_t remaining) {
            const auto length = std::min(remaining, max_step);
            return device().kind == tensor::DeviceKind::CUDA && length >= 16 && length < max_step
                       ? std::bit_floor(length)
                       : length;
        };
        if (step_length(tokens.size()) < tokens.size()) {
            std::vector<Tensor> outputs;
            std::vector<Tensor> hiddens;
            for (std::size_t offset = 0, count = 0; offset < tokens.size(); offset += count) {
                count = step_length(tokens.size() - offset);
                const bool last = offset + count == tokens.size();
                Tensor chunk_hidden;
                auto chunk_out = stage_host_step(tokens.subspan(offset, count), state, all_logits,
                                                 last ? select : std::nullopt, hidden ? &chunk_hidden : nullptr);
                if (!chunk_out) return chunk_out;
                if (all_logits || last) outputs.push_back(std::move(*chunk_out));
                if (hidden && (all_logits || last)) hiddens.push_back(std::move(chunk_hidden));
            }
            if (hidden) {
                *hidden = hiddens.size() == 1 ? std::move(hiddens.front()) : context_->concat(hiddens, 0);
                context_->synchronize();
            }
            if (outputs.size() == 1) return std::move(outputs.front());
            auto joined = context_->concat(outputs, 0);
            context_->synchronize();
            return joined;
        }
        auto& context = *context_;
        const auto T = static_cast<std::int64_t>(tokens.size());
        const auto start = state.position;
        // Multi-token all-logits steps are speculative verification blocks: keep enough CPU state to roll back
        // to any accepted prefix without disturbing the prompt rewind point.
        const bool record_rewind = device().kind == tensor::DeviceKind::CPU && all_logits && T > 1;
        for (auto& visit : state.visits) visit.record_rewind = record_rewind;
        if (record_rewind) snapshot_verification(state);
        else state.verify_end = std::min(state.verify_end, start);
        auto& step = T == 1 ? state.decode_step : state.prefill_step;
        auto& token_host = step_input(step.token, {T}, DType::I32);
        std::ranges::copy(tokens, require(token_host.data<std::int32_t>()).begin());
        auto& pos_host = step_input(step.positions, {T}, DType::I32);
        const auto pos_span = require(pos_host.data<std::int32_t>());
        for (std::int64_t t = 0; t < T; ++t) pos_span[t] = static_cast<std::int32_t>(start + t);
        Tensor local_pos_host;
        if (ring_local) {
            local_pos_host = step_input(step.local_positions, {T}, DType::I32);
            const auto lspan = require(local_pos_host.data<std::int32_t>());
            for (std::int64_t t = 0; t < T; ++t)
                lspan[t] = static_cast<std::int32_t>((start + static_cast<std::size_t>(t)) % state.local_capacity);
        }
        auto& slot_host = step_input(step.conv_slot, {1}, DType::I32);
        require(slot_host.data<std::int32_t>())[0] = static_cast<std::int32_t>(start % CONV_WIDTH);
        std::vector<Tensor> to_stage{token_host, pos_host, slot_host};
        if (T > 1) {
            const auto K = std::min<std::int64_t>(T, CONV_WIDTH);
            auto& tail_host = step_input(step.tail_slots, {K}, DType::I32);
            const auto tail_span = require(tail_host.data<std::int32_t>());
            for (std::int64_t i = 0; i < K; ++i)
                tail_span[i] = static_cast<std::int32_t>((start + T - K + i) % CONV_WIDTH);
            to_stage.push_back(tail_host);
        }
        if (ring_local) to_stage.push_back(local_pos_host);
        const auto staged = context.stage_inputs(to_stage);
        const Tensor tail_arg = T > 1 ? staged[3] : Tensor{};
        const Tensor local_pos_arg = ring_local ? staged.back() : Tensor{};
        auto output =
            run_forward(staged[0], staged[1], local_pos_arg, staged[2], tail_arg, state, T, all_logits, select, hidden);
        context.synchronize();
        state.position += tokens.size();
        if (record_rewind) state.verify_end = state.position;
        return output;
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

auto Inkling2Impl::forward(std::span<const std::int32_t> tokens, LanguageModelState& base, bool all_logits,
                           tensor::Tensor* hidden) -> Result<Tensor> {
    try {
        return stage_host_step(tokens, dynamic_cast<Inkling2State&>(base), all_logits, std::nullopt, hidden);
    } catch (const std::bad_cast&) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Inkling-2 state"});
    }
}

auto Inkling2Impl::select_token(std::span<const std::int32_t> tokens, LanguageModelState& base, bool all_tokens,
                                tensor::Tensor* hidden) -> Result<Tensor> {
    try {
        return stage_host_step(tokens, dynamic_cast<Inkling2State&>(base), all_tokens, ops::TokenOutput::HOST, hidden);
    } catch (const std::bad_cast&) {
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Inkling-2 state"});
    }
}

auto Inkling2Impl::forward_device_token(const Tensor& token, LanguageModelState& base) -> Result<Tensor> {
    try {
        auto& state = dynamic_cast<Inkling2State&>(base);
        if (token.dtype() != DType::I32 || token.dimensions() != 1 || token.size(0) != 1 ||
            token.device() != device() || state.position >= state.capacity)
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Inkling-2 device token or state"});
        auto& context = *context_;
        const auto start = state.position;
        auto& pos_host = step_input(state.decode_step.positions, {1}, DType::I32);
        require(pos_host.data<std::int32_t>())[0] = static_cast<std::int32_t>(start);
        auto& slot_host = step_input(state.decode_step.conv_slot, {1}, DType::I32);
        require(slot_host.data<std::int32_t>())[0] = static_cast<std::int32_t>(start % CONV_WIDTH);
        const bool ring_local =
            ring_caches(device()) && state.local_capacity > 0 && state.local_capacity < state.capacity;
        std::vector<Tensor> to_stage{pos_host, slot_host};
        if (ring_local) {
            auto& lpos = step_input(state.decode_step.local_positions, {1}, DType::I32);
            require(lpos.data<std::int32_t>())[0] = static_cast<std::int32_t>(start % state.local_capacity);
            to_stage.push_back(lpos);
        }
        const auto staged = context.stage_inputs(to_stage);
        if (weight("embed.weight").device() != device()) context.synchronize();
        const auto output_mode =
            supports_device_tokens() ? ops::TokenOutput::DEVICE : ops::TokenOutput::HOST;
        auto selected = run_forward(token, staged[0], ring_local ? staged[2] : Tensor{}, staged[1], {}, state, 1, false,
                                    output_mode, nullptr);
        state.position += 1;
        return require(require(selected.to(device())).reshape({1}));
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    } catch (const std::exception& error) {
        return std::unexpected(Error{ErrorCode::RUNTIME, error.what()});
    }
}

} // namespace kidi::model
