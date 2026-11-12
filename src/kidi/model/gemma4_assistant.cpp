#include "kidi/model/gemma4_assistant.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace kidi::model {
using ops::require;
using tensor::DType;
using tensor::Tensor;
namespace {
KIDI_MODULE(AssistantVocabulary);
class AssistantVocabularyImpl : public Module {
public:
    AssistantVocabularyImpl(int hidden, int vocabulary, int centroids, int top_k)
        : centroids_(hidden, centroids, true, false),
          vocabulary_(vocabulary),
          centroids_count_(centroids),
          top_k_(top_k) {
        register_module("centroids", centroids_);
        register_parameter("token_ordering", ordering_, {vocabulary}, DType::I64);
        clusters_.resize(centroids);
        candidates_.reserve(static_cast<std::size_t>(top_k) * (vocabulary / centroids));
    }
    auto select(ops::Context& context, const Tensor& hidden, const layers::Linear& lm_head) -> std::int32_t {
        const auto logits = centroids_->forward(context, hidden);
        const auto values = require(logits.data<float>());
        if (std::ranges::any_of(values, [](float value) { return !std::isfinite(value); }))
            throw ops::Failure({ErrorCode::RUNTIME, "nonfinite Gemma MTP centroid scores"});
        std::iota(clusters_.begin(), clusters_.end(), 0);
        std::partial_sort(clusters_.begin(), clusters_.begin() + top_k_, clusters_.end(), [&](int left, int right) {
            return values[left] > values[right] || (values[left] == values[right] && left < right);
        });
        const auto ordering = require(std::as_const(ordering_).data<std::int64_t>());
        const auto per_cluster = vocabulary_ / centroids_count_;
        candidates_.clear();
        for (int cluster = 0; cluster < top_k_; ++cluster)
            for (int row = 0; row < per_cluster; ++row)
                candidates_.push_back(static_cast<std::int32_t>(ordering[clusters_[cluster] * per_cluster + row]));
        return lm_head->select(context, hidden, candidates_);
    }

private:
    layers::Linear centroids_;
    Tensor ordering_;
    int vocabulary_, centroids_count_, top_k_;
    std::vector<std::int32_t> clusters_, candidates_;
};
} // namespace

struct Gemma4AssistantImpl::State {
    ops::Context context;
    int hidden, backbone, vocabulary, window, maximum_position;
    layers::Linear pre_projection, post_projection, lm_head;
    layers::RmsNorm norm;
    ModuleList<layers::Gemma4BlockImpl> layers;
    AssistantVocabulary masked_embedding{nullptr};
    std::vector<int> kind;
    std::array<layers::KeyValue, 2> caches;
    std::array<Tensor, 2> masks;
    std::array<std::array<Tensor, 2>, 2> angles;
    std::array<std::size_t, 2> key_starts{};
    std::size_t prefix = 0, capacity = 0;

    explicit State(const YAML::Node& config)
        : context(module_device),
          hidden(config["text_config"]["hidden_size"].as<int>()),
          backbone(config["backbone_hidden_size"].as<int>()),
          vocabulary(config["text_config"]["vocab_size"].as<int>()),
          window(config["text_config"]["sliding_window"].as<int>()),
          maximum_position(config["text_config"]["max_position_embeddings"].as<int>()),
          pre_projection(2 * backbone, hidden, true, false),
          post_projection(hidden, backbone, true, false),
          lm_head(hidden, vocabulary, true, false),
          norm(hidden, config["text_config"]["rms_norm_eps"].as<float>()) {
        const auto text = config["text_config"];
        for (const auto& layer : text["layer_types"]) {
            const int type = layer.as<std::string>() == "full_attention" ? 1 : 0;
            kind.push_back(type);
            layers->push_back(layers::Gemma4Block(
                hidden, text["intermediate_size"].as<int>(), text["num_attention_heads"].as<int>(),
                text["num_key_value_heads"].as<int>(), text[type ? "global_head_dim" : "head_dim"].as<int>(), 0,
                text["rms_norm_eps"].as<float>(), true));
        }
        if (config["use_ordered_embeddings"].as<bool>(false))
            masked_embedding = AssistantVocabulary(hidden, vocabulary, config["num_centroids"].as<int>(),
                                                   config["centroid_intermediate_top_k"].as<int>());
    }
};

auto Gemma4AssistantImpl::checkpoint_config() -> checkpoint::ConfigAdapter {
    return {"config.json", [](const checkpoint::ConfigSource& source) -> Result<YAML::Node> {
                if (source.document["model_type"].as<std::string>("") != "gemma4_assistant")
                    return std::unexpected(Error{ErrorCode::UNSUPPORTED, "checkpoint is not a Gemma 4 assistant"});
                constexpr std::array<std::string_view, 1> candidates{"model.safetensors"};
                const auto weights = require(source.weights(candidates));
                if (weights["format"].as<std::string>() != "safetensors")
                    return std::unexpected(Error{ErrorCode::UNSUPPORTED, "Gemma MTP requires safetensors weights"});
                YAML::Node result;
                result["model"] = YAML::Clone(source.document);
                result["weights_file"] = weights["path"];
                return result;
            }};
}
auto Gemma4AssistantImpl::validate_config(const YAML::Node& config, const YAML::Node& target) -> Result<void> {
    try {
        const auto text = config["text_config"];
        if (config["model_type"].as<std::string>() != "gemma4_assistant" || config["quantization_config"] ||
            text["quantization_config"] || text["enable_moe_block"].as<bool>(false) ||
            text["attention_bias"].as<bool>(false) || text["attention_k_eq_v"].as<bool>(false) ||
            text["use_double_wide_mlp"].as<bool>(false) || text["hidden_size_per_layer_input"].as<int>(0) != 0 ||
            text["vocab_size_per_layer_input"].as<int>(0) != 0 ||
            text["hidden_activation"].as<std::string>() != "gelu_pytorch_tanh")
            return std::unexpected(Error{ErrorCode::UNSUPPORTED, "unsupported Gemma MTP assistant architecture"});
        for (const auto* name :
             {"hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
              "head_dim", "global_head_dim", "vocab_size", "sliding_window", "max_position_embeddings"})
            if (text[name].as<int>() <= 0)
                return std::unexpected(
                    Error{ErrorCode::INVALID_MANIFEST, "invalid Gemma MTP dimension: " + std::string(name)});
        if (config["backbone_hidden_size"].as<int>() != target["hidden_size"].as<int>() ||
            text["num_kv_shared_layers"].as<int>() != text["num_hidden_layers"].as<int>() ||
            text["num_attention_heads"].as<int>() % text["num_key_value_heads"].as<int>() ||
            text["layer_types"].size() != text["num_hidden_layers"].as<std::size_t>())
            return std::unexpected(
                Error{ErrorCode::INVALID_MANIFEST, "invalid Gemma MTP sharing or backbone dimensions"});
        for (const auto* name : {"num_key_value_heads", "head_dim", "global_head_dim", "vocab_size"})
            if (text[name].as<int>() != target[name].as<int>())
                return std::unexpected(
                    Error{ErrorCode::INVALID_MANIFEST, "Gemma MTP target mismatch: " + std::string(name)});
        const auto epsilon = text["rms_norm_eps"].as<float>();
        if (!std::isfinite(epsilon) || epsilon <= 0)
            return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "invalid Gemma MTP normalization epsilon"});
        for (const auto& layer : text["layer_types"]) {
            const auto type = layer.as<std::string>();
            if (type != "sliding_attention" && type != "full_attention")
                return std::unexpected(Error{ErrorCode::UNSUPPORTED, "unknown Gemma MTP attention type"});
            const auto rope = text["rope_parameters"][type], reference = target["rope_parameters"][type];
            if (rope["rope_theta"].as<float>() != reference["rope_theta"].as<float>() ||
                rope["partial_rotary_factor"].as<float>(1.F) != reference["partial_rotary_factor"].as<float>(1.F) ||
                rope["rope_type"].as<std::string>() != reference["rope_type"].as<std::string>())
                return std::unexpected(
                    Error{ErrorCode::UNSUPPORTED, "Gemma MTP rotary configuration differs from target"});
        }
        if (config["use_ordered_embeddings"].as<bool>(false)) {
            const auto clusters = config["num_centroids"].as<int>(),
                       top_k = config["centroid_intermediate_top_k"].as<int>();
            if (clusters <= 0 || text["vocab_size"].as<int>() % clusters || top_k <= 0 || top_k > clusters)
                return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, "invalid Gemma MTP vocabulary clusters"});
        }
        return {};
    } catch (const YAML::Exception& error) {
        return std::unexpected(Error{ErrorCode::INVALID_MANIFEST, error.what()});
    }
}
Gemma4AssistantImpl::Gemma4AssistantImpl(const YAML::Node& config, const YAML::Node& target_config) {
    require(validate_config(config, target_config));
    if (!device().capabilities().compute.host() || module_dtype != DType::F32)
        throw ops::Failure({ErrorCode::UNSUPPORTED, "Gemma MTP assistant currently requires CPU FP32 execution"});
    impl_ = std::make_unique<State>(config);
    register_module("pre_projection", impl_->pre_projection);
    register_module("post_projection", impl_->post_projection);
    register_module("lm_head", impl_->lm_head);
    register_module("norm", impl_->norm);
    register_module("layers", impl_->layers);
    if (impl_->masked_embedding) register_module("masked_embedding", impl_->masked_embedding);
}
Gemma4AssistantImpl::~Gemma4AssistantImpl() = default;
auto Gemma4AssistantImpl::create(const YAML::Node& config, const YAML::Node& target_config) -> Result<Gemma4Assistant> {
    try {
        return Gemma4Assistant(config, target_config);
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}
auto Gemma4AssistantImpl::set_checkpoint(const checkpoint::Weights& weights) -> Result<void> {
    try {
        StateDict state;
        const auto checkpoint = require(weights.state_dict());
        for (const auto& [key, tensor] : checkpoint) {
            auto name = key.starts_with("model.") ? key.substr(6) : key;
            if (name == "embed_tokens.weight") name = "lm_head.weight";
            auto value = tensor;
            if (value.dtype() != DType::I64) {
                if (value.dtype() != DType::F32 && value.dtype() != DType::BF16 && value.dtype() != DType::F16)
                    throw ops::Failure({ErrorCode::UNSUPPORTED, "Gemma MTP requires floating-point assistant weights"});
                if (value.dtype() != DType::F32) value = impl_->context.cast(value, DType::F32);
            }
            if (const auto existing = state.find(name); existing != state.end()) {
                if (!std::ranges::equal(require(existing->second.host_bytes()), require(value.host_bytes())))
                    throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "conflicting tied Gemma MTP head weights"});
            } else
                state.emplace(std::move(name), std::move(value));
        }
        for (std::size_t layer = 0; layer < impl_->layers->size(); ++layer) {
            const auto prefix = "layers." + std::to_string(layer) + ".mlp.";
            const auto gate = state.find(prefix + "gate_proj.weight"), up = state.find(prefix + "up_proj.weight");
            if (gate == state.end() || up == state.end())
                throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "missing Gemma MTP feed-forward weights"});
            const std::array pieces{gate->second, up->second};
            state.emplace(prefix + "gate_up_proj.weight", impl_->context.concat(pieces, 0));
            state.erase(gate);
            state.erase(up);
        }
        if (impl_->masked_embedding) {
            const auto ordering = state.find("masked_embedding.token_ordering");
            if (ordering == state.end() || ordering->second.dtype() != DType::I64 ||
                ordering->second.numel() != static_cast<std::size_t>(impl_->vocabulary))
                throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Gemma MTP token ordering"});
            std::vector<bool> seen(impl_->vocabulary);
            for (const auto token : require(std::as_const(ordering->second).data<std::int64_t>())) {
                if (token < 0 || token >= impl_->vocabulary || seen[token])
                    throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Gemma MTP token ordering is not a permutation"});
                seen[token] = true;
            }
        }
        impl_->context.synchronize();
        require(set_state(state));
        impl_->context = ops::Context(device());
        impl_->prefix = 0;
        return {};
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}
auto Gemma4AssistantImpl::forward(Gemma4Impl& target, const Tensor& hidden, std::int32_t token, Gemma4State& state)
    -> Result<Gemma4Draft> {
    try {
        if (target.device() != device() || !target.accelerator().empty() || !state.position ||
            state.position >= state.capacity || state.position >= static_cast<std::size_t>(impl_->maximum_position) ||
            (hidden.dtype() != DType::F32 && hidden.dtype() != DType::BF16) || hidden.dimensions() != 3 ||
            hidden.size(0) != 1 || hidden.size(1) != 1 || hidden.size(2) != impl_->backbone || token < 0 ||
            token >= impl_->vocabulary)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid Gemma MTP input or target state"});
        auto& context = impl_->context;
        if (impl_->prefix != state.position || impl_->capacity != state.capacity) {
            const auto extent = std::min(state.capacity, ((state.position + 127) / 128) * 128);
            for (int type = 0; type < 2; ++type) {
                const auto start = type == 0 && state.position > static_cast<std::size_t>(impl_->window) + 1
                                       ? ((state.position - impl_->window - 1) / 128) * 128
                                       : 0;
                std::vector<float> mask(extent - start, -1e9F);
                for (std::size_t column = start; column < state.position; ++column)
                    if (type || state.position - 1 - column <= static_cast<std::size_t>(impl_->window))
                        mask[column - start] = 0.F;
                impl_->masks[type] = require(
                    Tensor::from_host({1, 1, 1, static_cast<std::int64_t>(mask.size())}, std::span<const float>(mask)));
                impl_->key_starts[type] = start;
                impl_->angles[type] = target.assistant_positions(type, state.position);
                const auto& source = target.assistant_cache(state, type);
                if (source.key.dtype() == DType::BF16) {
                    const auto shape = std::vector<std::int64_t>{1, static_cast<std::int64_t>(extent),
                                                                 static_cast<std::int64_t>(source.key.size(2))};
                    const auto bytes = extent * source.key.size(2) * tensor::element_size(DType::BF16);
                    auto key = require(Tensor::from_bytes(shape, DType::BF16,
                                                          require(source.key.host_bytes()).first(bytes), device()));
                    auto value = require(Tensor::from_bytes(shape, DType::BF16,
                                                            require(source.value.host_bytes()).first(bytes), device()));
                    impl_->caches[type] = {context.cast(key, DType::F32), context.cast(value, DType::F32)};
                } else {
                    impl_->caches[type] = source;
                }
            }
            impl_->prefix = state.position;
            impl_->capacity = state.capacity;
        }
        const auto target_hidden = hidden.dtype() == DType::F32 ? hidden : context.cast(hidden, DType::F32);
        const std::array inputs{target.assistant_embedding(context, token), target_hidden};
        auto value = impl_->pre_projection->forward(context, context.concat(inputs, -1));
        // All drafts in a round share the target history and the same RoPE position, as trained by Gemma MTP.
        for (std::size_t layer = 0; layer < impl_->layers->size(); ++layer) {
            const auto type = impl_->kind[layer];
            value = impl_->layers->at(layer)->forward(context, value, {}, impl_->caches[type], state.position - 1,
                                                      impl_->masks[type], impl_->angles[type][0],
                                                      impl_->angles[type][1], impl_->key_starts[type]);
        }
        value = impl_->norm->forward(context, value);
        const auto projected = impl_->post_projection->forward(context, value);
        std::int32_t selected;
        if (impl_->masked_embedding)
            selected = impl_->masked_embedding->select(context, value, impl_->lm_head);
        else {
            const auto output = impl_->lm_head->select(context, value);
            selected = require(output.data<std::int32_t>())[0];
        }
        context.synchronize();
        if (selected < 0 || selected >= impl_->vocabulary)
            throw ops::Failure({ErrorCode::RUNTIME, "Gemma MTP returned an invalid draft token"});
        return Gemma4Draft{projected, selected};
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}
auto Gemma4AssistantImpl::draft(Gemma4Impl& target, const Tensor& hidden, std::int32_t last_token, Gemma4State& state,
                                std::size_t max_drafts, const inference::GreedyState::StopPredicate& is_stop,
                                std::vector<std::int32_t>& drafts) -> Result<void> {
    auto current_hidden = hidden;
    auto token = last_token;
    for (std::size_t index = 0; index < max_drafts; ++index) {
        auto output = forward(target, current_hidden, token, state);
        if (!output) return std::unexpected(output.error());
        current_hidden = std::move(output->hidden);
        token = output->token;
        drafts.push_back(token);
        if (is_stop && is_stop(token)) break;
    }
    return {};
}
auto Gemma4AssistantImpl::preparation_ns() const -> std::uint64_t { return impl_->context.preparation_ns(); }
auto Gemma4AssistantImpl::release_workspaces() -> void {
    impl_->context.release_workspaces();
    impl_->prefix = 0;
}
} // namespace kidi::model
