#include "kidi/model/model.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "kidi/checkpoint/ggml/dequantize.h"
#include "kidi/ops/context.h"

namespace kidi::model {
namespace {

auto expand_to_float32(const tensor::Tensor& value) -> Result<tensor::Tensor> {
    if (!value.is_host_accessible()) {
        try {
            ops::Context context(value.device());
            auto cast = context.cast(value, tensor::DType::F32);
            context.synchronize();
            return cast;
        } catch (const ops::Failure& error) {
            return std::unexpected(error.error());
        }
    }
    const auto bytes = value.host_bytes();
    if (!bytes) return std::unexpected(bytes.error());
    auto output = tensor::Tensor::empty({value.shape().begin(), value.shape().end()}, tensor::DType::F32);
    if (!output) return std::unexpected(output.error());
    auto destination = output->data<float>();
    if (!destination) return std::unexpected(destination.error());
    for (std::size_t index = 0; index < destination->size(); ++index) {
        std::uint16_t bits = 0;
        std::memcpy(&bits, bytes->data() + index * sizeof(bits), sizeof(bits));
        (*destination)[index] = value.dtype() == tensor::DType::F16
                                    ? checkpoint::ggml::detail::fp16_to_fp32(bits)
                                    : std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
    }
    value.release_pages(false);
    return output;
}

auto normalize_for_slot(std::string_view name, tensor::Tensor value, const core::ParameterSpec& slot)
    -> Result<tensor::Tensor> {
    if (slot.dtype == tensor::DType::U8 && value.dtype() == tensor::DType::I8) {
        const auto bytes = std::as_const(value).host_bytes();
        if (!bytes) return std::unexpected(bytes.error());
        return tensor::Tensor::from_blob({value.shape().begin(), value.shape().end()}, tensor::DType::U8, *bytes,
                                         std::make_shared<tensor::Tensor>(std::move(value)));
    }
    const bool compact_scale = slot.dtype == tensor::DType::F32 && value.dtype() == tensor::DType::F16 &&
                               (name.ends_with("weight_scale") || name.ends_with("embedding_scale"));
    if (!compact_scale && slot.dtype == tensor::DType::F32 &&
        (value.dtype() == tensor::DType::F16 || value.dtype() == tensor::DType::BF16)) {
        return expand_to_float32(value);
    }
    if (!compact_scale && slot.dtype == tensor::DType::F32 && value.dtype() != tensor::DType::F32) {
        return std::unexpected(
            Error{ErrorCode::UNSUPPORTED, "unsupported checkpoint parameter dtype: " + std::string(name)});
    }
    return value;
}

} // namespace

auto Model::map_checkpoint(const checkpoint::Weights& weights) const -> Result<checkpoint::Weights> {
    const auto specs = state_mapping_specs();
    return weights.mapped(specs);
}

auto Model::prepare_checkpoint(const checkpoint::Weights& weights, StateDict* extras) const -> Result<StateDict> {
    auto mapped = map_checkpoint(weights);
    if (!mapped) return std::unexpected(std::move(mapped.error()));
    const auto slots = parameter_specs();
    StateDict state;
    for (const auto& name : mapped->names()) {
        auto value = mapped->tensor(name);
        if (!value) return std::unexpected(std::move(value.error()));
        const auto slot = slots.find(name);
        if (slot == slots.end()) {
            if (!extras)
                return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "unexpected checkpoint parameter: " + name});
            extras->emplace(name, std::move(*value));
            continue;
        }
        auto normalized = normalize_for_slot(name, std::move(*value), slot->second);
        if (!normalized) return std::unexpected(std::move(normalized.error()));
        state.emplace(name, std::move(*normalized));
    }
    return state;
}

auto Model::set_checkpoint(const checkpoint::Weights& weights) -> Result<void> {
    auto state = prepare_checkpoint(weights);
    if (!state) return std::unexpected(std::move(state.error()));
    return set_state(*state);
}

auto LanguageModel::rewind(LanguageModelState&, std::size_t) -> Result<void> {
    return std::unexpected(Error{ErrorCode::UNSUPPORTED, "model does not support KV cache rewind"});
}

auto LanguageModel::prefill(std::span<const std::int32_t> tokens, LanguageModelState& state) -> Result<void> {
    if (tokens.empty()) return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "prefill requires nonempty tokens"});
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        auto logits = forward(tokens.subspan(index, 1), state, false);
        if (!logits) return std::unexpected(std::move(logits.error()));
    }
    return {};
}

auto LanguageModel::select_token(std::span<const std::int32_t> tokens, LanguageModelState& state, bool all_tokens,
                                 tensor::Tensor* hidden) -> Result<tensor::Tensor> {
    auto logits = forward(tokens, state, all_tokens, hidden);
    if (!logits) return std::unexpected(std::move(logits.error()));
    try {
        ops::Context context(logits->device());
        const auto rows = all_tokens ? static_cast<std::int64_t>(tokens.size()) : std::int64_t{1};
        auto flat = context.reshape(*logits, {rows, vocabulary_size()});
        auto selected = context.greedy_token(flat);
        context.synchronize();
        return selected.to(tensor::Device::cpu());
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}

auto LanguageModel::forward_token(std::span<const std::int32_t> tokens, LanguageModelState& state,
                                  tensor::Tensor* hidden) -> Result<std::int32_t> {
    auto selected = select_token(tokens, state, false, hidden);
    if (!selected) return std::unexpected(std::move(selected.error()));
    const auto values = std::as_const(*selected).data<std::int32_t>();
    if (!values || values->empty() || (*values)[0] < 0)
        return std::unexpected(Error{ErrorCode::RUNTIME, "invalid selected token score"});
    return (*values)[0];
}

auto CausalLanguageModel::fork_state(const LanguageModelState&, std::size_t, std::size_t)
    -> Result<std::unique_ptr<LanguageModelState>> {
    return std::unexpected(Error{ErrorCode::UNSUPPORTED, "model does not support prefix cache forking"});
}

auto CausalLanguageModel::select_batch_tokens(std::span<const std::int32_t> tokens,
                                              std::span<LanguageModelState* const> states) -> Result<tensor::Tensor> {
    if (tokens.size() != states.size() || tokens.empty())
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "batch token count must match states"});
    std::vector<std::int32_t> chosen;
    chosen.reserve(tokens.size());
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (!states[index]) return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "null request state in batch"});
        auto token = forward_token(tokens.subspan(index, 1), *states[index]);
        if (!token) return std::unexpected(std::move(token.error()));
        chosen.push_back(*token);
    }
    return tensor::Tensor::from_host({static_cast<std::int64_t>(chosen.size())}, std::span<const std::int32_t>(chosen));
}

auto CausalLanguageModel::forward_device_token(const tensor::Tensor& token, LanguageModelState& state)
    -> Result<tensor::Tensor> {
    auto host = token.to(tensor::Device::cpu());
    if (!host) return std::unexpected(std::move(host.error()));
    const auto ids = std::as_const(*host).data<std::int32_t>();
    if (!ids || ids->size() != 1)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "expected scalar I32 token tensor"});
    auto selected = select_token(*ids, state, false);
    if (!selected) return std::unexpected(std::move(selected.error()));
    return selected->to(device());
}

auto CausalLanguageModel::forward_batch_tokens(std::span<const std::int32_t> tokens,
                                               std::span<LanguageModelState* const> states)
    -> Result<std::vector<std::int32_t>> {
    auto selected = select_batch_tokens(tokens, states);
    if (!selected) return std::unexpected(std::move(selected.error()));
    const auto values = std::as_const(*selected).data<std::int32_t>();
    if (!values || values->size() != tokens.size() ||
        std::ranges::any_of(*values, [](auto token) { return token < 0; }))
        return std::unexpected(Error{ErrorCode::RUNTIME, "invalid batched token selection"});
    return std::vector<std::int32_t>(values->begin(), values->end());
}

auto EncoderDecoderModel::forward(std::span<const std::int32_t> tokens, LanguageModelState& state, bool all_logits,
                                  tensor::Tensor* hidden) -> Result<tensor::Tensor> {
    if (all_logits || hidden)
        return std::unexpected(Error{ErrorCode::UNSUPPORTED, "encoder-decoder decode returns next-token logits only"});
    const auto* bound = dynamic_cast<const EncoderDecoderState*>(&state);
    if (!bound || !bound->memory)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "decoder state is not bound to encoder memory"});
    if (bound->batch != 1)
        return std::unexpected(
            Error{ErrorCode::INVALID_ARGUMENT, "single-sequence forward requires batch size 1; use decode_step"});
    return decode_step(*bound->memory, tokens, 1, &state, nullptr);
}

auto SpeechRecognitionModel::select_token(const tensor::Tensor& logits, const tensor::Tensor& mask) -> tensor::Tensor {
    ops::Context context(logits.device());
    return select_masked_token(context, logits, mask);
}

auto SpeechRecognitionModel::select_language(const tensor::Tensor& logits, const tensor::Tensor& allowed_ids)
    -> tensor::Tensor {
    ops::Context context(logits.device());
    return select_allowed_token(context, logits, allowed_ids);
}

auto SpeechRecognitionModel::select_masked_token(ops::Context& context, const tensor::Tensor& logits,
                                                 const tensor::Tensor& mask) const -> tensor::Tensor {
    auto selected = context.greedy_token(context.add(logits, mask));
    context.synchronize();
    return selected;
}

auto SpeechRecognitionModel::select_allowed_token(ops::Context& context, const tensor::Tensor& logits,
                                                  const tensor::Tensor& allowed_ids) const -> tensor::Tensor {
    const auto rows = context.reshape(logits, {vocabulary_size()});
    const auto allowed = context.gather(rows, allowed_ids);
    auto selected = context.greedy_token(context.reshape(allowed, {1, static_cast<std::int64_t>(allowed_ids.numel())}));
    context.synchronize();
    return selected;
}

} // namespace kidi::model
