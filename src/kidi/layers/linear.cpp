#include "kidi/layers/linear.h"
#include "kidi/runtime/ynn/graph.h"
#include "kidi/search/hnsw.h"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <vector>

namespace kidi::layers {
using ops::require;
using tensor::DType;

struct LinearImpl::HnswState {
    search::HnswIndex index;
    Tensor indexed_weight;
    std::size_t ef_search;
    std::vector<search::HnswWorkspace> search_workspaces;
    struct Slot {
        Tensor weights, selected;
        std::vector<std::vector<std::uint32_t>> candidates;
        std::vector<std::int32_t> union_ids;
        std::vector<std::optional<Error>> errors;
    };
    std::map<std::size_t, Slot> slots;

    HnswState(search::HnswIndex idx, Tensor bound, std::size_t ef)
        : index(std::move(idx)), indexed_weight(std::move(bound)), ef_search(ef) {}
};

namespace {
auto check_precision(tensor::DType precision) -> void {
    if (precision != tensor::DType::F32 && precision != tensor::DType::BF16 && precision != tensor::DType::I8)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "unsupported parameter precision"});
}
auto parameter_device(bool packed) -> tensor::Device {
    const auto kind =
        module_dtype == tensor::DType::I8 || packed ? tensor::ParameterKind::QUANTIZED : tensor::ParameterKind::FLOAT;
    return module_device.parameter_device(tensor::ParameterRole::PROJECTION, kind);
}
auto search_setting(const char* name, std::size_t fallback, std::size_t maximum) -> std::size_t {
    const auto* value = std::getenv(name);
    if (!value) return fallback;
    const std::string_view text(value);
    std::size_t parsed = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (error != std::errc{} || end != text.data() + text.size() || !parsed || parsed > maximum)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT,
                            std::string(name) + " must be an integer between 1 and " + std::to_string(maximum)});
    return parsed;
}
} // namespace

auto QuantizedState::decode_vector(const Tensor& values, std::string_view name, tensor::Device target) const
    -> const Tensor& {
    if (decoded.defined()) return decoded;
    if (values.dtype() != tensor::DType::I8 || values.dimensions() != 1 || scale.dtype() != tensor::DType::F32 ||
        scale.numel() != 1)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid quantized vector: " + std::string(name)});
    const auto scale_bytes = require(scale.copy_to_host());
    const auto multiplier = *reinterpret_cast<const float*>(scale_bytes.data());
    if (!std::isfinite(multiplier) || multiplier <= 0)
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid quantized scale: " + std::string(name)});
    const auto source = require(values.copy_to_host());
    const auto integers = std::span(reinterpret_cast<const std::int8_t*>(source.data()), values.numel());
    std::vector<float> output(integers.size());
    for (std::size_t index = 0; index < integers.size(); ++index) output[index] = integers[index] * multiplier;
    decoded =
        require(Tensor::from_host({static_cast<std::int64_t>(output.size())}, std::span<const float>(output), target));
    return decoded;
}

LinearImpl::LinearImpl(std::int32_t input_size, std::int32_t output_size, bool transpose, bool bias,
                       std::int32_t packed_bits, bool quantized_bias, std::int32_t group_size)
    : packed_bits_(packed_bits),
      input_size_(input_size),
      group_size_(group_size ? group_size : input_size),
      transpose_(transpose),
      has_bias_(bias) {
    const auto storage = parameter_device(packed_bits != 0);
    if (packed_bits) {
        if ((packed_bits != 2 && packed_bits != 4 && packed_bits != 8) || !transpose || bias || group_size_ <= 0 ||
            group_size_ % (8 / packed_bits) || input_size % group_size_)
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "invalid packed linear construction"});
        register_parameter("weight", weight_, {output_size, input_size / (8 / packed_bits)}, tensor::DType::U8,
                           allocate_parameters, storage);
        register_parameter("weight_scale", packed_scale_, {output_size, input_size / group_size_}, tensor::DType::F32,
                           allocate_parameters, storage);
        register_parameter("input_activation_scale", input_scale_, {}, tensor::DType::F32, allocate_parameters,
                           tensor::Device::cpu());
        register_parameter("output_activation_scale", output_scale_, {}, tensor::DType::F32, allocate_parameters,
                           tensor::Device::cpu());
        return;
    }
    check_precision(module_dtype);
    register_parameter("weight", weight_,
                       transpose ? std::vector<std::int64_t>{output_size, input_size}
                                 : std::vector<std::int64_t>{input_size, output_size},
                       module_dtype, allocate_parameters, storage);
    if (quantized_bias && (!bias || module_dtype != tensor::DType::I8))
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "quantized linear bias requires INT8 weights and a bias"});
    if (bias && quantized_bias) {
        bias_qstate_.emplace();
        const auto vector_storage =
            module_device.parameter_device(tensor::ParameterRole::PROJECTION, tensor::ParameterKind::PREPARED_VECTOR);
        register_parameter("bias", bias_, {output_size}, tensor::DType::I8, allocate_parameters, vector_storage);
        register_parameter("bias_scale", bias_qstate_->scale, {1}, tensor::DType::F32, allocate_parameters,
                           vector_storage);
    } else if (bias) {
        register_parameter("bias", bias_, {output_size}, tensor::DType::F32, allocate_parameters, storage);
    } else if (module_dtype == tensor::DType::I8)
        bias_ = require(Tensor::zeros({output_size}, tensor::DType::F32, storage));
    if (module_dtype == tensor::DType::I8) {
        qstate_.emplace();
        register_parameter("scale", qstate_->scale, {output_size, 1}, tensor::DType::F32, allocate_parameters, storage);
        if (allocate_parameters) {
            const std::vector<float> ones(output_size, 1.F);
            const auto bytes = std::as_bytes(std::span(ones));
            require(qstate_->scale.copy_from_host(bytes));
            if (bias_qstate_) require(bias_qstate_->scale.copy_from_host(bytes.first(sizeof(float))));
        }
    }
}

LinearImpl::~LinearImpl() = default;

auto LinearImpl::bias() const -> const Tensor& {
    return bias_qstate_ ? bias_qstate_->decode_vector(bias_, "linear bias") : bias_;
}

auto LinearImpl::has_hnsw() const noexcept -> bool { return static_cast<bool>(hnsw_); }

auto LinearImpl::release_workspaces() const -> void {
    if (!hnsw_) return;
    hnsw_->slots.clear();
    hnsw_->search_workspaces.clear();
}

auto LinearImpl::check_hnsw_binding() const -> void {
    if (!hnsw_) return;
    if (weight_.storage_identity() != hnsw_->indexed_weight.storage_identity() ||
        weight_.storage_offset() != hnsw_->indexed_weight.storage_offset() ||
        weight_.dtype() != hnsw_->indexed_weight.dtype() ||
        !std::ranges::equal(weight_.shape(), hnsw_->indexed_weight.shape()))
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "output weights changed; reload the matching HNSW graph"});
}

auto LinearImpl::bind_hnsw(const std::filesystem::path& path, const Tensor& packed_zero) -> Result<void> {
    try {
        if (path.empty() || !transpose_ || has_bias_ || qstate_ || !device().capabilities().compute.host())
            throw ops::Failure({ErrorCode::UNSUPPORTED, "HNSW linear requires a transposed bias-free CPU projection"});
        auto index = [&]() -> search::HnswIndex {
            if (packed_bits_) {
                if (!input_scale_.defined() || input_scale_.numel() != 1)
                    throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "HNSW linear activation scale is not bound"});
                const float in_scale = require(std::as_const(input_scale_).data<float>())[0];
                return require(search::HnswIndex::load_quantized(path, weight_, packed_scale_, packed_zero,
                                                                 packed_bits_, group_size_, in_scale));
            }
            if (weight_.dtype() != DType::F32 && weight_.dtype() != DType::BF16)
                throw ops::Failure(
                    {ErrorCode::UNSUPPORTED, "HNSW head requires FP32, BF16, or packed quantized weights"});
            return require(search::HnswIndex::load(path, weight_));
        }();
        const auto vocabulary = static_cast<std::size_t>(weight_.size(0));
        const auto ef = search_setting("KIDI_HNSW_EF", index.ef_search(), vocabulary);
        if (index.candidates() > std::min<std::size_t>(1024, vocabulary))
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "HNSW LM-head candidate count exceeds 1024"});
        if (ef < index.candidates())
            throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "HNSW search effort is smaller than the candidate count"});
        require(runtime::ynn::reserve_thread_pool(runtime::ynn::thread_count()));
        hnsw_ = std::make_unique<HnswState>(std::move(index), weight_, ef);
        return {};
    } catch (const ops::Failure& error) {
        return std::unexpected(error.error());
    }
}

auto LinearImpl::forward(ops::Context& context, const Tensor& input) const -> Tensor {
    if (!weight_.defined() || (has_bias_ && !bias_.defined()))
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "linear state has not been initialized"});
    if (packed_bits_) {
        return context.packed_linear(input, weight_, packed_scale_, packed_bits_, group_size_,
                                     require(input_scale_.data<float>())[0], require(output_scale_.data<float>())[0]);
    }
    if (qstate_) return context.quantized_linear(input, weight_, qstate_->scale, bias(), transpose_);
    return context.linear(input, weight_, bias(), transpose_);
}

auto LinearImpl::select(ops::Context& context, const Tensor& query, ops::TokenOutput output) const -> Tensor {
    if (!hnsw_) return context.greedy_token(forward(context, query), output);
    check_hnsw_binding();
    if (!context.device().capabilities().compute.host() || query.device() != context.device() || !query.defined() ||
        !query.is_contiguous() || query.dimensions() < 2 || query.size(-1) != input_size_ || !query.numel() ||
        (query.dtype() != DType::F32 && query.dtype() != DType::BF16))
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "HNSW linear query shape, dtype, or device mismatch"});
    context.synchronize();
    context.profile_phase("lm_head");
    const auto rows = query.numel() / query.size(-1);
    const auto& index = hnsw_->index;
    const auto width = static_cast<std::size_t>(input_size_), candidates = index.candidates();
    const auto capacity = rows * candidates;
    auto& slot = hnsw_->slots[rows];
    if (!slot.selected.defined()) {
        slot.selected = require(Tensor::empty({static_cast<std::int64_t>(rows)}, DType::I32));
        slot.errors.resize(rows);
        if (!packed_bits_) {
            slot.candidates.resize(rows);
            slot.union_ids.reserve(capacity);
            for (auto& list : slot.candidates) list.reserve(candidates);
            slot.weights = require(
                Tensor::empty({static_cast<std::int64_t>(capacity), input_size_}, index.vectors().dtype(), device()));
        }
    }
    if (hnsw_->search_workspaces.size() < rows) hnsw_->search_workspaces.resize(rows);
    const auto activation =
        (!packed_bits_ && index.vectors().dtype() == DType::BF16) ? context.cast(query, DType::BF16) : query;
    const auto floating = activation.dtype() == DType::F32 ? activation : context.cast(activation, DType::F32);
    context.synchronize();
    const auto queries = require(floating.data<float>());
    auto selected = require(slot.selected.data<std::int32_t>());
    std::ranges::fill(slot.errors, std::nullopt);
    const auto search = [&](std::size_t row) {
        auto found = index.search(queries.subspan(row * width, width), hnsw_->search_workspaces[row], candidates,
                                  hnsw_->ef_search);
        if (!found) {
            slot.errors[row] = std::move(found.error());
            return;
        }
        if (packed_bits_) {
            selected[row] = static_cast<std::int32_t>(found->front().id);
            return;
        }
        auto& list = slot.candidates[row];
        list.clear();
        for (const auto& neighbor : *found) list.push_back(neighbor.id);
    };
    if (rows == 1)
        search(0);
    else
        require(runtime::ynn::parallel_for(rows, search));
    for (std::size_t row = 0; row < rows; ++row)
        if (slot.errors[row]) throw ops::Failure(std::move(*slot.errors[row]));
    if (packed_bits_) return slot.selected;
    const auto row_bytes = width * tensor::element_size(index.vectors().dtype());
    slot.union_ids.clear();
    for (std::size_t row = 0; row < rows; ++row)
        for (auto token : slot.candidates[row]) slot.union_ids.push_back(static_cast<std::int32_t>(token));
    std::ranges::sort(slot.union_ids);
    slot.union_ids.erase(std::unique(slot.union_ids.begin(), slot.union_ids.end()), slot.union_ids.end());
    const auto weight_bytes = require(slot.weights.host_bytes());
    require(
        tensor::gather_rows(index.vectors(), slot.union_ids, weight_bytes.first(slot.union_ids.size() * row_bytes)));
    for (std::size_t row = slot.union_ids.size(); row < capacity; ++row)
        std::copy_n(weight_bytes.begin(), row_bytes, weight_bytes.begin() + row * row_bytes);
    const auto scores = context.cast(context.matmul(activation, slot.weights, true), DType::F32);
    context.synchronize();
    const auto score_values = require(scores.data<float>());
    for (std::size_t row = 0; row < rows; ++row) {
        float best = -std::numeric_limits<float>::infinity();
        selected[row] = -1;
        for (auto token : slot.candidates[row]) {
            const auto found = std::ranges::lower_bound(slot.union_ids, static_cast<std::int32_t>(token));
            if (found == slot.union_ids.end() || *found != static_cast<std::int32_t>(token))
                throw ops::Failure({ErrorCode::RUNTIME, "candidate is absent from the projection union"});
            const auto score = score_values[row * capacity + static_cast<std::size_t>(found - slot.union_ids.begin())];
            if (!std::isfinite(score)) throw ops::Failure({ErrorCode::RUNTIME, "nonfinite HNSW candidate score"});
            if (score > best || (score == best && static_cast<std::int32_t>(token) < selected[row])) {
                best = score;
                selected[row] = static_cast<std::int32_t>(token);
            }
        }
        if (selected[row] < 0) throw ops::Failure({ErrorCode::RUNTIME, "HNSW head selected no candidate"});
    }
    return slot.selected;
}

} // namespace kidi::layers
