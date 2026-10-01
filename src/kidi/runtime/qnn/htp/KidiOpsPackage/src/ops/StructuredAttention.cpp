#include "AttentionHvx.h"
#include "AttentionMath.h"
#ifdef __hexagon__
#include "AttentionCrouton.h"
#endif

#include "HTP/core/float16.h"
#include "HTP/core/qhpi.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#define KIDI_STRINGIZE_DETAIL(value) #value
#define KIDI_STRINGIZE(value) KIDI_STRINGIZE_DETAIL(value)
#define KIDI_PACKAGE_NAME KIDI_STRINGIZE(THIS_PKG_NAME)

namespace {

using kidi::qnn::htp::online_softmax_weights;

struct AttentionView {
    Float16* output;
    const Float16* query;
    const Float16* current_key;
    const Float16* current_value;
    const std::uint8_t* key_cache;
    const std::uint8_t* value_cache;
    const std::int32_t* positions;
    std::uint32_t query_length;
    std::uint32_t query_heads;
    std::uint32_t head_dim;
    std::uint32_t capacity;
    float attention_scale;
    std::int32_t window;
    float key_scale;
    std::int32_t key_offset;
    float value_scale;
    std::int32_t value_offset;
};

auto direct_data(const QHPI_Tensor* tensor) -> const void* {
    return qhpi_tensor_is_indirect(tensor) ? nullptr : qhpi_tensor_raw_data(tensor);
}

auto direct_data(QHPI_Tensor* tensor) -> void* {
    return qhpi_tensor_is_indirect(tensor) ? nullptr : qhpi_tensor_raw_data(tensor);
}

auto make_view(std::uint32_t num_outputs, QHPI_Tensor** outputs, std::uint32_t num_inputs,
               const QHPI_Tensor* const* inputs, AttentionView& view, bool direct_inputs = true) -> bool {
    if (num_outputs != 1 || num_inputs != 8 || !outputs || !inputs) return false;
    const auto query_shape = qhpi_tensor_shape(inputs[0]);
    const auto key_shape = qhpi_tensor_shape(inputs[1]);
    const auto value_shape = qhpi_tensor_shape(inputs[2]);
    const auto key_cache_shape = qhpi_tensor_shape(inputs[3]);
    const auto value_cache_shape = qhpi_tensor_shape(inputs[4]);
    const auto positions_shape = qhpi_tensor_shape(inputs[5]);
    const auto output_shape = qhpi_tensor_shape(outputs[0]);
    if (query_shape.rank != 4 || key_shape.rank != 4 || value_shape.rank != 4 || key_cache_shape.rank != 4 ||
        value_cache_shape.rank != 4 || positions_shape.rank != 4 || output_shape.rank != 4)
        return false;

    view.query_length = query_shape.dims[1];
    view.query_heads = query_shape.dims[2];
    view.head_dim = query_shape.dims[3];
    view.capacity = key_cache_shape.dims[1];
    if (query_shape.dims[0] != 1 || view.query_length == 0 || view.query_heads == 0 ||
        (view.head_dim != 256 && view.head_dim != 512) || key_shape.dims[0] != 1 ||
        key_shape.dims[1] != view.query_length || key_shape.dims[2] != 1 || key_shape.dims[3] != view.head_dim ||
        value_shape.dims[0] != 1 || value_shape.dims[1] != view.query_length || value_shape.dims[2] != 1 ||
        value_shape.dims[3] != view.head_dim || key_cache_shape.dims[0] != 1 || key_cache_shape.dims[2] != 1 ||
        key_cache_shape.dims[3] != view.head_dim || value_cache_shape.dims[0] != 1 ||
        value_cache_shape.dims[1] != view.capacity || value_cache_shape.dims[2] != 1 ||
        value_cache_shape.dims[3] != view.head_dim || positions_shape.dims[3] != view.query_length ||
        output_shape.dims[0] != 1 || output_shape.dims[1] != view.query_length ||
        output_shape.dims[2] != view.query_heads || output_shape.dims[3] != view.head_dim)
        return false;
    view.output = static_cast<Float16*>(direct_data(outputs[0]));
    view.output = static_cast<Float16*>(direct_data(outputs[0]));
    view.query = static_cast<const Float16*>(direct_data(inputs[0]));
    view.current_key = static_cast<const Float16*>(direct_data(inputs[1]));
    view.current_value = static_cast<const Float16*>(direct_data(inputs[2]));
    view.key_cache = static_cast<const std::uint8_t*>(direct_data(inputs[3]));
    view.value_cache = static_cast<const std::uint8_t*>(direct_data(inputs[4]));
    view.positions = static_cast<const std::int32_t*>(direct_data(inputs[5]));
    const auto* scale = static_cast<const float*>(direct_data(inputs[6]));
    const auto* window = static_cast<const std::int32_t*>(direct_data(inputs[7]));
    if (!view.output || !view.key_cache || !view.value_cache || !view.positions || !scale || !window ||
        (direct_inputs && (!view.query || !view.current_key || !view.current_value)))
        return false;

    const auto key_quant = qhpi_tensor_quant_parameters(inputs[3]);
    const auto value_quant = qhpi_tensor_quant_parameters(inputs[4]);
    view.attention_scale = scale[0];
    view.window = window[0];
    view.key_scale = key_quant.stepsize;
    view.key_offset = -key_quant.zero_offset;
    view.value_scale = value_quant.stepsize;
    view.value_offset = -value_quant.zero_offset;
    return true;
}

auto task_range(std::uint32_t tasks, std::uint32_t slices,
                std::uint32_t slice) -> std::pair<std::uint32_t, std::uint32_t> {
    if (slices <= 1) return {0, tasks};
    const auto first = static_cast<std::uint32_t>((static_cast<std::uint64_t>(tasks) * slice) / slices);
    const auto end = static_cast<std::uint32_t>((static_cast<std::uint64_t>(tasks) * (slice + 1)) / slices);
    return {first, end - first};
}

auto dequantize(std::uint8_t value, float scale, std::int32_t offset) -> float {
    return scale * static_cast<float>(static_cast<std::int32_t>(value) + offset);
}

void structured_attention_scalar(const AttentionView& view, std::uint32_t first_task, std::uint32_t task_count) {
    const std::int32_t first_position = view.positions[0];
    for (std::uint32_t task = first_task; task < first_task + task_count; ++task) {
        const std::uint32_t query_index = task / view.query_heads;
        const std::uint32_t head = task % view.query_heads;
        const std::int32_t end = std::min(view.positions[query_index] + 1, static_cast<std::int32_t>(view.capacity));
        const std::int32_t start = view.window > 0 ? std::max<std::int32_t>(0, end - view.window) : 0;
        const auto* query_row =
            view.query + (query_index * view.query_heads + head) * static_cast<std::size_t>(view.head_dim);
        std::array<float, 512> accumulator{};
        float maximum = 0.0f;
        float denominator = 0.0f;
        for (std::int32_t token = start; token < end; ++token) {
            float score = 0.0f;
            if (token < first_position) {
                const auto* key_row = view.key_cache + static_cast<std::size_t>(token) * view.head_dim;
                for (std::uint32_t channel = 0; channel < view.head_dim; ++channel)
                    score += static_cast<float>(query_row[channel]) *
                             dequantize(key_row[channel], view.key_scale, view.key_offset);
            } else {
                const auto* key_row =
                    view.current_key + static_cast<std::size_t>(token - first_position) * view.head_dim;
                for (std::uint32_t channel = 0; channel < view.head_dim; ++channel)
                    score += static_cast<float>(query_row[channel]) * static_cast<float>(key_row[channel]);
            }
            score *= view.attention_scale;
            const auto weights = online_softmax_weights(maximum, score, denominator == 0.0f);
            denominator = denominator * weights.previous + weights.current;
            if (token < first_position) {
                const auto* value_row = view.value_cache + static_cast<std::size_t>(token) * view.head_dim;
                for (std::uint32_t channel = 0; channel < view.head_dim; ++channel)
                    accumulator[channel] =
                        accumulator[channel] * weights.previous +
                        weights.current * dequantize(value_row[channel], view.value_scale, view.value_offset);
            } else {
                const auto* value_row =
                    view.current_value + static_cast<std::size_t>(token - first_position) * view.head_dim;
                for (std::uint32_t channel = 0; channel < view.head_dim; ++channel)
                    accumulator[channel] = accumulator[channel] * weights.previous +
                                           weights.current * static_cast<float>(value_row[channel]);
            }
            maximum = weights.maximum;
        }
        const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
        auto* output_row =
            view.output + (query_index * view.query_heads + head) * static_cast<std::size_t>(view.head_dim);
        for (std::uint32_t channel = 0; channel < view.head_dim; ++channel)
            output_row[channel] = accumulator[channel] * inverse;
    }
}

auto structured_attention_hvx_kernel(QHPI_RuntimeHandle* handle, std::uint32_t num_outputs, QHPI_Tensor** outputs,
                                     std::uint32_t num_inputs, const QHPI_Tensor* const* inputs) -> std::uint32_t {
    AttentionView view{};
    if (!make_view(num_outputs, outputs, num_inputs, inputs, view)) return QHPI_ERROR_FATAL;
    const auto [first_task, task_count] =
        task_range(view.query_length * view.query_heads, qhpi_num_slices(handle), qhpi_slice_number(handle));
    if (task_count == 0) return QHPI_SUCCESS;
#ifdef __hexagon__
    if (view.head_dim == 256)
        kidi::qnn::htp::structured_attention_hvx<4>(
            view.output, view.query, view.current_key, view.current_value, view.key_cache, view.value_cache,
            view.positions, view.attention_scale, view.window, view.query_length, view.query_heads, view.capacity,
            view.key_scale, view.key_offset, view.value_scale, view.value_offset, first_task, task_count);
    else
        kidi::qnn::htp::structured_attention_hvx<8>(
            view.output, view.query, view.current_key, view.current_value, view.key_cache, view.value_cache,
            view.positions, view.attention_scale, view.window, view.query_length, view.query_heads, view.capacity,
            view.key_scale, view.key_offset, view.value_scale, view.value_offset, first_task, task_count);
#else
    structured_attention_scalar(view, first_task, task_count);
#endif
    return QHPI_SUCCESS;
}

auto structured_attention_crouton_kernel(QHPI_RuntimeHandle* handle, std::uint32_t num_outputs, QHPI_Tensor** outputs,
                                         std::uint32_t num_inputs, const QHPI_Tensor* const* inputs) -> std::uint32_t {
    AttentionView view{};
    if (!make_view(num_outputs, outputs, num_inputs, inputs, view, false)) return QHPI_ERROR_FATAL;
    const auto [first_task, task_count] =
        task_range(view.query_length * view.query_heads, qhpi_num_slices(handle), qhpi_slice_number(handle));
    if (task_count == 0) return QHPI_SUCCESS;
#ifdef __hexagon__
    const kidi::qnn::htp::Crouton16Reader query(inputs[0]);
    const kidi::qnn::htp::Crouton16Reader current_key(inputs[1]);
    const kidi::qnn::htp::Crouton16Reader current_value(inputs[2]);
    if (view.head_dim == 256)
        kidi::qnn::htp::structured_attention_crouton<4>(
            view.output, query, current_key, current_value, view.key_cache, view.value_cache, view.positions,
            view.attention_scale, view.window, view.query_length, view.query_heads, view.capacity, view.key_scale,
            view.key_offset, view.value_scale, view.value_offset, first_task, task_count);
    else
        kidi::qnn::htp::structured_attention_crouton<8>(
            view.output, query, current_key, current_value, view.key_cache, view.value_cache, view.positions,
            view.attention_scale, view.window, view.query_length, view.query_heads, view.capacity, view.key_scale,
            view.key_offset, view.value_scale, view.value_offset, first_task, task_count);
    return QHPI_SUCCESS;
#else
    (void)handle;
    return QHPI_UNSUPPORTED;
#endif
}

auto structured_attention_hvx_matches(const QHPI_Op*, std::uint32_t num_inputs,
                                      const QHPI_Tensor* const* inputs) -> std::uint32_t {
    if (num_inputs != 8 || !inputs) return 0;
    const auto query_shape = qhpi_tensor_shape(inputs[0]);
    return query_shape.rank == 4 && (query_shape.dims[3] == 256 || query_shape.dims[3] == 512);
}

auto structured_attention_cost(QHPI_RuntimeHandle*, std::uint32_t num_inputs,
                               const QHPI_Tensor* const* inputs) -> float {
    if (num_inputs != 8 || !inputs) return 1.0f;
    const auto query = qhpi_tensor_shape(inputs[0]);
    const auto cache = qhpi_tensor_shape(inputs[3]);
    if (query.rank != 4 || cache.rank != 4) return 1.0f;
    return static_cast<float>(query.dims[1]) * query.dims[2] * cache.dims[1] * query.dims[3] * 0.03125f;
}

QHPI_Tensor_Signature_v1 hvx_input_signatures[] = {
    {QHPI_FLOAT16, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_TCM_ONLY},
    {QHPI_FLOAT16, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_TCM_ONLY},
    {QHPI_FLOAT16, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_TCM_ONLY},
    {QHPI_QUINT8, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_QUINT8, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_ONLY},
    {QHPI_INT32, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_OR_TCM},
    {QHPI_FLOAT32, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_OR_TCM},
    {QHPI_INT32, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_OR_TCM},
};

QHPI_Tensor_Signature_v1 hvx_output_signatures[] = {
    {QHPI_FLOAT16, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_TCM_ONLY},
};

QHPI_Kernel_v1 kernel{};
QHPI_OpInfo_v1 operation{};

} // namespace

auto describeStructuredAttention() -> QHPI_OpInfo_v1 {
    kernel.function_name = KIDI_PACKAGE_NAME "::structured_attention_hvx";
    kernel.function = structured_attention_hvx_kernel;
    kernel.resources = QHPI_RESOURCE_HVX;
    kernel.multithreaded = true;
    kernel.min_inputs = 8;
    kernel.input_signature = hvx_input_signatures;
    kernel.min_outputs = 1;
    kernel.output_signature = hvx_output_signatures;
    kernel.cost_function = structured_attention_cost;
    kernel.predicate = structured_attention_hvx_matches;

    operation.name = KIDI_PACKAGE_NAME "::StructuredAttention";
    operation.num_kernels = 1;
    operation.kernels = &kernel;
    return operation;
}
