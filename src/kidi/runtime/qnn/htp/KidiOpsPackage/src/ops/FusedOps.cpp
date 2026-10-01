#include "FusedOps.h"

#include "HTP/core/float16.h"
#include "HTP/core/qhpi.h"

#include <cstdint>
#include <utility>

#define KIDI_STRINGIZE_DETAIL(value) #value
#define KIDI_STRINGIZE(value) KIDI_STRINGIZE_DETAIL(value)
#define KIDI_PACKAGE_NAME KIDI_STRINGIZE(THIS_PKG_NAME)

namespace {

using namespace kidi::qnn::htp;

auto element_count(const QHPI_Shape& shape) -> std::size_t {
    std::size_t count = 1;
    for (std::uint32_t dimension = 0; dimension < shape.rank; ++dimension) count *= shape.dims[dimension];
    return count;
}

auto direct(const QHPI_Tensor* tensor) -> void* {
    return qhpi_tensor_is_indirect(tensor) ? nullptr : qhpi_tensor_raw_data(tensor);
}

auto slice_range(std::size_t tasks, QHPI_RuntimeHandle* handle) -> std::pair<std::size_t, std::size_t> {
    const auto slices = qhpi_num_slices(handle);
    const auto slice = qhpi_slice_number(handle);
    if (slices <= 1) return {0, tasks};
    return {tasks * slice / slices, tasks * (slice + 1) / slices};
}

/// Inputs: FC output [.., 2I] holding gate then up codes. Output: GELU(gate) * up codes [.., I].
auto gelu_multiply_kernel(QHPI_RuntimeHandle* handle, std::uint32_t num_outputs, QHPI_Tensor** outputs,
                          std::uint32_t num_inputs, const QHPI_Tensor* const* inputs) -> std::uint32_t {
    if (num_inputs != 1 || num_outputs != 1) return QHPI_ERROR_FATAL;
    const auto input_shape = qhpi_tensor_shape(inputs[0]);
    const auto output_shape = qhpi_tensor_shape(outputs[0]);
    if (input_shape.rank == 0 || output_shape.rank != input_shape.rank) return QHPI_ERROR_FATAL;
    const auto width = output_shape.dims[output_shape.rank - 1];
    const auto rows = element_count(output_shape) / width;
    if (input_shape.dims[input_shape.rank - 1] != 2 * width || element_count(input_shape) != 2 * rows * width)
        return QHPI_ERROR_FATAL;
    const auto* source = static_cast<const std::uint8_t*>(direct(inputs[0]));
    auto* target = static_cast<std::uint8_t*>(direct(outputs[0]));
    if (!source || !target) return QHPI_ERROR_FATAL;
    const auto input_quant = qhpi_tensor_quant_parameters(inputs[0]);
    const auto output_quant = qhpi_tensor_quant_parameters(outputs[0]);

    constexpr std::size_t BLOCK = 1024;
    const bool vector = width % 128 == 0;
    const std::size_t blocks_per_row = (width + BLOCK - 1) / BLOCK;
    const auto [first, last] = slice_range(rows * blocks_per_row, handle);
    for (auto task = first; task < last; ++task) {
        const auto row = task / blocks_per_row;
        const auto begin = task % blocks_per_row * BLOCK;
        const auto count = std::min<std::size_t>(BLOCK, width - begin);
        const auto* gate = source + row * 2 * width + begin;
        const auto* up = gate + width;
        auto* output = target + row * width + begin;
#ifdef __hexagon__
        if (vector) {
            gelu_multiply_hvx(gate, up, output, count, input_quant.zero_offset, input_quant.stepsize,
                              output_quant.zero_offset, output_quant.stepsize);
            continue;
        }
#endif
        (void)vector;
        gelu_multiply_scalar(gate, up, output, count, input_quant.zero_offset, input_quant.stepsize,
                             output_quant.zero_offset, output_quant.stepsize);
    }
    return QHPI_SUCCESS;
}

/// Inputs: x, {epsilon, has_residual}, weight, residual, scale. Output: x * rsqrt(mean(x^2) + eps) * weight,
/// or (residual + that) * scale when has_residual is nonzero.
auto rms_norm_kernel(QHPI_RuntimeHandle* handle, std::uint32_t num_outputs, QHPI_Tensor** outputs,
                     std::uint32_t num_inputs, const QHPI_Tensor* const* inputs) -> std::uint32_t {
    if (num_outputs != 1 || num_inputs != 5) return QHPI_ERROR_FATAL;
    const auto shape = qhpi_tensor_shape(inputs[0]);
    if (shape.rank == 0) return QHPI_ERROR_FATAL;
    const auto width = shape.dims[shape.rank - 1];
    const auto rows = element_count(shape) / width;
    // The lowering only emits this op for widths that are multiples of one HVX vector.
    if (width % 64 != 0 || element_count(qhpi_tensor_shape(inputs[2])) != width ||
        element_count(qhpi_tensor_shape(outputs[0])) != rows * width)
        return QHPI_ERROR_FATAL;
    const auto* input = static_cast<const Float16*>(direct(inputs[0]));
    const auto* parameters = static_cast<const float*>(direct(inputs[1]));
    const auto* weight = static_cast<const Float16*>(direct(inputs[2]));
    const auto* residual = static_cast<const Float16*>(direct(inputs[3]));
    const auto* scale = static_cast<const Float16*>(direct(inputs[4]));
    auto* output = static_cast<Float16*>(direct(outputs[0]));
    if (!input || !parameters || !weight || !residual || !scale || !output ||
        element_count(qhpi_tensor_shape(inputs[1])) != 2)
        return QHPI_ERROR_FATAL;
    const float epsilon = parameters[0];
    if (parameters[1] == 0.0F) residual = nullptr;
    const float factor = static_cast<float>(scale[0]);

    const auto [first, last] = slice_range(rows, handle);
    for (auto row = first; row < last; ++row) {
        const auto offset = row * width;
        const auto* row_residual = residual ? residual + offset : nullptr;
#ifdef __hexagon__
        rms_norm_hvx(input + offset, weight, row_residual, factor, output + offset, width, epsilon);
#else
        rms_norm_scalar(input + offset, weight, row_residual, factor, output + offset, width, epsilon);
#endif
    }
    return QHPI_SUCCESS;
}

auto cost(QHPI_RuntimeHandle*, std::uint32_t num_inputs, const QHPI_Tensor* const* inputs) -> float {
    return num_inputs ? static_cast<float>(element_count(qhpi_tensor_shape(inputs[0]))) : 1.0F;
}

QHPI_Tensor_Signature_v1 gelu_inputs[] = {
    {QHPI_QUINT8, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_OR_TCM},
};
QHPI_Tensor_Signature_v1 gelu_outputs[] = {
    {QHPI_QUINT8, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_OR_TCM},
};
QHPI_Tensor_Signature_v1 norm_inputs[] = {
    {QHPI_FLOAT16, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_OR_TCM},
    {QHPI_FLOAT32, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_OR_TCM},
    {QHPI_FLOAT16, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_OR_TCM},
    {QHPI_FLOAT16, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_OR_TCM},
    {QHPI_FLOAT16, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_OR_TCM},
};
QHPI_Tensor_Signature_v1 norm_outputs[] = {
    {QHPI_FLOAT16, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_OR_TCM},
};

QHPI_Kernel_v1 gelu_kernel{};
QHPI_Kernel_v1 norm_kernel{};
QHPI_OpInfo_v1 operations[2]{};

} // namespace

auto describeFusedOps() -> std::pair<QHPI_OpInfo_v1, QHPI_OpInfo_v1> {
    gelu_kernel.function_name = KIDI_PACKAGE_NAME "::gelu_multiply_hvx";
    gelu_kernel.function = gelu_multiply_kernel;
    gelu_kernel.resources = QHPI_RESOURCE_HVX;
    gelu_kernel.multithreaded = true;
    gelu_kernel.min_inputs = 1;
    gelu_kernel.input_signature = gelu_inputs;
    gelu_kernel.min_outputs = 1;
    gelu_kernel.output_signature = gelu_outputs;
    gelu_kernel.cost_function = cost;

    norm_kernel.function_name = KIDI_PACKAGE_NAME "::rms_norm_hvx";
    norm_kernel.function = rms_norm_kernel;
    norm_kernel.resources = QHPI_RESOURCE_HVX;
    norm_kernel.multithreaded = true;
    norm_kernel.min_inputs = 5;
    norm_kernel.input_signature = norm_inputs;
    norm_kernel.min_outputs = 1;
    norm_kernel.output_signature = norm_outputs;
    norm_kernel.cost_function = cost;

    operations[0].name = KIDI_PACKAGE_NAME "::GeluMultiply";
    operations[0].num_kernels = 1;
    operations[0].kernels = &gelu_kernel;
    operations[1].name = KIDI_PACKAGE_NAME "::RmsNorm";
    operations[1].num_kernels = 1;
    operations[1].kernels = &norm_kernel;
    return {operations[0], operations[1]};
}
