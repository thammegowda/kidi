// Hopper (sm_90a) W8A8 GEMM: TMA loads, warp-specialized producer/consumer warpgroups, asynchronous wgmma, and a
// fused asymmetric-activation epilogue. Adapted in compact form from vLLM's CUTLASS 3 scaled_mm/Machete kernels
// (Apache-2.0, Lucas Wilkinson and the vLLM contributors; see tahoma/libs/machete) on NVIDIA CUTLASS (BSD-3-Clause).
//
// out[m, n] = float(acc[m, n] - row_zero[m] * weight_sum[n]) * row_scale[m] * col_scale[n] + bias[n], in the same
// rounding order as the portable kernel in int8_gemm.cu so both paths produce identical results.

#include "cuda/int8_gemm.h"

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>

#include <stdexcept>
#include <string>

namespace kidi::cuda {
namespace {

using namespace cute;
namespace fusion = cutlass::epilogue::fusion;

template <template <class> class Op, class Output, class Input>
using Compute = fusion::Sm90Compute<Op, Output, Input, cutlass::FloatRoundStyle::round_to_nearest>;

template <class ElementD, class TileShape, class ClusterShape, class KernelSchedule, class EpilogueSchedule>
struct W8A8Sm90 {
    using RowVector = fusion::Sm90ColBroadcast<0, TileShape, float>;
    using RowZero = fusion::Sm90ColBroadcast<0, TileShape, int32_t>;
    using ColVector = fusion::Sm90RowBroadcast<0, TileShape, float>;
    using WeightSum = fusion::Sm90RowBroadcast<0, TileShape, int32_t>;
    using ZeroTerm = fusion::Sm90EVT<Compute<cutlass::multiplies, int32_t, int32_t>, RowZero, WeightSum>;
    using Centered = fusion::Sm90EVT<Compute<cutlass::minus, float, int32_t>, fusion::Sm90AccFetch, ZeroTerm>;
    using RowScaled = fusion::Sm90EVT<Compute<cutlass::multiplies, float, float>, Centered, RowVector>;
    using Output = fusion::Sm90EVT<Compute<cutlass::homogeneous_multiply_add, ElementD, float>, ColVector,
                                   RowScaled, ColVector>;

    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, int32_t, float, void, cutlass::layout::RowMajor, AlignmentD,
        ElementD, cutlass::layout::RowMajor, AlignmentD, EpilogueSchedule, Output>::CollectiveOp;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, int8_t, cutlass::layout::RowMajor, 16, int8_t,
        cutlass::layout::ColumnMajor, 16, int32_t, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
        cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>>;

    static void run(int rows, int columns, int width, const std::int8_t* input, const std::int8_t* weights,
                    const float* row_scales, const int* row_zeros, const int* weight_sums, const float* col_scales,
                    const float* bias, void* output, cudaStream_t stream) {
        using Kernel = typename Gemm::GemmKernel;
        // A [M,K], B [N,K] and D [M,N] are all packed with the last dimension contiguous.
        const typename Kernel::StrideA a_stride{static_cast<int64_t>(width), _1{}, int64_t{0}};
        const typename Kernel::StrideB b_stride{static_cast<int64_t>(width), _1{}, int64_t{0}};
        const typename Kernel::StrideD d_stride{static_cast<int64_t>(columns), _1{}, int64_t{0}};
        typename Output::Arguments fused{
            {col_scales}, {{{}, {{row_zeros}, {weight_sums}, {}}, {}}, {row_scales}, {}}, {bias}, {}};
        auto* d = static_cast<ElementD*>(output);
        typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                           {rows, columns, width, 1},
                                           {input, a_stride, weights, b_stride},
                                           {fused, nullptr, d_stride, d, d_stride}};
        Gemm gemm;
        if (gemm.can_implement(arguments) != cutlass::Status::kSuccess)
            throw std::runtime_error("CUDA: SM90 W8A8 GEMM cannot implement this shape");
        if (Gemm::get_workspace_size(arguments) != 0)
            throw std::runtime_error("CUDA: SM90 W8A8 GEMM unexpectedly requires a workspace");
        auto status = gemm.initialize(arguments, nullptr, stream);
        if (status == cutlass::Status::kSuccess) status = gemm.run(stream);
        if (status != cutlass::Status::kSuccess)
            throw std::runtime_error(std::string("CUDA: SM90 W8A8 GEMM failed: ") + cutlassGetStatusString(status));
    }
};

using Pingpong = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using PingpongEpilogue = cutlass::epilogue::TmaWarpSpecialized;
template <class ElementD>
using LargeTile = W8A8Sm90<ElementD, Shape<_128, _128, _128>, Shape<_2, _1, _1>, Pingpong, PingpongEpilogue>;

} // namespace

auto w8a8_gemm_sm90(int rows, int columns, int width, const std::int8_t* input, const std::int8_t* weights,
                    const float* row_scales, const int* row_zeros, const int* weight_sums, const float* col_scales,
                    const float* bias, void* output, bool bfloat_out, cudaStream_t stream) -> void {
    if (bfloat_out)
        LargeTile<cutlass::bfloat16_t>::run(rows, columns, width, input, weights, row_scales, row_zeros, weight_sums,
                                            col_scales, bias, output, stream);
    else
        LargeTile<float>::run(rows, columns, width, input, weights, row_scales, row_zeros, weight_sums, col_scales,
                              bias, output, stream);
}

} // namespace kidi::cuda
