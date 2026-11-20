#include "cuda/int8_gemm.h"

#include <cutlass/gemm/device/gemm.h>
#include <stdexcept>
#include <string>

namespace kidi::cuda {

auto launch_int8_gemm(int rows, int columns, int width, const std::int8_t* input, const std::int8_t* weights,
                      std::int32_t* output, cudaStream_t stream) -> void {
    using Gemm = cutlass::gemm::device::Gemm<
        std::int8_t, cutlass::layout::RowMajor, std::int8_t, cutlass::layout::ColumnMajor, std::int32_t,
        cutlass::layout::RowMajor, std::int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<32, 64, 64>, cutlass::gemm::GemmShape<16, 32, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>, cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 16, 16>;
    Gemm gemm;
    const typename Gemm::Arguments arguments{
        {rows, columns, width}, {input, width}, {weights, width}, {output, columns}, {output, columns}, {1, 0}};
    const auto status = gemm(arguments, nullptr, stream);
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error(std::string("CUDA: CUTLASS INT8 GEMM failed: ") + cutlassGetStatusString(status));
}

} // namespace kidi::cuda