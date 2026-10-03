#pragma once

#include "kidi/tensor/tensor.h"

namespace kidi::tensor {
struct WebGpuBufferView {
    std::uint32_t handle;
    std::size_t offset_bytes;
    /// Packed bit width whose kernel layout the buffer already holds (see `adopt_web_gpu_buffer`), else 0.
    std::int32_t packed_layout = 0;
};
auto web_gpu_buffer(const Tensor& tensor) -> Result<WebGpuBufferView>;
/// Wraps a buffer the JavaScript runtime already filled, taking ownership of its handle. `packed_layout` records
/// that 2- or 4-bit packed weights were uploaded in the kernels' interleaved layout.
auto adopt_web_gpu_buffer(std::uint32_t handle, std::vector<std::int64_t> shape, DType dtype,
                          std::int32_t packed_layout) -> Result<Tensor>;
/// Submits recorded GPU work without waiting. Results reach host tensors after JavaScript awaits the runtime once
/// the current Wasm call returns.
auto web_gpu_synchronize() -> Result<void>;
auto web_gpu_error() -> Error;
/// Waits for submitted GPU work and queued reads, returning nonzero on failure. Only test binaries linked with JSPI
/// install one. The browser runtime never waits inside Wasm, so without it synchronization only submits work and host
/// copies of WebGPU tensors fail.
using WebGpuWait = int (*)();
auto set_web_gpu_wait(WebGpuWait wait) -> void;
} // namespace kidi::tensor
