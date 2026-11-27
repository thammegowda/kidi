#pragma once

#include <cuda_bf16.h>
#include <cstdint>

namespace kidi::cuda::ptx {

template <typename Element, int Count>
struct alignas(sizeof(Element) * Count <= 16 ? sizeof(Element) * Count : 16) Fragment {
    Element values[Count];

    __device__ __forceinline__ void clear() {
#pragma unroll
        for (int index = 0; index < Count; ++index) values[index] = Element{0};
    }

    __device__ __forceinline__ Element& operator[](int index) { return values[index]; }
    __device__ __forceinline__ const Element& operator[](int index) const { return values[index]; }
};

template <bool CacheGlobal = false>
__device__ __forceinline__ void cp_async_zfill_16(void* destination, const void* source, bool valid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const auto shared_addr = static_cast<unsigned>(__cvta_generic_to_shared(destination));
    const int copy_bytes = valid ? 16 : 0;
    if constexpr (CacheGlobal) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :
                     : "r"(shared_addr), "l"(source), "r"(copy_bytes));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                     :
                     : "r"(shared_addr), "l"(source), "r"(copy_bytes));
    }
#else
    *static_cast<uint4*>(destination) = valid ? *static_cast<const uint4*>(source) : make_uint4(0, 0, 0, 0);
#endif
}

__device__ __forceinline__ void cp_async_4(void* destination, const void* source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const auto shared_addr = static_cast<unsigned>(__cvta_generic_to_shared(destination));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" : : "r"(shared_addr), "l"(source));
#else
    *static_cast<unsigned*>(destination) = *static_cast<const unsigned*>(source);
#endif
}

__device__ __forceinline__ void cp_async_fence() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int Remaining>
__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if constexpr (Remaining == 0) {
        asm volatile("cp.async.wait_all;\n" ::);
    } else {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(Remaining));
    }
#endif
}

template <bool Transpose = false>
__device__ __forceinline__ void ldmatrix_x4(Fragment<unsigned, 4>& destination, const void* shared_source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    const auto shared_addr = static_cast<unsigned>(__cvta_generic_to_shared(shared_source));
    if constexpr (Transpose) {
        asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(destination[0]), "=r"(destination[1]), "=r"(destination[2]), "=r"(destination[3])
                     : "r"(shared_addr));
    } else {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(destination[0]), "=r"(destination[1]), "=r"(destination[2]), "=r"(destination[3])
                     : "r"(shared_addr));
    }
#else
    (void)destination;
    (void)shared_source;
    __trap();
#endif
}

__device__ __forceinline__ void mma_bf16_m16n8k16(Fragment<float, 4>& d, const Fragment<unsigned, 4>& a,
                                                  const Fragment<unsigned, 2>& b, const Fragment<float, 4>& c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#else
    (void)d;
    (void)a;
    (void)b;
    (void)c;
    __trap();
#endif
}

__device__ __forceinline__ void mma_s8_m16n8k32(Fragment<int, 4>& d, const Fragment<unsigned, 4>& a,
                                                const Fragment<unsigned, 2>& b, const Fragment<int, 4>& c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
#else
    (void)d;
    (void)a;
    (void)b;
    (void)c;
    __trap();
#endif
}

} // namespace kidi::cuda::ptx
