#include <metal_stdlib>
using namespace metal;
struct ScatterGeometry {
    ulong capacity, row_bytes, indices, rows;
    uint copy_mode; // 0 = bytes, 1 = uint (4B), 2 = uint4 (16B)
};
kernel void scatter_bytes(device uchar* destination [[buffer(0)]],
                          device const uchar* updates [[buffer(1)]],
                          device const int* indices [[buffer(2)]],
                          constant ScatterGeometry& geometry [[buffer(3)]],
                          device atomic_int* invalid [[buffer(4)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint simd [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    if (row >= geometry.rows) return;
    threadgroup uint shared_flags[8];
    const ulong count = geometry.indices;
    bool bad = false;
    for (ulong k = tid; k < count; k += 256) {
        const int idx = indices[k];
        if (idx < 0 || ulong(idx) >= geometry.capacity) bad = true;
    }
    const bool bad_simd = simd_any(bad);
    if (lane == 0) shared_flags[simd] = bad_simd ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool any_bad = false;
    for (uint s = 0; s < 8; ++s) any_bad |= (shared_flags[s] != 0u);
    if (any_bad) {
        if (row == 0 && tid == 0) atomic_store_explicit(invalid, 1, memory_order_relaxed);
        return;
    }

    const ulong slot = row % count;
    const int target_idx = indices[slot];
    bool superseded = false;
    for (ulong later = slot + 1 + tid; later < count; later += 256)
        if (indices[later] == target_idx) superseded = true;
    const bool sup_simd = simd_any(superseded);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) shared_flags[simd] = sup_simd ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool any_sup = false;
    for (uint s = 0; s < 8; ++s) any_sup |= (shared_flags[s] != 0u);
    if (any_sup) return;

    const ulong batch = row / count;
    const ulong dst_off = (batch * geometry.capacity + ulong(target_idx)) * geometry.row_bytes;
    const ulong src_off = row * geometry.row_bytes;
    if (geometry.copy_mode == 2u) {
        device uint4* dst4 = (device uint4*)(destination + dst_off);
        device const uint4* src4 = (device const uint4*)(updates + src_off);
        const ulong n = geometry.row_bytes >> 4;
        for (ulong i = tid; i < n; i += 256) dst4[i] = src4[i];
    } else if (geometry.copy_mode == 1u) {
        device uint* dst1 = (device uint*)(destination + dst_off);
        device const uint* src1 = (device const uint*)(updates + src_off);
        const ulong n = geometry.row_bytes >> 2;
        for (ulong i = tid; i < n; i += 256) dst1[i] = src1[i];
    } else {
        device uchar* dst0 = destination + dst_off;
        device const uchar* src0 = updates + src_off;
        for (ulong i = tid; i < geometry.row_bytes; i += 256) dst0[i] = src0[i];
    }
}
