#include <metal_stdlib>
using namespace metal;
kernel void scatter_bytes(device uchar* destination [[buffer(0)]],
                          device const uchar* updates [[buffer(1)]],
                          device const int* indices [[buffer(2)]],
                          constant ulong4& sizes [[buffer(3)]],
                          device atomic_int* invalid [[buffer(4)]],
                          uint element [[thread_position_in_grid]]) {
    if (element >= sizes.w) return;
    for (ulong index = 0; index < sizes.z; ++index) {
        if (indices[index] < 0 || ulong(indices[index]) >= sizes.x) {
            if (element == 0) atomic_store_explicit(invalid, 1, memory_order_relaxed);
            return;
        }
    }
    const ulong row = element / sizes.y;
    const ulong index = row % sizes.z;
    for (ulong later = index + 1; later < sizes.z; ++later)
        if (indices[later] == indices[index]) return;
    const ulong batch = row / sizes.z;
    const ulong offset = (batch * sizes.x + ulong(indices[index])) * sizes.y + element % sizes.y;
    destination[offset] = updates[element];
}
