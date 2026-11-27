#include <metal_stdlib>
using namespace metal;

// Codebook tables `vq3d_lut[128]` and `vq4d_lut[128]` (float4) are prepended by quantized_linear.mm
// from ynn::vq3d_codebook() / ynn::vq4d_codebook() so CPU and Metal share the exact same constants.

constant uint VQ_DIM [[function_constant(0)]];

struct HadamardGeometry {
    uint rows, src_cols, dst_cols, block;
};

// Fast normalized Walsh-Hadamard transform along rows (supports block=64, block=32, or block=0 copy/pad).
// Can run in-place (src == dst when src_cols == dst_cols) or out-of-place with zero padding (dst_cols > src_cols).
kernel void hadamard_rows_f32(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                              constant HadamardGeometry& geom [[buffer(2)]],
                              uint2 group [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group.y;
    if (row >= geom.rows) return;
    if (geom.block == 0) {
        const uint col = group.x * 64 + tid;
        if (col < geom.dst_cols)
            dst[ulong(row) * geom.dst_cols + col] =
                col < geom.src_cols ? src[ulong(row) * geom.src_cols + col] : 0.0f;
        return;
    }
    const uint b = geom.block; // 64 or 32
    const uint c0 = group.x * b;
    if (c0 >= geom.src_cols) {
        const uint col = c0 + tid;
        if (tid < b && col < geom.dst_cols) dst[ulong(row) * geom.dst_cols + col] = 0.0f;
        return;
    }
    threadgroup float buf[64];
    float v = tid < b ? src[ulong(row) * geom.src_cols + c0 + tid] : 0.0f;

    // 5 warp-level butterfly stages (distances 1, 2, 4, 8, 16) via simd_shuffle_xor — zero shared memory!
    for (uint dist = 1; dist < 32; dist <<= 1) {
        const float partner = simd_shuffle_xor(v, dist);
        v = (lane & dist) ? (partner - v) : (v + partner);
    }
    if (b == 64) {
        buf[tid] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float other = buf[tid ^ 32];
        v = (simd == 0) ? (v + other) : (other - v);
        v *= 0.125f; // 1 / sqrt(64)
        dst[ulong(row) * geom.dst_cols + c0 + tid] = v;
    } else if (tid < 32) {
        v *= 0.176776695f; // 1 / sqrt(32)
        dst[ulong(row) * geom.dst_cols + c0 + tid] = v;
    }
    // Zero-pad any trailing columns when dst_cols > src_cols (last block only)
    if (c0 + b >= geom.src_cols && geom.dst_cols > geom.src_cols) {
        const uint pad_col = geom.src_cols + tid;
        if (pad_col < geom.dst_cols) dst[ulong(row) * geom.dst_cols + pad_col] = 0.0f;
    }
}

struct VqGeometry {
    uint tokens, rows, k_pad, n_groups, row_codes, top, experts;
    float gate_cap, up_cap;
};

struct ExpertBucketGeometry {
    uint pairs, experts, tile_size;
};

inline bool valid_expert_id(float id, uint experts) {
    return id >= 0.f && id < float(experts) && id == trunc(id);
}

// Per-expert bins followed by a compact table of nonempty [begin, end) tiles.
kernel void bucket_expert_routes(device const float* routes [[buffer(0)]], device uint* work [[buffer(1)]],
                                 constant ExpertBucketGeometry& geom [[buffer(2)]],
                                 device atomic_int* error [[buffer(3)]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint counts[256];
    for (uint e = tid; e < geom.experts; e += 128)
        atomic_store_explicit(counts + e, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint pair = tid; pair < geom.pairs; pair += 128) {
        const float id = routes[pair * 2];
        if (!valid_expert_id(id, geom.experts)) {
            atomic_store_explicit(error, 2, memory_order_relaxed);
            continue;
        }
        const auto expert = uint(id);
        const uint slot = atomic_fetch_add_explicit(counts + expert, 1u, memory_order_relaxed);
        work[expert * geom.pairs + slot] = pair;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        device uint* tiles = work + geom.experts * geom.pairs;
        uint n = 0;
        for (uint e = 0; e < geom.experts; ++e) {
            const uint end = e * geom.pairs + atomic_load_explicit(counts + e, memory_order_relaxed);
            for (uint i = e * geom.pairs; i < end; i += geom.tile_size) {
                tiles[1 + 2 * n] = i;
                tiles[2 + 2 * n] = min(end, i + geom.tile_size);
                ++n;
            }
        }
        tiles[0] = n;
    }
}

// Dot product of one 16-byte scale group (48 weights for VQ3D, 64 weights for VQ4D) against x_rot.
inline float vq_group_dot(const device uchar* gc, const device float* xb, threadgroup const float4* lut) {
    // Load all 16 codes in one 128-bit vector transaction.
    const uchar4 c0 = *(const device uchar4*)(gc);
    const uchar4 c1 = *(const device uchar4*)(gc + 4);
    const uchar4 c2 = *(const device uchar4*)(gc + 8);
    const uchar4 c3 = *(const device uchar4*)(gc + 12);
    const uchar codes[16] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w,
                             c2.x, c2.y, c2.z, c2.w, c3.x, c3.y, c3.z, c3.w};
    float acc = 0.0f;
    if (VQ_DIM == 3) {
#pragma unroll
        for (uint v = 0; v < 16; ++v) {
            const uchar code = codes[v];
            const float4 w = lut[code & 0x7Fu];
            const float d = fma(xb[3 * v], w.x, fma(xb[3 * v + 1], w.y, xb[3 * v + 2] * w.z));
            acc += (code & 0x80u) ? -d : d;
        }
    } else {
        const device packed_float4* x4 = (const device packed_float4*)xb;
#pragma unroll
        for (uint v = 0; v < 16; ++v) {
            const uchar code = codes[v];
            const float4 w = lut[code & 0x7Fu];
            const float d = dot(float4(x4[v]), w);
            acc += (code & 0x80u) ? -d : d;
        }
    }
    return acc;
}

// Routed MoE expert gate_up projection (rotated output): y_rot[pair, row] for pair = t * top + slot.
kernel void vq_expert_gate_rot(device const float* x_rot [[buffer(0)]], device const uchar* values [[buffer(1)]],
                               device const uchar* scales [[buffer(2)]], device const float* global_scale [[buffer(3)]],
                               device const float* routes [[buffer(4)]], device float* y_rot [[buffer(5)]],
                               constant VqGeometry& geom [[buffer(6)]],
                               uint2 group [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint simd [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float4 lut[128];
    lut[tid] = (VQ_DIM == 3) ? vq3d_lut[tid] : vq4d_lut[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = group.x * 4 + simd;
    const uint pair = group.y;
    const uint pairs = geom.tokens * geom.top;
    if (row >= geom.rows || pair >= pairs) return;

    const uint t = pair / geom.top;
    const uint expert = uint(routes[pair * 2]);
    const ulong row_idx = ulong(expert) * geom.rows + row;
    const device uchar* r_codes = values + row_idx * geom.row_codes;
    const device uchar* r_scales = scales + row_idx * geom.n_groups;
    const device float* xr = x_rot + ulong(t) * geom.k_pad;
    const uint g_elems = (VQ_DIM == 3) ? 48u : 64u;

    float total = 0.0f;
    for (uint g = lane; g < geom.n_groups; g += 32) {
        const float s = float(r_scales[g]);
        if (s != 0.0f) total = fma(vq_group_dot(r_codes + g * 16, xr + g * g_elems, lut), s, total);
    }
    const float sum = simd_sum(total) * global_scale[row_idx];
    if (lane == 0) y_rot[ulong(pair) * geom.rows + row] = sum;
}

// Softcapped SiLU(gate) * up * route_weight epilogue after un-rotating gate_up [pairs, 2*H] -> hidden [pairs, H].
kernel void vq_expert_silu_epilogue(device const float* gate_up [[buffer(0)]], device const float* routes [[buffer(1)]],
                                    device float* hidden [[buffer(2)]], constant VqGeometry& geom [[buffer(3)]],
                                    uint tid [[thread_position_in_grid]]) {
    const uint h = geom.rows / 2;
    const uint total = geom.tokens * geom.top * h;
    if (tid >= total) return;
    const uint pair = tid / h;
    const uint i = tid % h;
    const float g = gate_up[ulong(pair) * geom.rows + i];
    const float u = gate_up[ulong(pair) * geom.rows + h + i];
    const float gc = precise::tanh(g / geom.gate_cap) * geom.gate_cap;
    const float uc = precise::tanh(u / geom.up_cap) * geom.up_cap;
    const float w = routes[pair * 2 + 1];
    hidden[tid] = (gc / (1.0f + precise::exp(-g))) * uc * w;
}

// Routed MoE expert down projection: sums across all `top` experts in rotated space -> y_rot[t, row].
// Caller runs ONE hadamard_rows_f32 on y_rot afterward (saving top-1 output rotations!).
kernel void vq_expert_down_rot(device const float* x_rot [[buffer(0)]], device const uchar* values [[buffer(1)]],
                               device const uchar* scales [[buffer(2)]], device const float* global_scale [[buffer(3)]],
                               device const float* routes [[buffer(4)]], device float* y_rot [[buffer(5)]],
                               constant VqGeometry& geom [[buffer(6)]],
                               uint2 group [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint simd [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float4 lut[128];
    lut[tid] = (VQ_DIM == 3) ? vq3d_lut[tid] : vq4d_lut[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = group.x * 4 + simd;
    const uint t = group.y;
    if (row >= geom.rows || t >= geom.tokens) return;

    const uint g_elems = (VQ_DIM == 3) ? 48u : 64u;
    float out_sum = 0.0f;
    for (uint slot = 0; slot < geom.top; ++slot) {
        const uint pair = t * geom.top + slot;
        const uint expert = uint(routes[pair * 2]);
        const ulong row_idx = ulong(expert) * geom.rows + row;
        const device uchar* r_codes = values + row_idx * geom.row_codes;
        const device uchar* r_scales = scales + row_idx * geom.n_groups;
        const device float* xr = x_rot + ulong(pair) * geom.k_pad;
        float total = 0.0f;
        for (uint g = lane; g < geom.n_groups; g += 32) {
            const float s = float(r_scales[g]);
            if (s != 0.0f) total = fma(vq_group_dot(r_codes + g * 16, xr + g * g_elems, lut), s, total);
        }
        out_sum += simd_sum(total) * global_scale[row_idx];
    }
    if (lane == 0) y_rot[ulong(t) * geom.rows + row] = out_sum;
}

// Pack [row, code-within-group, group] so neighboring SIMD lanes can read adjacent vectors.
kernel void vq_interleave_groups(device const float* src [[buffer(0)]], device float4* target [[buffer(1)]],
                                constant VqGeometry& geom [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
    const uint v = p.x / geom.n_groups, group = p.x % geom.n_groups;
    const device float* x = src + ulong(p.y) * geom.k_pad + (group * 16 + v) * VQ_DIM;
    target[ulong(p.y) * 16 * geom.n_groups + p.x] = float4(x[0], x[1], x[2], VQ_DIM == 4 ? x[3] : 0.f);
}

inline float vq_interleaved_dot(float4 x, float4 y) {
    return VQ_DIM == 3 ? fma(x.x, y.x, fma(x.y, y.y, x.z * y.z)) : dot(x, y);
}

template <bool PAIRED>
inline float4 vq_interleaved_dots(device const uchar* c0, device const uchar* c1, device const float4* x0,
                                  device const float4* x1, threadgroup const float4* lut, uint stride) {
    const uint4 code0 = *reinterpret_cast<device const uint4*>(c0);
    const uint4 code1 = *reinterpret_cast<device const uint4*>(c1);
    float4 total = 0.f;
#pragma unroll
    for (uint v = 0; v < 16; ++v) {
        const uint c = (code0[v >> 2] >> ((v & 3) * 8)) & 255u;
        const uint d = (code1[v >> 2] >> ((v & 3) * 8)) & 255u;
        const float4 a = x0[v * stride], w0 = lut[c & 127u], w1 = lut[d & 127u];
        const float p0 = vq_interleaved_dot(a, w0), p1 = vq_interleaved_dot(a, w1);
        total.x += (c & 128u) ? -p0 : p0;
        total.y += (d & 128u) ? -p1 : p1;
        if (PAIRED) {
            const float4 b = x1[v * stride];
            const float p2 = vq_interleaved_dot(b, w0), p3 = vq_interleaved_dot(b, w1);
            total.z += (c & 128u) ? -p2 : p2;
            total.w += (d & 128u) ? -p3 : p3;
        }
    }
    return total;
}

kernel void vq_interleaved_expert(device const float4* x [[buffer(0)]], device const uchar* values [[buffer(1)]],
                                  device const uchar* scales [[buffer(2)]], device const float* global [[buffer(3)]],
                                  device const uint* work [[buffer(4)]], device float* output [[buffer(5)]],
                                  constant VqGeometry& geom [[buffer(6)]], constant uint& gated [[buffer(7)]],
                                  uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                  uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint pairs = geom.tokens * geom.top;
    const device uint* tiles = work + geom.experts * pairs;
    if (group.y >= tiles[0]) return;
    threadgroup float4 lut[128];
    lut[tid] = VQ_DIM == 3 ? vq3d_lut[tid] : vq4d_lut[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row = group.x * 8 + 2 * simd;
    if (row >= geom.rows) return;
    const uint next = min(row + 1, geom.rows - 1);
    const uint start = tiles[1 + 2 * group.y], end = tiles[2 + 2 * group.y];
    const bool two = start + 1 < end;
    const uint p0 = work[start], p1 = work[start + uint(two)];
    const uint divisor = gated ? geom.top : 1;
    const device float4* x0 = x + ulong(p0 / divisor) * 16 * geom.n_groups;
    const device float4* x1 = x + ulong(p1 / divisor) * 16 * geom.n_groups;
    const ulong expert = start / pairs;
    const ulong r0 = expert * geom.rows + row, r1 = expert * geom.rows + next;
    const device uchar* c0 = values + r0 * geom.row_codes;
    const device uchar* c1 = values + r1 * geom.row_codes;
    const device uchar* s0 = scales + r0 * geom.n_groups;
    const device uchar* s1 = scales + r1 * geom.n_groups;
    float4 total = 0.f;
    for (uint g = lane; g < geom.n_groups; g += 32) {
        if (!(s0[g] || s1[g])) continue;
        const auto d = two ? vq_interleaved_dots<true>(c0 + 16 * g, c1 + 16 * g, x0 + g, x1 + g, lut, geom.n_groups)
                           : vq_interleaved_dots<false>(c0 + 16 * g, c1 + 16 * g, x0 + g, x1 + g, lut, geom.n_groups);
        if (s0[g]) {
            total.x = fma(d.x, float(s0[g]), total.x);
            if (two) total.z = fma(d.z, float(s0[g]), total.z);
        }
        if (s1[g]) {
            total.y = fma(d.y, float(s1[g]), total.y);
            if (two) total.w = fma(d.w, float(s1[g]), total.w);
        }
    }
    const float u = simd_sum(total.x), v = simd_sum(total.y);
    const float a = simd_sum(total.z), b = simd_sum(total.w);
    if (lane == 0) {
        output[ulong(p0) * geom.rows + row] = gated ? u * global[r0] : u;
        if (next != row) output[ulong(p0) * geom.rows + next] = gated ? v * global[r1] : v;
        if (two) {
            output[ulong(p1) * geom.rows + row] = gated ? a * global[r0] : a;
            if (next != row) output[ulong(p1) * geom.rows + next] = gated ? b * global[r1] : b;
        }
    }
}

kernel void vq_reduce_expert_rows(device const float* partial [[buffer(0)]], device const float* global [[buffer(1)]],
                                  device const float* routes [[buffer(2)]], device float* output [[buffer(3)]],
                                  constant VqGeometry& geom [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= geom.tokens * geom.rows) return;
    const uint t = i / geom.rows, row = i % geom.rows;
    // Preserve the single-token kernel's scale multiply-add and route-slot order.
    float out_sum = 0.f;
    for (uint slot = 0; slot < geom.top; ++slot) {
        const uint pair = t * geom.top + slot;
        const float id = routes[pair * 2];
        if (!valid_expert_id(id, geom.experts)) return; // The bucket pass already recorded the error.
        out_sum += partial[ulong(pair) * geom.rows + row] * global[ulong(uint(id)) * geom.rows + row];
    }
    output[i] = out_sum;
}
