/*
Adapted from https://github.com/ggml-org/llama.cpp
Revision: 680a036285273a3ff56032ec5d7f3352609eba4f (GGML 0.25.3).
Block layouts: ggml/src/ggml-common.h.
Reference dequantizers: ggml/src/ggml-quants.c.
FP16 conversion: ggml/src/ggml-impl.h.

Adaptations: C++ namespace, inline single-block functions, std::bit_cast instead
of union punning, and removal of unused encoding/runtime facilities. Upstream
naming is retained. The caller validates lengths and copies unaligned blocks.
When updating, compare these routines with the recorded revision and rerun the
weights import tests; do not silently substitute new quantization layouts.

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#include <bit>
#include <cstdint>
#include <cstring>

namespace ggml_read {

inline float fp16_to_fp32(uint16_t h) {
    const uint32_t w = (uint32_t)h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;
    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = 0x1.0p-112f;
    const float normalized_value = std::bit_cast<float>((two_w >> 4) + exp_offset) * exp_scale;
    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float denormalized_value = std::bit_cast<float>((two_w >> 17) | magic_mask) - 0.5f;
    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign | (two_w < denormalized_cutoff ? std::bit_cast<uint32_t>(denormalized_value)
                                                                : std::bit_cast<uint32_t>(normalized_value));
    return std::bit_cast<float>(result);
}

struct block_q4_0 {
    uint16_t d;
    uint8_t qs[16];
};
struct block_q4_1 {
    uint16_t d, m;
    uint8_t qs[16];
};
struct block_q5_0 {
    uint16_t d;
    uint8_t qh[4], qs[16];
};
struct block_q5_1 {
    uint16_t d, m;
    uint8_t qh[4], qs[16];
};
struct block_q8_0 {
    uint16_t d;
    int8_t qs[32];
};
static_assert(sizeof(block_q4_0) == 18 && sizeof(block_q4_1) == 20);
static_assert(sizeof(block_q5_0) == 22 && sizeof(block_q5_1) == 24 && sizeof(block_q8_0) == 34);

inline void dequantize(const block_q4_0& x, float* y) {
    const float d = fp16_to_fp32(x.d);
    for (int j = 0; j < 16; j++) {
        const int x0 = (x.qs[j] & 0x0F) - 8;
        const int x1 = (x.qs[j] >> 4) - 8;
        y[j] = x0 * d;
        y[j + 16] = x1 * d;
    }
}

inline void dequantize(const block_q4_1& x, float* y) {
    const float d = fp16_to_fp32(x.d), m = fp16_to_fp32(x.m);
    for (int j = 0; j < 16; j++) {
        const int x0 = x.qs[j] & 0x0F;
        const int x1 = x.qs[j] >> 4;
        y[j] = x0 * d + m;
        y[j + 16] = x1 * d + m;
    }
}

inline void dequantize(const block_q5_0& x, float* y) {
    const float d = fp16_to_fp32(x.d);
    uint32_t qh;
    memcpy(&qh, x.qh, sizeof(qh));
    for (int j = 0; j < 16; j++) {
        const uint8_t xh_0 = ((qh >> j) << 4) & 0x10;
        const uint8_t xh_1 = (qh >> (j + 12)) & 0x10;
        const int x0 = ((x.qs[j] & 0x0F) | xh_0) - 16;
        const int x1 = ((x.qs[j] >> 4) | xh_1) - 16;
        y[j] = x0 * d;
        y[j + 16] = x1 * d;
    }
}

inline void dequantize(const block_q5_1& x, float* y) {
    const float d = fp16_to_fp32(x.d), m = fp16_to_fp32(x.m);
    uint32_t qh;
    memcpy(&qh, x.qh, sizeof(qh));
    for (int j = 0; j < 16; j++) {
        const uint8_t xh_0 = ((qh >> j) << 4) & 0x10;
        const uint8_t xh_1 = (qh >> (j + 12)) & 0x10;
        const int x0 = (x.qs[j] & 0x0F) | xh_0;
        const int x1 = (x.qs[j] >> 4) | xh_1;
        y[j] = x0 * d + m;
        y[j + 16] = x1 * d + m;
    }
}

inline void dequantize(const block_q8_0& x, float* y) {
    const float d = fp16_to_fp32(x.d);
    for (int j = 0; j < 32; j++) y[j] = x.qs[j] * d;
}

} // namespace ggml_read