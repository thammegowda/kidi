#include "kidi/runtime/mps/quantized_linear.h"
#include "kidi/tensor/metal.h"
#include "quantized_linear_metal_source.h"
#include "vq_kernels_metal_source.h"
#include "mpp_kernels_metal_source.h"
#include "ynnpack/composites/vq_linear.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <mutex>
#include <map>
#include <sstream>
#include <tuple>
#include <string>
#include <utility>

namespace kidi::runtime::mps {
namespace {
using tensor::Device;
using tensor::DType;
using tensor::Tensor;

constexpr std::size_t VQ_PAIRED_EXPERT_TILE = 2;

struct Pipelines {
    id<MTLDevice> device;
    id<MTLComputePipelineState> quantize;
    id<MTLComputePipelineState> linear_packed;
    id<MTLComputePipelineState> linear_tiled;
    id<MTLComputePipelineState> packed_i8w8;
    id<MTLComputePipelineState> packed_gate_up;
    id<MTLLibrary> library;
};

auto pipelines() -> Result<std::shared_ptr<Pipelines>> {
    static std::mutex mutex;
    static std::shared_ptr<Pipelines> cached;
    std::scoped_lock lock(mutex);
    if (cached) return cached;
    @autoreleasepool {
        auto result = std::make_shared<Pipelines>();
        result->device = MTLCreateSystemDefaultDevice();
        if (!result->device) return std::unexpected(Error{ErrorCode::UNSUPPORTED, "no Metal device"});
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        NSError* error = nil;
        id<MTLLibrary> library =
            [result->device newLibraryWithSource:[NSString stringWithUTF8String:KIDI_QUANTIZED_LINEAR_METAL_SOURCE]
                                         options:options
                                           error:&error];
        if (!library)
            return std::unexpected(Error{ErrorCode::RUNTIME, "compile INT8 Metal kernels: " +
                                                                 std::string(error.localizedDescription.UTF8String)});
        result->quantize =
            [result->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"quantize_rows"]
                                                          error:&error];
        result->linear_tiled =
            [result->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"linear_a8w8_tiled"]
                                                          error:&error];
        result->linear_packed =
            [result->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"linear_a8w8_packed"]
                                                          error:&error];
        result->packed_i8w8 =
            [result->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"packed_gemm_i8w8"]
                                                          error:&error];
        result->packed_gate_up =
            [result->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"packed_gate_up_i8w8"]
                                                          error:&error];
        result->library = library;
        if (!result->quantize || !result->linear_tiled || !result->linear_packed || !result->packed_i8w8 ||
            !result->packed_gate_up)
            return std::unexpected(Error{ErrorCode::RUNTIME, "create INT8 Metal pipelines"});
        cached = result;
        return result;
    }
}
struct PackedPipelines {
    id<MTLComputePipelineState> gemv, gemm;
};
auto packed_pipelines(std::uint32_t bits, std::uint32_t group_size, bool output_scale)
    -> Result<std::shared_ptr<PackedPipelines>> {
    static std::mutex mutex;
    static std::map<std::tuple<std::uint32_t, std::uint32_t, bool>, std::shared_ptr<PackedPipelines>> cache;
    std::scoped_lock lock(mutex);
    const auto key = std::tuple{bits, group_size, output_scale};
    if (const auto found = cache.find(key); found != cache.end()) return found->second;
    auto shared = pipelines();
    if (!shared) return std::unexpected(std::move(shared.error()));
    MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
    [constants setConstantValue:&bits type:MTLDataTypeUInt atIndex:0];
    [constants setConstantValue:&group_size type:MTLDataTypeUInt atIndex:1];
    [constants setConstantValue:&output_scale type:MTLDataTypeBool atIndex:2];
    NSError* error = nil;
    auto result = std::make_shared<PackedPipelines>();
    id<MTLFunction> gemv = [(*shared)->library newFunctionWithName:@"packed_gemv"
                                                    constantValues:constants
                                                             error:&error];
    id<MTLFunction> gemm = [(*shared)->library newFunctionWithName:@"packed_gemm"
                                                    constantValues:constants
                                                             error:&error];
    if (!gemv || !gemm) return std::unexpected(Error{ErrorCode::RUNTIME, "specialize packed Metal functions"});
    result->gemv = [(*shared)->device newComputePipelineStateWithFunction:gemv error:&error];
    result->gemm = [(*shared)->device newComputePipelineStateWithFunction:gemm error:&error];
    if (!result->gemv || !result->gemm)
        return std::unexpected(Error{ErrorCode::RUNTIME, "compile packed Metal pipelines"});
    cache.emplace(key, result);
    return result;
}
auto bind(id<MTLComputeCommandEncoder> encoder, const Tensor& tensor, NSUInteger index) -> Result<void> {
    auto buffer = tensor::metal_buffer(tensor);
    if (!buffer) return std::unexpected(std::move(buffer.error()));
    [encoder setBuffer:(__bridge id<MTLBuffer>)buffer->handle offset:buffer->offset_bytes atIndex:index];
    return {};
}
void append_vq_codebooks(std::ostringstream& source) {
    source << "#include <metal_stdlib>\nusing namespace metal;\n";
    source << std::fixed << std::setprecision(8);
    const auto emit = [&](const char* name, const float (*table)[4]) {
        source << "constant float4 " << name << "[128] = {\n";
        for (int i = 0; i < 128; ++i)
            source << "float4(" << table[i][0] << "f," << table[i][1] << "f," << table[i][2] << "f," << table[i][3]
                   << "f),";
        source << "};\n";
    };
    emit("vq3d_lut", ynn::vq3d_codebook());
    emit("vq4d_lut", ynn::vq4d_codebook());
}

auto vq_library() -> Result<id<MTLLibrary>> {
    static std::mutex mutex;
    static id<MTLLibrary> cached = nil;
    std::scoped_lock lock(mutex);
    if (cached) return cached;
    auto shared = pipelines();
    if (!shared) return std::unexpected(std::move(shared.error()));
    std::ostringstream source;
    append_vq_codebooks(source);
    source << KIDI_VQ_KERNELS_METAL_SOURCE;
    @autoreleasepool {
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        NSError* error = nil;
        cached = [(*shared)->device newLibraryWithSource:[NSString stringWithUTF8String:source.str().c_str()]
                                                 options:options
                                                   error:&error];
        if (!cached)
            return std::unexpected(Error{
                ErrorCode::RUNTIME, "compile VQ Metal kernels: " + std::string(error.localizedDescription.UTF8String)});
        return cached;
    }
}

auto vq_pipeline(const char* kernel, std::uint32_t dim) -> Result<id<MTLComputePipelineState>> {
    static std::mutex mutex;
    static std::map<std::pair<std::string, std::uint32_t>, id<MTLComputePipelineState>> cache;
    std::scoped_lock lock(mutex);
    const auto key = std::pair{std::string(kernel), dim};
    if (const auto found = cache.find(key); found != cache.end()) return found->second;
    auto library = vq_library();
    if (!library) return std::unexpected(std::move(library.error()));
    MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
    [constants setConstantValue:&dim type:MTLDataTypeUInt atIndex:0];
    NSError* error = nil;
    id<MTLFunction> function = [*library newFunctionWithName:[NSString stringWithUTF8String:kernel]
                                              constantValues:constants
                                                       error:&error];
    id<MTLComputePipelineState> pipeline =
        function ? [[*library device] newComputePipelineStateWithFunction:function error:&error] : nil;
    if (!pipeline) return std::unexpected(Error{ErrorCode::RUNTIME, "create VQ Metal pipeline " + std::string(kernel)});
    cache.emplace(key, pipeline);
    return pipeline;
}

// Metal 4 (Apple M5+ Neural Accelerator) FP16-tile library; returns nil on pre-Metal-4 devices.
auto mpp_library() -> id<MTLLibrary> {
    static std::mutex mutex;
    static bool initialized = false;
    static id<MTLLibrary> cached = nil;
    std::scoped_lock lock(mutex);
    if (initialized) return cached;
    initialized = true;
    auto shared = pipelines();
    if (!shared) return nil;
    // MTLGPUFamilyMetal4 == 5002.
    if (![(*shared)->device supportsFamily:static_cast<MTLGPUFamily>(5002)]) return nil;
    std::ostringstream source;
    source << KIDI_MPP_KERNELS_METAL_SOURCE;
    @autoreleasepool {
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = static_cast<MTLLanguageVersion>((4 << 16) | 0);
        options.mathMode = MTLMathModeSafe;
        NSError* error = nil;
        cached = [(*shared)->device newLibraryWithSource:[NSString stringWithUTF8String:source.str().c_str()]
                                                 options:options
                                                   error:&error];
        if (!cached && error)
            std::fprintf(stderr, "[kidi] Metal 4 prefill library unavailable: %s\n",
                         error.localizedDescription.UTF8String);
        return cached;
    }
}
auto mpp_pipeline(id<MTLLibrary> library, const char* kernel) -> Result<id<MTLComputePipelineState>> {
    static std::mutex mutex;
    static std::map<std::pair<void*, std::string>, id<MTLComputePipelineState>> cache;
    std::scoped_lock lock(mutex);
    const auto key = std::pair{(__bridge void*)library, std::string(kernel)};
    if (const auto found = cache.find(key); found != cache.end()) return found->second;
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:kernel]];
    id<MTLComputePipelineState> pipeline =
        function ? [[library device] newComputePipelineStateWithFunction:function error:&error] : nil;
    if (!pipeline)
        return std::unexpected(
            Error{ErrorCode::RUNTIME, "create Metal 4 pipeline " + std::string(kernel) + ": " +
                                          (error ? std::string(error.localizedDescription.UTF8String) : "not found")});
    cache.emplace(key, pipeline);
    return pipeline;
}

struct HadamardGeometry {
    std::uint32_t rows, src_cols, dst_cols, block;
};

struct VqGeometry {
    std::uint32_t tokens, rows, k_pad, n_groups, row_codes, top, experts;
    float gate_cap, up_cap;
};

auto expert_tiles(std::size_t pairs, std::size_t experts, std::size_t tile) -> std::size_t {
    // Each expert takes one tile for its first pair, then another for every full tile of extra pairs.
    const auto active = std::min(pairs, experts);
    return active + (pairs - active) / tile;
}

auto encode_expert_buckets(CommandBatch& batch, id<MTLComputeCommandEncoder> encoder, const Tensor& routes,
                           Tensor& work, std::uint32_t experts, std::uint32_t tile) -> Result<void> {
    auto group = vq_pipeline("bucket_expert_routes", 3);
    if (!group) return std::unexpected(std::move(group.error()));
    auto error = batch.index_error_buffer();
    if (!error) return std::unexpected(std::move(error.error()));
    const struct {
        std::uint32_t pairs, experts, tile_size;
    } geom{static_cast<std::uint32_t>(routes.size(0) * routes.size(1)), experts, tile};
    [encoder setComputePipelineState:*group];
    auto status = bind(encoder, routes, 0);
    if (status) status = bind(encoder, work, 1);
    if (!status) return status;
    [encoder setBytes:&geom length:sizeof(geom) atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)*error offset:0 atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return {};
}

auto encode_hadamard_rows(id<MTLComputeCommandEncoder> encoder, const Tensor& src, Tensor& dst, std::uint32_t rows,
                          std::uint32_t src_cols, std::uint32_t dst_cols, std::uint32_t block) -> Result<void> {
    auto pipe = vq_pipeline("hadamard_rows_f32", 3);
    if (!pipe) return std::unexpected(std::move(pipe.error()));
    const HadamardGeometry geom{rows, src_cols, dst_cols, block};
    [encoder setComputePipelineState:*pipe];
    auto status = bind(encoder, src, 0);
    if (status) status = bind(encoder, dst, 1);
    [encoder setBytes:&geom length:sizeof(geom) atIndex:2];
    const std::uint32_t tile = block ? block : 64u;
    const std::uint32_t groups_x = (dst_cols + tile - 1) / tile;
    if (status)
        [encoder dispatchThreadgroups:MTLSizeMake(groups_x, rows, 1)
                threadsPerThreadgroup:MTLSizeMake(block == 32 ? 32 : 64, 1, 1)];
    return status;
}
} // namespace

auto vq_expert_route_words(std::size_t tokens, std::size_t top, std::size_t experts) -> std::int64_t {
    if (tokens < 16 || tokens > 512 || !top || top > 16 || !experts || experts > 256) return 0;
    const auto pairs = tokens * top;
    if (pairs > 4096) return 0;
    return static_cast<std::int64_t>(experts * pairs + 1 + 2 * expert_tiles(pairs, experts, VQ_PAIRED_EXPERT_TILE));
}

auto encode_vq_experts(CommandBatch& batch, const Tensor& input, const Tensor& values, const Tensor& scales,
                       const Tensor& global, const Tensor& routes, Tensor& x_rot_scratch, Tensor& interleaved,
                       Tensor& pair_scratch, Tensor& route_scratch, Tensor& output, std::int32_t dim, bool gated,
                       float gate_cap, float up_cap) -> Result<void> {
    if ((dim != 3 && dim != 4) || input.dtype() != DType::F32 || output.dtype() != DType::F32 ||
        routes.dtype() != DType::F32 || values.dtype() != DType::U8 || scales.dtype() != DType::U8 ||
        global.dtype() != DType::F32 || values.dimensions() != 3 || scales.dimensions() != 3 ||
        routes.dimensions() != 3 || routes.size(2) != 2)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Metal VQ expert operands"});
    const auto tokens = static_cast<std::uint32_t>(routes.size(0));
    const auto top = static_cast<std::uint32_t>(routes.size(1));
    const auto experts = static_cast<std::uint32_t>(values.size(0));
    const auto rows = static_cast<std::uint32_t>(values.size(1));
    const auto cols = static_cast<std::uint32_t>(input.size(-1));
    const auto k_pad = static_cast<std::uint32_t>(ynn::vq_padded_cols(dim, cols));
    const auto had_col = static_cast<std::uint32_t>(ynn::vq_col_hadamard_block(cols));
    const auto had_row = static_cast<std::uint32_t>(ynn::vq_row_hadamard_block(rows));

    auto handle = batch.compute_encoder();
    if (!handle) return std::unexpected(std::move(handle.error()));
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)*handle;

    const std::uint32_t in_rows = gated ? tokens : (tokens * top);
    const Tensor* x_buf = &input;
    if (had_col != 0 || k_pad != cols) {
        auto st = encode_hadamard_rows(encoder, input, x_rot_scratch, in_rows, cols, k_pad, had_col);
        if (!st) return st;
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        x_buf = &x_rot_scratch;
    }

    const auto n_groups = static_cast<std::uint32_t>(scales.size(2));
    const VqGeometry geom{tokens, rows,    k_pad,    n_groups, static_cast<std::uint32_t>(values.size(2)),
                          top,    experts, gate_cap, up_cap};

    Result<void> status;
    if (interleaved.defined()) {
        auto pack = vq_pipeline("vq_interleave_groups", static_cast<std::uint32_t>(dim));
        if (!pack) return std::unexpected(std::move(pack.error()));
        [encoder setComputePipelineState:*pack];
        status = bind(encoder, *x_buf, 0);
        if (status) status = bind(encoder, interleaved, 1);
        if (!status) return status;
        [encoder setBytes:&geom length:sizeof(geom) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(16u * n_groups, in_rows, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

        status = encode_expert_buckets(batch, encoder, routes, route_scratch, experts, VQ_PAIRED_EXPERT_TILE);
        if (!status) return status;
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        auto projection = vq_pipeline("vq_interleaved_expert", static_cast<std::uint32_t>(dim));
        if (!projection) return std::unexpected(std::move(projection.error()));
        [encoder setComputePipelineState:*projection];
        status = bind(encoder, interleaved, 0);
        if (status) status = bind(encoder, values, 1);
        if (status) status = bind(encoder, scales, 2);
        if (status) status = bind(encoder, global, 3);
        if (status) status = bind(encoder, route_scratch, 4);
        if (status) status = bind(encoder, pair_scratch, 5);
        if (!status) return status;
        const std::uint32_t gate = gated;
        [encoder setBytes:&geom length:sizeof(geom) atIndex:6];
        [encoder setBytes:&gate length:sizeof(gate) atIndex:7];
        [encoder dispatchThreadgroups:MTLSizeMake((rows + 7) / 8,
                                                  expert_tiles(tokens * top, experts, VQ_PAIRED_EXPERT_TILE), 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        if (!gated) {
            auto reduce = vq_pipeline("vq_reduce_expert_rows", static_cast<std::uint32_t>(dim));
            if (!reduce) return std::unexpected(std::move(reduce.error()));
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [encoder setComputePipelineState:*reduce];
            status = bind(encoder, pair_scratch, 0);
            if (status) status = bind(encoder, global, 1);
            if (status) status = bind(encoder, routes, 2);
            if (status) status = bind(encoder, output, 3);
            if (!status) return status;
            [encoder setBytes:&geom length:sizeof(geom) atIndex:4];
            [encoder dispatchThreads:MTLSizeMake(output.numel(), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
    } else {
        auto pipe = vq_pipeline(gated ? "vq_expert_gate_rot" : "vq_expert_down_rot", static_cast<std::uint32_t>(dim));
        if (!pipe) return std::unexpected(std::move(pipe.error()));
        [encoder setComputePipelineState:*pipe];
        status = bind(encoder, *x_buf, 0);
        if (status) status = bind(encoder, values, 1);
        if (status) status = bind(encoder, scales, 2);
        if (status) status = bind(encoder, global, 3);
        if (status) status = bind(encoder, routes, 4);
        if (status) status = bind(encoder, gated ? pair_scratch : output, 5);
        if (!status) return status;
        [encoder setBytes:&geom length:sizeof(geom) atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake((rows + 3) / 4, gated ? (tokens * top) : tokens, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }

    if (gated) {
        if (had_row != 0) {
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            status = encode_hadamard_rows(encoder, pair_scratch, pair_scratch, tokens * top, rows, rows, had_row);
            if (!status) return status;
        }
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        auto ep = vq_pipeline("vq_expert_silu_epilogue", static_cast<std::uint32_t>(dim));
        if (!ep) return std::unexpected(std::move(ep.error()));
        [encoder setComputePipelineState:*ep];
        status = bind(encoder, pair_scratch, 0);
        if (status) status = bind(encoder, routes, 1);
        if (status) status = bind(encoder, output, 2);
        [encoder setBytes:&geom length:sizeof(geom) atIndex:3];
        if (status)
            [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(tokens) * top * (rows / 2), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else if (had_row != 0) {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        status = encode_hadamard_rows(encoder, output, output, tokens, rows, rows, had_row);
    }
    return status;
}

auto encode_relative_attention_mpp(CommandBatch& batch, std::span<const std::int64_t> attributes, TensorInputs inputs,
                                   Tensor& output) -> Result<bool> {
    id<MTLLibrary> mpp = mpp_library();
    if (!mpp) return false;
    const auto tokens = static_cast<std::uint32_t>(inputs[0].size(0));
    const auto heads = static_cast<std::uint32_t>(attributes[0]);
    const auto width = static_cast<std::uint32_t>(inputs[0].size(1) / heads);
    if (tokens < 16 || width != 128) return false;
    auto pipeline = mpp_pipeline(mpp, "relative_attention_mpp");
    if (!pipeline) return false;
    auto handle = batch.compute_encoder();
    if (!handle) return std::unexpected(std::move(handle.error()));
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)*handle;
    const auto as_float = [&](std::size_t index) {
        return std::bit_cast<float>(static_cast<std::int32_t>(attributes[index]));
    };
    const bool has_positions = inputs.size() == 7 || inputs.size() == 9;
    const bool has_scales = inputs.size() == 9;
    const std::uint32_t kv_dtype = inputs[1].dtype() == DType::F32 ? 0u : (inputs[1].dtype() == DType::BF16 ? 1u : 2u);
    const std::uint32_t scale_stride = has_scales ? static_cast<std::uint32_t>(inputs[7].size(1)) : 1u;
    const std::uint32_t scale_group =
        has_scales ? static_cast<std::uint32_t>(inputs[1].size(1) / inputs[7].size(1)) : 32u;
    if (has_scales && scale_group != 32u) return false;
    const std::uint32_t ring_mode = attributes.size() >= 8 && attributes[7] != 0 ? 1u : 0u;
    const struct {
        std::uint32_t tokens, heads, key_heads, start, window, extent, width, key_stride, capacity, has_positions;
        float log_alpha, log_floor;
        std::uint32_t kv_dtype, scale_stride, scale_group, ring_mode;
    } geometry{tokens,
               heads,
               static_cast<std::uint32_t>(attributes[1]),
               static_cast<std::uint32_t>(attributes[2]),
               static_cast<std::uint32_t>(attributes[3]),
               static_cast<std::uint32_t>(attributes[4]),
               width,
               static_cast<std::uint32_t>(inputs[1].size(1)),
               static_cast<std::uint32_t>(inputs[1].size(0)),
               static_cast<std::uint32_t>(has_positions),
               as_float(5),
               as_float(6),
               kv_dtype,
               scale_stride,
               scale_group,
               ring_mode};
    [encoder setComputePipelineState:*pipeline];
    Result<void> status;
    for (std::size_t index = 0; index < 6 && status; ++index) status = bind(encoder, inputs[index], index);
    if (status) status = bind(encoder, inputs[has_positions ? 6 : 0], 6);
    if (status) status = bind(encoder, output, 7);
    [encoder setBytes:&geometry length:sizeof(geometry) atIndex:8];
    if (status) status = bind(encoder, inputs[has_scales ? 7 : 0], 9);
    if (status) status = bind(encoder, inputs[has_scales ? 8 : 0], 10);
    if (status)
        [encoder dispatchThreadgroups:MTLSizeMake((tokens + 31) / 32, heads, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    if (!status) return std::unexpected(std::move(status.error()));
    return true;
}

auto encode_expand_packed_weight(CommandBatch& batch, const Tensor& weight, const Tensor& scales, Tensor& output,
                                 std::int32_t bits, std::int32_t group_size) -> Result<void> {
    if (output.dtype() != DType::F16 || output.dimensions() != 2 || output.numel() > UINT32_MAX ||
        (bits != 2 && bits != 4 && bits != 8) || group_size <= 0 || output.size(1) % group_size ||
        (weight.dtype() != DType::U8 && !(bits == 8 && weight.dtype() == DType::I8)) || weight.dimensions() != 2 ||
        scales.dtype() != DType::F32 || scales.dimensions() != 2 || weight.size(0) != output.size(0) ||
        weight.size(1) * (8 / bits) != output.size(1) || scales.size(0) != output.size(0) ||
        scales.size(1) != output.size(1) / group_size)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid streamed packed weight geometry"});
    static std::mutex mutex;
    static std::map<std::pair<std::uint32_t, std::uint32_t>, id<MTLComputePipelineState>> cached;
    id<MTLComputePipelineState> pipeline;
    {
        std::scoped_lock lock(mutex);
        const auto key = std::pair<std::uint32_t, std::uint32_t>{bits, group_size};
        if (const auto found = cached.find(key); found != cached.end())
            pipeline = found->second;
        else {
            auto shared = pipelines();
            if (!shared) return std::unexpected(std::move(shared.error()));
            MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
            [constants setConstantValue:&key.first type:MTLDataTypeUInt atIndex:0];
            [constants setConstantValue:&key.second type:MTLDataTypeUInt atIndex:1];
            NSError* error = nil;
            auto function = [(*shared)->library newFunctionWithName:@"expand_packed_weight"
                                                     constantValues:constants
                                                              error:&error];
            pipeline = function ? [(*shared)->device newComputePipelineStateWithFunction:function error:&error] : nil;
            if (!pipeline)
                return std::unexpected(Error{ErrorCode::RUNTIME, "compile streamed packed weight expansion"});
            cached.emplace(key, pipeline);
        }
    }
    const struct {
        std::uint32_t rows, width, columns;
        float input_scale, output_scale;
    } geometry{1, static_cast<std::uint32_t>(output.size(1)), static_cast<std::uint32_t>(output.size(0)), 0, 0};
    MPSCommandBuffer* buffer = (__bridge MPSCommandBuffer*)batch.native_handle();
    auto encoder = [buffer computeCommandEncoder];
    if (!encoder) return std::unexpected(Error{ErrorCode::RUNTIME, "create streamed packed weight encoder"});
    [encoder setComputePipelineState:pipeline];
    auto result = bind(encoder, weight, 0);
    if (result) result = bind(encoder, scales, 1);
    if (result) result = bind(encoder, output, 2);
    [encoder setBytes:&geometry length:sizeof(geometry) atIndex:3];
    if (result)
        [encoder dispatchThreads:MTLSizeMake(output.numel(), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    return result;
}

auto encode_packed_linear(CommandBatch& batch, const Tensor& input, const Tensor& weight, const Tensor& scales,
                          Tensor& output, std::int32_t bits, std::int32_t group_size, float input_scale,
                          float output_scale, bool vector_projection) -> Result<void> {
    const bool quantized_input = input.dtype() == DType::I8;
    if (!input.defined() || input.dimensions() < 2 || !input.size(-1) ||
        (input.dtype() != DType::F32 && !quantized_input) || (!quantized_input && input_scale != 0) ||
        (quantized_input && (bits != 8 || group_size != input.size(-1) || input_scale <= 0 || vector_projection ||
                             input.numel() / input.size(-1) < 4)))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid packed Metal linear input"});
    if (input.numel() > UINT32_MAX || output.numel() > UINT32_MAX || weight.numel() > UINT32_MAX)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "packed Metal linear exceeds indexing bounds"});
    id<MTLComputePipelineState> gemv = nil, gemm = nil;
    if (quantized_input) {
        auto shared = pipelines();
        if (!shared) return std::unexpected(std::move(shared.error()));
        gemm = (*shared)->packed_i8w8;
    } else {
        auto shared = packed_pipelines(bits, group_size, output_scale > 0);
        if (!shared) return std::unexpected(std::move(shared.error()));
        gemv = (*shared)->gemv;
        gemm = (*shared)->gemm;
    }
    const struct {
        std::uint32_t rows, width, columns;
        float input_scale, output_scale;
    } geometry{static_cast<std::uint32_t>(input.numel() / input.size(-1)), static_cast<std::uint32_t>(input.size(-1)),
               static_cast<std::uint32_t>(weight.size(0)), input_scale, output_scale};
    MPSCommandBuffer* buffer = (__bridge MPSCommandBuffer*)batch.native_handle();
    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
    if (!encoder) return std::unexpected(Error{ErrorCode::RUNTIME, "create packed projection encoder"});
    const bool tiled = geometry.rows >= 4 && !vector_projection;
    const bool wide_tiled = tiled && quantized_input;
    const auto columns_per_group = wide_tiled ? 64u : tiled ? 32u : 4u;
    [encoder setComputePipelineState:tiled ? gemm : gemv];
    auto status = bind(encoder, input, 0);
    if (status) status = bind(encoder, weight, 1);
    if (status) status = bind(encoder, scales, 2);
    if (status) status = bind(encoder, output, 3);
    [encoder setBytes:&geometry length:sizeof(geometry) atIndex:4];
    if (status)
        [encoder dispatchThreadgroups:MTLSizeMake((geometry.columns + columns_per_group - 1) / columns_per_group,
                                                  wide_tiled ? (geometry.rows + 63) / 64
                                                  : tiled    ? (geometry.rows + 31) / 32
                                                             : geometry.rows,
                                                  1)
                threadsPerThreadgroup:MTLSizeMake(32, wide_tiled ? 16 : 4, 1)];
    [encoder endEncoding];
    return status;
}

auto encode_packed_gate_up(CommandBatch& batch, const Tensor& input, const Tensor& weight, const Tensor& scales,
                           Tensor& output, float input_scale, float output_scale, float hidden_scale) -> Result<void> {
    if (!input.defined() || input.device() != Device::apple_gpu() || input.dtype() != DType::I8 ||
        input.dimensions() < 2 || !input.size(-1) || !input.numel() || input.numel() > UINT32_MAX ||
        !std::isfinite(input_scale) || input_scale <= 0 || !output.defined() ||
        output.device() != Device::apple_gpu() || output.dtype() != DType::I8 || output.numel() > UINT32_MAX ||
        !std::isfinite(hidden_scale) || hidden_scale <= 0)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid packed Metal gate/up activation"});
    if (!weight.defined() || weight.dtype() != DType::U8 || weight.dimensions() != 2 || !weight.size(0) ||
        weight.size(0) % 2 || scales.dtype() != DType::F32 || scales.dimensions() != 2 ||
        scales.size(0) != weight.size(0) || scales.size(1) != 1 || !std::isfinite(output_scale) || output_scale <= 0)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid packed Metal gate/up parameters"});
    const auto rows = input.numel() / input.size(-1);
    const auto width = input.size(-1);
    const auto columns = weight.size(0) / 2;
    auto expected_shape = std::vector<std::int64_t>(input.shape().begin(), input.shape().end());
    expected_shape.back() = columns;
    if (!std::ranges::equal(output.shape(), expected_shape))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid packed Metal gate/up output"});
    if (weight.size(1) != width)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid packed Metal gate/up parameters"});
    auto shared = pipelines();
    if (!shared) return std::unexpected(std::move(shared.error()));
    const struct {
        std::uint32_t rows, width, columns;
        float input_scale, output_scale, hidden_scale;
    } geometry{static_cast<std::uint32_t>(rows),
               static_cast<std::uint32_t>(width),
               static_cast<std::uint32_t>(columns),
               input_scale,
               output_scale,
               hidden_scale};
    MPSCommandBuffer* buffer = (__bridge MPSCommandBuffer*)batch.native_handle();
    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
    if (!encoder) return std::unexpected(Error{ErrorCode::RUNTIME, "create packed gate/up encoder"});
    [encoder setComputePipelineState:(*shared)->packed_gate_up];
    auto status = bind(encoder, input, 0);
    if (status) status = bind(encoder, weight, 1);
    if (status) status = bind(encoder, scales, 2);
    if (status) status = bind(encoder, output, 3);
    [encoder setBytes:&geometry length:sizeof(geometry) atIndex:4];
    if (status)
        [encoder dispatchThreadgroups:MTLSizeMake((columns + 63) / 64, (rows + 63) / 64, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 16, 1)];
    [encoder endEncoding];
    return status;
}

struct QuantizedLinear::Impl {
    std::shared_ptr<Pipelines> pipelines;
    Tensor weights, packed_weights, scales, bias, quantized, activation_scales, zeros;
    std::size_t width, columns, rows = 0;
};
QuantizedLinear::QuantizedLinear(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
QuantizedLinear::QuantizedLinear(QuantizedLinear&&) noexcept = default;
auto QuantizedLinear::operator=(QuantizedLinear&&) noexcept -> QuantizedLinear& = default;
QuantizedLinear::~QuantizedLinear() = default;
auto QuantizedLinear::create(const Tensor& weights, const Tensor& scales, const Tensor& bias)
    -> Result<QuantizedLinear> {
    if (!weights.defined() || weights.dtype() != DType::I8 || weights.dimensions() != 2 || weights.size(0) == 0 ||
        weights.size(0) > 65536 || weights.size(1) == 0 || weights.numel() > UINT32_MAX || !weights.is_contiguous() ||
        !scales.defined() || scales.dtype() != DType::F32 || scales.dimensions() != 2 ||
        scales.size(0) != weights.size(1) || scales.size(1) != 1 || !scales.is_contiguous() || !bias.defined() ||
        bias.dtype() != DType::F32 || bias.dimensions() != 1 || bias.size(0) != weights.size(1) ||
        !bias.is_contiguous())
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Metal quantized linear parameters"});
    auto shared = pipelines();
    if (!shared) return std::unexpected(std::move(shared.error()));
    auto impl = std::make_unique<Impl>();
    impl->pipelines = *shared;
    impl->width = weights.size(0);
    impl->columns = weights.size(1);
    auto device_weights = weights.to(Device::apple_gpu());
    auto device_scales = scales.to(Device::apple_gpu());
    auto device_bias = bias.to(Device::apple_gpu());
    if (!device_weights) return std::unexpected(std::move(device_weights.error()));
    if (!device_scales) return std::unexpected(std::move(device_scales.error()));
    if (!device_bias) return std::unexpected(std::move(device_bias.error()));
    impl->weights = std::move(*device_weights);
    impl->scales = std::move(*device_scales);
    impl->bias = std::move(*device_bias);
    return QuantizedLinear(std::move(impl));
}
auto QuantizedLinear::prepare(const Tensor& input) -> Result<void> {
    if (!input.defined() || input.device() != Device::apple_gpu() || input.dtype() != DType::F32 ||
        input.dimensions() < 2 || input.size(-1) != impl_->width || !input.is_contiguous() || input.numel() == 0 ||
        input.numel() > UINT32_MAX)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Metal quantized linear input"});
    auto rows = input.numel() / impl_->width;
    if (rows > UINT32_MAX / impl_->columns)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "Metal quantized linear output too large"});
    if (impl_->rows == rows) return {};
    if (rows < 4 && !impl_->packed_weights.defined()) {
        const auto stride = (impl_->width + 3) & ~std::size_t{3};
        if (impl_->columns > UINT32_MAX / stride)
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "packed Metal weights exceed indexing bounds"});
        auto packed = Tensor::zeros({static_cast<std::int64_t>(impl_->columns), static_cast<std::int64_t>(stride)},
                                    DType::I8, Device::apple_gpu());
        if (!packed) return std::unexpected(std::move(packed.error()));
        auto destination = packed->data<std::int8_t>();
        auto source = std::as_const(impl_->weights).data<std::int8_t>();
        if (!source) return std::unexpected(std::move(source.error()));
        if (!destination) return std::unexpected(std::move(destination.error()));
        for (std::size_t column_base = 0; column_base < impl_->columns; column_base += 32)
            for (std::size_t channel_base = 0; channel_base < impl_->width; channel_base += 32)
                for (std::size_t column = column_base; column < std::min(column_base + 32, impl_->columns); ++column)
                    for (std::size_t channel = channel_base; channel < std::min(channel_base + 32, impl_->width);
                         ++channel)
                        (*destination)[column * stride + channel] = (*source)[channel * impl_->columns + column];
        impl_->packed_weights = std::move(*packed);
    }
    auto quantized = Tensor::empty({static_cast<std::int64_t>(rows), static_cast<std::int64_t>(impl_->width)},
                                   DType::I8, Device::apple_gpu());
    auto scales = Tensor::empty({static_cast<std::int64_t>(rows)}, DType::F32, Device::apple_gpu());
    auto zeros = Tensor::empty({static_cast<std::int64_t>(rows)}, DType::I32, Device::apple_gpu());
    if (!quantized) return std::unexpected(std::move(quantized.error()));
    if (!scales) return std::unexpected(std::move(scales.error()));
    if (!zeros) return std::unexpected(std::move(zeros.error()));
    impl_->quantized = std::move(*quantized);
    impl_->activation_scales = std::move(*scales);
    impl_->zeros = std::move(*zeros);
    impl_->rows = rows;
    return {};
}
auto QuantizedLinear::encode(CommandBatch& batch, const Tensor& input, Tensor& output) -> Result<void> {
    auto status = prepare(input);
    if (!status) return status;
    auto expected_shape = std::vector<std::int64_t>(input.shape().begin(), input.shape().end());
    expected_shape.back() = static_cast<std::int64_t>(impl_->columns);
    if (!output.defined() || output.device() != Device::apple_gpu() || output.dtype() != DType::F32 ||
        !output.is_contiguous() || !std::ranges::equal(output.shape(), expected_shape))
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "invalid Metal quantized linear output"});
    MPSCommandBuffer* buffer = (__bridge MPSCommandBuffer*)batch.native_handle();
    const struct {
        std::uint32_t rows, width, columns;
    } geometry = {static_cast<std::uint32_t>(impl_->rows), static_cast<std::uint32_t>(impl_->width),
                  static_cast<std::uint32_t>(impl_->columns)};
    id<MTLComputeCommandEncoder> quantize = [buffer computeCommandEncoder];
    if (!quantize) return std::unexpected(Error{ErrorCode::RUNTIME, "create quantization encoder"});
    [quantize setComputePipelineState:impl_->pipelines->quantize];
    status = bind(quantize, input, 0);
    if (status) status = bind(quantize, impl_->quantized, 1);
    if (status) status = bind(quantize, impl_->activation_scales, 2);
    if (status) status = bind(quantize, impl_->zeros, 3);
    [quantize setBytes:&geometry length:sizeof(geometry) atIndex:4];
    if (status)
        [quantize dispatchThreadgroups:MTLSizeMake(impl_->rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    if (!status) {
        [quantize endEncoding];
        return status;
    }
    [quantize memoryBarrierWithScope:MTLBarrierScopeBuffers];
    id<MTLComputeCommandEncoder> linear = quantize;
    const bool tiled = impl_->rows >= 4;
    [linear setComputePipelineState:tiled ? impl_->pipelines->linear_tiled : impl_->pipelines->linear_packed];
    status = bind(linear, impl_->quantized, 0);
    if (status) status = bind(linear, tiled ? impl_->weights : impl_->packed_weights, 1);
    if (status) status = bind(linear, impl_->activation_scales, 2);
    if (status) status = bind(linear, impl_->zeros, 3);
    if (status) status = bind(linear, impl_->scales, 4);
    if (status) status = bind(linear, impl_->bias, 5);
    if (status) status = bind(linear, output, 6);
    [linear setBytes:&geometry length:sizeof(geometry) atIndex:7];
    if (status)
        [linear dispatchThreadgroups:MTLSizeMake(tiled ? (impl_->columns + 31) / 32 : (impl_->columns + 3) / 4,
                                                 tiled ? (impl_->rows + 31) / 32 : impl_->rows, 1)
               threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    [linear endEncoding];
    return status;
}
auto QuantizedLinear::run(const Tensor& input, Tensor& output) -> Result<void> {
    auto batch = CommandBatch::create();
    if (!batch) return std::unexpected(std::move(batch.error()));
    auto status = encode(*batch, input, output);
    if (!status) return status;
    return batch->finish();
}
} // namespace kidi::runtime::mps