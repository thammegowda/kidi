#include "kidi/runtime/mps/quantized_linear.h"
#include "kidi/ops/context.h"
#include "kidi/ops/quantization.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>

namespace {
using namespace kidi;
using ops::require;
using tensor::Tensor;
auto median(std::vector<double> values) -> double {
    std::ranges::sort(values);
    return values[values.size() / 2];
}
auto benchmark(std::int64_t width, std::int64_t columns, int bits, int group, std::int64_t rows = 1) -> void {
    auto original = require(Tensor::empty({columns, width}, tensor::DType::F32));
    auto weights = require(original.data<float>());
    for (std::size_t index = 0; index < weights.size(); ++index)
        weights[index] = (static_cast<int>((index * 17) % 15) - 7) * 0.125F;
    auto packed = require(ops::pack_weight(original, bits, group));
    const auto device = tensor::Device::apple_gpu();
    packed.values = require(packed.values.to(device));
    packed.scales = require(packed.scales.to(device));
    auto input = require(Tensor::empty({rows, width}, tensor::DType::F32, device));
    auto output = require(Tensor::empty({rows, columns}, tensor::DType::F32, device));
    const float input_scale = rows > 1 ? 0.125F : 0.F, output_scale = rows > 1 ? 0.25F : 0.F;
    auto inputs = require(input.data<float>());
    for (std::size_t index = 0; index < inputs.size(); ++index)
        inputs[index] = (static_cast<int>(index % 17) - 8) * 0.125F;
    std::vector<double> host, gpu;
    for (int iteration = -4; iteration < 15; ++iteration) {
        const auto started = std::chrono::steady_clock::now();
        auto batch = require(runtime::mps::CommandBatch::create());
        MPSCommandBuffer* wrapper = (__bridge MPSCommandBuffer*)batch.native_handle();
        id<MTLCommandBuffer> command = wrapper.commandBuffer;
        for (int invocation = 0; invocation < 16; ++invocation)
            require(runtime::mps::encode_packed_linear(batch, input, packed.values, packed.scales, output, bits, group,
                                                       input_scale, output_scale));
        require(batch.finish());
        if (iteration >= 0) {
            host.push_back(
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count() / 16);
            if (command.GPUEndTime <= command.GPUStartTime) throw std::runtime_error("missing GPU command timestamps");
            gpu.push_back((command.GPUEndTime - command.GPUStartTime) * 1e6 / 16);
        }
    }
    const auto bytes = require(packed.values.data<std::uint8_t>());
    const auto scales = require(packed.scales.data<float>());
    const auto result = require(output.data<float>());
    for (auto row : {std::int64_t{0}, rows - 1})
        for (auto column : {std::int64_t{0}, columns / 2, columns - 1}) {
            double expected = 0;
            for (std::int64_t channel = 0; channel < width; ++channel) {
                const auto offset = column * width + channel;
                const auto raw = (bytes[offset / (8 / bits)] >> ((offset % (8 / bits)) * bits)) & ((1 << bits) - 1);
                const auto weight = (raw ^ (1 << (bits - 1))) - (1 << (bits - 1));
                expected += inputs[row * width + channel] * weight * scales[column * (width / group) + channel / group];
            }
            if (output_scale)
                expected = std::clamp(std::nearbyint(expected / output_scale), -128., 127.) * output_scale;
            const auto tolerance = rows == 1 ? 0.001 + std::abs(expected) * 0.0001 : 0.01 + std::abs(expected) * 0.002;
            if (!std::isfinite(result[row * columns + column]) ||
                std::abs(expected - result[row * columns + column]) > tolerance)
                throw std::runtime_error("packed projection numeric mismatch");
        }
    std::cout << "packed|rows=" << rows << "|width=" << width << "|columns=" << columns << "|bits=" << bits
              << "|group=" << group << "|host_us=" << median(host) << "|gpu_us=" << median(gpu)
              << "|effective_GBps=" << (packed.values.nbytes() + packed.scales.nbytes()) / median(gpu) / 1000 << '\n';
    if (rows > 1) {
        ops::Context context(device);
        std::vector<double> cached;
        Tensor actual;
        for (int iteration = -4; iteration < 15; ++iteration) {
            const auto started = std::chrono::steady_clock::now();
            for (int invocation = 0; invocation < 16; ++invocation)
                actual =
                    context.packed_linear(input, packed.values, packed.scales, bits, group, input_scale, output_scale);
            context.synchronize();
            if (iteration >= 0)
                cached.push_back(
                    std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count() / 16);
        }
        const auto values = require(actual.data<float>());
        for (std::size_t index = 0; index < values.size(); ++index)
            if (!std::isfinite(values[index]) || std::abs(values[index] - result[index]) > output_scale)
                throw std::runtime_error("cached prefill differs from packed projection");
        std::cout << "cached_fp16|rows=" << rows << "|width=" << width << "|columns=" << columns << "|bits=" << bits
                  << "|host_us=" << median(cached) << '\n';
    }
}
auto cache_benchmark(std::int64_t rows) -> void {
    const auto device = tensor::Device::apple_gpu();
    auto source = require(Tensor::empty({1, rows, 512}, tensor::DType::F32, device));
    auto values = require(source.data<float>());
    std::iota(values.begin(), values.end(), 0.F);
    auto destination = require(Tensor::zeros({1, 2048, 512}, tensor::DType::F32, device));
    auto indices = require(Tensor::empty({rows}, tensor::DType::I32, device));
    auto index_values = require(indices.data<std::int32_t>());
    std::iota(index_values.begin(), index_values.end(), 128);
    for (bool contiguous : {false, true}) {
        std::vector<double> times;
        for (int iteration = -2; iteration < 7; ++iteration) {
            auto batch = require(runtime::mps::CommandBatch::create());
            MPSCommandBuffer* wrapper = (__bridge MPSCommandBuffer*)batch.native_handle();
            id<MTLCommandBuffer> command = wrapper.commandBuffer;
            for (int invocation = 0; invocation < 16; ++invocation) {
                if (contiguous)
                    require(batch.copy_slice_(source, destination, 1, source.nbytes(), destination.nbytes(),
                                              128 * 512 * sizeof(float)));
                else
                    require(batch.scatter_(destination, source, indices));
            }
            require(batch.finish());
            if (command.GPUEndTime <= command.GPUStartTime) throw std::runtime_error("missing GPU timestamps");
            if (iteration >= 0) times.push_back((command.GPUEndTime - command.GPUStartTime) * 1e6 / 16);
        }
        const auto actual = require(destination.data<float>()).subspan(128 * 512, source.numel());
        if (!std::ranges::equal(actual, values)) throw std::runtime_error("cache copy mismatch");
        std::cout << "cache|rows=" << rows << "|contiguous=" << contiguous << "|gpu_us=" << median(times) << '\n';
    }
}
} // namespace
auto main(int argc, char** argv) -> int {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "cache") {
            for (auto rows : {1, 128, 256, 512, 1024}) cache_benchmark(rows);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "prefill") {
            for (auto rows : {128, 256, 512, 1024}) {
                benchmark(1536, 256, 8, 1536, rows);
                benchmark(1536, 6144, 4, 1536, rows);
                benchmark(12288, 1536, 2, 12288, rows);
            }
            return 0;
        }
        benchmark(1536, 256, 8, 1536);
        benchmark(1536, 2048, 8, 1536);
        benchmark(1536, 6144, 4, 128);
        benchmark(1536, 12288, 4, 128);
        benchmark(12288, 1536, 4, 128);
        benchmark(1536, 262144, 8, 1536);
        benchmark(1536, 12288, 2, 1536);
        benchmark(12288, 1536, 2, 12288);
        benchmark(1536, 262144, 2, 1536);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}