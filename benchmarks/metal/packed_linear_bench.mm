#include "kidi/runtime/mps/quantized_linear.h"
#include "kidi/ops/context.h"
#include "kidi/ops/quantization.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

namespace {
using namespace kidi;
using ops::require;
using tensor::Tensor;
auto median(std::vector<double> values) -> double {
    std::ranges::sort(values);
    return values[values.size() / 2];
}
auto benchmark(std::int64_t width, std::int64_t columns, int bits, int group) -> void {
    auto original = require(Tensor::empty({columns, width}, tensor::DType::F32));
    auto weights = require(original.data<float>());
    for (std::size_t index = 0; index < weights.size(); ++index)
        weights[index] = (static_cast<int>((index * 17) % 15) - 7) * 0.125F;
    auto packed = require(ops::pack_weight(original, bits, group));
    const auto device = tensor::Device::apple_gpu();
    packed.values = require(packed.values.to(device));
    packed.scales = require(packed.scales.to(device));
    auto input = require(Tensor::empty({1, width}, tensor::DType::F32, device));
    auto output = require(Tensor::empty({1, columns}, tensor::DType::F32, device));
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
            require(
                runtime::mps::encode_packed_linear(batch, input, packed.values, packed.scales, output, bits, group));
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
    for (auto column : {std::int64_t{0}, columns / 2, columns - 1}) {
        double expected = 0;
        for (std::int64_t channel = 0; channel < width; ++channel) {
            const auto offset = column * width + channel;
            const auto raw = (bytes[offset / (8 / bits)] >> ((offset % (8 / bits)) * bits)) & ((1 << bits) - 1);
            const auto weight = (raw ^ (1 << (bits - 1))) - (1 << (bits - 1));
            expected += inputs[channel] * weight * scales[column * (width / group) + channel / group];
        }
        if (!std::isfinite(result[column]) || std::abs(expected - result[column]) > 0.001 + std::abs(expected) * 0.0001)
            throw std::runtime_error("packed GEMV numeric mismatch");
    }
    std::cout << "packed|width=" << width << "|columns=" << columns << "|bits=" << bits << "|group=" << group
              << "|host_us=" << median(host) << "|gpu_us=" << median(gpu)
              << "|effective_GBps=" << (packed.values.nbytes() + packed.scales.nbytes()) / median(gpu) / 1000 << '\n';
}
} // namespace
auto main() -> int {
    try {
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