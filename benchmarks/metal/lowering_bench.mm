#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ynnpack.h>

#include "kidi/runtime/ynn/graph.h"

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t INPUT_SIZE = 768;
constexpr std::size_t OUTPUT_SIZE = 64'000;
constexpr std::size_t CPU_THREADS = 8;
constexpr std::size_t DEFAULT_WARMUPS = 5;
constexpr std::size_t DEFAULT_ITERATIONS = 30;
constexpr std::size_t DEFAULT_INFLIGHT = 16;

struct Summary {
    double median_us;
    double p95_us;
};

auto parse_positive(const char* value, const char* name) -> std::size_t {
    const auto parsed = std::stoull(value);
    if (parsed == 0) throw std::invalid_argument(std::string(name) + " must be positive");
    return static_cast<std::size_t>(parsed);
}

auto summarize(std::vector<double> samples) -> Summary {
    std::ranges::sort(samples);
    const auto middle = samples.size() / 2;
    const auto median = samples.size() % 2 == 0 ? (samples[middle - 1] + samples[middle]) / 2.0 : samples[middle];
    const auto p95_index = static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(samples.size()))) - 1;
    return {.median_us = median, .p95_us = samples[p95_index]};
}

template <typename Function>
auto measure_us(Function&& function) -> double {
    const auto started = Clock::now();
    std::forward<Function>(function)();
    return std::chrono::duration<double, std::micro>(Clock::now() - started).count();
}

auto require(const kidi::Result<void>& result) -> void {
    if (!result) throw std::runtime_error(result.error().message);
}

auto make_values(std::size_t size, std::uint32_t seed, float scale) -> std::vector<float> {
    std::vector<float> result(size);
    for (std::size_t index = 0; index < size; ++index) {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        const auto unit = static_cast<float>(seed >> 8) / 8'388'607.5F - 1.0F;
        result[index] = unit * scale;
    }
    return result;
}

auto to_bf16(std::span<const float> values) -> std::vector<std::uint16_t> {
    std::vector<std::uint16_t> result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        auto bits = std::bit_cast<std::uint32_t>(values[index]);
        bits += 0x7FFFU + ((bits >> 16) & 1U);
        result[index] = static_cast<std::uint16_t>(bits >> 16);
    }
    return result;
}

struct YnnProjection {
    kidi::runtime::ynn::Executable executable;
    std::vector<float> output;
    double compile_ms;
};

auto make_ynn_projection(std::size_t rows, const std::vector<float>& weights, const std::vector<float>& bias)
    -> kidi::Result<YnnProjection> {
    kidi::runtime::ynn::set_thread_count(CPU_THREADS);
    const auto started = Clock::now();
    auto graph = kidi::runtime::ynn::Graph::create(2);
    if (!graph) return std::unexpected(std::move(graph.error()));

    const std::array<std::size_t, 2> input_shape = {rows, INPUT_SIZE};
    const std::array<std::size_t, 2> output_shape = {rows, OUTPUT_SIZE};
    const std::array<std::size_t, 2> weight_shape = {INPUT_SIZE, OUTPUT_SIZE};
    const std::array<std::size_t, 1> bias_shape = {OUTPUT_SIZE};
    std::uint32_t input_id = 0;
    std::uint32_t output_id = 1;
    std::uint32_t weight_id = YNN_INVALID_VALUE_ID;
    std::uint32_t bias_id = YNN_INVALID_VALUE_ID;
    auto status = kidi::runtime::ynn::check_status(
        ynn_define_tensor(graph->get(), ynn_type_fp32, input_shape.size(), input_shape.data(), nullptr,
                          YNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id),
        "define benchmark input");
    if (status) {
        status = kidi::runtime::ynn::check_status(
            ynn_define_tensor(graph->get(), ynn_type_fp32, output_shape.size(), output_shape.data(), nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id),
            "define benchmark output");
    }
    if (status) {
        status = kidi::runtime::ynn::check_status(ynn_define_tensor(graph->get(), ynn_type_fp32, weight_shape.size(),
                                                                    weight_shape.data(), weights.data(), 0, &weight_id),
                                                  "define benchmark weight");
    }
    if (status) {
        status = kidi::runtime::ynn::check_status(ynn_define_tensor(graph->get(), ynn_type_fp32, bias_shape.size(),
                                                                    bias_shape.data(), bias.data(), 0, &bias_id),
                                                  "define benchmark bias");
    }
    if (status) {
        status = kidi::runtime::ynn::check_status(
            ynn_define_dot(graph->get(), 1, input_id, weight_id, bias_id, &output_id, 0),
            "define benchmark projection");
    }
    if (!status) return std::unexpected(std::move(status.error()));

    auto executable = std::move(*graph).compile();
    if (!executable) return std::unexpected(std::move(executable.error()));
    if (auto reshape = executable->reshape(); !reshape) return std::unexpected(std::move(reshape.error()));
    const auto compile_ms = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    return YnnProjection{
        .executable = std::move(*executable),
        .output = std::vector<float>(rows * OUTPUT_SIZE),
        .compile_ms = compile_ms,
    };
}

auto run(std::size_t rows, std::size_t warmups, std::size_t iterations, std::size_t inflight, bool device_argmax,
         bool metal_bf16) -> int {
    const auto input = make_values(rows * INPUT_SIZE, 0x13579BDFU, 1.0F);
    const auto weights = make_values(INPUT_SIZE * OUTPUT_SIZE, 0x2468ACE1U, 0.03F);
    const auto bias = make_values(OUTPUT_SIZE, 0x10203040U, 0.1F);

    auto ynn = make_ynn_projection(rows, weights, bias);
    if (!ynn) throw std::runtime_error(ynn.error().message);
    require(ynn->executable.bind(0, const_cast<float*>(input.data())));
    require(ynn->executable.bind(1, ynn->output.data()));

    id<MTLDevice> metal_device = MTLCreateSystemDefaultDevice();
    if (metal_device == nil) throw std::runtime_error("no Metal device is available");
    id<MTLCommandQueue> command_queue = [metal_device newCommandQueue];
    if (command_queue == nil) throw std::runtime_error("cannot create Metal command queue");

    MPSShape* input_shape = @[ @(rows), @(INPUT_SIZE) ];
    MPSShape* weight_shape = @[ @(INPUT_SIZE), @(OUTPUT_SIZE) ];
    MPSShape* bias_shape = @[ @(OUTPUT_SIZE) ];
    MPSShape* output_shape = device_argmax ? @[ @(rows), @1 ] : @[ @(rows), @(OUTPUT_SIZE) ];

    const auto metal_compile_started = Clock::now();
    MPSGraph* graph = [MPSGraph new];
    graph.options = MPSGraphOptionsNone;
    const auto metal_data_type = metal_bf16 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32;
    const auto metal_input = metal_bf16 ? to_bf16(input) : std::vector<std::uint16_t>{};
    const auto metal_weights = metal_bf16 ? to_bf16(weights) : std::vector<std::uint16_t>{};
    const auto metal_bias = metal_bf16 ? to_bf16(bias) : std::vector<std::uint16_t>{};
    MPSGraphTensor* input_tensor = [graph placeholderWithShape:input_shape dataType:metal_data_type name:@"input"];
    NSData* weight_data = [NSData
        dataWithBytesNoCopy:metal_bf16 ? static_cast<void*>(const_cast<std::uint16_t*>(metal_weights.data()))
                                       : static_cast<void*>(const_cast<float*>(weights.data()))
                     length:weights.size() * (metal_bf16 ? sizeof(std::uint16_t) : sizeof(float))freeWhenDone:NO];
    NSData* bias_data =
        [NSData dataWithBytesNoCopy:metal_bf16 ? static_cast<void*>(const_cast<std::uint16_t*>(metal_bias.data()))
                                               : static_cast<void*>(const_cast<float*>(bias.data()))
                             length:bias.size() * (metal_bf16 ? sizeof(std::uint16_t) : sizeof(float))freeWhenDone:NO];
    MPSGraphTensor* weight_tensor = [graph constantWithData:weight_data shape:weight_shape dataType:metal_data_type];
    MPSGraphTensor* bias_tensor = [graph constantWithData:bias_data shape:bias_shape dataType:metal_data_type];
    MPSGraphTensor* product = [graph matrixMultiplicationWithPrimaryTensor:input_tensor
                                                           secondaryTensor:weight_tensor
                                                                      name:@"projection"];
    MPSGraphTensor* logits_tensor = [graph additionWithPrimaryTensor:product
                                                     secondaryTensor:bias_tensor
                                                                name:@"logits"];
    MPSGraphTensor* selection_tensor =
        metal_bf16 ? [graph castTensor:logits_tensor toType:MPSDataTypeFloat32 name:@"selection_fp32"] : logits_tensor;
    MPSGraphTensor* output_tensor =
        device_argmax ? [graph reductionArgMaximumWithTensor:selection_tensor axis:1 name:@"token"] : selection_tensor;

    MPSGraphShapedType* input_type = [[MPSGraphShapedType alloc] initWithShape:input_shape dataType:metal_data_type];
    MPSGraphCompilationDescriptor* compilation = [MPSGraphCompilationDescriptor new];
    compilation.optimizationLevel = MPSGraphOptimizationLevel1;
    compilation.waitForCompilationCompletion = YES;
    MPSGraphDevice* graph_device = [MPSGraphDevice deviceWithMTLDevice:metal_device];
    MPSGraphExecutable* executable = [graph compileWithDevice:graph_device
                                                        feeds:@{input_tensor : input_type}
                                                targetTensors:@[ output_tensor ]
                                             targetOperations:nil
                                        compilationDescriptor:compilation];
    if (executable == nil) throw std::runtime_error("MPSGraph compilation failed");
    executable.options = MPSGraphOptionsNone;
    const auto metal_compile_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - metal_compile_started).count();

    id<MTLBuffer> input_buffer =
        [metal_device newBufferWithBytes:metal_bf16 ? static_cast<const void*>(metal_input.data())
                                                    : static_cast<const void*>(input.data())
                                  length:input.size() * (metal_bf16 ? sizeof(std::uint16_t) : sizeof(float))options
                                        :MTLResourceStorageModeShared];
    const auto metal_output_size = rows * (device_argmax ? sizeof(std::int32_t) : OUTPUT_SIZE * sizeof(float));
    id<MTLBuffer> output_buffer = [metal_device newBufferWithLength:metal_output_size
                                                            options:MTLResourceStorageModeShared];
    if (input_buffer == nil || output_buffer == nil) throw std::runtime_error("Metal buffer allocation failed");
    MPSGraphTensorData* input_tensor_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:input_buffer
                                                                                    shape:input_shape
                                                                                 dataType:metal_data_type];
    MPSGraphTensorData* output_tensor_data =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:output_buffer
                                                shape:output_shape
                                             dataType:device_argmax ? MPSDataTypeInt32 : MPSDataTypeFloat32];
    MPSGraphExecutableExecutionDescriptor* execution = [MPSGraphExecutableExecutionDescriptor new];
    execution.waitUntilCompleted = YES;
    MPSGraphExecutableExecutionDescriptor* device_resident_execution = [MPSGraphExecutableExecutionDescriptor new];
    device_resident_execution.waitUntilCompleted = NO;
    std::atomic_bool async_failed = false;
    std::atomic_size_t async_completed = 0;
    std::mutex async_mutex;
    std::condition_variable async_condition;
    auto* async_failed_ptr = &async_failed;
    auto* async_completed_ptr = &async_completed;
    auto* async_condition_ptr = &async_condition;
    device_resident_execution.completionHandler = ^(NSArray<MPSGraphTensorData*>*, NSError* error) {
      if (error != nil) async_failed_ptr->store(true, std::memory_order_relaxed);
      async_completed_ptr->fetch_add(1, std::memory_order_release);
      async_condition_ptr->notify_one();
    };

    NSMutableArray<MPSGraphTensorData*>* async_outputs = [NSMutableArray arrayWithCapacity:inflight];
    NSMutableArray<id<MTLBuffer>>* async_output_buffers = [NSMutableArray arrayWithCapacity:inflight];
    for (std::size_t index = 0; index < inflight; ++index) {
        id<MTLBuffer> buffer = [metal_device newBufferWithLength:metal_output_size
                                                         options:MTLResourceStorageModeShared];
        if (buffer == nil) throw std::runtime_error("Metal async output buffer allocation failed");
        MPSGraphTensorData* data =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                                                    shape:output_shape
                                                 dataType:device_argmax ? MPSDataTypeInt32 : MPSDataTypeFloat32];
        [async_output_buffers addObject:buffer];
        [async_outputs addObject:data];
    }

    const auto invoke_metal = [&] {
        @autoreleasepool {
            NSArray<MPSGraphTensorData*>* results = [executable runWithMTLCommandQueue:command_queue
                                                                           inputsArray:@[ input_tensor_data ]
                                                                          resultsArray:@[ output_tensor_data ]
                                                                   executionDescriptor:execution];
            if (results.count != 1) throw std::runtime_error("MPSGraph returned an unexpected result count");
        }
    };
    std::vector<std::int32_t> ynn_tokens(rows);
    const auto invoke_ynn = [&] {
        require(ynn->executable.invoke());
        if (device_argmax) {
            for (std::size_t row = 0; row < rows; ++row) {
                const auto begin = ynn->output.begin() + static_cast<std::ptrdiff_t>(row * OUTPUT_SIZE);
                ynn_tokens[row] =
                    static_cast<std::int32_t>(std::distance(begin, std::max_element(begin, begin + OUTPUT_SIZE)));
            }
        }
    };
    const auto invoke_metal_device_resident = [&](std::size_t count) {
        @autoreleasepool {
            MPSCommandBuffer* command_buffer = [MPSCommandBuffer commandBufferFromCommandQueue:command_queue];
            for (std::size_t invocation = 0; invocation < count; ++invocation) {
                NSArray<MPSGraphTensorData*>* results = [executable encodeToCommandBuffer:command_buffer
                                                                              inputsArray:@[ input_tensor_data ]
                                                                             resultsArray:@[ output_tensor_data ]
                                                                      executionDescriptor:device_resident_execution];
                if (results.count != 1) throw std::runtime_error("MPSGraph returned an unexpected result count");
            }
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (command_buffer.status == MTLCommandBufferStatusError) {
                throw std::runtime_error("Metal command buffer execution failed");
            }
        }
    };
    const auto invoke_ynn_batch = [&](std::size_t count) {
        for (std::size_t invocation = 0; invocation < count; ++invocation) invoke_ynn();
    };
    const auto invoke_metal_async_trailing = [&](std::size_t count) {
        @autoreleasepool {
            async_failed.store(false, std::memory_order_relaxed);
            async_completed.store(0, std::memory_order_relaxed);
            for (std::size_t invocation = 0; invocation < count; ++invocation) {
                MPSGraphTensorData* result = async_outputs[invocation % async_outputs.count];
                NSArray<MPSGraphTensorData*>* results =
                    [executable runAsyncWithMTLCommandQueue:command_queue
                                                inputsArray:@[ input_tensor_data ]
                                               resultsArray:@[ result ]
                                        executionDescriptor:device_resident_execution];
                if (results.count != 1) throw std::runtime_error("MPSGraph returned an unexpected result count");
            }
            id<MTLCommandBuffer> fence = [command_queue commandBuffer];
            [fence commit];
            [fence waitUntilCompleted];
            if (fence.status == MTLCommandBufferStatusError || async_failed.load(std::memory_order_relaxed)) {
                throw std::runtime_error("asynchronous Metal execution failed");
            }
            std::unique_lock lock(async_mutex);
            async_condition.wait(lock, [&] { return async_completed.load(std::memory_order_acquire) >= count; });
        }
    };

    for (std::size_t warmup = 0; warmup < warmups; ++warmup) {
        invoke_ynn();
        invoke_metal();
    }

    invoke_ynn();
    invoke_metal();
    double max_abs_error = 0.0;
    double squared_error = 0.0;
    std::size_t selection_mismatches = 0;
    if (device_argmax) {
        const auto* metal_tokens = static_cast<const std::int32_t*>(output_buffer.contents);
        for (std::size_t row = 0; row < rows; ++row) {
            if (metal_tokens[row] != ynn_tokens[row]) {
                ++selection_mismatches;
                std::cerr << "argmax mismatch row=" << row << " ynn=" << ynn_tokens[row]
                          << " metal=" << metal_tokens[row] << " mps_dtype=" << output_tensor.dataType
                          << " mps_shape=" << output_tensor.shape.description.UTF8String << '\n';
            }
        }
        if (selection_mismatches != 0) throw std::runtime_error("MPSGraph argmax differs from YNNPACK");
    } else {
        const auto* metal_output = static_cast<const float*>(output_buffer.contents);
        for (std::size_t index = 0; index < ynn->output.size(); ++index) {
            const auto difference = static_cast<double>(ynn->output[index]) - metal_output[index];
            max_abs_error = std::max(max_abs_error, std::abs(difference));
            squared_error += difference * difference;
        }
        const auto error_threshold = metal_bf16 ? 0.25 : 1.0e-3;
        if (max_abs_error > error_threshold) {
            throw std::runtime_error("MPSGraph output exceeds the numerical error threshold");
        }
    }
    const auto rms_error = device_argmax ? 0.0 : std::sqrt(squared_error / static_cast<double>(ynn->output.size()));

    std::vector<double> ynn_samples;
    std::vector<double> metal_samples;
    std::vector<double> ynn_sustained_samples;
    std::vector<double> metal_device_resident_samples;
    std::vector<double> metal_async_trailing_samples;
    ynn_samples.reserve(iterations);
    metal_samples.reserve(iterations);
    ynn_sustained_samples.reserve(iterations);
    metal_device_resident_samples.reserve(iterations);
    metal_async_trailing_samples.reserve(iterations);
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        if (iteration % 2 == 0) {
            ynn_samples.push_back(measure_us(invoke_ynn));
            metal_samples.push_back(measure_us(invoke_metal));
        } else {
            metal_samples.push_back(measure_us(invoke_metal));
            ynn_samples.push_back(measure_us(invoke_ynn));
        }
    }
    invoke_metal_device_resident(inflight);
    invoke_metal_async_trailing(inflight);
    if (device_argmax) {
        for (std::size_t invocation = 0; invocation < inflight; ++invocation) {
            const auto* tokens = static_cast<const std::int32_t*>(async_output_buffers[invocation].contents);
            for (std::size_t row = 0; row < rows; ++row) {
                if (tokens[row] != ynn_tokens[row]) {
                    throw std::runtime_error("asynchronous MPSGraph argmax differs from YNNPACK");
                }
            }
        }
    }
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        if (iteration % 2 == 0) {
            ynn_sustained_samples.push_back(measure_us([&] { invoke_ynn_batch(inflight); }) / inflight);
            metal_device_resident_samples.push_back(measure_us([&] { invoke_metal_device_resident(inflight); }) /
                                                    inflight);
            metal_async_trailing_samples.push_back(measure_us([&] { invoke_metal_async_trailing(inflight); }) /
                                                   inflight);
        } else {
            metal_async_trailing_samples.push_back(measure_us([&] { invoke_metal_async_trailing(inflight); }) /
                                                   inflight);
            metal_device_resident_samples.push_back(measure_us([&] { invoke_metal_device_resident(inflight); }) /
                                                    inflight);
            ynn_sustained_samples.push_back(measure_us([&] { invoke_ynn_batch(inflight); }) / inflight);
        }
    }

    const auto ynn_summary = summarize(std::move(ynn_samples));
    const auto metal_summary = summarize(std::move(metal_samples));
    const auto ynn_sustained_summary = summarize(std::move(ynn_sustained_samples));
    const auto metal_device_resident_summary = summarize(std::move(metal_device_resident_samples));
    const auto metal_async_trailing_summary = summarize(std::move(metal_async_trailing_samples));
    std::cout << std::fixed << std::setprecision(3) << "metal_lowering|device=" << metal_device.name.UTF8String
              << "|rows=" << rows << "|input_size=" << INPUT_SIZE << "|output_size=" << OUTPUT_SIZE
              << "|metal_precision=" << (metal_bf16 ? "bf16" : "fp32")
              << "|output_mode=" << (device_argmax ? "argmax" : "logits") << "|warmups=" << warmups
              << "|iterations=" << iterations << "|ynn_threads=" << CPU_THREADS << "|mps_optimization_level=1"
              << "|buffers_reused=1|timed_host_copies=0|inflight=" << inflight << "|ynn_compile_ms=" << ynn->compile_ms
              << "|metal_compile_ms=" << metal_compile_ms << "|ynn_sync_median_us=" << ynn_summary.median_us
              << "|ynn_sync_p95_us=" << ynn_summary.p95_us << "|metal_sync_median_us=" << metal_summary.median_us
              << "|metal_sync_p95_us=" << metal_summary.p95_us
              << "|sync_speedup=" << ynn_summary.median_us / metal_summary.median_us
              << "|ynn_sustained_median_us=" << ynn_sustained_summary.median_us
              << "|metal_device_resident_median_us=" << metal_device_resident_summary.median_us
              << "|device_resident_speedup="
              << ynn_sustained_summary.median_us / metal_device_resident_summary.median_us
              << "|metal_async_trailing_median_us=" << metal_async_trailing_summary.median_us
              << "|async_trailing_speedup=" << ynn_sustained_summary.median_us / metal_async_trailing_summary.median_us
              << "|metal_sync_rows_per_sec=" << static_cast<double>(rows) * 1.0e6 / metal_summary.median_us
              << "|metal_device_resident_rows_per_sec="
              << static_cast<double>(rows) * 1.0e6 / metal_device_resident_summary.median_us << std::scientific
              << std::setprecision(6) << "|max_abs_error=" << max_abs_error << "|rms_error=" << rms_error
              << "|selection_mismatches=" << selection_mismatches << '\n';
    return 0;
}

} // namespace

auto main(int argc, char** argv) -> int {
    @autoreleasepool {
        try {
            const auto rows = argc > 1 ? parse_positive(argv[1], "rows") : std::size_t{1};
            const auto iterations = argc > 2 ? parse_positive(argv[2], "iterations") : DEFAULT_ITERATIONS;
            const auto warmups = argc > 3 ? parse_positive(argv[3], "warmups") : DEFAULT_WARMUPS;
            const auto inflight = argc > 4 ? parse_positive(argv[4], "inflight") : DEFAULT_INFLIGHT;
            const std::string output_mode = argc > 5 ? argv[5] : "logits";
            const std::string metal_precision = argc > 6 ? argv[6] : "fp32";
            if (argc > 7 || (output_mode != "logits" && output_mode != "argmax") ||
                (metal_precision != "fp32" && metal_precision != "bf16")) {
                std::cerr << "usage: kidi_metal_lowering_bench [rows] [iterations] [warmups] [inflight] "
                             "[logits|argmax] [fp32|bf16]\n";
                return 2;
            }
            return run(rows, warmups, iterations, inflight, output_mode == "argmax", metal_precision == "bf16");
        } catch (const std::exception& error) {
            std::cerr << "kidi_metal_lowering_bench: " << error.what() << '\n';
            return 1;
        }
    }
}