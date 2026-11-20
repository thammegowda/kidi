#include "cuda/runtime.h"
#include <cuda_runtime_api.h>
#if defined(KIDI_CUDA_USE_CUBLAS)
#include <cublas_v2.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace kidi::cuda;
using Buffer = std::unique_ptr<void, decltype(&release)>;

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct Event {
    cudaEvent_t value{};
    Event() { check(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

template <typename Function>
float measure_launch(Function&& launch, int device, int repeats) {
    for (int iteration = 0; iteration < 3; ++iteration) launch();
    synchronize(device);
    Event begin, end;
    std::array<float, 5> samples{};
    for (auto& sample : samples) {
        check(cudaEventRecord(begin.value));
        for (int iteration = 0; iteration < repeats; ++iteration) launch();
        check(cudaEventRecord(end.value));
        check(cudaEventSynchronize(end.value));
        float milliseconds = 0;
        check(cudaEventElapsedTime(&milliseconds, begin.value, end.value));
        sample = milliseconds * 1000 / repeats;
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

float measure(Program& program, const std::array<Tensor, 4>& inputs, int count, void* output, int repeats) {
    return measure_launch([&] { program.run(inputs.data(), count, output); }, inputs[0].device, repeats);
}

#if defined(KIDI_CUDA_USE_CUBLAS)
void benchmark_bfloat_vector(int width, int columns, int device) {
    std::vector<std::uint16_t> values(width), weights(static_cast<std::size_t>(columns) * width);
    const auto fill = [](auto& data, int multiplier) {
        for (std::size_t index = 0; index < data.size(); ++index) {
            const float number = static_cast<float>(static_cast<int>(index * multiplier % 257) - 128) * 0.003125F;
            std::uint32_t bits = 0;
            std::memcpy(&bits, &number, sizeof(bits));
            data[index] = static_cast<std::uint16_t>(bits >> 16);
        }
    };
    fill(values, 17);
    fill(weights, 11);
    Buffer input(allocate(values.size() * sizeof(std::uint16_t), device), release),
        weight(allocate(weights.size() * sizeof(std::uint16_t), device), release),
        output(allocate(columns * sizeof(std::uint16_t), device), release);
    copy_to_device(input.get(), values.data(), values.size() * sizeof(std::uint16_t), device);
    copy_to_device(weight.get(), weights.data(), weights.size() * sizeof(std::uint16_t), device);
    std::array<Tensor, 4> inputs{};
    inputs[0] = {input.get(), Type::BF16, 3, {1, 1, width}, device};
    inputs[1] = {weight.get(), Type::BF16, 2, {columns, width}, device};
    Descriptor descriptor;
    descriptor.operation = Op::LINEAR;
    descriptor.input_count = 2;
    descriptor.inputs[0] = inputs[0];
    descriptor.inputs[1] = inputs[1];
    descriptor.output.type = Type::BF16;
    descriptor.attribute_count = 1;
    descriptor.attributes[0] = 1;
    Program automatic(descriptor, device);
    cublasHandle_t handle{};
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS benchmark initialization failed");
    const float alpha = 1, beta = 0;
    const auto baseline = [&] {
        const auto status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, columns, 1, width, &alpha, weight.get(),
                                         CUDA_R_16BF, width, input.get(), CUDA_R_16BF, width, &beta, output.get(),
                                         CUDA_R_16BF, columns, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS vector benchmark failed");
    };
    const auto baseline_us = measure_launch(baseline, device, 10);
    std::vector<std::uint16_t> expected(columns), actual(columns);
    copy_to_host(expected.data(), output.get(), expected.size() * sizeof(std::uint16_t), device);
    const auto automatic_us = measure(automatic, inputs, 2, output.get(), 10);
    copy_to_host(actual.data(), output.get(), actual.size() * sizeof(std::uint16_t), device);
    cublasDestroy(handle);
    const auto decode = [](std::uint16_t rounded) {
        const std::uint32_t bits = static_cast<std::uint32_t>(rounded) << 16;
        float number = 0;
        std::memcpy(&number, &bits, sizeof(number));
        return number;
    };
    float maximum_error = 0;
    double maximum_math_error = 0, maximum_cublas_math_error = 0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const auto target = decode(expected[index]), result = decode(actual[index]);
        const auto error = std::abs(result - target);
        double oracle = 0, magnitude = 0;
        for (int contraction = 0; contraction < width; ++contraction) {
            const double product =
                static_cast<double>(decode(values[contraction])) * decode(weights[index * width + contraction]);
            oracle += product;
            magnitude += std::abs(product);
        }
        const double fp32_error = (width / 512 + 16) * (std::numeric_limits<float>::epsilon() / 2) * magnitude;
        const double bound = std::abs(oracle) / 256 + fp32_error + 1e-6;
        if (!std::isfinite(result) || std::abs(result - oracle) > bound)
            throw std::runtime_error("BF16 vector exceeds the FP32 accumulation/BF16 rounding error bound");
        maximum_math_error = std::max(maximum_math_error, std::abs(result - oracle));
        maximum_cublas_math_error = std::max(maximum_cublas_math_error, std::abs(target - oracle));
        maximum_error = std::max(maximum_error, error);
    }
    std::cout << "{\"operation\":\"bf16_vector\",\"width\":" << width << ",\"columns\":" << columns
              << ",\"cublas_us\":" << baseline_us << ",\"automatic_us\":" << automatic_us
              << ",\"speedup\":" << baseline_us / automatic_us << ",\"maximum_error\":" << maximum_error
              << ",\"maximum_math_error\":" << maximum_math_error
              << ",\"maximum_cublas_math_error\":" << maximum_cublas_math_error << "}\n"
              << std::flush;
}
#endif

void benchmark_mixed_matrix(int rows, int device) {
    constexpr int width = 2560, columns = 10752;
    std::vector<std::uint16_t> values(static_cast<std::size_t>(rows) * width), weights(columns * width);
    const auto encode = [](float value) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return static_cast<std::uint16_t>(bits >> 16);
    };
    for (std::size_t index = 0; index < values.size(); ++index)
        values[index] = encode(static_cast<float>(static_cast<int>(index * 17 % 257) - 113) * 0.00321F);
    for (std::size_t index = 0; index < weights.size(); ++index)
        weights[index] = encode(static_cast<float>(static_cast<int>(index * 11 % 251) - 125) * 0.00123F);
    Buffer input(allocate(values.size() * sizeof(values[0]), device), release),
        weight(allocate(weights.size() * sizeof(weights[0]), device), release),
        output(allocate(static_cast<std::size_t>(rows) * columns * sizeof(float), device), release);
    copy_to_device(input.get(), values.data(), values.size() * sizeof(values[0]), device);
    copy_to_device(weight.get(), weights.data(), weights.size() * sizeof(weights[0]), device);
    std::array<Tensor, 4> inputs{};
    inputs[0] = {input.get(), Type::BF16, 3, {1, rows, width}, device};
    inputs[1] = {weight.get(), Type::BF16, 2, {columns, width}, device};
    Descriptor descriptor;
    descriptor.operation = Op::LINEAR;
    descriptor.input_count = 2;
    std::copy(inputs.begin(), inputs.end(), descriptor.inputs.begin());
    descriptor.attribute_count = 1;
    descriptor.attributes[0] = 1;
    descriptor.immutable_weights = true;
    descriptor.reference_projection = true;
    Program reference(descriptor, device);
    const auto reference_us = measure(reference, inputs, 2, output.get(), 2);
    std::vector<float> expected(static_cast<std::size_t>(rows) * columns), actual(expected.size());
    copy_to_host(expected.data(), output.get(), expected.size() * sizeof(float), device);
    descriptor.reference_projection = false;
    Program automatic(descriptor, device);
    const auto automatic_us = measure(automatic, inputs, 2, output.get(), 2);
    copy_to_host(actual.data(), output.get(), actual.size() * sizeof(float), device);
    float maximum_error = 0, scaled_error = 0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index])) throw std::runtime_error("non-finite mixed matrix result");
        const auto error = std::abs(actual[index] - expected[index]);
        maximum_error = std::max(maximum_error, error);
        scaled_error = std::max(scaled_error, error / (1.F + std::abs(expected[index])));
    }
    if (scaled_error > 1e-4F) throw std::runtime_error("mixed matrix exceeds FP32 reduction tolerance");
    std::cout << "{\"operation\":\"bf16_matrix\",\"rows\":" << rows << ",\"width\":" << width
              << ",\"columns\":" << columns << ",\"reference_us\":" << reference_us
              << ",\"automatic_us\":" << automatic_us << ",\"speedup\":" << reference_us / automatic_us
              << ",\"maximum_error\":" << maximum_error << ",\"scaled_error\":" << scaled_error << "}\n"
              << std::flush;
}

void benchmark(int rows, int width, int columns, int device, int bits = 8, bool calibrated = false) {
    std::vector<float> input(static_cast<std::size_t>(rows) * width), scales(columns), bias(columns);
    std::vector<std::int8_t> weights(static_cast<std::size_t>(columns) * width);
    for (std::size_t index = 0; index < input.size(); ++index)
        input[index] = index / width == 3 ? (index % 2 ? -1000.F : 1000.F)
                                          : static_cast<float>(static_cast<int>(index * 17 % 257) - 113) * 0.03125F;
    for (std::size_t index = 0; index < weights.size(); ++index)
        weights[index] = static_cast<std::int8_t>(static_cast<int>(index * 11 % (bits == 8 ? 255 : 1 << bits)) -
                                                  (bits == 8 ? 127 : 1 << (bits - 1)));
    const bool packed = bits != 8 || calibrated;
    std::vector<std::uint8_t> packed_weights(weights.size() / (8 / bits), 0);
    if (packed)
        for (std::size_t index = 0; index < weights.size(); ++index)
            packed_weights[index / (8 / bits)] |= (static_cast<unsigned int>(weights[index]) & ((1 << bits) - 1))
                                                  << ((index % (8 / bits)) * bits);
    for (int index = 0; index < columns; ++index) {
        scales[index] = (index % 7 + 1) * 0.0001F;
        bias[index] = (index % 5 - 2) * 0.01F;
    }
    Buffer input_device(allocate(input.size() * sizeof(float), device), release);
    Buffer weight_device(allocate(packed ? packed_weights.size() : weights.size(), device), release);
    Buffer scale_device(allocate(scales.size() * sizeof(float), device), release);
    Buffer bias_device(allocate(bias.size() * sizeof(float), device), release);
    Buffer output(allocate(static_cast<std::size_t>(rows) * columns * sizeof(float), device), release);
    copy_to_device(input_device.get(), input.data(), input.size() * sizeof(float), device);
    copy_to_device(weight_device.get(), packed ? static_cast<const void*>(packed_weights.data()) : weights.data(),
                   packed ? packed_weights.size() : weights.size(), device);
    copy_to_device(scale_device.get(), scales.data(), scales.size() * sizeof(float), device);
    copy_to_device(bias_device.get(), bias.data(), bias.size() * sizeof(float), device);
    std::array<Tensor, 4> inputs{
        {{input_device.get(), Type::F32, 2, {rows, width}, device},
         {weight_device.get(), packed ? Type::U8 : Type::I8, 2, {columns, width / (packed ? 8 / bits : 1)}, device},
         {scale_device.get(), Type::F32, 2, {columns, 1}, device},
         {bias_device.get(), Type::F32, 1, {columns}, device}}};
    Descriptor descriptor;
    descriptor.operation = packed ? Op::PACKED_LINEAR : Op::QUANTIZED_LINEAR;
    descriptor.inputs[0] = inputs[0];
    descriptor.inputs[1] = inputs[1];
    descriptor.inputs[2] = inputs[2];
    descriptor.inputs[3] = inputs[3];
    descriptor.input_count = packed ? 3 : 4;
    descriptor.attribute_count = packed ? 3 : 1;
    descriptor.attributes[0] = packed ? bits : 1;
    if (packed) descriptor.attributes[1] = width;
    if (calibrated) {
        descriptor.epsilon = 0.125F;
        const float output_scale = 0.0625F;
        std::int32_t encoded = 0;
        std::memcpy(&encoded, &output_scale, sizeof(encoded));
        descriptor.attributes[2] = encoded;
    }
    descriptor.immutable_weights = true;
    descriptor.reference_projection = true;
    Program reference(descriptor, device);
    const auto reference_us = measure(reference, inputs, descriptor.input_count, output.get(), 10);
    std::vector<float> expected(static_cast<std::size_t>(rows) * columns), actual(expected.size());
    copy_to_host(expected.data(), output.get(), expected.size() * sizeof(float), device);
    descriptor.reference_projection = false;
    Program automatic(descriptor, device);
    const auto automatic_us = measure(automatic, inputs, descriptor.input_count, output.get(), 10);
    copy_to_host(actual.data(), output.get(), actual.size() * sizeof(float), device);
    float maximum_error = 0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index])) throw std::runtime_error("non-finite projection result");
        maximum_error = std::max(maximum_error, std::abs(actual[index] - expected[index]));
    }
    if (maximum_error > 1e-4F) throw std::runtime_error("projection differs from reference");
    std::cout << "{\"rows\":" << rows << ",\"width\":" << width << ",\"columns\":" << columns << ",\"bits\":" << bits
              << ",\"calibrated\":" << (calibrated ? "true" : "false") << ",\"device\":" << device
              << ",\"reference_us\":" << reference_us << ",\"automatic_us\":" << automatic_us
              << ",\"speedup\":" << reference_us / automatic_us << ",\"maximum_error\":" << maximum_error << "}\n"
              << std::flush;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const int device = argc > 1 ? std::stoi(argv[1]) : 0;
        if (!kidi::cuda::available(device)) throw std::runtime_error("CUDA device unavailable");
        check(cudaSetDevice(device));
        if (argc > 2 && std::string(argv[2]) == "mixed") {
            for (const auto rows : {1, 4, 32}) benchmark_mixed_matrix(rows, device);
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "bf16") {
#if defined(KIDI_CUDA_USE_CUBLAS)
            for (const auto columns : {256, 1536, 2048, 12288, 24576, 262144})
                benchmark_bfloat_vector(1536, columns, device);
            benchmark_bfloat_vector(12288, 1536, device);
            benchmark_bfloat_vector(6144, 1536, device);
            benchmark_bfloat_vector(2048, 1536, device);
            benchmark_bfloat_vector(4096, 1536, device);
            return 0;
#else
            throw std::runtime_error("The BF16 vector comparison requires a cuBLAS build");
#endif
        }
        for (const auto rows : {1, 4, 8, 16, 128}) {
            benchmark(rows, 768, 3072, device);
            benchmark(rows, 1024, 3072, device);
        }
        for (const auto rows : {1, 32}) {
            benchmark(rows, 2560, 10240, device, 4, true);
            benchmark(rows, 1024, 3072, device, 2, true);
            benchmark(rows, 1024, 3072, device, 8, true);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}