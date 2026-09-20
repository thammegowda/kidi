#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kidi/runtime/mps/quantized_linear.h"

namespace {
using Clock = std::chrono::steady_clock;
using kidi::tensor::Device;
using kidi::tensor::DType;
using kidi::tensor::Tensor;

template <typename Type>
auto take(kidi::Result<Type> result) -> Type {
    if (!result) throw std::runtime_error(result.error().message);
    return std::move(*result);
}
auto check(kidi::Result<void> result) -> void {
    if (!result) throw std::runtime_error(result.error().message);
}
auto median(std::vector<double> values) -> double {
    std::ranges::sort(values);
    return (values[(values.size() - 1) / 2] + values[values.size() / 2]) / 2;
}
auto benchmark(std::int64_t width, std::int64_t columns, std::int64_t rows) -> void {
    std::vector<std::int8_t> weights(width * columns);
    std::vector<float> inputs(rows * width), scales(columns, 1.0F / 128), bias(columns, 0.125F);
    for (std::size_t index = 0; index < weights.size(); ++index)
        weights[index] = static_cast<std::int8_t>(static_cast<int>((index * 73 + 19) % 255) - 127);
    for (std::size_t index = 0; index < inputs.size(); ++index)
        inputs[index] = static_cast<float>(static_cast<int>(index % 256) - 128) / 128;
    auto weight_tensor = take(Tensor::from_host({width, columns}, std::span<const std::int8_t>(weights)));
    auto scale_tensor = take(Tensor::from_host({columns, 1}, std::span<const float>(scales)));
    auto bias_tensor = take(Tensor::from_host({columns}, std::span<const float>(bias)));
    auto input = take(Tensor::from_host({rows, width}, std::span<const float>(inputs), Device::apple_gpu()));
    auto output = take(Tensor::empty({rows, columns}, DType::F32, Device::apple_gpu()));
    auto kernel = take(kidi::runtime::mps::QuantizedLinear::create(weight_tensor, scale_tensor, bias_tensor));
    check(kernel.prepare(input));
    for (int warmup = 0; warmup < 20; ++warmup) check(kernel.run(input, output));
    auto values = take(output.data<float>());
    for (std::int64_t row = 0; row < rows; ++row) {
        for (const auto column : {std::int64_t{0}, columns / 2, columns - 1}) {
            std::int32_t sum = 0;
            for (std::int64_t channel = 0; channel < width; ++channel)
                sum += static_cast<std::int32_t>(inputs[row * width + channel] * 128) *
                       weights[channel * columns + column];
            const auto expected = static_cast<float>(sum) / 16384 + bias[column];
            if (!std::isfinite(values[row * columns + column]) || values[row * columns + column] != expected)
                throw std::runtime_error("INT8 benchmark numeric mismatch");
        }
    }
    std::vector<double> sync, queued;
    for (int iteration = 0; iteration < 40; ++iteration) {
        auto start = Clock::now();
        check(kernel.run(input, output));
        sync.push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count());
        start = Clock::now();
        auto batch = take(kidi::runtime::mps::CommandBatch::create());
        for (int invocation = 0; invocation < 16; ++invocation) check(kernel.encode(batch, input, output));
        check(batch.finish());
        queued.push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count() / 16);
    }
    std::cout << std::fixed << std::setprecision(3) << "int8_linear|rows=" << rows << "|width=" << width
              << "|columns=" << columns << "|sync_us=" << median(sync) << "|queued_us=" << median(queued)
              << "|weights_bytes=" << weights.size() << "|numeric_check=pass\n";
}
} // namespace
auto main(int argc, char** argv) -> int {
    try {
        const auto rows = argc > 1 ? std::stoll(argv[1]) : 1;
        if (rows < 1 || rows > 64) throw std::invalid_argument("rows must be 1..64");
        benchmark(768, 768, rows);
        benchmark(768, 2304, rows);
        benchmark(768, 2048, rows);
        benchmark(2048, 768, rows);
        benchmark(768, 64000, rows);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}