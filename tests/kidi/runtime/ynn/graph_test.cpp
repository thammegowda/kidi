#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>

#include <ynnpack.h>

#include "kidi/runtime/ynn/graph.h"
#include "kidi/tensor/tensor.h"

namespace {

auto run(kidi::runtime::ynn::Executable& executable, std::span<float> input, std::span<float> output) -> bool {
    auto status = executable.bind(0, input.data());
    if (status) status = executable.bind(1, output.data());
    if (status) status = executable.invoke();
    if (!status) std::cerr << status.error().message << '\n';
    return status.has_value();
}

auto run(kidi::runtime::ynn::Executable& executable, const kidi::tensor::Tensor& input, kidi::tensor::Tensor& output)
    -> bool {
    auto status = executable.bind(0, input);
    if (status) status = executable.bind(1, output);
    if (status) status = executable.invoke();
    if (!status) std::cerr << status.error().message << '\n';
    return status.has_value();
}

} // namespace

auto main() -> int {
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
    if (kidi::runtime::ynn::supported_arch_flags() == 0) {
        std::cerr << "Apple arm64 YNNPACK feature detection returned no supported kernels\n";
        return 1;
    }
#endif

    auto graph = kidi::runtime::ynn::Graph::create(2);
    if (!graph) {
        std::cerr << graph.error().message << '\n';
        return 1;
    }

    constexpr std::array<std::size_t, 2> SHAPE = {2, 2};
    std::uint32_t input_id = 0;
    std::uint32_t output_id = 1;
    auto status =
        kidi::runtime::ynn::check_status(ynn_define_tensor(graph->get(), ynn_type_fp32, SHAPE.size(), SHAPE.data(),
                                                           nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id),
                                         "define input");
    if (status) {
        status =
            kidi::runtime::ynn::check_status(ynn_define_tensor(graph->get(), ynn_type_fp32, SHAPE.size(), nullptr,
                                                               nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id),
                                             "define output");
    }
    if (status) {
        status =
            kidi::runtime::ynn::check_status(ynn_define_copy(graph->get(), input_id, &output_id, 0), "define copy");
    }
    if (!status) {
        std::cerr << status.error().message << '\n';
        return 1;
    }

    auto invalid_threads = std::move(*graph).compile(std::numeric_limits<std::size_t>::max());
    if (invalid_threads || invalid_threads.error().code != kidi::ErrorCode::INVALID_ARGUMENT) return 1;
    kidi::runtime::ynn::set_thread_count(2);
    auto executable = std::move(*graph).compile(1);
    if (kidi::runtime::ynn::thread_count() != 2) return 1;
    if (!executable) {
        std::cerr << executable.error().message << '\n';
        return 1;
    }

    status = executable->reshape();
    if (!status) {
        std::cerr << status.error().message << '\n';
        return 1;
    }
    std::array input = {1.0F, 2.0F, 3.0F, 4.0F};
    std::array<float, 4> output{};
    if (!run(*executable, input, output) || output != input) return 1;

    input = {4.0F, 3.0F, 2.0F, 1.0F};
    output = {};
    if (!run(*executable, input, output) || output != input) return 1;

    constexpr std::array large_values = {5.0F, 6.0F, 7.0F, 8.0F};
    auto large_input = kidi::tensor::Tensor::from_host({2, 2}, std::span<const float>(large_values));
    auto large_output = kidi::tensor::Tensor::zeros({2, 2}, kidi::tensor::DType::F32);
    if (!large_input || !large_output) {
        std::cerr << (large_input ? large_output.error().message : large_input.error().message) << '\n';
        return 1;
    }
    if (!run(*executable, *large_input, *large_output)) return 1;
    auto output_values = large_output->data<float>();
    if (!output_values || !std::ranges::equal(*output_values, large_values)) return 1;

    const auto output_shape = executable->shape(output_id);
    if (!output_shape || *output_shape != std::vector<std::size_t>(SHAPE.begin(), SHAPE.end())) {
        std::cerr << "unexpected output shape\n";
        return 1;
    }
    return 0;
}