#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <vector>

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

auto test_broadcast_dot(bool broadcast) -> bool {
    constexpr std::size_t HEADS = 2, GROUPS = 2, LENGTH = 259, WIDTH = 40;
    const auto right_groups = broadcast ? std::size_t{1} : GROUPS;
    constexpr std::array<std::size_t, 5> LEFT_SHAPE{1, HEADS, GROUPS, 1, LENGTH};
    const std::array<std::size_t, 3> right_shape{1, LENGTH, HEADS * right_groups * WIDTH};
    const std::array<std::size_t, 5> right_reshape{1, LENGTH, HEADS, right_groups, WIDTH};
    constexpr std::array<std::int32_t, 5> AXES{0, 2, 3, 1, 4};
    auto graph = kidi::runtime::ynn::Graph::create(3);
    if (!graph) return false;
    const auto check = [](ynn_status status) {
        if (status == ynn_status_success) return true;
        std::cerr << "broadcast dot definition failed: " << status << '\n';
        return false;
    };
    std::uint32_t left_id = 0, right_id = 1, output_id = 2;
    auto reshaped = YNN_INVALID_VALUE_ID, transposed = YNN_INVALID_VALUE_ID;
    if (!check(ynn_define_tensor(graph->get(), ynn_type_fp32, LEFT_SHAPE.size(), LEFT_SHAPE.data(), nullptr,
                                 YNN_VALUE_FLAG_EXTERNAL_INPUT, &left_id)) ||
        !check(ynn_define_tensor(graph->get(), ynn_type_fp32, right_shape.size(), right_shape.data(), nullptr,
                                 YNN_VALUE_FLAG_EXTERNAL_INPUT, &right_id)) ||
        !check(ynn_define_tensor(graph->get(), ynn_type_fp32, 0, nullptr, nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                                 &output_id)) ||
        !check(ynn_define_static_reshape(graph->get(), right_reshape.size(), right_reshape.data(), right_id, &reshaped,
                                         0)) ||
        !check(ynn_define_static_transpose(graph->get(), AXES.size(), AXES.data(), reshaped, &transposed, 0)) ||
        !check(ynn_define_dot(graph->get(), 1, left_id, transposed, YNN_INVALID_VALUE_ID, &output_id, 0)))
        return false;
    auto executable = std::move(*graph).compile(1);
    if (!executable) {
        std::cerr << executable.error().message << '\n';
        return false;
    }
    std::array<float, HEADS * GROUPS * LENGTH> left{};
    std::vector<float> right(LENGTH * HEADS * right_groups * WIDTH);
    std::array<float, HEADS * GROUPS * WIDTH> output{};
    for (std::size_t row = 0; row < HEADS * GROUPS; ++row) left[row * LENGTH + LENGTH - 1 - row] = 1.F;
    for (std::size_t index = 0; index < right.size(); ++index) right[index] = static_cast<int>(index % 17) - 8;
    auto status = executable->reshape();
    if (status) status = executable->bind(left_id, left.data());
    if (status) status = executable->bind(right_id, right.data());
    if (status) status = executable->bind(output_id, output.data());
    if (status) status = executable->invoke();
    if (!status) {
        std::cerr << status.error().message << '\n';
        return false;
    }
    for (std::size_t index = 0; index < output.size(); ++index) {
        const auto row = index / WIDTH, head = row / GROUPS, group = row % GROUPS;
        const auto right_row = ((LENGTH - 1 - row) * HEADS + head) * right_groups + (broadcast ? 0 : group);
        if (output[index] != right[right_row * WIDTH + index % WIDTH]) {
            std::cerr << "broadcast dot mismatch at " << index << " broadcast=" << broadcast << '\n';
            return false;
        }
    }
    return true;
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
    if (!test_broadcast_dot(false) || !test_broadcast_dot(true)) return 1;
    return 0;
}