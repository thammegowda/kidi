#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>

#include <ynnpack.h>

#include "kidi/runtime/ynn.h"

namespace {

bool run(kidi::runtime::YnnExecutable& executable, std::span<float> input, std::span<float> output,
         std::span<const std::size_t> shape) {
    auto status = executable.set_shape(0, shape);
    if (status) status = executable.reshape();
    if (status) status = executable.bind(0, input.data());
    if (status) status = executable.bind(1, output.data());
    if (status) status = executable.invoke();
    if (!status) std::cerr << status.error().message << '\n';
    return status.has_value();
}

} // namespace

int main() {
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
    if (kidi::runtime::ynn_supported_arch_flags() == 0) {
        std::cerr << "Apple arm64 YNNPACK feature detection returned no supported kernels\n";
        return 1;
    }
#endif

    auto graph = kidi::runtime::YnnGraph::create(2);
    if (!graph) {
        std::cerr << graph.error().message << '\n';
        return 1;
    }

    constexpr std::array<std::size_t, 2> DYNAMIC_SHAPE = {0, 2};
    std::uint32_t input_id = 0;
    std::uint32_t output_id = 1;
    auto status = kidi::runtime::check_ynn_status(
        ynn_define_tensor(graph->get(), ynn_type_fp32, DYNAMIC_SHAPE.size(), DYNAMIC_SHAPE.data(), nullptr,
                          YNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id),
        "define input");
    if (status) {
        status = kidi::runtime::check_ynn_status(
            ynn_define_tensor(graph->get(), ynn_type_fp32, DYNAMIC_SHAPE.size(), nullptr, nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id),
            "define output");
    }
    if (status) {
        status = kidi::runtime::check_ynn_status(ynn_define_copy(graph->get(), input_id, &output_id, 0), "define copy");
    }
    if (!status) {
        std::cerr << status.error().message << '\n';
        return 1;
    }

    auto executable = std::move(*graph).compile();
    if (!executable) {
        std::cerr << executable.error().message << '\n';
        return 1;
    }

    constexpr std::array<std::size_t, 2> SMALL_SHAPE = {1, 2};
    std::array small_input = {1.0F, 2.0F};
    std::array<float, 2> small_output{};
    if (!run(*executable, small_input, small_output, SMALL_SHAPE) || small_output != small_input) return 1;

    small_input = {3.0F, 4.0F};
    small_output = {};
    if (!run(*executable, small_input, small_output, SMALL_SHAPE) || small_output != small_input) return 1;

    constexpr std::array<std::size_t, 2> LARGE_SHAPE = {2, 2};
    std::array large_input = {5.0F, 6.0F, 7.0F, 8.0F};
    std::array<float, 4> large_output{};
    if (!run(*executable, large_input, large_output, LARGE_SHAPE) || large_output != large_input) return 1;

    const auto output_shape = executable->shape(output_id);
    if (!output_shape || *output_shape != std::vector<std::size_t>(LARGE_SHAPE.begin(), LARGE_SHAPE.end())) {
        std::cerr << "unexpected output shape after invalidation\n";
        return 1;
    }
    return 0;
}