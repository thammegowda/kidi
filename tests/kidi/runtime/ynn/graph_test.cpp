#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <vector>

#include <ynnpack.h>

#include "kidi/ops/context.h"
#include "kidi/runtime/ynn/feed_forward_cache.h"
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
#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
    const auto cache_root = std::filesystem::temp_directory_path() / "kidi-ynn-feed-forward-cache-test";
    std::filesystem::remove_all(cache_root);
    std::filesystem::create_directories(cache_root);
    const auto source = cache_root / "weights.bin";
    std::ofstream(source, std::ios::binary).write("source", 6);
    kidi::runtime::ynn::FeedForwardCache cache;
    cache.configure(cache_root / "cache", source);
    kidi::runtime::ynn::FeedForwardCacheSpec cache_spec{
        .bits = 4, .input_size = 8, .intermediate_size = 16, .source_hash = 17};
    const std::array<std::uint8_t, 6> cached_bytes{1, 2, 3, 5, 8, 13};
    cache.store(cache_spec, cached_bytes);
    const auto cached = cache.load(cache_spec, cached_bytes.size());
    if (!cached || !cached.owner || !std::ranges::equal(cached.bytes, cached_bytes)) return 1;
    auto incompatible = cache_spec;
    incompatible.bits = 8;
    if (cache.load(incompatible, cached_bytes.size())) return 1;
    const auto source_time = std::filesystem::last_write_time(source);
    std::ofstream(source, std::ios::binary | std::ios::trunc).write("change", 6);
    std::filesystem::last_write_time(source, source_time);
    auto replaced = cache_spec;
    replaced.source_hash = 18;
    kidi::runtime::ynn::FeedForwardCache replacement;
    replacement.configure(cache_root / "cache", source);
    if (replacement.load(replaced, cached_bytes.size())) return 1;
    std::ofstream(source, std::ios::binary | std::ios::app).write("changed", 7);
    kidi::runtime::ynn::FeedForwardCache stale;
    stale.configure(cache_root / "cache", source);
    if (stale.load(cache_spec, cached_bytes.size())) return 1;
    const auto read_only_parent = cache_root / "read-only";
    std::filesystem::create_directories(read_only_parent);
    std::filesystem::permissions(read_only_parent,
                                 std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec);
    kidi::runtime::ynn::FeedForwardCache uncreatable;
    uncreatable.configure(read_only_parent / "nested", source);
    std::filesystem::permissions(read_only_parent, std::filesystem::perms::owner_all);
    if (uncreatable.enabled() || uncreatable.writable()) return 1;
    ::setenv("KIDI_FFN_CACHE", "0", 1);
    kidi::runtime::ynn::FeedForwardCache disabled;
    disabled.configure(cache_root / "disabled", source);
    ::unsetenv("KIDI_FFN_CACHE");
    if (disabled.enabled() || disabled.writable()) return 1;

    constexpr std::int32_t ROWS = 32, INPUT_SIZE = 128, INTERMEDIATE = 2056, BITS = 4;
    std::vector<float> activations(ROWS * INPUT_SIZE);
    for (std::size_t index = 0; index < activations.size(); ++index)
        activations[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * 0.0625F;
    std::vector<std::uint8_t> gate_up_bytes(2 * INTERMEDIATE * (INPUT_SIZE / 2));
    for (std::size_t index = 0; index < gate_up_bytes.size(); ++index)
        gate_up_bytes[index] = static_cast<std::uint8_t>((index * 13 + 5) & 0xFFU);
    std::vector<std::uint8_t> down_bytes(INPUT_SIZE * (INTERMEDIATE / 2));
    for (std::size_t index = 0; index < down_bytes.size(); ++index)
        down_bytes[index] = static_cast<std::uint8_t>((index * 7 + 3) & 0xFFU);
    std::vector<float> gate_up_scales(2 * INTERMEDIATE), down_scales(INPUT_SIZE);
    for (std::size_t index = 0; index < gate_up_scales.size(); ++index)
        gate_up_scales[index] = 0.00390625F * static_cast<float>(1 + index % 5);
    for (std::size_t index = 0; index < down_scales.size(); ++index)
        down_scales[index] = 0.00390625F * static_cast<float>(1 + index % 7);
    const auto input_tensor =
        kidi::ops::require(kidi::tensor::Tensor::from_host({1, ROWS, INPUT_SIZE}, std::span<const float>(activations)));
    const auto gate_up_tensor = kidi::ops::require(kidi::tensor::Tensor::from_host(
        {2 * INTERMEDIATE, INPUT_SIZE / 2}, std::span<const std::uint8_t>(gate_up_bytes)));
    const auto down_tensor = kidi::ops::require(
        kidi::tensor::Tensor::from_host({INPUT_SIZE, INTERMEDIATE / 2}, std::span<const std::uint8_t>(down_bytes)));
    const auto gate_up_scale_tensor = kidi::ops::require(
        kidi::tensor::Tensor::from_host({2 * INTERMEDIATE, 1}, std::span<const float>(gate_up_scales)));
    const auto down_scale_tensor =
        kidi::ops::require(kidi::tensor::Tensor::from_host({INPUT_SIZE, 1}, std::span<const float>(down_scales)));
    const auto run_ffn = [&](kidi::ops::Context& context) {
        return context.gated_feed_forward(input_tensor, gate_up_tensor, gate_up_scale_tensor, down_tensor,
                                          down_scale_tensor, BITS, INPUT_SIZE, INTERMEDIATE, 0.125F, 0.0625F, 0.03125F,
                                          0.125F);
    };
    kidi::ops::Context reference_context(kidi::tensor::Device::cpu());
    const auto reference_ffn = run_ffn(reference_context);
    const auto ffn_cache_dir = cache_root / "ffn-multi-tile";
    kidi::ops::Context populate_context(kidi::tensor::Device::cpu());
    populate_context.configure_cache(ffn_cache_dir, source);
    const auto populated_ffn = run_ffn(populate_context);
    if (std::filesystem::is_empty(ffn_cache_dir)) return 1;
    std::filesystem::permissions(ffn_cache_dir,
                                 std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec);
    kidi::ops::Context cached_context(kidi::tensor::Device::cpu());
    cached_context.configure_cache(ffn_cache_dir, source);
    const auto cached_ffn = run_ffn(cached_context);
    std::filesystem::permissions(ffn_cache_dir, std::filesystem::perms::owner_all);
    const auto expected_values = kidi::ops::require(reference_ffn.data<float>());
    const auto populated_values = kidi::ops::require(populated_ffn.data<float>());
    const auto cached_values = kidi::ops::require(cached_ffn.data<float>());
    if (!std::ranges::equal(expected_values, populated_values) || !std::ranges::equal(expected_values, cached_values))
        return 1;
    std::filesystem::remove_all(cache_root);
#endif

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