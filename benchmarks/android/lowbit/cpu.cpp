#include "lowbit.h"

#include "kidi/runtime/ynn/graph.h"
#include "ynnpack/composites/composites.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace kidi::bench {
namespace {

using Clock = std::chrono::steady_clock;

// Pack unpacked signed `bits`-bit weights [n][k] into sub-byte packed [n][k / (8 / bits)] for symmetric_linear.
auto pack_subbyte(const std::vector<std::int8_t>& weight, std::uint32_t n, std::uint32_t k, int bits)
    -> std::vector<std::uint8_t> {
    const auto per_byte = static_cast<std::uint32_t>(8 / bits);
    const auto row_bytes = k / per_byte;
    const auto mask = static_cast<std::uint8_t>((1U << bits) - 1);
    std::vector<std::uint8_t> packed(static_cast<std::size_t>(n) * row_bytes, 0);
    for (std::uint32_t row = 0; row < n; ++row) {
        for (std::uint32_t col = 0; col < k; ++col) {
            const auto v = static_cast<std::uint8_t>(weight[static_cast<std::size_t>(row) * k + col]) & mask;
            packed[static_cast<std::size_t>(row) * row_bytes + col / per_byte] |=
                static_cast<std::uint8_t>(v << ((col % per_byte) * bits));
        }
    }
    return packed;
}

} // namespace

auto run_cpu(const Problem& problem, const Options& options) -> Measurement {
    Measurement measurement;
    measurement.device = "cpu-ynnpack";
    const auto total_out = static_cast<std::size_t>(problem.m) * problem.n;
    measurement.output.resize(total_out);
    runtime::ynn::set_thread_count(options.threads);
    const auto start = Clock::now();

    std::vector<float> float_out(total_out);
    const bool use_sym = ::ynn::is_symmetric_linear_supported(problem.bits) && (problem.k % 32 == 0);
    std::vector<std::uint8_t> sym_tiles;
    std::vector<std::int8_t> sym_act;
    std::vector<float> sym_act_scale(problem.m, 1.0F);
    ::ynn::dense_linear_plan plan{};
    ::ynn::aligned_byte_buffer dense_packed;
    std::vector<std::int32_t> dense_sums;
    std::vector<float> dense_a_scales(problem.m, 1.0F);
    std::vector<std::int32_t> dense_a_zps(problem.m, 0);
    bool dense_transposed_a = false;
    std::size_t dense_tile_m = 1, dense_a_bytes = 0, dense_scratch_bytes = 0;
    ::ynn::aligned_byte_buffer dense_a_packed;
    std::vector<std::int32_t> dense_scratch;

    if (use_sym) {
        const auto subbyte = pack_subbyte(problem.weight, problem.n, problem.k, problem.bits);
        const auto tiles = ::ynn::blockwise_linear_tile_count(problem.n);
        sym_tiles.resize(::ynn::packed_symmetric_linear_size(problem.bits, problem.n, problem.k, problem.k));
        ::ynn::pack_symmetric_linear_tiles(problem.bits, problem.n, problem.k, 0, problem.k, subbyte.data(),
                                           problem.factor.data(), false, 0, tiles, sym_tiles.data());
        const auto padded_k = ::ynn::symmetric_linear_padded_block(problem.k);
        sym_act.resize(static_cast<std::size_t>(problem.m) * padded_k, 0);
        for (std::uint32_t r = 0; r < problem.m; ++r) {
            std::copy_n(problem.input.data() + static_cast<std::size_t>(r) * problem.k, problem.k,
                        sym_act.data() + static_cast<std::size_t>(r) * padded_k);
        }
    } else {
        if (::ynn::make_dense_linear_plan(ynn_type_int8, ynn_type_int8, ynn_type_int32, problem.n, problem.k, &plan) !=
            ynn_status_success) {
            throw std::runtime_error("failed to create INT8 dense linear plan");
        }
        dense_packed.resize(plan.total_bytes);
        dense_sums.resize(problem.n);
        ::ynn::pack_dense_linear_tiles(plan, true, problem.weight.data(), problem.k, 0, plan.blocks_n,
                                       dense_packed.data(), dense_sums.data());
        ::ynn::dense_linear_select_kernel(plan, problem.m, &dense_transposed_a, &dense_tile_m, &dense_a_bytes);
        dense_a_packed.resize(dense_a_bytes);
        dense_scratch_bytes = ::ynn::quantized_linear_scratch_bytes(plan, problem.m, dense_tile_m);
        dense_scratch.resize(dense_scratch_bytes / sizeof(std::int32_t));
    }

    const auto invoke = [&] {
        if (use_sym) {
            const auto tiles = ::ynn::blockwise_linear_tile_count(problem.n);
            if (options.threads > 1 && tiles > 1) {
                (void)runtime::ynn::parallel_ranges(tiles, 1, [&](std::size_t t_begin, std::size_t t_end) {
                    ::ynn::symmetric_linear_f32(problem.bits, problem.m, problem.n, problem.k, problem.k, 1.0F,
                                                sym_tiles.data(), sym_act.data(), sym_act_scale.data(),
                                                float_out.data(), problem.n, t_begin, t_end);
                });
            } else {
                ::ynn::symmetric_linear_f32(problem.bits, problem.m, problem.n, problem.k, problem.k, 1.0F,
                                            sym_tiles.data(), sym_act.data(), sym_act_scale.data(), float_out.data(),
                                            problem.n, 0, tiles);
            }
        } else {
            const void* eff_a = problem.input.data();
            if (dense_transposed_a) {
                ::ynn::dense_linear_pack_a(plan, problem.m, dense_tile_m, problem.input.data(), problem.k,
                                           dense_a_packed.data());
                eff_a = dense_a_packed.data();
            }
            if (options.threads > 1 && plan.blocks_n > 1) {
                (void)runtime::ynn::parallel_ranges_with_worker(
                    plan.blocks_n, 1, dense_scratch_bytes,
                    [&](const runtime::ynn::WorkerContext& w, std::size_t t_begin, std::size_t t_end) {
                        ::ynn::quantized_linear_run_tiles(plan, problem.m, dense_transposed_a, dense_tile_m, eff_a,
                                                          problem.k, dense_packed.data(), dense_sums.data(),
                                                          dense_a_scales.data(), dense_a_zps.data(),
                                                          problem.factor.data(), nullptr, float_out.data(), problem.n,
                                                          reinterpret_cast<std::int32_t*>(w.scratch), t_begin, t_end);
                    });
            } else {
                ::ynn::quantized_linear_run_tiles(plan, problem.m, dense_transposed_a, dense_tile_m, eff_a, problem.k,
                                                  dense_packed.data(), dense_sums.data(), dense_a_scales.data(),
                                                  dense_a_zps.data(), problem.factor.data(), nullptr, float_out.data(),
                                                  problem.n, dense_scratch.data(), 0, plan.blocks_n);
            }
        }
        for (std::size_t i = 0; i < total_out; ++i) {
            measurement.output[i] = static_cast<std::int8_t>(std::clamp(std::nearbyint(float_out[i]), -128.0F, 127.0F));
        }
    };

    measurement.prepare_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();

    for (int iteration = 0; iteration < options.warmups + options.runs; ++iteration) {
        const auto begin = Clock::now();
        invoke();
        const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
        if (iteration >= options.warmups) measurement.wall_ms.push_back(elapsed);
    }
    measurement.details = {{"threads", options.threads}, {"arch", runtime::ynn::supported_arch_names()}};
    return measurement;
}

} // namespace kidi::bench
