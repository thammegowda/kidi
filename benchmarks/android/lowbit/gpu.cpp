#include "lowbit.h"
#include "vulkan.h"

#include <array>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

namespace kidi::bench {

auto run_gpu(const Problem& problem, const Options& options) -> Measurement {
    using Clock = std::chrono::steady_clock;
    Measurement measurement;
    const auto start = Clock::now();
    vk::Context vulkan;
    measurement.device = vulkan.properties.deviceName;

    const bool gemv = problem.m <= vk::GEMV_MAX_ROWS;
    if (!gemv && problem.n % vk::GEMM_COLUMNS != 0)
        throw std::runtime_error("GPU GEMM requires N divisible by " + std::to_string(vk::GEMM_COLUMNS));
    const auto weights = vk::pack_weights(problem.weight, problem.n, problem.k, problem.bits, !gemv);
    auto& input = vulkan.upload(problem.input.data(), problem.input.size());
    auto& packed = vulkan.upload(weights.data(), weights.size() * sizeof(std::uint32_t));
    auto& factor = vulkan.upload(problem.factor.data(), problem.factor.size() * sizeof(float));
    auto& output = vulkan.buffer(static_cast<VkDeviceSize>(problem.m) * problem.n);
    std::memset(output.mapped, 0x7F, output.size);

    const auto set_layout = vulkan.set_layout(4);
    const auto layout = vulkan.pipeline_layout(set_layout, 3 * sizeof(std::uint32_t));
    const auto pipeline = vulkan.pipeline(vk::spirv(gemv ? vk::Kernel::GEMV : vk::Kernel::GEMM, problem.bits), layout);
    const auto set =
        vulkan.descriptor_set(vulkan.descriptor_pool(1, 4), set_layout, {&input, &packed, &factor, &output});
    const auto queries = vulkan.query_pool(2);
    const auto commands = vulkan.command_buffer();

    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vk::check(vkBeginCommandBuffer(commands, &begin), "begin commands");
    vkCmdResetQueryPool(commands, queries, 0, 2);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    const std::array<std::uint32_t, 3> shape{problem.m, problem.k, problem.n};
    vkCmdPushConstants(commands, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shape), shape.data());
    vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
    if (gemv)
        vkCmdDispatch(commands, problem.n / vk::GEMV_COLUMNS, 1, 1);
    else
        vkCmdDispatch(commands, problem.n / vk::GEMM_COLUMNS, (problem.m + vk::GEMM_ROWS - 1) / vk::GEMM_ROWS, 1);
    vk::compute_barrier(commands);
    vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
    vk::check(vkEndCommandBuffer(commands), "end commands");
    measurement.prepare_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();

    for (int iteration = 0; iteration < options.warmups + options.runs; ++iteration) {
        const auto started = Clock::now();
        vulkan.submit({&commands, 1}, true);
        const auto wall = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
        if (iteration < options.warmups) continue;
        measurement.wall_ms.push_back(wall);
        measurement.device_ms.push_back(vulkan.elapsed_ms(queries));
    }
    const auto* result = static_cast<const std::int8_t*>(output.mapped);
    measurement.output.assign(result, result + output.size);
    measurement.details = {{"kernel", gemv ? "gemv" : "gemm"}, {"api", "vulkan-1.3"}, {"int8_dot", true}};
    return measurement;
}

} // namespace kidi::bench
