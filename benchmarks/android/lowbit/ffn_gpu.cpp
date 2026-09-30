#include "ffn.h"
#include "vulkan.h"

#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace kidi::bench {
namespace {

using Clock = std::chrono::steady_clock;

struct Dispatch {
    VkPipeline pipeline{};
    VkPipelineLayout layout{};
    VkDescriptorSet set{};
    std::array<std::uint32_t, 4> constants{};
    std::uint32_t constant_bytes = 0;
    std::array<std::uint32_t, 3> groups{1, 1, 1};
};

auto encode(VkCommandBuffer commands, const Dispatch& dispatch) -> void {
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, dispatch.pipeline);
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, dispatch.layout, 0, 1, &dispatch.set, 0, nullptr);
    vkCmdPushConstants(commands, dispatch.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, dispatch.constant_bytes,
                       dispatch.constants.data());
    vkCmdDispatch(commands, dispatch.groups[0], dispatch.groups[1], dispatch.groups[2]);
    vk::compute_barrier(commands);
}

// Records dispatches with a leading barrier so resubmissions are ordered after earlier queue work.
auto record(VkCommandBuffer commands, std::span<const Dispatch> dispatches, VkQueryPool queries,
            VkCommandBufferUsageFlags usage) -> void {
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = usage;
    vk::check(vkBeginCommandBuffer(commands, &begin), "begin commands");
    if (queries) {
        vkCmdResetQueryPool(commands, queries, 0, 2);
        vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
    }
    vk::compute_barrier(commands);
    for (const auto& dispatch : dispatches) encode(commands, dispatch);
    if (queries) vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
    vk::check(vkEndCommandBuffer(commands), "end commands");
}

} // namespace

auto run_ffn_gpu(const FfnStack& stack, const Options& options) -> FfnMeasurement {
    const auto& mode = options.mode;
    if (mode != "eager" && mode != "eager-async" && mode != "encode" && mode != "replay" && mode != "replay-queued")
        throw std::runtime_error("GPU FFN modes are eager, eager-async, encode, replay, and replay-queued");
    FfnMeasurement measurement;
    const auto start = Clock::now();
    vk::Context vulkan;
    measurement.device = vulkan.properties.deviceName;
    const auto m = stack.m, hidden = stack.hidden, widest = stack.max_intermediate();
    const auto layers = stack.layers.size();
    const bool gemv = m <= vk::GEMV_MAX_ROWS;

    const auto projection_set = vulkan.set_layout(4);
    const auto projection_layout = vulkan.pipeline_layout(projection_set, 3 * sizeof(std::uint32_t));
    const auto activation_set = vulkan.set_layout(2);
    const auto activation_layout = vulkan.pipeline_layout(activation_set, 4 * sizeof(std::uint32_t));
    const auto kernel = gemv ? vk::Kernel::GEMV : vk::Kernel::GEMM;
    const std::array projection_pipelines{vulkan.pipeline(vk::spirv(kernel, 4), projection_layout),
                                          vulkan.pipeline(vk::spirv(kernel, 2), projection_layout)};
    const auto activation_pipeline = vulkan.pipeline(vk::spirv(vk::Kernel::GELU_MULTIPLY, 8), activation_layout);
    const auto pool =
        vulkan.descriptor_pool(static_cast<std::uint32_t>(3 * layers), static_cast<std::uint32_t>(10 * layers));

    auto& input = vulkan.buffer(static_cast<VkDeviceSize>(m) * hidden);
    auto& gate_up = vulkan.buffer(static_cast<VkDeviceSize>(m) * 2 * widest);
    auto& activation = vulkan.buffer(static_cast<VkDeviceSize>(m) * widest);
    std::array<vk::Buffer*, 2> outputs{&vulkan.buffer(static_cast<VkDeviceSize>(m) * hidden),
                                       &vulkan.buffer(static_cast<VkDeviceSize>(m) * hidden)};

    const auto projection = [&](const vk::Buffer& in, std::span<const std::int8_t> weights, std::uint32_t k,
                                std::uint32_t n, int bits, std::span<const float> factor, const vk::Buffer& out) {
        if (!gemv && n % vk::GEMM_COLUMNS != 0) throw std::runtime_error("GPU GEMM requires N divisible by 256");
        const auto packed = vk::pack_weights(weights, n, k, bits, !gemv);
        auto& weight_buffer = vulkan.upload(packed.data(), packed.size() * sizeof(std::uint32_t));
        auto& factor_buffer = vulkan.upload(factor.data(), factor.size_bytes());
        Dispatch dispatch;
        dispatch.pipeline = projection_pipelines[bits == 4 ? 0 : 1];
        dispatch.layout = projection_layout;
        dispatch.set = vulkan.descriptor_set(pool, projection_set, {&in, &weight_buffer, &factor_buffer, &out});
        dispatch.constants = {m, k, n, 0};
        dispatch.constant_bytes = 3 * sizeof(std::uint32_t);
        dispatch.groups =
            gemv ? std::array<std::uint32_t, 3>{n / vk::GEMV_COLUMNS, 1, 1}
                 : std::array<std::uint32_t, 3>{n / vk::GEMM_COLUMNS, (m + vk::GEMM_ROWS - 1) / vk::GEMM_ROWS, 1};
        return dispatch;
    };
    std::vector<Dispatch> dispatches;
    for (std::size_t index = 0; index < layers; ++index) {
        const auto& layer = stack.layers[index];
        const auto intermediate = layer.intermediate;
        const auto& layer_input = index == 0 ? input : *outputs[(index - 1) % 2];
        dispatches.push_back(projection(layer_input, ffn_weights(stack, index, 0), hidden, 2 * intermediate, layer.bits,
                                        layer.gate_up_factor, gate_up));
        Dispatch gelu;
        gelu.pipeline = activation_pipeline;
        gelu.layout = activation_layout;
        gelu.set = vulkan.descriptor_set(pool, activation_set, {&gate_up, &activation});
        gelu.constants = {m, intermediate, std::bit_cast<std::uint32_t>(layer.gate_up_scale),
                          std::bit_cast<std::uint32_t>(1.F / layer.hidden_scale)};
        gelu.constant_bytes = 4 * sizeof(std::uint32_t);
        gelu.groups = {(m * intermediate / 4 + vk::GELU_LOCAL_SIZE - 1) / vk::GELU_LOCAL_SIZE, 1, 1};
        dispatches.push_back(gelu);
        dispatches.push_back(projection(activation, ffn_weights(stack, index, 1), intermediate, hidden, layer.bits,
                                        layer.down_factor, *outputs[index % 2]));
    }
    const auto& final_output = *outputs[(layers - 1) % 2];

    const auto queries = vulkan.query_pool(2);
    const auto step_commands = vulkan.command_buffer();
    std::vector<VkCommandBuffer> op_commands;
    if (mode == "replay" || mode == "replay-queued")
        record(step_commands, dispatches, queries, VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);
    if (mode == "eager" || mode == "eager-async")
        for (const auto& dispatch : dispatches) {
            op_commands.push_back(vulkan.command_buffer());
            record(op_commands.back(), {&dispatch, 1}, VK_NULL_HANDLE, 0);
        }
    const auto depth = mode == "replay-queued" ? static_cast<std::size_t>(options.queue_depth) : 1;
    const std::vector<VkCommandBuffer> queued(depth, step_commands);
    measurement.prepare_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    measurement.launches_per_step = mode.starts_with("eager") ? dispatches.size() : 1;

    // Returns wall time per step; the host rewrites the input buffer in place before launching.
    const auto step = [&](const std::vector<std::int8_t>& values) {
        std::memcpy(input.mapped, values.data(), values.size());
        const auto begin = Clock::now();
        if (mode == "eager") {
            for (auto commands : op_commands) vulkan.submit({&commands, 1}, true);
        } else if (mode == "eager-async") {
            for (std::size_t index = 0; index < op_commands.size(); ++index)
                vulkan.submit({&op_commands[index], 1}, index + 1 == op_commands.size());
        } else if (mode == "encode") {
            vk::check(vkResetCommandBuffer(step_commands, 0), "reset commands");
            record(step_commands, dispatches, queries, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
            vulkan.submit({&step_commands, 1}, true);
        } else {
            vulkan.submit(queued, true);
        }
        return std::chrono::duration<double, std::milli>(Clock::now() - begin).count() / static_cast<double>(depth);
    };
    const bool timestamps = mode == "encode" || mode.starts_with("replay");
    for (int iteration = 0; iteration < options.warmups + options.runs; ++iteration) {
        const auto elapsed = step(stack.inputs[iteration % 2]);
        if (iteration < options.warmups) continue;
        measurement.step_ms.push_back(elapsed);
        if (timestamps) measurement.device_ms.push_back(vulkan.elapsed_ms(queries));
    }
    for (int index = 0; index < 2; ++index) {
        step(stack.inputs[index]);
        const auto* result = static_cast<const std::int8_t*>(final_output.mapped);
        measurement.outputs[index].assign(result, result + static_cast<std::size_t>(m) * hidden);
    }
    measurement.details = {
        {"kernel", gemv ? "gemv" : "gemm"}, {"dispatches", dispatches.size()}, {"queue_depth", depth}};
    return measurement;
}

} // namespace kidi::bench
