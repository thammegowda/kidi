#include <vulkan/vulkan.h>
#include <nlohmann/json.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

namespace {
constexpr std::uint32_t WIDTH = 2048, ROWS = 1024;

auto check(VkResult result, const char* operation) -> void {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(operation) + ": " + std::to_string(result));
}

struct Probe {
    VkInstance instance{};
    VkPhysicalDevice physical{};
    VkDevice device{};
    VkQueue queue{};
    VkCommandPool commands{};
    VkDescriptorPool descriptors{};
    VkDescriptorSetLayout descriptor_layout{};
    VkPipelineLayout layout{};
    VkShaderModule shader{};
    VkPipeline pipeline{};
    VkQueryPool queries{};
    VkFence fence{};
    std::array<VkBuffer, 4> buffers{};
    std::array<VkDeviceMemory, 4> memory{};
    std::array<void*, 4> mapped{};

    ~Probe() {
        if (device) {
            vkDeviceWaitIdle(device);
            if (fence) vkDestroyFence(device, fence, nullptr);
            if (queries) vkDestroyQueryPool(device, queries, nullptr);
            if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
            if (shader) vkDestroyShaderModule(device, shader, nullptr);
            if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
            if (descriptors) vkDestroyDescriptorPool(device, descriptors, nullptr);
            if (descriptor_layout) vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
            if (commands) vkDestroyCommandPool(device, commands, nullptr);
            for (std::size_t index = 0; index < buffers.size(); ++index) {
                if (mapped[index]) vkUnmapMemory(device, memory[index]);
                if (buffers[index]) vkDestroyBuffer(device, buffers[index], nullptr);
                if (memory[index]) vkFreeMemory(device, memory[index], nullptr);
            }
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    auto create_buffer(std::size_t index, VkDeviceSize size) -> void {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        check(vkCreateBuffer(device, &info, nullptr, &buffers[index]), "create buffer");
        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(device, buffers[index], &requirements);
        VkPhysicalDeviceMemoryProperties properties;
        vkGetPhysicalDeviceMemoryProperties(physical, &properties);
        std::uint32_t type = properties.memoryTypeCount;
        constexpr auto FLAGS = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (std::uint32_t candidate = 0; candidate < properties.memoryTypeCount; ++candidate) {
            if ((requirements.memoryTypeBits & (1U << candidate)) &&
                (properties.memoryTypes[candidate].propertyFlags & FLAGS) == FLAGS) {
                type = candidate;
                break;
            }
        }
        if (type == properties.memoryTypeCount) throw std::runtime_error("no coherent host-visible GPU memory");
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = type;
        check(vkAllocateMemory(device, &allocation, nullptr, &memory[index]), "allocate memory");
        check(vkBindBufferMemory(device, buffers[index], memory[index], 0), "bind memory");
        check(vkMapMemory(device, memory[index], 0, size, 0, &mapped[index]), "map memory");
    }
};

auto run(const char* shader_file, std::string_view variant) -> void {
    Probe probe;
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "Kidi Adreno compute probe";
    application.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instance{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance.pApplicationInfo = &application;
    check(vkCreateInstance(&instance, nullptr, &probe.instance), "create instance");
    std::uint32_t device_count = 0;
    check(vkEnumeratePhysicalDevices(probe.instance, &device_count, nullptr), "enumerate GPUs");
    std::vector<VkPhysicalDevice> devices(device_count);
    check(vkEnumeratePhysicalDevices(probe.instance, &device_count, devices.data()), "enumerate GPUs");
    VkPhysicalDeviceProperties properties{};
    for (auto candidate : devices) {
        vkGetPhysicalDeviceProperties(candidate, &properties);
        if (std::string_view(properties.deviceName).find("Adreno") != std::string_view::npos) {
            probe.physical = candidate;
            break;
        }
    }
    if (!probe.physical) throw std::runtime_error("Adreno GPU not found; CPU fallback is forbidden");
    VkPhysicalDeviceVulkan13Features available{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &available;
    vkGetPhysicalDeviceFeatures2(probe.physical, &features);
    VkPhysicalDeviceVulkan13Properties properties13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    VkPhysicalDeviceSubgroupProperties subgroups{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    subgroups.pNext = &properties13;
    VkPhysicalDeviceProperties2 extended{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    extended.pNext = &subgroups;
    vkGetPhysicalDeviceProperties2(probe.physical, &extended);
    if (variant != "baseline" &&
        (subgroups.subgroupSize != 64 || !(subgroups.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT)))
        throw std::runtime_error("probe requires 64-lane arithmetic subgroups");
    if (variant == "dot" &&
        (!available.shaderIntegerDotProduct || !properties13.integerDotProduct4x8BitPackedSignedAccelerated))
        throw std::runtime_error("packed signed INT8 dot product is not hardware accelerated");
    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(probe.physical, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(probe.physical, &family_count, families.data());
    std::uint32_t family = family_count;
    for (std::uint32_t index = 0; index < family_count; ++index)
        if ((families[index].queueFlags & VK_QUEUE_COMPUTE_BIT) && families[index].timestampValidBits) {
            family = index;
            break;
        }
    if (family == family_count) throw std::runtime_error("no timestamp-capable compute queue");
    constexpr float PRIORITY = 1.F;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &PRIORITY;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    VkPhysicalDeviceVulkan13Features enabled{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    enabled.shaderIntegerDotProduct = variant == "dot";
    device_info.pNext = &enabled;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    check(vkCreateDevice(probe.physical, &device_info, nullptr, &probe.device), "create GPU device");
    vkGetDeviceQueue(probe.device, family, 0, &probe.queue);

    const std::array<VkDeviceSize, 4> sizes{WIDTH * sizeof(float), WIDTH * ROWS / 2, ROWS * sizeof(float),
                                            ROWS * sizeof(float)};
    for (std::size_t index = 0; index < sizes.size(); ++index) probe.create_buffer(index, sizes[index]);
    auto* activation = static_cast<float*>(probe.mapped[0]);
    auto* packed = static_cast<std::uint32_t*>(probe.mapped[1]);
    auto* scales = static_cast<float*>(probe.mapped[2]);
    auto* output = static_cast<float*>(probe.mapped[3]);
    std::fill_n(packed, WIDTH * ROWS / 8, 0U);
    for (std::uint32_t channel = 0; channel < WIDTH; ++channel)
        activation[channel] = (static_cast<int>(channel % 31) - 15) * 0.0625F;
    std::vector<float> expected(ROWS);
    for (std::uint32_t row = 0; row < ROWS; ++row) {
        scales[row] = 0.03125F;
        output[row] = std::numeric_limits<float>::quiet_NaN();
        for (std::uint32_t channel = 0; channel < WIDTH; ++channel) {
            const auto weight = static_cast<int>((row * 17 + channel * 5) % 16) - 8;
            const auto offset = row * WIDTH + channel;
            packed[offset / 8] |= static_cast<std::uint32_t>(weight & 15) << (4 * (offset % 8));
            expected[row] += activation[channel] * weight * scales[row];
        }
    }

    std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
    for (std::uint32_t index = 0; index < bindings.size(); ++index)
        bindings[index] = {index, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo descriptor_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptor_info.bindingCount = bindings.size();
    descriptor_info.pBindings = bindings.data();
    check(vkCreateDescriptorSetLayout(probe.device, &descriptor_info, nullptr, &probe.descriptor_layout),
          "create descriptors");
    const VkPushConstantRange constants{VK_SHADER_STAGE_COMPUTE_BIT, 0, 2 * sizeof(std::uint32_t)};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &probe.descriptor_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &constants;
    check(vkCreatePipelineLayout(probe.device, &layout_info, nullptr, &probe.layout), "create layout");

    std::ifstream input(shader_file, std::ios::binary | std::ios::ate);
    const auto bytes = input.tellg();
    if (!input || bytes <= 0 || bytes % 4) throw std::runtime_error("invalid SPIR-V file");
    std::vector<std::uint32_t> code(static_cast<std::size_t>(bytes) / 4);
    input.seekg(0);
    if (!input.read(reinterpret_cast<char*>(code.data()), bytes)) throw std::runtime_error("truncated shader");
    VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader_info.codeSize = static_cast<std::size_t>(bytes);
    shader_info.pCode = code.data();
    check(vkCreateShaderModule(probe.device, &shader_info, nullptr, &probe.shader), "create shader");
    VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = probe.shader;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = probe.layout;
    check(vkCreateComputePipelines(probe.device, {}, 1, &pipeline_info, nullptr, &probe.pipeline), "compile pipeline");

    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    check(vkCreateDescriptorPool(probe.device, &pool_info, nullptr, &probe.descriptors), "create pool");
    VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = probe.descriptors;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &probe.descriptor_layout;
    VkDescriptorSet descriptor_set{};
    check(vkAllocateDescriptorSets(probe.device, &set_info, &descriptor_set), "allocate descriptors");
    std::array<VkDescriptorBufferInfo, 4> buffer_info{};
    std::array<VkWriteDescriptorSet, 4> writes{};
    for (std::size_t index = 0; index < writes.size(); ++index) {
        buffer_info[index] = {probe.buffers[index], 0, sizes[index]};
        writes[index] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[index].dstSet = descriptor_set;
        writes[index].dstBinding = index;
        writes[index].descriptorCount = 1;
        writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[index].pBufferInfo = &buffer_info[index];
    }
    vkUpdateDescriptorSets(probe.device, writes.size(), writes.data(), 0, nullptr);
    VkCommandPoolCreateInfo command_pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    command_pool.queueFamilyIndex = family;
    check(vkCreateCommandPool(probe.device, &command_pool, nullptr, &probe.commands), "create command pool");
    VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = probe.commands;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = 1;
    VkCommandBuffer commands{};
    check(vkAllocateCommandBuffers(probe.device, &allocation, &commands), "allocate commands");
    VkQueryPoolCreateInfo query_info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    query_info.queryCount = 2;
    check(vkCreateQueryPool(probe.device, &query_info, nullptr, &probe.queries), "create timestamps");
    const VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    check(vkCreateFence(probe.device, &fence_info, nullptr, &probe.fence), "create fence");

    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    check(vkBeginCommandBuffer(commands, &begin), "begin commands");
    vkCmdResetQueryPool(commands, probe.queries, 0, 2);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, probe.pipeline);
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, probe.layout, 0, 1, &descriptor_set, 0, nullptr);
    const std::array dimensions{WIDTH, ROWS};
    vkCmdPushConstants(commands, probe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(dimensions), dimensions.data());
    vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, probe.queries, 0);
    vkCmdDispatch(commands, ROWS, 1, 1);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0,
                         nullptr, 0, nullptr);
    vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, probe.queries, 1);
    check(vkEndCommandBuffer(commands), "end commands");

    nlohmann::json runs = nlohmann::json::array();
    for (int iteration = 0; iteration < 21; ++iteration) {
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commands;
        const auto started = std::chrono::steady_clock::now();
        check(vkQueueSubmit(probe.queue, 1, &submit, probe.fence), "submit GPU work");
        check(vkWaitForFences(probe.device, 1, &probe.fence, VK_TRUE, 10000000000ULL), "wait for GPU");
        const auto wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        std::array<std::uint64_t, 2> timestamps{};
        check(vkGetQueryPoolResults(probe.device, probe.queries, 0, 2, sizeof(timestamps), timestamps.data(),
                                    sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
              "read timestamps");
        for (std::size_t row = 0; row < ROWS; ++row)
            if (!std::isfinite(output[row]) || output[row] != expected[row])
                throw std::runtime_error("GPU Q4 projection differs from exact reference at " + std::to_string(row));
        const auto mask = families[family].timestampValidBits == 64
                              ? ~std::uint64_t{0}
                              : (std::uint64_t{1} << families[family].timestampValidBits) - 1;
        const auto gpu_ms = ((timestamps[1] - timestamps[0]) & mask) * properties.limits.timestampPeriod / 1e6;
        runs.push_back({{"warmup", iteration == 0},
                        {"gpu_ms", std::round(gpu_ms * 1e5) / 1e5},
                        {"submit_wait_ms", std::round(wall * 1e5) / 1e5}});
        check(vkResetFences(probe.device, 1, &probe.fence), "reset fence");
    }
    std::cout << nlohmann::json{{"device", properties.deviceName},
                                {"variant", variant},
                                {"calibrated_activation_scale", variant == "dot" ? 0.0625 : 0.0},
                                {"backend", "vulkan"},
                                {"workload", "signed-q4-fp32-matvec"},
                                {"width", WIDTH},
                                {"rows", ROWS},
                                {"weights_bytes", sizes[1]},
                                {"exact_outputs", true},
                                {"cpu_fallback", false},
                                {"runs", runs}}
                     .dump()
              << '\n';
}
} // namespace

auto main(int argc, char** argv) -> int {
    try {
        const std::string_view variant = argc == 3 ? argv[2] : "baseline";
        if ((argc != 2 && argc != 3) || (variant != "baseline" && variant != "subgroup" && variant != "dot"))
            throw std::runtime_error("usage: kidi_vulkan_probe SHADER.spv [baseline|subgroup|dot]");
        run(argv[1], variant);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}