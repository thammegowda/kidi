#include "vulkan.h"

#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

namespace kidi::bench::vk {
namespace {

constexpr std::uint32_t GEMV_BITS8[] = {
#include "gemv_bits8.inc"
};
constexpr std::uint32_t GEMV_BITS4[] = {
#include "gemv_bits4.inc"
};
constexpr std::uint32_t GEMV_BITS2[] = {
#include "gemv_bits2.inc"
};
constexpr std::uint32_t GEMM_BITS8[] = {
#include "gemm_bits8.inc"
};
constexpr std::uint32_t GEMM_BITS4[] = {
#include "gemm_bits4.inc"
};
constexpr std::uint32_t GEMM_BITS2[] = {
#include "gemm_bits2.inc"
};
constexpr std::uint32_t GELU_MULTIPLY[] = {
#include "gelu_multiply.inc"
};

} // namespace

auto check(VkResult result, const char* operation) -> void {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(operation) + ": VkResult " + std::to_string(result));
}

auto spirv(Kernel kernel, int bits) -> std::span<const std::uint32_t> {
    if (kernel == Kernel::GELU_MULTIPLY) return GELU_MULTIPLY;
    const bool gemv = kernel == Kernel::GEMV;
    switch (bits) {
        case 8:
            return gemv ? std::span<const std::uint32_t>(GEMV_BITS8) : std::span<const std::uint32_t>(GEMM_BITS8);
        case 4:
            return gemv ? std::span<const std::uint32_t>(GEMV_BITS4) : std::span<const std::uint32_t>(GEMM_BITS4);
        case 2:
            return gemv ? std::span<const std::uint32_t>(GEMV_BITS2) : std::span<const std::uint32_t>(GEMM_BITS2);
        default:
            throw std::runtime_error("unsupported weight bits");
    }
}

auto pack_weights(std::span<const std::int8_t> weights, std::uint32_t n, std::uint32_t k, int bits,
                  bool k_major) -> std::vector<std::uint32_t> {
    const auto values_per_word = static_cast<std::uint32_t>(32 / bits);
    const auto words_per_column = k / values_per_word;
    const auto mask = (1U << bits) - 1;
    std::vector<std::uint32_t> packed(static_cast<std::size_t>(n) * words_per_column);
    for (std::uint32_t column = 0; column < n; ++column)
        for (std::uint32_t word = 0; word < words_per_column; ++word) {
            std::uint32_t value = 0;
            for (std::uint32_t element = 0; element < values_per_word; ++element) {
                const auto weight =
                    static_cast<std::uint32_t>(static_cast<std::uint8_t>(
                        weights[static_cast<std::size_t>(column) * k + word * values_per_word + element])) &
                    mask;
                value |= weight << ((element % 4) * 8 + (element / 4) * bits);
            }
            const auto index = k_major ? static_cast<std::size_t>(word) * n + column
                                       : static_cast<std::size_t>(column) * words_per_column + word;
            packed[index] = value;
        }
    return packed;
}

Context::Context() {
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "kidi-lowbit-bench";
    application.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &application;
    check(vkCreateInstance(&instance_info, nullptr, &instance_), "create instance");
    std::uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "enumerate GPUs");
    std::vector<VkPhysicalDevice> candidates(count);
    check(vkEnumeratePhysicalDevices(instance_, &count, candidates.data()), "enumerate GPUs");
    for (auto candidate : candidates) {
        vkGetPhysicalDeviceProperties(candidate, &properties);
        if (std::string_view(properties.deviceName).find("Adreno") != std::string_view::npos) {
            physical_ = candidate;
            break;
        }
    }
    if (!physical_) throw std::runtime_error("Adreno GPU not found; CPU fallback is forbidden");

    VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &features13;
    vkGetPhysicalDeviceFeatures2(physical_, &features);
    VkPhysicalDeviceVulkan13Properties properties13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    VkPhysicalDeviceSubgroupProperties subgroups{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    subgroups.pNext = &properties13;
    VkPhysicalDeviceProperties2 extended{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    extended.pNext = &subgroups;
    vkGetPhysicalDeviceProperties2(physical_, &extended);
    if (!features13.shaderIntegerDotProduct || !properties13.integerDotProduct4x8BitPackedSignedAccelerated)
        throw std::runtime_error("packed signed INT8 dot product is not hardware accelerated");
    if (subgroups.subgroupSize != 64 || !(subgroups.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT))
        throw std::runtime_error("GEMV kernel requires 64-lane arithmetic subgroups");

    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &count, families.data());
    family_ = count;
    for (std::uint32_t index = 0; index < count; ++index)
        if ((families[index].queueFlags & VK_QUEUE_COMPUTE_BIT) && families[index].timestampValidBits) {
            family_ = index;
            timestamp_bits_ = families[index].timestampValidBits;
            break;
        }
    if (family_ == count) throw std::runtime_error("no timestamp-capable compute queue");

    constexpr float PRIORITY = 1.F;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = family_;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &PRIORITY;
    VkPhysicalDeviceVulkan13Features enabled{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    enabled.shaderIntegerDotProduct = VK_TRUE;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.pNext = &enabled;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    check(vkCreateDevice(physical_, &device_info, nullptr, &device), "create device");
    vkGetDeviceQueue(device, family_, 0, &queue_);

    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = family_;
    check(vkCreateCommandPool(device, &pool_info, nullptr, &commands_), "create command pool");
    const VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    check(vkCreateFence(device, &fence_info, nullptr, &fence_), "create fence");
}

Context::~Context() {
    if (device) {
        vkDeviceWaitIdle(device);
        for (auto& buffer : buffers_) {
            if (buffer.mapped) vkUnmapMemory(device, buffer.memory);
            if (buffer.buffer) vkDestroyBuffer(device, buffer.buffer, nullptr);
            if (buffer.memory) vkFreeMemory(device, buffer.memory, nullptr);
        }
        for (auto pipeline : pipelines_) vkDestroyPipeline(device, pipeline, nullptr);
        for (auto shader : shaders_) vkDestroyShaderModule(device, shader, nullptr);
        for (auto pool : descriptor_pools_) vkDestroyDescriptorPool(device, pool, nullptr);
        for (auto layout : pipeline_layouts_) vkDestroyPipelineLayout(device, layout, nullptr);
        for (auto layout : set_layouts_) vkDestroyDescriptorSetLayout(device, layout, nullptr);
        for (auto pool : query_pools_) vkDestroyQueryPool(device, pool, nullptr);
        if (fence_) vkDestroyFence(device, fence_, nullptr);
        if (commands_) vkDestroyCommandPool(device, commands_, nullptr);
        vkDestroyDevice(device, nullptr);
    }
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

auto Context::buffer(VkDeviceSize size) -> Buffer& {
    auto& target = buffers_.emplace_back();
    target.size = size;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    check(vkCreateBuffer(device, &info, nullptr, &target.buffer), "create buffer");
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device, target.buffer, &requirements);
    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(physical_, &memory);
    constexpr auto HOST = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    std::uint32_t chosen = memory.memoryTypeCount;
    for (auto wanted : {HOST | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, HOST}) {
        for (std::uint32_t type = 0; type < memory.memoryTypeCount && chosen == memory.memoryTypeCount; ++type)
            if ((requirements.memoryTypeBits & (1U << type)) &&
                (memory.memoryTypes[type].propertyFlags & wanted) == wanted)
                chosen = type;
        if (chosen != memory.memoryTypeCount) break;
    }
    if (chosen == memory.memoryTypeCount) throw std::runtime_error("no host-visible GPU memory");
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = chosen;
    check(vkAllocateMemory(device, &allocation, nullptr, &target.memory), "allocate memory");
    check(vkBindBufferMemory(device, target.buffer, target.memory, 0), "bind memory");
    check(vkMapMemory(device, target.memory, 0, size, 0, &target.mapped), "map memory");
    return target;
}

auto Context::upload(const void* data, VkDeviceSize size) -> Buffer& {
    auto& target = buffer(size);
    std::memcpy(target.mapped, data, size);
    return target;
}

auto Context::set_layout(std::uint32_t bindings) -> VkDescriptorSetLayout {
    std::vector<VkDescriptorSetLayoutBinding> entries(bindings);
    for (std::uint32_t index = 0; index < bindings; ++index)
        entries[index] = {index, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    info.bindingCount = bindings;
    info.pBindings = entries.data();
    auto& layout = set_layouts_.emplace_back();
    check(vkCreateDescriptorSetLayout(device, &info, nullptr, &layout), "create set layout");
    return layout;
}

auto Context::pipeline_layout(VkDescriptorSetLayout set_layout, std::uint32_t push_bytes) -> VkPipelineLayout {
    const VkPushConstantRange constants{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes};
    VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    info.setLayoutCount = 1;
    info.pSetLayouts = &set_layout;
    info.pushConstantRangeCount = 1;
    info.pPushConstantRanges = &constants;
    auto& layout = pipeline_layouts_.emplace_back();
    check(vkCreatePipelineLayout(device, &info, nullptr, &layout), "create pipeline layout");
    return layout;
}

auto Context::pipeline(std::span<const std::uint32_t> code, VkPipelineLayout layout) -> VkPipeline {
    VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader_info.codeSize = code.size_bytes();
    shader_info.pCode = code.data();
    auto& shader = shaders_.emplace_back();
    check(vkCreateShaderModule(device, &shader_info, nullptr, &shader), "create shader");
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = shader;
    info.stage.pName = "main";
    info.layout = layout;
    auto& pipeline = pipelines_.emplace_back();
    check(vkCreateComputePipelines(device, {}, 1, &info, nullptr, &pipeline), "compile pipeline");
    return pipeline;
}

auto Context::descriptor_pool(std::uint32_t sets, std::uint32_t descriptors) -> VkDescriptorPool {
    const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptors};
    VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    info.maxSets = sets;
    info.poolSizeCount = 1;
    info.pPoolSizes = &size;
    auto& pool = descriptor_pools_.emplace_back();
    check(vkCreateDescriptorPool(device, &info, nullptr, &pool), "create descriptor pool");
    return pool;
}

auto Context::descriptor_set(VkDescriptorPool pool, VkDescriptorSetLayout layout,
                             std::initializer_list<const Buffer*> buffers) -> VkDescriptorSet {
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = pool;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &layout;
    VkDescriptorSet set{};
    check(vkAllocateDescriptorSets(device, &allocate, &set), "allocate descriptor set");
    std::vector<VkDescriptorBufferInfo> infos;
    std::vector<VkWriteDescriptorSet> writes;
    infos.reserve(buffers.size());
    for (const auto* buffer : buffers) {
        infos.push_back({buffer->buffer, 0, buffer->size});
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set;
        write.dstBinding = static_cast<std::uint32_t>(writes.size());
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &infos.back();
        writes.push_back(write);
    }
    vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    return set;
}

auto Context::command_buffer() -> VkCommandBuffer {
    VkCommandBufferAllocateInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    info.commandPool = commands_;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = 1;
    VkCommandBuffer commands{};
    check(vkAllocateCommandBuffers(device, &info, &commands), "allocate command buffer");
    return commands;
}

auto Context::query_pool(std::uint32_t queries) -> VkQueryPool {
    VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = queries;
    auto& pool = query_pools_.emplace_back();
    check(vkCreateQueryPool(device, &info, nullptr, &pool), "create query pool");
    return pool;
}

auto Context::submit(std::span<const VkCommandBuffer> commands, bool wait_for_completion) -> void {
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = static_cast<std::uint32_t>(commands.size());
    submit.pCommandBuffers = commands.data();
    // Only the last submission of a step signals the fence; earlier ones are ordered by in-buffer barriers.
    check(vkQueueSubmit(queue_, 1, &submit, wait_for_completion ? fence_ : VK_NULL_HANDLE), "submit");
    if (wait_for_completion) {
        fence_pending_ = true;
        wait();
    }
}

auto Context::wait() -> void {
    if (!fence_pending_) return;
    check(vkWaitForFences(device, 1, &fence_, VK_TRUE, 60'000'000'000ULL), "wait for GPU");
    check(vkResetFences(device, 1, &fence_), "reset fence");
    fence_pending_ = false;
}

auto Context::elapsed_ms(VkQueryPool pool) -> double {
    std::array<std::uint64_t, 2> timestamps{};
    check(vkGetQueryPoolResults(device, pool, 0, 2, sizeof(timestamps), timestamps.data(), sizeof(std::uint64_t),
                                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
          "read timestamps");
    const auto mask = timestamp_bits_ == 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << timestamp_bits_) - 1;
    return ((timestamps[1] - timestamps[0]) & mask) * properties.limits.timestampPeriod / 1e6;
}

auto compute_barrier(VkCommandBuffer commands) -> void {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr,
                         0, nullptr);
}

} // namespace kidi::bench::vk
