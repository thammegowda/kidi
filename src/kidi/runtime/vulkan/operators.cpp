#include "kidi/runtime/operator.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kidi/ops/context.h"

namespace kidi::runtime {
namespace {

using ops::require;
using tensor::Device;
using tensor::DeviceKind;
using tensor::DType;
using tensor::Tensor;

constexpr std::uint32_t GEMV_BITS8[] = {
#include "packed_gemv_bits8.inc"
};
constexpr std::uint32_t GEMV_BITS4[] = {
#include "packed_gemv_bits4.inc"
};
constexpr std::uint32_t GEMV_BITS2[] = {
#include "packed_gemv_bits2.inc"
};
constexpr std::uint32_t GEMV_I8_BITS8[] = {
#include "packed_gemv_i8_bits8.inc"
};
constexpr std::uint32_t GEMV_I8_BITS4[] = {
#include "packed_gemv_i8_bits4.inc"
};
constexpr std::uint32_t GEMV_I8_BITS2[] = {
#include "packed_gemv_i8_bits2.inc"
};
constexpr std::uint32_t GEMM_BITS8[] = {
#include "packed_gemm_bits8.inc"
};
constexpr std::uint32_t GEMM_BITS4[] = {
#include "packed_gemm_bits4.inc"
};
constexpr std::uint32_t GEMM_BITS2[] = {
#include "packed_gemm_bits2.inc"
};
constexpr std::uint32_t GEMM_I8_BITS8[] = {
#include "packed_gemm_i8_bits8.inc"
};
constexpr std::uint32_t GEMM_I8_BITS4[] = {
#include "packed_gemm_i8_bits4.inc"
};
constexpr std::uint32_t GEMM_I8_BITS2[] = {
#include "packed_gemm_i8_bits2.inc"
};
constexpr std::uint32_t GELU_MULTIPLY[] = {
#include "gelu_multiply.inc"
};

enum class Kernel { GEMV, GEMM, GEMV_I8, GEMM_I8, GELU_MULTIPLY };

auto spirv(Kernel kernel, int bits) -> std::span<const std::uint32_t> {
    if (kernel == Kernel::GELU_MULTIPLY) return GELU_MULTIPLY;
    const bool gemv = kernel == Kernel::GEMV || kernel == Kernel::GEMV_I8;
    const bool output_i8 = kernel == Kernel::GEMV_I8 || kernel == Kernel::GEMM_I8;
    switch (bits) {
        case 8:
            if (gemv)
                return output_i8 ? std::span<const std::uint32_t>(GEMV_I8_BITS8)
                                 : std::span<const std::uint32_t>(GEMV_BITS8);
            return output_i8 ? std::span<const std::uint32_t>(GEMM_I8_BITS8)
                             : std::span<const std::uint32_t>(GEMM_BITS8);
        case 4:
            if (gemv)
                return output_i8 ? std::span<const std::uint32_t>(GEMV_I8_BITS4)
                                 : std::span<const std::uint32_t>(GEMV_BITS4);
            return output_i8 ? std::span<const std::uint32_t>(GEMM_I8_BITS4)
                             : std::span<const std::uint32_t>(GEMM_BITS4);
        case 2:
            if (gemv)
                return output_i8 ? std::span<const std::uint32_t>(GEMV_I8_BITS2)
                                 : std::span<const std::uint32_t>(GEMV_BITS2);
            return output_i8 ? std::span<const std::uint32_t>(GEMM_I8_BITS2)
                             : std::span<const std::uint32_t>(GEMM_BITS2);
        default:
            throw ops::Failure({ErrorCode::UNSUPPORTED, "unsupported Vulkan packed projection precision"});
    }
}

auto check(VkResult result, std::string_view operation) -> void {
    if (result != VK_SUCCESS)
        throw ops::Failure(
            {ErrorCode::RUNTIME, std::string(operation) + " failed with VkResult " + std::to_string(result)});
}

struct GpuBuffer {
    VkDevice device{};
    VkBuffer buffer{};
    VkDeviceMemory memory{};
    void* mapped{};
    VkDeviceSize size{};

    GpuBuffer() = default;
    GpuBuffer(const GpuBuffer&) = delete;
    auto operator=(const GpuBuffer&) -> GpuBuffer& = delete;
    GpuBuffer(GpuBuffer&& other) noexcept
        : device(std::exchange(other.device, {})),
          buffer(std::exchange(other.buffer, {})),
          memory(std::exchange(other.memory, {})),
          mapped(std::exchange(other.mapped, {})),
          size(std::exchange(other.size, {})) {}
    auto operator=(GpuBuffer&& other) noexcept -> GpuBuffer& {
        if (this != &other) {
            release();
            device = std::exchange(other.device, {});
            buffer = std::exchange(other.buffer, {});
            memory = std::exchange(other.memory, {});
            mapped = std::exchange(other.mapped, {});
            size = std::exchange(other.size, {});
        }
        return *this;
    }
    ~GpuBuffer() { release(); }

    auto release() -> void {
        if (!device) return;
        if (mapped) vkUnmapMemory(device, memory);
        if (buffer) vkDestroyBuffer(device, buffer, nullptr);
        if (memory) vkFreeMemory(device, memory, nullptr);
        device = {};
        buffer = {};
        memory = {};
        mapped = {};
        size = {};
    }
};

class GpuContext {
public:
    static auto instance() -> GpuContext& {
        static GpuContext context;
        return context;
    }

    auto buffer(VkDeviceSize size) -> GpuBuffer {
        GpuBuffer target;
        target.device = device_;
        target.size = std::max<VkDeviceSize>(size, 1);
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = target.size;
        info.usage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        check(vkCreateBuffer(device_, &info, nullptr, &target.buffer), "create Vulkan buffer");
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device_, target.buffer, &requirements);
        VkPhysicalDeviceMemoryProperties memory{};
        vkGetPhysicalDeviceMemoryProperties(physical_, &memory);
        constexpr auto HOST = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        auto chosen = memory.memoryTypeCount;
        for (auto wanted : {HOST | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, HOST}) {
            for (std::uint32_t type = 0; type < memory.memoryTypeCount && chosen == memory.memoryTypeCount; ++type)
                if ((requirements.memoryTypeBits & (1U << type)) &&
                    (memory.memoryTypes[type].propertyFlags & wanted) == wanted)
                    chosen = type;
            if (chosen != memory.memoryTypeCount) break;
        }
        if (chosen == memory.memoryTypeCount)
            throw ops::Failure({ErrorCode::RUNTIME, "no host-visible Vulkan memory type"});
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = chosen;
        check(vkAllocateMemory(device_, &allocation, nullptr, &target.memory), "allocate Vulkan memory");
        check(vkBindBufferMemory(device_, target.buffer, target.memory, 0), "bind Vulkan buffer");
        check(vkMapMemory(device_, target.memory, 0, target.size, 0, &target.mapped), "map Vulkan buffer");
        return target;
    }

    auto upload(std::span<const std::byte> bytes) -> GpuBuffer {
        auto result = buffer(bytes.size());
        std::memcpy(result.mapped, bytes.data(), bytes.size());
        return result;
    }

    auto pipeline(Kernel kernel, int bits) -> VkPipeline {
        const auto kernel_index = static_cast<std::size_t>(kernel);
        const auto bits_index = bits == 2 ? 0U : bits == 4 ? 1U : 2U;
        auto& cached = pipelines_[kernel_index][bits_index];
        if (cached) return cached;
        const auto code = spirv(kernel, bits);
        VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shader_info.codeSize = code.size_bytes();
        shader_info.pCode = code.data();
        VkShaderModule shader{};
        check(vkCreateShaderModule(device_, &shader_info, nullptr, &shader), "create Vulkan shader");
        shaders_.push_back(shader);
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        info.stage.module = shader;
        info.stage.pName = "main";
        info.layout = kernel == Kernel::GELU_MULTIPLY ? gelu_pipeline_layout_ : pipeline_layout_;
        check(vkCreateComputePipelines(device_, {}, 1, &info, nullptr, &cached), "create Vulkan pipeline");
        return cached;
    }

    auto dispatch_gated(int bits, const GpuBuffer& activations, const GpuBuffer& gate_up_weights,
                        const GpuBuffer& gate_up_factors, const GpuBuffer& down_weights, const GpuBuffer& down_factors,
                        GpuBuffer& gate_up, GpuBuffer& hidden, GpuBuffer& output, std::uint32_t rows,
                        std::uint32_t input_width, std::uint32_t intermediate, std::uint32_t columns,
                        float gate_up_scale, float inverse_hidden_scale, float down_output_scale) -> void {
        std::scoped_lock lock(mutex_);
        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10};
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 3;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        VkDescriptorPool pool{};
        check(vkCreateDescriptorPool(device_, &pool_info, nullptr, &pool), "create Vulkan descriptor pool");
        const auto gate_set =
            descriptor_set(pool, set_layout_, {&activations, &gate_up_weights, &gate_up_factors, &gate_up});
        const auto gelu_set = descriptor_set(pool, gelu_set_layout_, {&gate_up, &hidden});
        const auto down_set = descriptor_set(pool, set_layout_, {&hidden, &down_weights, &down_factors, &output});

        check(vkResetCommandPool(device_, commands_, 0), "reset Vulkan command pool");
        auto command = command_buffer();
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(command, &begin), "begin Vulkan command buffer");
        const auto projection = rows < 4 ? Kernel::GEMV_I8 : Kernel::GEMM_I8;
        encode_projection(command, projection, bits, gate_set, rows, input_width, 2 * intermediate, 0.F);
        compute_barrier(command, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        encode_gelu(command, gelu_set, rows, intermediate, gate_up_scale, inverse_hidden_scale);
        compute_barrier(command, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        encode_projection(command, rows < 4 ? Kernel::GEMV : Kernel::GEMM, bits, down_set, rows, intermediate, columns,
                          down_output_scale);
        compute_barrier(command, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT, VK_PIPELINE_STAGE_HOST_BIT);
        check(vkEndCommandBuffer(command), "end Vulkan command buffer");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        check(vkQueueSubmit(queue_, 1, &submit, fence_), "submit Vulkan work");
        check(vkWaitForFences(device_, 1, &fence_, VK_TRUE, 60'000'000'000ULL), "wait for Vulkan work");
        check(vkResetFences(device_, 1, &fence_), "reset Vulkan fence");
        vkDestroyDescriptorPool(device_, pool, nullptr);
    }

    auto dispatch(Kernel kernel, int bits, const GpuBuffer& activations, const GpuBuffer& weights,
                  const GpuBuffer& factors, GpuBuffer& output, std::uint32_t rows, std::uint32_t width,
                  std::uint32_t columns, float output_scale) -> void {
        std::scoped_lock lock(mutex_);
        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        VkDescriptorPool pool{};
        check(vkCreateDescriptorPool(device_, &pool_info, nullptr, &pool), "create Vulkan descriptor pool");
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = pool;
        allocate.descriptorSetCount = 1;
        allocate.pSetLayouts = &set_layout_;
        VkDescriptorSet set{};
        check(vkAllocateDescriptorSets(device_, &allocate, &set), "allocate Vulkan descriptor set");
        const std::array<const GpuBuffer*, 4> buffers{&activations, &weights, &factors, &output};
        std::array<VkDescriptorBufferInfo, 4> infos{};
        std::array<VkWriteDescriptorSet, 4> writes{};
        for (std::uint32_t index = 0; index < buffers.size(); ++index) {
            infos[index] = {buffers[index]->buffer, 0, buffers[index]->size};
            writes[index] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[index].dstSet = set;
            writes[index].dstBinding = index;
            writes[index].descriptorCount = 1;
            writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[index].pBufferInfo = &infos[index];
        }
        vkUpdateDescriptorSets(device_, writes.size(), writes.data(), 0, nullptr);

        check(vkResetCommandPool(device_, commands_, 0), "reset Vulkan command pool");
        auto command = command_buffer();
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(command, &begin), "begin Vulkan command buffer");
        encode_projection(command, kernel, bits, set, rows, width, columns, output_scale);
        compute_barrier(command, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT, VK_PIPELINE_STAGE_HOST_BIT);
        check(vkEndCommandBuffer(command), "end Vulkan command buffer");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        check(vkQueueSubmit(queue_, 1, &submit, fence_), "submit Vulkan work");
        check(vkWaitForFences(device_, 1, &fence_, VK_TRUE, 60'000'000'000ULL), "wait for Vulkan work");
        check(vkResetFences(device_, 1, &fence_), "reset Vulkan fence");
        vkDestroyDescriptorPool(device_, pool, nullptr);
    }

private:
    auto descriptor_set(VkDescriptorPool pool, VkDescriptorSetLayout layout,
                        std::initializer_list<const GpuBuffer*> buffers) -> VkDescriptorSet {
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = pool;
        allocate.descriptorSetCount = 1;
        allocate.pSetLayouts = &layout;
        VkDescriptorSet set{};
        check(vkAllocateDescriptorSets(device_, &allocate, &set), "allocate Vulkan descriptor set");
        std::vector<VkDescriptorBufferInfo> infos;
        std::vector<VkWriteDescriptorSet> writes;
        infos.reserve(buffers.size());
        writes.reserve(buffers.size());
        std::uint32_t binding = 0;
        for (const auto* buffer : buffers) {
            infos.push_back({buffer->buffer, 0, buffer->size});
            auto& write = writes.emplace_back(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
            write.dstSet = set;
            write.dstBinding = binding++;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo = &infos.back();
        }
        vkUpdateDescriptorSets(device_, writes.size(), writes.data(), 0, nullptr);
        return set;
    }

    auto command_buffer() -> VkCommandBuffer {
        VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        command_info.commandPool = commands_;
        command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_info.commandBufferCount = 1;
        VkCommandBuffer command{};
        check(vkAllocateCommandBuffers(device_, &command_info, &command), "allocate Vulkan command buffer");
        return command;
    }

    static auto compute_barrier(VkCommandBuffer command, VkAccessFlags source, VkAccessFlags target,
                                VkPipelineStageFlags target_stage) -> void {
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = source;
        barrier.dstAccessMask = target;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, target_stage, 0, 1, &barrier, 0, nullptr, 0,
                             nullptr);
    }

    auto encode_projection(VkCommandBuffer command, Kernel kernel, int bits, VkDescriptorSet set, std::uint32_t rows,
                           std::uint32_t width, std::uint32_t columns, float output_scale) -> void {
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline(kernel, bits));
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &set, 0, nullptr);
        const struct {
            std::uint32_t rows, width, columns;
            float output_scale;
        } push{rows, width, columns, output_scale};
        vkCmdPushConstants(command, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        const auto gemv = kernel == Kernel::GEMV || kernel == Kernel::GEMV_I8;
        const auto groups_x = gemv ? columns / 8 : columns / 256;
        const auto groups_y = gemv ? 1 : (rows + 15) / 16;
        vkCmdDispatch(command, groups_x, groups_y, 1);
    }

    auto encode_gelu(VkCommandBuffer command, VkDescriptorSet set, std::uint32_t rows, std::uint32_t intermediate,
                     float gate_up_scale, float inverse_hidden_scale) -> void {
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline(Kernel::GELU_MULTIPLY, 8));
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, gelu_pipeline_layout_, 0, 1, &set, 0, nullptr);
        const struct {
            std::uint32_t rows, intermediate;
            float gate_up_scale, inverse_hidden_scale;
        } push{rows, intermediate, gate_up_scale, inverse_hidden_scale};
        vkCmdPushConstants(command, gelu_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(command, (rows * intermediate / 4 + 63) / 64, 1, 1);
    }

    GpuContext() {
        VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        application.pApplicationName = "kidi-vulkan-runtime";
        application.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instance_info.pApplicationInfo = &application;
        check(vkCreateInstance(&instance_info, nullptr, &instance_), "create Vulkan instance");
        select_device();
        create_device();
        VkDescriptorSetLayoutBinding bindings[4]{};
        for (std::uint32_t index = 0; index < 4; ++index)
            bindings[index] = {index, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        set_info.bindingCount = 4;
        set_info.pBindings = bindings;
        check(vkCreateDescriptorSetLayout(device_, &set_info, nullptr, &set_layout_), "create Vulkan set layout");
        VkDescriptorSetLayoutBinding gelu_bindings[2]{};
        for (std::uint32_t index = 0; index < 2; ++index)
            gelu_bindings[index] = {index, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo gelu_set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        gelu_set_info.bindingCount = 2;
        gelu_set_info.pBindings = gelu_bindings;
        check(vkCreateDescriptorSetLayout(device_, &gelu_set_info, nullptr, &gelu_set_layout_),
              "create Vulkan GELU set layout");
        VkPushConstantRange constants{VK_SHADER_STAGE_COMPUTE_BIT, 0, 4 * sizeof(std::uint32_t)};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts = &set_layout_;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &constants;
        check(vkCreatePipelineLayout(device_, &layout_info, nullptr, &pipeline_layout_),
              "create Vulkan pipeline layout");
        layout_info.pSetLayouts = &gelu_set_layout_;
        check(vkCreatePipelineLayout(device_, &layout_info, nullptr, &gelu_pipeline_layout_),
              "create Vulkan GELU pipeline layout");
        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = family_;
        check(vkCreateCommandPool(device_, &pool_info, nullptr, &commands_), "create Vulkan command pool");
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check(vkCreateFence(device_, &fence_info, nullptr, &fence_), "create Vulkan fence");
    }

    ~GpuContext() {
        if (device_) {
            vkDeviceWaitIdle(device_);
            for (auto& row : pipelines_)
                for (auto pipeline : row)
                    if (pipeline) vkDestroyPipeline(device_, pipeline, nullptr);
            for (auto shader : shaders_) vkDestroyShaderModule(device_, shader, nullptr);
            if (fence_) vkDestroyFence(device_, fence_, nullptr);
            if (commands_) vkDestroyCommandPool(device_, commands_, nullptr);
            if (gelu_pipeline_layout_) vkDestroyPipelineLayout(device_, gelu_pipeline_layout_, nullptr);
            if (pipeline_layout_) vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
            if (gelu_set_layout_) vkDestroyDescriptorSetLayout(device_, gelu_set_layout_, nullptr);
            if (set_layout_) vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
            vkDestroyDevice(device_, nullptr);
        }
        if (instance_) vkDestroyInstance(instance_, nullptr);
    }

    auto select_device() -> void {
        std::uint32_t count = 0;
        check(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "enumerate Vulkan devices");
        std::vector<VkPhysicalDevice> devices(count);
        check(vkEnumeratePhysicalDevices(instance_, &count, devices.data()), "enumerate Vulkan devices");
        for (auto candidate : devices) {
            VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
            VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            features.pNext = &features13;
            vkGetPhysicalDeviceFeatures2(candidate, &features);
            VkPhysicalDeviceVulkan13Properties properties13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
            VkPhysicalDeviceSubgroupProperties subgroups{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
            subgroups.pNext = &properties13;
            VkPhysicalDeviceProperties2 extended{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            extended.pNext = &subgroups;
            vkGetPhysicalDeviceProperties2(candidate, &extended);
            if (!features13.shaderIntegerDotProduct || !properties13.integerDotProduct4x8BitPackedSignedAccelerated ||
                subgroups.subgroupSize != 64 || !(subgroups.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT))
                continue;
            std::uint32_t family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, nullptr);
            std::vector<VkQueueFamilyProperties> families(family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, families.data());
            for (std::uint32_t index = 0; index < family_count; ++index)
                if (families[index].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    physical_ = candidate;
                    family_ = index;
                    return;
                }
        }
        throw ops::Failure({ErrorCode::UNSUPPORTED, "no suitable Vulkan device for packed projection"});
    }

    auto create_device() -> void {
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
        check(vkCreateDevice(physical_, &device_info, nullptr, &device_), "create Vulkan device");
        vkGetDeviceQueue(device_, family_, 0, &queue_);
    }

    std::mutex mutex_;
    VkInstance instance_{};
    VkPhysicalDevice physical_{};
    VkDevice device_{};
    VkQueue queue_{};
    std::uint32_t family_{};
    VkDescriptorSetLayout set_layout_{};
    VkDescriptorSetLayout gelu_set_layout_{};
    VkPipelineLayout pipeline_layout_{};
    VkPipelineLayout gelu_pipeline_layout_{};
    VkCommandPool commands_{};
    VkFence fence_{};
    std::array<std::array<VkPipeline, 3>, 5> pipelines_{};
    std::vector<VkShaderModule> shaders_;
};

auto cpu_alias(const Tensor& tensor) -> Tensor {
    if (tensor.device().kind == DeviceKind::CPU) return tensor;
    const auto bytes = require(tensor.host_bytes());
    auto owner = std::shared_ptr<const void>(bytes.data(), [](const void*) {});
    return require(Tensor::from_blob({tensor.shape().begin(), tensor.shape().end()}, tensor.dtype(), bytes, owner));
}

auto cpu_aliases(TensorInputs inputs) -> std::vector<Tensor> {
    std::vector<Tensor> result;
    result.reserve(inputs.size());
    for (std::size_t index = 0; index < inputs.size(); ++index) result.push_back(cpu_alias(inputs[index]));
    return result;
}

auto copy_into(const Tensor& source, Tensor& destination) -> void {
    if (source.dtype() != destination.dtype() || !std::ranges::equal(source.shape(), destination.shape()))
        throw ops::Failure({ErrorCode::INVALID_ARGUMENT, "Vulkan CPU bridge output shape or dtype mismatch"});
    const auto bytes = require(source.host_bytes());
    auto target = require(destination.host_bytes());
    std::memcpy(target.data(), bytes.data(), bytes.size());
}

class VulkanOperator final : public Operator {
public:
    VulkanOperator(std::unique_ptr<Operator> cpu, Device device, std::vector<std::vector<std::int64_t>> shapes,
                   std::vector<DType> dtypes)
        : cpu_(std::move(cpu)), device_(device), shapes_(std::move(shapes)), dtypes_(std::move(dtypes)) {}

    auto run(TensorInputs inputs) -> Tensor override {
        auto output = require(Tensor::empty(shapes_.back(), dtypes_.back(), device_));
        run_into(inputs, {&output, 1});
        return output;
    }

    auto run_(TensorInputs inputs, Tensor& destination) -> Tensor override {
        auto output = run(inputs);
        copy_into(output, destination);
        return destination;
    }

    auto run_pair(TensorInputs inputs) -> std::array<Tensor, 2> override {
        std::array<Tensor, 2> outputs{require(Tensor::empty(shapes_[0], dtypes_[0], device_)),
                                      require(Tensor::empty(shapes_[1], dtypes_[1], device_))};
        run_into(inputs, outputs);
        return outputs;
    }

    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override {
        auto aliases = cpu_aliases(inputs);
        std::vector<Tensor> cpu_outputs;
        cpu_outputs.reserve(outputs.size());
        for (std::size_t index = 0; index < outputs.size(); ++index)
            cpu_outputs.push_back(require(Tensor::empty(shapes_[index], dtypes_[index], Device::cpu())));
        cpu_->run_into(aliases, cpu_outputs);
        for (std::size_t index = 0; index < outputs.size(); ++index) copy_into(cpu_outputs[index], outputs[index]);
    }

    auto allocations() const -> AllocationStats override { return cpu_->allocations(); }

private:
    std::unique_ptr<Operator> cpu_;
    Device device_;
    std::vector<std::vector<std::int64_t>> shapes_;
    std::vector<DType> dtypes_;
};

class RmsRotaryOperator final : public Operator {
public:
    RmsRotaryOperator(std::unique_ptr<Operator> norm, std::unique_ptr<Operator> rotary, std::vector<std::int64_t> shape,
                      Device device)
        : norm_(std::move(norm)), rotary_(std::move(rotary)), shape_(std::move(shape)), device_(device) {}

    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(shape_, DType::F32, device_);
        compute(inputs, output);
        return output;
    }

    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "RMS rotary is not an in-place operation"});
    }

    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { compute(inputs, outputs[0]); }

    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto compute(TensorInputs inputs, Tensor& output) -> void {
        auto aliases = cpu_aliases(inputs);
        auto normalized = require(Tensor::empty(shape_, DType::F32, Device::cpu()));
        norm_->run_into({&aliases[0], &aliases[1]}, {&normalized, 1});
        auto rotated = require(Tensor::empty(shape_, DType::F32, Device::cpu()));
        rotary_->run_into({&normalized, &aliases[2], &aliases[3]}, {&rotated, 1});
        copy_into(rotated, output);
    }

    std::unique_ptr<Operator> norm_;
    std::unique_ptr<Operator> rotary_;
    std::vector<std::int64_t> shape_;
    Device device_;
    OutputPool pool_;
};

auto interleave_word(std::uint32_t input, int bits) -> std::uint32_t {
    if (bits == 8) return input;
    std::uint32_t output = 0;
    const auto slots = static_cast<std::uint32_t>(32 / bits);
    const auto mask = (1u << bits) - 1u;
    for (std::uint32_t slot = 0; slot < slots; ++slot)
        output |= ((input >> (slot * static_cast<std::uint32_t>(bits))) & mask)
                  << (8 * (slot % 4) + static_cast<std::uint32_t>(bits) * (slot / 4));
    return output;
}

auto load_word(const std::byte* source) -> std::uint32_t {
    std::uint32_t result = 0;
    std::memcpy(&result, source, sizeof(result));
    return result;
}

auto gemv_weights(const Tensor& weight, int bits, std::size_t columns, std::size_t width)
    -> std::vector<std::uint32_t> {
    const auto bytes = require(weight.host_bytes());
    const auto bytes_per_row = width / (8 / bits);
    const auto words_per_column = width * static_cast<std::size_t>(bits) / 32;
    std::vector<std::uint32_t> result(columns * words_per_column);
    for (std::size_t column = 0; column < columns; ++column)
        for (std::size_t word = 0; word < words_per_column; ++word)
            result[column * words_per_column + word] =
                interleave_word(load_word(bytes.data() + column * bytes_per_row + word * 4), bits);
    return result;
}

auto gemm_weights(const Tensor& weight, int bits, std::size_t columns, std::size_t width)
    -> std::vector<std::uint32_t> {
    const auto bytes = require(weight.host_bytes());
    const auto bytes_per_row = width / (8 / bits);
    const auto parts = static_cast<std::size_t>(bits / 2);
    const auto groups = width / 16;
    std::vector<std::uint32_t> result(groups * parts * columns);
    for (std::size_t column = 0; column < columns; ++column)
        for (std::size_t group = 0; group < groups; ++group)
            for (std::size_t part = 0; part < parts; ++part) {
                const auto word = group * parts + part;
                result[(group * parts + part) * columns + column] =
                    interleave_word(load_word(bytes.data() + column * bytes_per_row + word * 4), bits);
            }
    return result;
}

auto factors(const Tensor& scales, float input_scale, float output_scale = 0.F) -> std::vector<float> {
    const auto values = require(scales.data<float>());
    std::vector<float> result(scales.size(0));
    const auto divisor = output_scale > 0.F ? output_scale : 1.F;
    for (std::size_t index = 0; index < result.size(); ++index) result[index] = values[index] * input_scale / divisor;
    return result;
}

auto quantize_words(const Tensor& input, std::uint32_t rows, std::uint32_t width, float scale)
    -> std::vector<std::uint32_t> {
    const auto values = require(input.data<float>());
    std::vector<std::uint32_t> result(static_cast<std::size_t>(rows) * width / 4);
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t word = 0; word < width / 4; ++word) {
            std::uint32_t packed = 0;
            for (std::size_t lane = 0; lane < 4; ++lane) {
                const auto index = row * width + word * 4 + lane;
                const auto rounded = std::nearbyint(values[index] / scale);
                const auto clipped = static_cast<std::int32_t>(std::clamp(rounded, -128.F, 127.F));
                packed |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(static_cast<std::int8_t>(clipped)))
                          << (8 * lane);
            }
            result[row * (width / 4) + word] = packed;
        }
    return result;
}

class PackedLinearOperator final : public Operator {
public:
    PackedLinearOperator(const OperatorSpec& spec, TensorInputs inputs, Device device)
        : device_(device),
          rows_(static_cast<std::uint32_t>(inputs[0].numel() / inputs[0].size(-1))),
          width_(static_cast<std::uint32_t>(inputs[0].size(-1))),
          columns_(static_cast<std::uint32_t>(inputs[1].size(0))),
          bits_(static_cast<int>(spec.attributes[0])),
          input_scale_(spec.epsilon),
          output_scale_(std::bit_cast<float>(static_cast<std::int32_t>(spec.attributes[2]))),
          kernel_(rows_ < 4 ? Kernel::GEMV : Kernel::GEMM) {
        shape_.assign(inputs[0].shape().begin(), inputs[0].shape().end());
        shape_.back() = columns_;
        auto& context = GpuContext::instance();
        const auto weight_words = kernel_ == Kernel::GEMV ? gemv_weights(inputs[1], bits_, columns_, width_)
                                                          : gemm_weights(inputs[1], bits_, columns_, width_);
        weight_ = context.upload(std::as_bytes(std::span<const std::uint32_t>(weight_words)));
        const auto scale_factors = factors(inputs[2], input_scale_);
        factors_ = context.upload(std::as_bytes(std::span<const float>(scale_factors)));
    }

    static auto supported(const OperatorSpec& spec, TensorInputs inputs) -> bool {
        const auto* enabled = std::getenv("KIDI_VULKAN_PACKED");
        if (!enabled || std::string_view(enabled) != "1") return false;
        if (spec.operation != Operation::PACKED_LINEAR || spec.epsilon <= 0 || inputs[2].size(1) != 1) return false;
        const auto rows = inputs[0].numel() / inputs[0].size(-1);
        const auto width = inputs[0].size(-1);
        const auto columns = inputs[1].size(0);
        if (width % 16 != 0) return false;
        if (rows >= 4) return false;
        return columns % 8 == 0;
    }

    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(shape_, DType::F32, device_);
        compute(inputs[0], output);
        return output;
    }

    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "packed linear is not an in-place operation"});
    }

    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { compute(inputs[0], outputs[0]); }

    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto quantize(const Tensor& input) const -> std::vector<std::uint32_t> {
        return quantize_words(input, rows_, width_, input_scale_);
    }

    auto compute(const Tensor& input, Tensor& output) const -> void {
        auto& context = GpuContext::instance();
        const auto quantized = quantize(input);
        auto activations = context.upload(std::as_bytes(std::span<const std::uint32_t>(quantized)));
        auto device_output = context.buffer(output.nbytes());
        context.dispatch(kernel_, bits_, activations, weight_, factors_, device_output, rows_, width_, columns_,
                         output_scale_);
        auto target = require(output.host_bytes());
        std::memcpy(target.data(), device_output.mapped, target.size());
    }

    Device device_;
    std::uint32_t rows_, width_, columns_;
    int bits_;
    float input_scale_, output_scale_;
    Kernel kernel_;
    std::vector<std::int64_t> shape_;
    GpuBuffer weight_, factors_;
    OutputPool pool_;
};

class GatedFeedForwardOperator final : public Operator {
public:
    GatedFeedForwardOperator(const OperatorSpec& spec, TensorInputs inputs, Device device)
        : device_(device),
          rows_(static_cast<std::uint32_t>(inputs[0].numel() / inputs[0].size(-1))),
          input_width_(static_cast<std::uint32_t>(spec.attributes[1])),
          intermediate_(static_cast<std::uint32_t>(spec.attributes[2])),
          columns_(static_cast<std::uint32_t>(inputs[3].size(0))),
          bits_(static_cast<int>(spec.attributes[0])),
          gate_input_scale_(spec.epsilon),
          gate_output_scale_(std::bit_cast<float>(static_cast<std::int32_t>(spec.attributes[3]))),
          down_input_scale_(std::bit_cast<float>(static_cast<std::int32_t>(spec.attributes[4]))),
          down_output_scale_(std::bit_cast<float>(static_cast<std::int32_t>(spec.attributes[5]))),
          kernel_(rows_ < 4 ? Kernel::GEMV : Kernel::GEMM) {
        shape_.assign(inputs[0].shape().begin(), inputs[0].shape().end());
        shape_.back() = columns_;
        auto& context = GpuContext::instance();
        const auto gate_words = kernel_ == Kernel::GEMV
                                    ? gemv_weights(inputs[1], bits_, 2 * intermediate_, input_width_)
                                    : gemm_weights(inputs[1], bits_, 2 * intermediate_, input_width_);
        gate_up_weights_ = context.upload(std::as_bytes(std::span<const std::uint32_t>(gate_words)));
        const auto gate_factors = factors(inputs[2], gate_input_scale_, gate_output_scale_);
        gate_up_factors_ = context.upload(std::as_bytes(std::span<const float>(gate_factors)));
        const auto down_words = kernel_ == Kernel::GEMV ? gemv_weights(inputs[3], bits_, columns_, intermediate_)
                                                        : gemm_weights(inputs[3], bits_, columns_, intermediate_);
        down_weights_ = context.upload(std::as_bytes(std::span<const std::uint32_t>(down_words)));
        const auto down_factors = factors(inputs[4], down_input_scale_);
        down_factors_ = context.upload(std::as_bytes(std::span<const float>(down_factors)));
        activations_ = context.buffer(static_cast<VkDeviceSize>(rows_) * input_width_);
        gate_up_ = context.buffer(static_cast<VkDeviceSize>(rows_) * 2 * intermediate_);
        hidden_ = context.buffer(static_cast<VkDeviceSize>(rows_) * intermediate_);
        output_ = context.buffer(static_cast<VkDeviceSize>(rows_) * columns_ * sizeof(float));
    }

    static auto supported(const OperatorSpec& spec, TensorInputs inputs) -> bool {
        const auto* disabled = std::getenv("KIDI_VULKAN_GATED");
        if (disabled && std::string_view(disabled) == "0") return false;
        if (spec.operation != Operation::GATED_FEED_FORWARD || spec.epsilon <= 0 || inputs[2].size(1) != 1 ||
            inputs[4].size(1) != 1)
            return false;
        const auto rows = inputs[0].numel() / inputs[0].size(-1);
        const auto input_width = static_cast<std::uint32_t>(spec.attributes[1]);
        const auto intermediate = static_cast<std::uint32_t>(spec.attributes[2]);
        const auto columns = inputs[3].size(0);
        if (input_width % 16 != 0 || intermediate % 16 != 0) return false;
        if (rows < 4) return (2 * intermediate) % 8 == 0 && columns % 8 == 0;
        return (2 * intermediate) % 256 == 0 && columns % 256 == 0;
    }

    auto run(TensorInputs inputs) -> Tensor override {
        auto output = pool_.acquire(shape_, DType::F32, device_);
        compute(inputs[0], output);
        return output;
    }

    auto run_(TensorInputs, Tensor&) -> Tensor override {
        throw ops::Failure({ErrorCode::UNSUPPORTED, "gated feed-forward is not an in-place operation"});
    }

    auto run_into(TensorInputs inputs, std::span<Tensor> outputs) -> void override { compute(inputs[0], outputs[0]); }

    auto allocations() const -> AllocationStats override { return pool_.allocations(); }

private:
    auto compute(const Tensor& input, Tensor& output) -> void {
        const auto quantized = quantize_words(input, rows_, input_width_, gate_input_scale_);
        std::memcpy(activations_.mapped, quantized.data(), quantized.size() * sizeof(std::uint32_t));
        GpuContext::instance().dispatch_gated(bits_, activations_, gate_up_weights_, gate_up_factors_, down_weights_,
                                              down_factors_, gate_up_, hidden_, output_, rows_, input_width_,
                                              intermediate_, columns_, gate_output_scale_, 1.F / down_input_scale_,
                                              down_output_scale_);
        auto target = require(output.host_bytes());
        std::memcpy(target.data(), output_.mapped, target.size());
    }

    Device device_;
    std::uint32_t rows_, input_width_, intermediate_, columns_;
    int bits_;
    float gate_input_scale_, gate_output_scale_, down_input_scale_, down_output_scale_;
    Kernel kernel_;
    std::vector<std::int64_t> shape_;
    GpuBuffer gate_up_weights_, gate_up_factors_, down_weights_, down_factors_, activations_, gate_up_, hidden_,
        output_;
    OutputPool pool_;
};

auto tensor_shape(const Tensor& input) -> std::vector<std::int64_t> {
    return {input.shape().begin(), input.shape().end()};
}

auto normalize_axis(std::int64_t axis, std::size_t rank) -> std::size_t {
    if (axis < 0) axis += static_cast<std::int64_t>(rank);
    return static_cast<std::size_t>(axis);
}

auto broadcast_shape(const Tensor& left, const Tensor& right) -> std::vector<std::int64_t> {
    const auto left_shape = left.shape(), right_shape = right.shape();
    const auto rank = std::max(left_shape.size(), right_shape.size());
    std::vector<std::int64_t> result(rank, 1);
    for (std::size_t index = 0; index < rank; ++index) {
        const auto left_axis = index + left_shape.size() >= rank ? left_shape[index + left_shape.size() - rank] : 1;
        const auto right_axis = index + right_shape.size() >= rank ? right_shape[index + right_shape.size() - rank] : 1;
        result[index] = std::max(left_axis, right_axis);
    }
    return result;
}

auto inferred_outputs(const OperatorSpec& spec, TensorInputs inputs)
    -> std::pair<std::vector<std::vector<std::int64_t>>, std::vector<DType>> {
    auto shape = tensor_shape(inputs[0]);
    auto dtype = spec.dtype;
    switch (spec.operation) {
        case Operation::ADD:
        case Operation::MULTIPLY:
            shape = broadcast_shape(inputs[0], inputs[1]);
            dtype = inputs[0].dtype();
            break;
        case Operation::CAST:
            break;
        case Operation::MATMUL:
        case Operation::LINEAR:
            shape.back() = inputs[1].size(spec.attributes[0] ? 0 : 1);
            dtype = DType::F32;
            break;
        case Operation::QUANTIZED_LINEAR:
            shape.back() = inputs[1].size(spec.attributes.empty() || !spec.attributes[0] ? 1 : 0);
            dtype = DType::F32;
            break;
        case Operation::PACKED_LINEAR:
            shape.back() = inputs[1].size(0);
            dtype = DType::F32;
            break;
        case Operation::GATED_FEED_FORWARD:
            shape.back() = spec.attributes[1];
            dtype = DType::F32;
            break;
        case Operation::GREEDY_TOKEN:
            shape = {static_cast<std::int64_t>(inputs[0].numel() / inputs[0].size(-1))};
            dtype = DType::I32;
            break;
        case Operation::EMBEDDING:
            shape = {1, static_cast<std::int64_t>(inputs[0].numel()), spec.attributes[0]};
            dtype = DType::F32;
            break;
        case Operation::ATTENTION:
            dtype = DType::F32;
            break;
        case Operation::TRANSPOSE: {
            auto original = shape;
            for (std::size_t index = 0; index < shape.size(); ++index)
                shape[index] = original[static_cast<std::size_t>(spec.attributes[index])];
            dtype = inputs[0].dtype();
            break;
        }
        case Operation::SLICE:
            shape[normalize_axis(spec.attributes[0], shape.size())] = spec.attributes[2];
            dtype = inputs[0].dtype();
            break;
        case Operation::GATHER: {
            const auto axis = normalize_axis(spec.attributes[0], inputs[0].dimensions());
            shape = tensor_shape(inputs[0]);
            if (inputs[1].dimensions() == 1) {
                shape[axis] = inputs[1].numel();
            } else {
                shape.erase(shape.begin() + static_cast<std::ptrdiff_t>(axis));
                shape.insert(shape.begin() + static_cast<std::ptrdiff_t>(axis), inputs[1].shape().begin(),
                             inputs[1].shape().end());
            }
            dtype = inputs[0].dtype();
            break;
        }
        case Operation::CONCAT: {
            const auto axis = normalize_axis(spec.attributes[0], inputs[0].dimensions());
            shape = tensor_shape(inputs[0]);
            shape[axis] = 0;
            for (std::size_t index = 0; index < inputs.size(); ++index) shape[axis] += inputs[index].size(axis);
            dtype = inputs[0].dtype();
            break;
        }
        case Operation::SCATTER:
            dtype = inputs[0].dtype();
            break;
        case Operation::RESIDUAL_NORM:
            return {{{shape}, {shape}}, {DType::F32, DType::F32}};
        case Operation::RMS_NORM:
        case Operation::RMS_NORM_RESIDUAL:
        case Operation::LAYER_NORM:
        case Operation::SOFTMAX:
        case Operation::LOG_SOFTMAX:
        case Operation::GELU:
        case Operation::TANH:
        case Operation::ROTARY:
        case Operation::STATIC_ROUND:
        case Operation::GELU_MULTIPLY:
        case Operation::RMS_ROTARY:
            dtype = DType::F32;
            break;
    }
    return {{{shape}}, {dtype}};
}

class VulkanBackend final : public OperatorBackend {
public:
    VulkanBackend() : cpu_(cpu_operators()) {}

    auto prepare(const OperatorSpec& spec, TensorInputs inputs) -> std::unique_ptr<Operator> override {
        if (spec.operation == Operation::RMS_ROTARY)
            return std::make_unique<RmsRotaryOperator>(
                cpu_->prepare({Operation::RMS_NORM, {}, DType::F32, spec.epsilon}, inputs.first(2)),
                cpu_->prepare({Operation::ROTARY, {}, DType::F32}, {&inputs[0], &inputs[2], &inputs[3]}),
                std::vector<std::int64_t>(inputs[0].shape().begin(), inputs[0].shape().end()), Device::vulkan());
        if (GatedFeedForwardOperator::supported(spec, inputs))
            return std::make_unique<GatedFeedForwardOperator>(spec, inputs, Device::vulkan());
        if (PackedLinearOperator::supported(spec, inputs))
            return std::make_unique<PackedLinearOperator>(spec, inputs, Device::vulkan());
        auto [shapes, dtypes] = inferred_outputs(spec, inputs);
        return std::make_unique<VulkanOperator>(cpu_->prepare(spec, inputs), Device::vulkan(), std::move(shapes),
                                                std::move(dtypes));
    }

    auto synchronize() -> void override { cpu_->synchronize(); }

    auto release_cached_buffers() -> void override { cpu_->release_cached_buffers(); }

    auto supports_replay() const noexcept -> bool override { return true; }

    auto copy_slice_(Tensor& destination, const Tensor& source, std::size_t outer, std::size_t source_bytes,
                     std::size_t destination_bytes, std::size_t offset_bytes) -> void override {
        auto target = require(destination.host_bytes());
        auto input = require(source.host_bytes());
        std::vector<std::byte> snapshot;
        const auto from = reinterpret_cast<std::uintptr_t>(input.data());
        const auto to = reinterpret_cast<std::uintptr_t>(target.data());
        if (from < to + target.size() && to < from + input.size()) {
            snapshot.assign(input.begin(), input.end());
            input = snapshot;
        }
        for (std::size_t block = 0; block < outer; ++block)
            std::memcpy(target.data() + block * destination_bytes + offset_bytes, input.data() + block * source_bytes,
                        source_bytes);
    }

private:
    std::unique_ptr<OperatorBackend> cpu_;
};

} // namespace

auto vulkan_operators() -> std::unique_ptr<OperatorBackend> { return std::make_unique<VulkanBackend>(); }

} // namespace kidi::runtime
