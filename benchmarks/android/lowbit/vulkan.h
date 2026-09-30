#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <deque>
#include <initializer_list>
#include <span>
#include <vector>

namespace kidi::bench::vk {

inline constexpr std::uint32_t GEMV_MAX_ROWS = 4;
inline constexpr std::uint32_t GEMV_COLUMNS = 8;
inline constexpr std::uint32_t GEMM_ROWS = 16;
inline constexpr std::uint32_t GEMM_COLUMNS = 256;
inline constexpr std::uint32_t GELU_LOCAL_SIZE = 64;

enum class Kernel { GEMV, GEMM, GELU_MULTIPLY };

auto check(VkResult result, const char* operation) -> void;
auto spirv(Kernel kernel, int bits) -> std::span<const std::uint32_t>;

// Packs [n][k] signed values into words of 32 / bits consecutive K values of one column. Byte `b` of a word holds
// values b, b + 4, b + 8, ... so masking whole bytes yields consecutive INT8x4 groups. GEMV stores words column-major
// ([column][word]); GEMM stores them K-major ([word][column]) so lanes load adjacent columns.
auto pack_weights(std::span<const std::int8_t> weights, std::uint32_t n, std::uint32_t k, int bits,
                  bool k_major) -> std::vector<std::uint32_t>;

struct Buffer {
    VkBuffer buffer{};
    VkDeviceMemory memory{};
    void* mapped{};
    VkDeviceSize size{};
};

// Owns one Adreno compute device and every object created through it. Buffers are host-visible and coherent so the
// host can update inputs in place between submissions.
class Context {
public:
    Context();
    ~Context();
    Context(const Context&) = delete;
    auto operator=(const Context&) -> Context& = delete;

    auto buffer(VkDeviceSize size) -> Buffer&;
    auto upload(const void* data, VkDeviceSize size) -> Buffer&;
    auto set_layout(std::uint32_t bindings) -> VkDescriptorSetLayout;
    auto pipeline_layout(VkDescriptorSetLayout set_layout, std::uint32_t push_bytes) -> VkPipelineLayout;
    auto pipeline(std::span<const std::uint32_t> code, VkPipelineLayout layout) -> VkPipeline;
    auto descriptor_pool(std::uint32_t sets, std::uint32_t descriptors) -> VkDescriptorPool;
    auto descriptor_set(VkDescriptorPool pool, VkDescriptorSetLayout layout,
                        std::initializer_list<const Buffer*> buffers) -> VkDescriptorSet;
    auto command_buffer() -> VkCommandBuffer;
    auto query_pool(std::uint32_t queries) -> VkQueryPool;
    auto submit(std::span<const VkCommandBuffer> commands, bool wait) -> void;
    auto wait() -> void;
    auto elapsed_ms(VkQueryPool pool) -> double;

    VkPhysicalDeviceProperties properties{};
    VkDevice device{};

private:
    VkInstance instance_{};
    VkPhysicalDevice physical_{};
    VkQueue queue_{};
    std::uint32_t family_{};
    std::uint32_t timestamp_bits_{};
    VkCommandPool commands_{};
    VkFence fence_{};
    bool fence_pending_ = false;
    std::deque<Buffer> buffers_;
    std::vector<VkDescriptorSetLayout> set_layouts_;
    std::vector<VkPipelineLayout> pipeline_layouts_;
    std::vector<VkShaderModule> shaders_;
    std::vector<VkPipeline> pipelines_;
    std::vector<VkDescriptorPool> descriptor_pools_;
    std::vector<VkQueryPool> query_pools_;
};

// Orders all prior compute writes (including earlier submissions on the queue) before later compute reads/writes.
auto compute_barrier(VkCommandBuffer commands) -> void;

} // namespace kidi::bench::vk
