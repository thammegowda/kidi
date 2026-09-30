#include "kidi/tensor/backend.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstring>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace kidi::tensor {
namespace {

auto check(VkResult result, std::string_view operation) -> void {
    if (result != VK_SUCCESS)
        throw Error{ErrorCode::RUNTIME, std::string(operation) + " failed with VkResult " + std::to_string(result)};
}

class VulkanHostStorage final : public Storage {
public:
    VulkanHostStorage(Device device, std::size_t size_bytes, std::size_t alignment)
        : device_(device), size_bytes_(size_bytes), writable_(true) {
        auto* allocation = ::operator new(std::max<std::size_t>(size_bytes, 1), std::align_val_t(alignment));
        owner_ = std::shared_ptr<const void>(allocation, [alignment](const void* pointer) {
            ::operator delete(const_cast<void*>(pointer), std::align_val_t(alignment));
        });
        data_ = static_cast<const std::byte*>(allocation);
    }

    VulkanHostStorage(Device device, std::span<const std::byte> bytes, std::shared_ptr<const void> owner)
        : device_(device), size_bytes_(bytes.size()), owner_(std::move(owner)), data_(bytes.data()), writable_(false) {}

    auto device() const noexcept -> Device override { return device_; }
    auto size_bytes() const noexcept -> std::size_t override { return size_bytes_; }
    auto mutable_data() noexcept -> std::byte* { return writable_ ? const_cast<std::byte*>(data_) : nullptr; }
    auto data() const noexcept -> const std::byte* { return data_; }
    auto writable() const noexcept -> bool { return writable_; }

private:
    Device device_;
    std::size_t size_bytes_;
    std::shared_ptr<const void> owner_;
    const std::byte* data_;
    bool writable_;
};

class Instance {
public:
    Instance() {
        VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        application.pApplicationName = "kidi-vulkan";
        application.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        info.pApplicationInfo = &application;
        check(vkCreateInstance(&info, nullptr, &instance_), "create Vulkan instance");
    }

    ~Instance() {
        if (instance_) vkDestroyInstance(instance_, nullptr);
    }

    Instance(const Instance&) = delete;
    auto operator=(const Instance&) -> Instance& = delete;

    auto get() const noexcept -> VkInstance { return instance_; }

private:
    VkInstance instance_{};
};

auto probe_device() -> std::string {
    Instance instance;
    std::uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(instance.get(), &count, nullptr), "enumerate Vulkan devices");
    if (count == 0) throw Error{ErrorCode::UNSUPPORTED, "no Vulkan physical devices found"};
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(instance.get(), &count, devices.data()), "enumerate Vulkan devices");
    for (auto device : devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(device, &properties);
        VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features.pNext = &features13;
        vkGetPhysicalDeviceFeatures2(device, &features);
        VkPhysicalDeviceVulkan13Properties properties13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
        VkPhysicalDeviceSubgroupProperties subgroups{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        subgroups.pNext = &properties13;
        VkPhysicalDeviceProperties2 extended{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        extended.pNext = &subgroups;
        vkGetPhysicalDeviceProperties2(device, &extended);
        if (!features13.shaderIntegerDotProduct || !properties13.integerDotProduct4x8BitPackedSignedAccelerated)
            continue;
        if (subgroups.subgroupSize != 64 || !(subgroups.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT))
            continue;
        std::uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, families.data());
        const auto has_compute = std::ranges::any_of(
            families, [](const auto& family) { return (family.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0; });
        if (has_compute) return properties.deviceName;
    }
    throw Error{ErrorCode::UNSUPPORTED,
                "no Vulkan compute device with accelerated signed packed INT8 dot product and 64-lane subgroups"};
}

auto vulkan_storage(Storage& storage) -> Result<VulkanHostStorage*> {
    auto* result = dynamic_cast<VulkanHostStorage*>(&storage);
    if (!result)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "storage does not belong to Vulkan backend"});
    return result;
}

auto vulkan_storage(const Storage& storage) -> Result<const VulkanHostStorage*> {
    const auto* result = dynamic_cast<const VulkanHostStorage*>(&storage);
    if (!result)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "storage does not belong to Vulkan backend"});
    return result;
}

auto check_bounds(std::size_t storage_size, std::size_t offset, std::size_t size) -> Result<void> {
    if (offset > storage_size || size > storage_size - offset)
        return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "tensor storage copy is out of bounds"});
    return {};
}

class VulkanBackend final : public Backend {
public:
    explicit VulkanBackend(std::string device_name) : device_name_(std::move(device_name)) {}

    auto name() const noexcept -> std::string_view override { return "vulkan"; }
    auto device_name(Device) const -> std::string override { return device_name_; }
    auto device_kind() const noexcept -> DeviceKind override { return DeviceKind::VULKAN; }
    auto is_available(Device device) const noexcept -> bool override {
        return device.kind == DeviceKind::VULKAN && device.index == 0;
    }
    auto is_host_accessible(Device device) const noexcept -> bool override { return is_available(device); }
    auto supports_execution() const noexcept -> bool override { return true; }
    auto unavailable_reason(Device) const -> std::string override {
        return "Vulkan device " + device_name_ + " is available only as vulkan:0";
    }

    auto allocate(Device device, std::size_t size_bytes,
                  std::size_t alignment) const -> Result<std::shared_ptr<Storage>> override {
        if (!is_available(device)) return std::unexpected(Error{ErrorCode::UNSUPPORTED, unavailable_reason(device)});
        alignment = std::max<std::size_t>(alignment, 64);
        if ((alignment & (alignment - 1)) != 0)
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "tensor alignment must be a power of two"});
        try {
            return std::shared_ptr<Storage>(new VulkanHostStorage(device, size_bytes, alignment));
        } catch (const std::bad_alloc&) {
            return std::unexpected(Error{ErrorCode::RUNTIME, "failed to allocate Vulkan tensor storage"});
        }
    }

    auto wrap_host(Device device, std::span<const std::byte> bytes,
                   std::shared_ptr<const void> owner) const -> Result<std::shared_ptr<Storage>> override {
        if (!is_available(device)) return std::unexpected(Error{ErrorCode::UNSUPPORTED, unavailable_reason(device)});
        if (!owner && !bytes.empty())
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "external tensor storage requires an owner"});
        return std::shared_ptr<Storage>(new VulkanHostStorage(device, bytes, std::move(owner)));
    }

    auto copy_from_host(Storage& destination, std::size_t destination_offset,
                        std::span<const std::byte> source) const -> Result<void> override {
        auto storage = vulkan_storage(destination);
        if (!storage) return std::unexpected(std::move(storage.error()));
        if (!(*storage)->writable())
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "cannot write to read-only tensor storage"});
        auto bounds = check_bounds((*storage)->size_bytes(), destination_offset, source.size());
        if (!bounds) return bounds;
        std::memcpy((*storage)->mutable_data() + destination_offset, source.data(), source.size());
        return {};
    }

    auto copy_to_host(const Storage& source, std::size_t source_offset,
                      std::span<std::byte> destination) const -> Result<void> override {
        auto storage = vulkan_storage(source);
        if (!storage) return std::unexpected(std::move(storage.error()));
        auto bounds = check_bounds((*storage)->size_bytes(), source_offset, destination.size());
        if (!bounds) return bounds;
        std::memcpy(destination.data(), (*storage)->data() + source_offset, destination.size());
        return {};
    }

    auto host_view(Storage& storage) const -> Result<std::span<std::byte>> override {
        auto typed = vulkan_storage(storage);
        if (!typed) return std::unexpected(std::move(typed.error()));
        if (!(*typed)->writable())
            return std::unexpected(Error{ErrorCode::INVALID_ARGUMENT, "tensor storage is read-only"});
        return std::span<std::byte>((*typed)->mutable_data(), (*typed)->size_bytes());
    }

    auto host_view(const Storage& storage) const -> Result<std::span<const std::byte>> override {
        auto typed = vulkan_storage(storage);
        if (!typed) return std::unexpected(std::move(typed.error()));
        return std::span<const std::byte>((*typed)->data(), (*typed)->size_bytes());
    }

    auto synchronize(Device device) const -> Result<void> override {
        if (!is_available(device)) return std::unexpected(Error{ErrorCode::UNSUPPORTED, unavailable_reason(device)});
        return {};
    }

private:
    std::string device_name_;
};

} // namespace

auto make_vulkan_backend() -> std::shared_ptr<Backend> {
    try {
        return std::make_shared<VulkanBackend>(probe_device());
    } catch (const Error& error) {
        return make_unavailable_backend(DeviceKind::VULKAN, "vulkan", error.message);
    } catch (const std::exception& error) {
        return make_unavailable_backend(DeviceKind::VULKAN, "vulkan", error.what());
    }
}

} // namespace kidi::tensor
