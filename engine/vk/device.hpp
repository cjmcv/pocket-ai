
/*!
* \brief 逻辑设备
*    vulkan操作的主体，大部分操作都基于逻辑设备进行
*  基于手动选定的物理设备，按用途需求进行创建。
*  同时构建 队列和命令池
*/

#ifndef PTK_ENGINE_VULKAN_DEVICE_HPP_
#define PTK_ENGINE_VULKAN_DEVICE_HPP_

#include <vulkan/vulkan.h>

#include <memory>
#include <vector>
#include "buffer.hpp"
#include "common.hpp"

namespace ptk {
namespace vk {

class Device {
public:
    static Device *Create(VkPhysicalDevice physical_device, 
                          VkQueueFlags queue_flags,
                          std::vector<const char*> &layers) {
        uint32_t queue_family_index = SelectQueueFamily(physical_device, queue_flags);

        float queue_priority = 1.0;
        VkDeviceQueueCreateInfo queue_create_info = {};
        queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_info.pNext = nullptr;
        queue_create_info.flags = 0;
        queue_create_info.queueFamilyIndex = queue_family_index;
        queue_create_info.queueCount = 1;
        queue_create_info.pQueuePriorities = &queue_priority;

        VkDeviceCreateInfo device_create_info = {};
        device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_create_info.pNext = nullptr;
        device_create_info.flags = 0;
        device_create_info.queueCreateInfoCount = 1;
        device_create_info.pQueueCreateInfos = &queue_create_info;
        device_create_info.enabledLayerCount = layers.size();
        device_create_info.ppEnabledLayerNames = layers.data();
        // device_create_info.enabledExtensionCount = extensions.size();
        // device_create_info.ppEnabledExtensionNames = extensions.data();
        device_create_info.pEnabledFeatures = nullptr;

        VkDevice device;
        vkCreateDevice(physical_device, &device_create_info, /*pAllocator=*/nullptr, &device);

        // Create Command pool
        VkCommandPool command_pool;
        VkCommandPoolCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        create_info.pNext = nullptr;
        create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        create_info.queueFamilyIndex = queue_family_index;
        vkCreateCommandPool(device, &create_info, /*pAllocator=*/nullptr, &command_pool);

        return new Device(device, command_pool, physical_device, queue_family_index);
    }
    ~Device() {
        vkDestroyCommandPool(device_, command_pool_, NULL);	
        vkDestroyDevice(device_, NULL);
    }

    inline VkDevice device() const { return device_; }
    inline VkCommandPool command_pool() const { return command_pool_; }
    inline VkPhysicalDeviceMemoryProperties &memory_properties() { return memory_properties_; }

    void QueueSubmitAndWait(VkCommandBuffer command_buffer) {
        VkFenceCreateInfo fence_create_info = {};
        fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_create_info.pNext = nullptr;
        fence_create_info.flags = 0;

        VkFence fence = VK_NULL_HANDLE;
        vkCreateFence(device_, &fence_create_info, /*pALlocator=*/nullptr, &fence);

        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;

        vkQueueSubmit(queue_, 1, &submit_info, fence);

        vkWaitForFences(device_, /*fenceCount=*/1, &fence, /*waitAll=*/true, /*timeout=*/UINT64_MAX);

        vkDestroyFence(device_, fence, /*pAllocator=*/nullptr);
    }
    
private:
    Device(VkDevice device, VkCommandPool command_pool, 
           VkPhysicalDevice physical_device, uint32_t queue_family_index)
           : device_(device), command_pool_(command_pool) {

        physical_device_ = physical_device;
        vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties_);
        // Get a handle to the only member of the queue family.
        vkGetDeviceQueue(device, queue_family_index, 0, &queue_);
    }

    // Retrieve the queue family in the physical device as needed 
    // and return the index of the corresponding queue family on the physical device
    static uint32_t SelectQueueFamily(VkPhysicalDevice physical_device, VkQueueFlags queue_flags){
        uint32_t count;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, nullptr);

        std::vector<VkQueueFamilyProperties> queue_families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, queue_families.data());

        for (uint32_t index = 0; index < count; ++index) {
            const VkQueueFamilyProperties &properties = queue_families[index];
            if (properties.queueCount > 0 &&
                ((properties.queueFlags & queue_flags) == queue_flags)) {
                return index;
            }
        }

        PTK_LOGE("Instance::SelectQueueFamily -> Cannot find queue family with required bits.");
        return 0;
    }

    VkPhysicalDevice physical_device_;
    VkDevice device_;
    VkCommandPool command_pool_;    
    
    // Get.
    VkPhysicalDeviceMemoryProperties memory_properties_;
    VkQueue queue_;
};

}  // namespace vk
}  // namespace ptk

#endif  // PTK_ENGINE_VULKAN_DEVICE_HPP_
