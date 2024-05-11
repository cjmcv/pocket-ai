/*!
* \brief CommandBuffer
*     Centered around VkCommandBuffer, created by the command pool and released when the command pool is released.
*     Mainly used for recording commands.
*/

#ifndef POCKET_AI_ENGINE_VULKAN_COMMAND_BUFFER_HPP_
#define POCKET_AI_ENGINE_VULKAN_COMMAND_BUFFER_HPP_

#include <vulkan/vulkan.h>

#include <memory>
#include <unordered_map>

#include "buffer.hpp"
#include "pipeline.hpp"

namespace pai {
namespace vk {

class CommandBuffer {
public:
    static CommandBuffer *Create(VkDevice device, VkCommandPool command_pool) {
        VkCommandBuffer command_buffer;
        
        // Now allocate a command buffer from the command pool.
        VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
        commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.commandPool = command_pool; // specify the command pool to allocate from.
        // if the command buffer is primary, it can be directly submitted to queues.
        // A secondary buffer has to be called from some primary command buffer, and cannot be directly
        // submitted to a queue. To keep things simple, we use a primary command buffer.
        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = 1;                                       // allocate a single command buffer.
        vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &command_buffer); // allocate command buffer.

        return new CommandBuffer(command_buffer);
    }
    ~CommandBuffer() = default;

    inline VkCommandBuffer command_buffer() const { return command_buffer_; };

    // Begins command buffer recording.
    void Begin() {
        vkResetCommandBuffer(command_buffer_, /*flags=*/0);

        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.pNext = nullptr;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        begin_info.pInheritanceInfo = nullptr;
        vkBeginCommandBuffer(command_buffer_, &begin_info);
    }
    // Ends command buffer recording.
    void End() {
        vkEndCommandBuffer(command_buffer_);
    }

    // Records a command to copy the src_buffer to dst_buffer.
    void CopyBuffer(const Buffer &src_buffer, size_t src_offset,
                    const Buffer &dst_buffer, size_t dst_offset, size_t length) {
        VkBufferCopy region = {};
        region.srcOffset = src_offset;
        region.dstOffset = dst_offset;
        region.size = length;
        vkCmdCopyBuffer(command_buffer_, 
                        src_buffer.buffer(),
                        dst_buffer.buffer(),
                        /*regionCount=*/1, &region);
    }

    // Records a command to bind the compute |pipeline| and resource descriptor
    // sets recorded in |bound_descriptor_sets| into this command buffer.
    void BindPipelineAndDescriptorSets(const Pipeline *pipeline, 
                                       std::vector<VkDescriptorSet> &&descriptor_sets) {

        vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                        pipeline->pipeline());

        for (uint32_t i=0; i<descriptor_sets.size(); i++) {
            vkCmdBindDescriptorSets(
                command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline->pipeline_layout(), 
                /*firstSet=*/i,
                /*descriptorSetCount=*/1,
                /*pDescriptorSets=*/&descriptor_sets[i],
                /*dynamicOffsetCount=*/0,
                /*pDynamicOffsets=*/nullptr);
        }
    }

    void PushConstant(VkPipelineLayout layout, const int params_size, const void *params) {
        vkCmdPushConstants(command_buffer_, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, params_size, params);
    }


    // Records a dispatch command.
    // x,y,z: groupCount
    // For example, X is the number of local workgroups to dispatch in the X dimension.
    void Dispatch(uint32_t x, uint32_t y, uint32_t z){
        vkCmdDispatch(command_buffer_, x, y, z);
    }

    // Records a pipeline barrier that synchronizes shader read from a compute
    // shader with shader write from a previous compute shader.
    void DispatchBarrier() {
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(command_buffer_,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                            &barrier, 0, nullptr, 0, nullptr);
    }

private:
    CommandBuffer(VkCommandBuffer command_buffer)
    : command_buffer_(command_buffer) {}

    VkCommandBuffer command_buffer_;
};

}  // namespace vk
}  // namespace pai

#endif  // POCKET_AI_ENGINE_VULKAN_COMMAND_BUFFER_HPP_
