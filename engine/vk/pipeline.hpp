/*!
* \brief Compute pipeline
*      Centered around VkPipeline, the setting of specialization constants is also here
*/

#ifndef POCKET_AI_ENGINE_VULKAN_PIPELINE_HPP_
#define POCKET_AI_ENGINE_VULKAN_PIPELINE_HPP_

#include <vulkan/vulkan.h>
#include "shader_module.hpp"
#include "common.hpp"

namespace pai {
namespace vk {

struct SpecConstant {
    uint32_t id;
    union {
        int32_t s32;
        uint32_t u32;
        float f32;
    } value;
};

struct SpecConstantData {
    // All packed specialization data
    std::vector<uint8_t> data;
    // Entry describing each specialization constant
    std::vector<VkSpecializationMapEntry> entries;
};

// Packs |spec_constants| into a byte buffer so that they can used for Vulkan
// API calls.
SpecConstantData PackSpecConstantData(std::vector<SpecConstant> &spec_constants) {

    size_t const_size = 4; // 只支持4字节，包含int32_t，uint32_t和float
    size_t total_size = const_size * spec_constants.size();

    std::vector<uint8_t> data(total_size);
    std::vector<VkSpecializationMapEntry> entries;
    entries.reserve(spec_constants.size());

    uint32_t index = 0; // Next available byte's index in the buffer
    for (const auto &spec_const : spec_constants) {
        uint8_t *ptr = data.data() + index;

        memcpy(ptr, &(spec_const.value.u32), const_size); // TODO: 优化union
        entries.emplace_back();
        entries.back().constantID = spec_const.id;
        entries.back().offset = index;
        entries.back().size = const_size;

        index += const_size;
    }

    return SpecConstantData{std::move(data), std::move(entries)};
}
    
class Pipeline {
public:

    union PushConstant {
        int i;
        float f;
    };
    // Creates a Vulkan compute pipeline
    static Pipeline *Create(VkDevice device, 
                           VkShaderModule shader_module, 
                           std::vector<VkDescriptorSetLayout> &set_layouts,
                           const char *entry_point, 
                           std::vector<SpecConstant> &spec_constants,
                           uint32_t push_constant_num){
  
        // Pack the specialization constant into an byte buffer
        SpecConstantData spec_constant_data = PackSpecConstantData(spec_constants);

        VkPipelineShaderStageCreateInfo shader_stage_create_info = {};
        shader_stage_create_info.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shader_stage_create_info.pNext = nullptr;
        shader_stage_create_info.flags = 0;
        shader_stage_create_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shader_stage_create_info.module = shader_module;
        shader_stage_create_info.pName = entry_point;

        VkSpecializationInfo spec_constant_info = {};
        // Update specialization information
        if (!spec_constants.empty()) {
            spec_constant_info.mapEntryCount = spec_constant_data.entries.size();
            spec_constant_info.pMapEntries = spec_constant_data.entries.data();
            spec_constant_info.dataSize = spec_constant_data.data.size();
            spec_constant_info.pData = spec_constant_data.data.data();
            shader_stage_create_info.pSpecializationInfo = &spec_constant_info;
        }
        else {
            shader_stage_create_info.pSpecializationInfo = nullptr;
        }

        VkPushConstantRange push_constant_range;
        push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_constant_range.offset = 0;
        push_constant_range.size = sizeof(PushConstant) * push_constant_num; 

        VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
        pipeline_layout_create_info.sType =
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_create_info.pNext = nullptr;
        pipeline_layout_create_info.flags = 0;
        pipeline_layout_create_info.setLayoutCount = set_layouts.size();
        pipeline_layout_create_info.pSetLayouts = set_layouts.data();
        if (push_constant_num == 0) {
            pipeline_layout_create_info.pushConstantRangeCount = 0;
            pipeline_layout_create_info.pPushConstantRanges = nullptr;
        }
        else {
            pipeline_layout_create_info.pushConstantRangeCount = 1;
            pipeline_layout_create_info.pPushConstantRanges = &push_constant_range;        
        }

        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        vkCreatePipelineLayout(device, &pipeline_layout_create_info,
                            /*pAllocator=*/nullptr, &pipeline_layout);

        VkComputePipelineCreateInfo pipeline_create_info = {};
        pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_create_info.pNext = nullptr;
        pipeline_create_info.flags = 0;
        pipeline_create_info.stage = shader_stage_create_info;
        pipeline_create_info.layout = pipeline_layout;
        pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
        pipeline_create_info.basePipelineIndex = 0;

        VkPipeline pipeline = VK_NULL_HANDLE;
        vkCreateComputePipelines(device, 
                                /*pipelineCache=*/VK_NULL_HANDLE,
                                /*createInfoCount=*/1, 
                                &pipeline_create_info,
                                /*pAllocator=*/nullptr, 
                                &pipeline);

        return new Pipeline(pipeline, device, pipeline_layout);
    }

    ~Pipeline() { 
        vkDestroyPipeline(device_, pipeline_, /*pAllocator=*/nullptr);
        vkDestroyPipelineLayout(device_, pipeline_layout_, /*pAllocator=*/nullptr);
    }

    inline VkPipeline pipeline() const { return pipeline_; }
    inline VkPipelineLayout pipeline_layout() const { return pipeline_layout_; }

private:
    Pipeline(VkPipeline pipeline, VkDevice device, VkPipelineLayout layout)
    : pipeline_(pipeline),
      device_(device),
      pipeline_layout_(layout) {}

    VkDevice device_;
    
    VkPipeline pipeline_;
    VkPipelineLayout pipeline_layout_;
};

}  // end of namespace vk
}  // end of namespace pai

#endif  // POCKET_AI_ENGINE_VULKAN_PIPELINE_HPP_
