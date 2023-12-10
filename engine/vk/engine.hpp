/*!
* \brief Engine is the outermost class of engine/vk
*/

#ifndef PTK_ENGINE_VULKAN_ENGINE_HPP_
#define PTK_ENGINE_VULKAN_ENGINE_HPP_

#include <map>

#include "common.hpp"
#include "instance.hpp"
#include "device.hpp"
#include "buffer.hpp"
#include "shader_module.hpp"
#include "pipeline.hpp"
#include "descriptor_pool.hpp"
#include "command_buffer.hpp"

namespace ptk {
namespace vk {

struct KernelParams {
    std::vector<VkDescriptorType> buffer_type;
    std::vector<SpecConstant> spec_constant;
    uint32_t push_constant_num;
};

typedef void (*SetpParamsFuncs)(KernelParams *params);

class Engine {
    // The main units required for kernel operation. 
    // Created by the Engine, one kernel corresponds to one ExecUnit.
    class ExecUnit {
    public:
        void GetGroupCount(const uint32_t width, const uint32_t height, const uint32_t channels) {
            uint32_t size_x = params_->spec_constant[0].value.u32;
            uint32_t size_y = params_->spec_constant[1].value.u32;
            uint32_t size_z = params_->spec_constant[2].value.u32;

            group_count_xyz_[0] = (width + size_x - 1) / size_x;
            group_count_xyz_[1] = (height + size_y - 1) / size_y;
            group_count_xyz_[2] = (channels + size_z - 1) / size_z;
        }

        void Run(uint32_t *num_xyz, std::vector<Buffer*> &input_buffers, const int push_constant_size, 
                 const void *push_constant, std::vector<Buffer*> &output_buffers) {
            static int idx = 0;
            idx++;
            printf("round idx: %d.\n", idx);

            // The binding order of the buffer needs to be consistent with the one in comp,
            // That is, "i=0" corresponds to the one in comp "binding=0"
            for (uint32_t i=0; i<input_buffers.size(); i++)
                descriptor_pool_->WriteBuffer(descriptor_set_, i, input_buffers[i]);
            for (uint32_t i=0; i<output_buffers.size(); i++)
                descriptor_pool_->WriteBuffer(descriptor_set_, input_buffers.size() + i, output_buffers[i]);

            command_buffer_->Begin();
            command_buffer_->BindPipelineAndDescriptorSets(pipeline_, {descriptor_set_});
            if (push_constant_size != 0)
                command_buffer_->PushConstant(pipeline_->pipeline_layout(), push_constant_size, push_constant);

            GetGroupCount(num_xyz[0], num_xyz[1], num_xyz[2]);
            command_buffer_->Dispatch(group_count_xyz_[0], group_count_xyz_[1], group_count_xyz_[2]);
            command_buffer_->End();

            device_->QueueSubmitAndWait(command_buffer_->command_buffer());
        }

        KernelParams *params_;
        uint32_t group_count_xyz_[3];

        Device *device_;
        ShaderModule *shader_module_;
        //
        Pipeline *pipeline_;
        DescriptorPool *descriptor_pool_;
        CommandBuffer *command_buffer_;
        //
        VkDescriptorSet descriptor_set_;
    };

public:
    void Init(std::string shaders_path, std::vector<std::pair<std::string, SetpParamsFuncs>> &shaders_params, 
              int physical_device_id = 0, bool enable_validation = false) {
        // vk_dispatcher_ = VulkanKernelDispatcher::GetInstance();
        instance_ = new Instance(enable_validation);
        std::vector<VkPhysicalDevice> phys_devices = instance_->EnumeratePhysicalDevices(true);
        device_ = Device::Create(phys_devices[physical_device_id], VK_QUEUE_COMPUTE_BIT, instance_->layers());

        // Register
        for (uint32_t i=0; i<shaders_params.size(); i++) {
            ExecUnit *res = new ExecUnit;
            res->params_ = new KernelParams;
            shaders_params[i].second(res->params_);
            exec_map_[shaders_params[i].first] = res;
        }
        // Setup
        std::unordered_map<std::string, ExecUnit*>::iterator it = exec_map_.begin();
        while (it != exec_map_.end()) {
            ExecUnit *res = it->second;
            KernelParams *params = res->params_;

            res->device_ = device_;
            std::string kernel_path = shaders_path + (std::string)"/" + it->first + ".spv";
            res->shader_module_ = ShaderModule::Create(device_->device(), params->buffer_type, kernel_path.c_str());
            //
            std::vector<VkDescriptorSetLayout> set_layouts = res->shader_module_->descriptor_set_layouts();
            res->pipeline_ = Pipeline::Create(device_->device(), res->shader_module_->shader_module(), 
                                              set_layouts, "main", params->spec_constant, params->push_constant_num); 
            //
            auto pool_sizes = res->shader_module_->CalculateDescriptorPoolSize();
            res->descriptor_pool_ = DescriptorPool::Create(device_->device(), res->shader_module_->num_sets(), pool_sizes);
            res->descriptor_pool_->AllocateDescriptorSets(set_layouts);
            res->descriptor_set_ = res->descriptor_pool_->GetDescriptorSet(set_layouts[0]);

            res->command_buffer_ = CommandBuffer::Create(device_->device(), device_->command_pool());

            it++;
        } 
        printf("Finish Engine::Init.\n");
    }

    void Deinit() {
        // Cleanup exec_map_
        std::unordered_map<std::string, ExecUnit*>::iterator it = exec_map_.begin();
        while (it != exec_map_.end()) {
            ExecUnit *res = it->second;
            delete res->command_buffer_;
            delete res->descriptor_pool_;
            delete res->pipeline_;
            delete res->shader_module_;
            delete res->params_;
            delete res;
            it++;
        }
        delete device_;
        delete instance_;
    }

    Buffer *CreateBuffer(uint32_t size) {
        return Buffer::Create(device_->device(),
                            device_->memory_properties(),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                            size);
    }

    void Run(std::string kernel_name, uint32_t *num_xyz, std::vector<Buffer*> &input_buffers, 
             const int push_constant_size, const void *push_constant, 
             std::vector<Buffer*> &output_buffers) {

        std::unordered_map<std::string, ExecUnit*>::iterator it = exec_map_.find(kernel_name);
        if (it == exec_map_.end()) {
            // PTK_LOGE("Can not find Op: %s.\n", kernel_name.c_str());
            printf("Can not find Op: %s.\n", kernel_name.c_str());
            return;
        }
        ExecUnit *res = it->second;
        res->Run(num_xyz, input_buffers, push_constant_size, push_constant, output_buffers);
    }

private:
    Instance *instance_;
    Device *device_;

    std::unordered_map<std::string, ExecUnit*> exec_map_;
};

} // namespace vk
} // namespace ptk

#endif // PTK_ENGINE_VULKAN_ENGINE_HPP_
