/*!
* \brief VulkanEngine 对外提供vulkan操作的最外层类，
*     而engine/vulkan内其他文件对外部不可见。
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

typedef enum DescriptorType {
    DESCRIPTOR_TYPE_STORAGE_IMAGE = 1, // 3,
    DESCRIPTOR_TYPE_UNIFORM_BUFFER = 2, // 6,
    DESCRIPTOR_TYPE_STORAGE_BUFFER = 3 // 7
} DescriptorType;

struct SpecializationConstant {
    uint32_t id;
    union {
        int32_t s32;
        uint32_t u32;
        float f32;
    } value;
};

// kernel固定参数，在kernel注册时指定，不需要依赖vulkan头文件和库
struct KernelParams {
    std::vector<DescriptorType> buffer_type;
    std::vector<SpecializationConstant> spec_constant;
    uint32_t push_constant_num;
};

typedef void (*SetpParamsFuncs)(KernelParams *params);

class VulkanEngine {
    // kernel运行所需资源单元。由Engine创建，一个kernel对应一个ExecUnit
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

            // buffer绑定顺序需要跟comp里一致, 即i与comp里对应的一致
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
            std::vector<VkDescriptorType> buffer_types;
            buffer_types.resize(params->buffer_type.size());
            for (uint32_t i=0; i<params->buffer_type.size(); i++) {
                if (params->buffer_type[i] == DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    buffer_types[i] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            }
            res->shader_module_ = ShaderModule::Create(device_->device(), buffer_types, kernel_path.c_str());
            //
            std::vector<SpecConstant> spec_constants;
            spec_constants.resize(params->spec_constant.size());
            for (uint32_t i=0; i<params->spec_constant.size(); i++) {
                spec_constants[i].id = params->spec_constant[i].id;
                spec_constants[i].value.u32 = params->spec_constant[i].value.u32;
            }
            std::vector<VkDescriptorSetLayout> set_layouts = res->shader_module_->descriptor_set_layouts();
            res->pipeline_ = Pipeline::Create(device_->device(), res->shader_module_->shader_module(), 
                                            set_layouts, "main", spec_constants, params->push_constant_num); 
            //
            auto pool_sizes = res->shader_module_->CalculateDescriptorPoolSize();
            res->descriptor_pool_ = DescriptorPool::Create(device_->device(), res->shader_module_->num_sets(), pool_sizes);
            res->descriptor_pool_->AllocateDescriptorSets(set_layouts);
            res->descriptor_set_ = res->descriptor_pool_->GetDescriptorSet(set_layouts[0]);

            res->command_buffer_ = CommandBuffer::Create(device_->device(), device_->command_pool());

            it++;
        } 
        printf("Finish VulkanEngine::Init.\n");
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
