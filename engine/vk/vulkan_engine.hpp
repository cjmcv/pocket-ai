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

// 该文件无vulkan依赖，kernel中不带入vulkan相关内容，与vulkan engine解耦。
// kernel的配置只描述了kernel的相关信息，engine里include了kernels_list后将里面的配置转换为vulkan相关类型进行使用。
// vulkan -> op 依赖 engine 依赖 kernel+kernel dispatcher。 host -> op 依赖 kernel。
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
    uint32_t workgroup_size[3];
};

// TODO: 删掉，在test里有
const int WIDTH = 3200; // Size of rendered mandelbrot set.
const int HEIGHT = 2400; // Size of renderered mandelbrot set.

// TODO：换个地方
void SetParamsMandelbrot(KernelParams *params) {
    params->buffer_type = {
        DESCRIPTOR_TYPE_STORAGE_BUFFER
    };
    params->spec_constant = {};
    params->push_constant_num = 0;
    // The workgroup_size is fixed in comp: layout(local_size_x = WG_X, local_size_y = WG_Y, local_size_z = 1) in;
    // 这里的设定只是计算dispatch的group数量，考虑用特化常量
    params->workgroup_size[0] = 32;
    params->workgroup_size[1] = 32;
    params->workgroup_size[2] = 1;
}

void SetParamsMatMulTiledFp32(KernelParams *params) {
    params->buffer_type = {
        DESCRIPTOR_TYPE_STORAGE_BUFFER,
        DESCRIPTOR_TYPE_STORAGE_BUFFER,
        DESCRIPTOR_TYPE_STORAGE_BUFFER // VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    };

    params->spec_constant = {
        {0, 640},
        {1, 640},
        {2, 640},
    };
    params->push_constant_num = 0;
    params->workgroup_size[0] = 16;
    params->workgroup_size[1] = 1;
    params->workgroup_size[2] = 1;
}

void SetParamsEngineTest(KernelParams *params) {
    params->buffer_type = {
        DESCRIPTOR_TYPE_STORAGE_BUFFER,
        DESCRIPTOR_TYPE_STORAGE_BUFFER,
        DESCRIPTOR_TYPE_STORAGE_BUFFER // VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    };

    params->spec_constant = {
        {0, 160},
        {1, 320}};
    SpecializationConstant c;
    c.id = 2;
    c.value.f32 = 640.123f;
    params->spec_constant.push_back(c);

    params->push_constant_num = 2;
    params->workgroup_size[0] = 16;
    params->workgroup_size[1] = 1;
    params->workgroup_size[2] = 1;
}

typedef void (*SetpParamsFuncs)(KernelParams *params);

class VulkanKernelDispatcher {
public:
    //
    KernelParams *CreateKernelParams(std::string kernel_name) {
        KernelParams *params = new KernelParams;

        std::map<std::string, SetpParamsFuncs>::iterator iter;
        iter = params_.find(kernel_name);
        if(iter != params_.end()) {
            PTK_LOGI("CreateKernelParams: %s.\n", kernel_name.c_str());
            iter->second(params);
        }
        else {
            printf("Can not find params.\n");
        }
        return params;
    }
	// Singleton mode. Only one KernelFactory exist.
	static VulkanKernelDispatcher *GetInstance() {
		static VulkanKernelDispatcher *dispatcher = new VulkanKernelDispatcher;
		return dispatcher;
	}

private:
	VulkanKernelDispatcher() {
        params_["mandelbrot"] = SetParamsMandelbrot;
        params_["matmul_tiled_fp32"] = SetParamsMatMulTiledFp32;
        params_["engine_test"] = SetParamsEngineTest;
    }

    std::map<std::string, SetpParamsFuncs> params_;
};

class VulkanEngine {
    // kernel运行所需资源单元。由Engine创建，一个kernel对应一个ExecUnit
    class ExecUnit {
    public:
        void Run(std::vector<Buffer*> &input_buffers, const int push_constant_size, 
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
            command_buffer_->Dispatch((uint32_t)ceil(WIDTH / float(params_->workgroup_size[0])), 
                                      (uint32_t)ceil(HEIGHT / float(params_->workgroup_size[1])), 
                                      1);
            command_buffer_->End();

            device_->QueueSubmitAndWait(command_buffer_->command_buffer());
        }

        KernelParams *params_;
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
    void Init(std::string shaders_path, int physical_device_id = 0, bool enable_validation = false) {
        vk_dispatcher_ = VulkanKernelDispatcher::GetInstance();
        instance_ = new Instance(enable_validation);
        std::vector<VkPhysicalDevice> phys_devices = instance_->EnumeratePhysicalDevices(true);
        device_ = Device::Create(phys_devices[physical_device_id], VK_QUEUE_COMPUTE_BIT, instance_->layers());

        SetKernelMap();
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

    void Run(std::string kernel_name, std::vector<Buffer*> &input_buffers, 
             const int push_constant_size, const void *push_constant, 
             std::vector<Buffer*> &output_buffers) {

        std::unordered_map<std::string, ExecUnit*>::iterator it = exec_map_.find(kernel_name);
        if (it == exec_map_.end()) {
            // PTK_LOGE("Can not find Op: %s.\n", kernel_name.c_str());
            printf("Can not find Op: %s.\n", kernel_name.c_str());
            return;
        }
        ExecUnit *res = it->second;
        res->Run(input_buffers, push_constant_size, push_constant, output_buffers);
    }

private:
    void SetKernelMap() {
        {
            ExecUnit *res = new ExecUnit;
            res->params_ = vk_dispatcher_->CreateKernelParams("mandelbrot");
            exec_map_["mandelbrot"] = res;
        }
        {
            ExecUnit *res = new ExecUnit;
            res->params_ = vk_dispatcher_->CreateKernelParams("matmul_tiled_fp32");
            exec_map_["matmul_tiled_fp32"] = res;
        }
        {
            ExecUnit *res = new ExecUnit;
            res->params_ = vk_dispatcher_->CreateKernelParams("engine_test");
            exec_map_["engine_test"] = res;
        }
    }

private:
    Instance *instance_;
    Device *device_;

    std::unordered_map<std::string, ExecUnit*> exec_map_;
    VulkanKernelDispatcher *vk_dispatcher_;
};

} // namespace vk
} // namespace ptk

#endif // PTK_ENGINE_VULKAN_ENGINE_HPP_
