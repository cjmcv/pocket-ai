#ifndef PTK_ENGINE_OPENCL_ENGINE_HPP_
#define PTK_ENGINE_OPENCL_ENGINE_HPP_

#include <unordered_map>
#include <string>

#include "common.hpp"
#include "kernel.hpp"
#include "platform.hpp"

namespace ptk {
namespace cl {

class Engine {
public:
    Engine() {}
    ~Engine() {}

    void Init(std::string kernels_path, 
              std::vector<std::tuple<std::string, std::string, pSetParamsFuncs>> &kernels_params, 
              uint32_t platform_id) {
        platform_ = new cl::Platform;
        platform_->GetInfos();
        platform_->GetDeviceId(platform_id, &device_);

        // Create the context
        cl_int err_code;
        context_ = clCreateContext(0, 1, &device_, NULL, NULL, &err_code);
        CL_CHECK(err_code);

        ///////////////////////////
        // Load CL source.
        // 一份kernel源码文件，可包含多个kernel函数。
        // 遍历输入的所有kernel参数，针对第一个元素，即源码文件名，创建loader，
        // 重名的只取一份
        for (uint32_t i=0; i<kernels_params.size(); i++) {
            std::string src_name = std::get<0>(kernels_params[i]);
            std::unordered_map<std::string, KernelLoader *>::iterator it = loaders_map_.find(src_name);
            if (it == loaders_map_.end()) {
                KernelLoader *loader = new KernelLoader;
                std::string kernel_file_name = kernels_path + "/" + src_name + ".cl";
                loader->Load(kernel_file_name.c_str());
                loader->CreateProgram(context_);
                
                loaders_map_[src_name] = loader;
                PTK_LOGS("Loaded programs: %s.\n", src_name.c_str());
            }
        }
        // printf("%s, %s, %p.\n", std::get<0>(kernels_params[0]).c_str(), std::get<1>(kernels_params[0]).c_str(), std::get<2>(kernels_params[0]));
        // 找到kernel函数名对应的源码文件名，取出loader并创建kernel。
        for (uint32_t i=0; i<kernels_params.size(); i++) {
            std::string src_name = std::get<0>(kernels_params[i]);
            std::unordered_map<std::string, KernelLoader *>::iterator it = loaders_map_.find(src_name);
            if (it == loaders_map_.end()) {
                PTK_LOGE("Can not find src file: %s.\n", src_name.c_str());
            }

            std::string kernel_name = std::get<1>(kernels_params[i]);
            kernels_map_[kernel_name] = it->second->CreateKernel(kernel_name, std::get<2>(kernels_params[i]));
            PTK_LOGS("Registered kernels: %s.\n", kernel_name.c_str());
        }

        queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &err_code);
    }

    void Deinit() {
        delete platform_;
        if (context_) CL_CHECK(clReleaseContext(context_));
        if (queue_) CL_CHECK(clReleaseCommandQueue(queue_));

        for (auto it=loaders_map_.begin(); it != loaders_map_.end(); it++) {
            it->second->UnLoad();
            delete it->second;
        }
    }

    Kernel *GetKernel(std::string kernel_name) {
        std::unordered_map<std::string, Kernel *>::iterator it = kernels_map_.find(kernel_name);
        if (it == kernels_map_.end()) {
            PTK_LOGE("Can not find Kernel: %s.\n", kernel_name.c_str());
        }

        cl::Kernel *kernel = it->second;
        kernel->ResourceBinding(context_, queue_);
        return kernel;
    }

    void AsyncRun(cl::Kernel *kernel, uint32_t work_dim, const size_t *global_work_size, const size_t *local_work_size, cl_event *ev) {
        clEnqueueNDRangeKernel(queue_, kernel->kernel(), work_dim, NULL, global_work_size, local_work_size, 0, NULL, ev);
    }

    void FinishQueue() {
        clFinish(queue_);
    }

private:
    Platform *platform_;
    cl_device_id device_;
    cl_context context_;
    cl_command_queue queue_;

    std::unordered_map<std::string, KernelLoader*> loaders_map_;
    std::unordered_map<std::string, Kernel*> kernels_map_;
};

} // namespace cl
} // namespace ptk

#endif //PTK_ENGINE_OPENCL_ENGINE_HPP_