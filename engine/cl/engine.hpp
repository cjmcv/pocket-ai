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
              std::vector<std::pair<std::string, pSetParamsFuncs>> &kernels_params, 
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
        loaders_.push_back(new cl::KernelLoader);
        std::string kernel_file_name = kernels_path + "/dot_product.cl";
        loaders_[0]->Load(kernel_file_name.c_str());
        loaders_[0]->CreateProgram(context_);

        for (uint32_t i=0; i<kernels_params.size(); i++) {
            kernels_map_[kernels_params[i].first] = loaders_[0]->CreateKernel(kernels_params[i].first, kernels_params[i].second);
        }

        queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &err_code);
    }

    void Deinit() {
        delete platform_;
        if (context_) CL_CHECK(clReleaseContext(context_));
        if (queue_) CL_CHECK(clReleaseCommandQueue(queue_));

        for (uint32_t i=0; i<loaders_.size(); i++) {
            loaders_[i]->UnLoad();
            delete loaders_[i];            
        }
    }

    Kernel *GetKernel(std::string kernel_name) {
        std::unordered_map<std::string, cl::Kernel *>::iterator it = kernels_map_.find(kernel_name);
        if (it == kernels_map_.end()) {
            PTK_LOGE("Can not find Kernel: %s.\n", kernel_name.c_str());
        }

        cl::Kernel *kernel = it->second;
        kernel->ResourceBinding(context_, queue_);
        return kernel;
    }

    void AsyncRun(std::string kernel_name, uint32_t work_dim, const size_t *global_work_size, const size_t *local_work_size, cl_event *ev) {
        std::unordered_map<std::string, cl::Kernel *>::iterator it = kernels_map_.find(kernel_name);
        if (it == kernels_map_.end()) {
            PTK_LOGE("Can not find Kernel: %s.\n", kernel_name.c_str());
            return;
        }
        cl::Kernel *kernel = it->second;
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

    std::vector<KernelLoader *> loaders_;
    std::unordered_map<std::string, cl::Kernel*> kernels_map_;
};

} // namespace cl
} // namespace ptk

#endif //PTK_ENGINE_OPENCL_ENGINE_HPP_