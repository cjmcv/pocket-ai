#ifndef PTK_ENGINE_OPENCL_KERNEL_HPP_
#define PTK_ENGINE_OPENCL_KERNEL_HPP_

#include "common.hpp"

namespace ptk {
namespace cl {

////////////////
// Class.
////////////////
struct KernelIOAttri {
    cl_mem_flags mem_flag;
    uint32_t args_size; // 与kernel绑定的函数出入参数大小
};

struct KernelIOBuffer {
    cl_mem mem;
    void *mapped_ptr;
    uint32_t size;
};

struct KernelParams {
    std::vector<KernelIOAttri> io_attri;
    std::vector<KernelIOBuffer> io_buffer;
};

typedef void (*pSetParamsFuncs)(ptk::cl::KernelParams *params);

class Kernel {
public:
    static Kernel *Create(cl_program program, std::string name, pSetParamsFuncs set_params) {
        cl_int err_code;
        cl_kernel clkernel = clCreateKernel(program, name.c_str(), &err_code);
        CL_CHECK(err_code);

        // Register
        KernelParams *params = new KernelParams;
        set_params(params);
        return new Kernel(name, clkernel, params);
    }

    Kernel(std::string name, cl_kernel kernel, KernelParams *params)
        : kernel_(kernel), name_(name), params_(params) {}
    ~Kernel() { 
        CL_CHECK(clReleaseKernel(kernel_)); 
        delete params_;
    }

    cl_kernel kernel() { return kernel_; }

    void ResourceBinding(cl_context context, cl_command_queue queue) {
        context_ = context;
        queue_ = queue;
    }

    void AdjustIoMemAttri4Mapp() {
        for (uint32_t i=0; i<params_->io_attri.size(); i++) {
            KernelIOAttri *io_attri = &params_->io_attri[i];
            // 将所有非NULL的 mem_flag 改为 map 常用的标志
            if (io_attri->mem_flag != NULL) {
                io_attri->mem_flag = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
            }
        }
    }

    void CreateBuffer(std::vector<size_t> &size) {
        cl_int err_code;
        params_->io_buffer.resize(params_->io_attri.size());
        for (uint32_t i=0; i<params_->io_attri.size(); i++) {
            KernelIOAttri *io_attri = &params_->io_attri[i];
            KernelIOBuffer *id_buffer = &params_->io_buffer[i];
            if (io_attri->mem_flag != NULL) {
                id_buffer->mem = clCreateBuffer(context_, io_attri->mem_flag, size[i], NULL, &err_code);
                CL_CHECK(err_code);
                CL_CHECK(clSetKernelArg(kernel_, i, io_attri->args_size, (void*)&id_buffer->mem));
                id_buffer->mapped_ptr = NULL;
            }
            else {
                CL_CHECK(clSetKernelArg(kernel_, i, io_attri->args_size, (void*)&size[i]));
                id_buffer->mem = NULL;
                id_buffer->mapped_ptr = NULL;
            }
            id_buffer->size = size[i];
        }
    }

    void ReleaseBuffer() {
        for (uint32_t i=0; i<params_->io_attri.size(); i++) {
            KernelIOBuffer *id_buffer = &params_->io_buffer[i];
            if (id_buffer->mem != NULL) {
                CL_CHECK(clReleaseMemObject(id_buffer->mem));
            }
        }
    }

    void WriteBuffer(cl_bool is_blocking, const void *host_ptr, uint32_t args_id) {
        if (args_id > params_->io_buffer.size() - 1) {
            PTK_LOGE("WriteBuffer -> args_id: %d is out of range.\n", args_id);
        }
        KernelIOBuffer *id_buffer = &params_->io_buffer[args_id];
        CL_CHECK(clEnqueueWriteBuffer(queue_, id_buffer->mem, is_blocking, 0, id_buffer->size, host_ptr, 0, NULL, NULL));
    }

    void ReadBuffer(cl_bool is_blocking, void *host_ptr, uint32_t args_id) {
        if (args_id > params_->io_buffer.size() - 1) {
            PTK_LOGE("ReadBuffer -> args_id: %d is out of range.\n", args_id);
        }
        KernelIOBuffer *id_buffer = &params_->io_buffer[args_id];
        CL_CHECK(clEnqueueReadBuffer(queue_, id_buffer->mem, is_blocking, 0, id_buffer->size, host_ptr, 0, NULL, NULL));
    }

    void *MapBuffer(cl_bool is_blocking, uint32_t args_id) {
        if (args_id > params_->io_buffer.size() - 1) {
            PTK_LOGE("MapBuffer -> args_id: %d is out of range.\n", args_id);
        }
        cl_int err_code;
        KernelIOBuffer *id_buffer = &params_->io_buffer[args_id];
        id_buffer->mapped_ptr = clEnqueueMapBuffer(queue_, id_buffer->mem, is_blocking, CL_MAP_WRITE, 0, id_buffer->size, 0, NULL, NULL, &err_code);
        CL_CHECK(err_code);
        return id_buffer->mapped_ptr;
    }

    void UnmapBuffer(uint32_t args_id) {
        if (args_id > params_->io_buffer.size() - 1) {
            PTK_LOGE("UnmapBuffer -> args_id: %d is out of range.\n", args_id);
        }
        KernelIOBuffer *id_buffer = &params_->io_buffer[args_id];
        CL_CHECK(clEnqueueUnmapMemObject(queue_, id_buffer->mem, id_buffer->mapped_ptr, 0, NULL, NULL));
    }

private:
    std::string name_;
    cl_kernel kernel_;
    KernelParams *params_;

    // binding params
    cl_context context_;
    cl_command_queue queue_;
};

// KernelLoader: It is used to get the kernel functions from the program file.
class KernelLoader {
public:
    KernelLoader() {
        err_code_ = CL_SUCCESS;
        program_length_ = 0;
        program_source_ = NULL;
        program_ = NULL;
    }

    // Load 
    bool Load(const char *source_file) {
        program_source_ = LoadProgSource(source_file, "", &program_length_);
        if (program_source_ == NULL) {
            PTK_LOGE("LoadProgSource %s Failed.\n", source_file); 
            return false;
        }
        return true;
    }

    void UnLoad() {
        if (program_) CL_CHECK(clReleaseProgram(program_));
        if (program_source_) free(program_source_);
    }

    bool CreateProgram(const cl_context &context) {
        if (program_) CL_CHECK(clReleaseProgram(program_));

        program_ = clCreateProgramWithSource(context, 1, (const char **)&program_source_, &program_length_, &err_code_);
        CL_CHECK(err_code_);
        CL_CHECK(clBuildProgram(program_, 0, NULL, NULL, NULL, NULL));

        return true;
    }

    Kernel *CreateKernel(std::string name, pSetParamsFuncs set_params) {
        return Kernel::Create(program_, name, set_params);
    }

private:
    //  Loads a Program file and prepends the preamble to the code.
    char* LoadProgSource(const char* file_name, const char* preamble, size_t* final_length){
        // Locals 
        FILE* file_stream = NULL;
        size_t source_length;

        // Open the OpenCL source code file
    #ifdef _WIN32   // Windows version
        if (fopen_s(&file_stream, file_name, "rb") != 0) {
            PTK_LOGS("Can not open the file : %s.\n", file_name);
            return NULL;
        }
    #else           // Linux version
        file_stream = fopen(file_name, "rb");
        if (file_stream == 0) {
            PTK_LOGS("Can not open the file : %s.\n", file_name);
            return NULL;
        }
    #endif

        size_t preamble_length = strlen(preamble);

        // get the length of the source code
        fseek(file_stream, 0, SEEK_END);
        source_length = ftell(file_stream);
        fseek(file_stream, 0, SEEK_SET);

        // allocate a buffer for the source code string and read it in
        char* source_string = (char *)malloc(source_length + preamble_length + 1);
        memcpy(source_string, preamble, preamble_length);
        if (fread((source_string)+preamble_length, source_length, 1, file_stream) != 1) {
            fclose(file_stream);
            free(source_string);
            return NULL;
        }

        // close the file and return the total length of the combined (preamble + source) string
        fclose(file_stream);
        if (final_length != 0) {
            *final_length = source_length + preamble_length;
        }
        source_string[source_length + preamble_length] = '\0';

        return source_string;
    }

private:
    cl_int err_code_;
    // Byte size of kernel code
    // Buffer to hold source for compilation
    size_t program_length_;
    char* program_source_;

    cl_program program_;
};

} // namespace cl
} // namespace ptk

#endif //PTK_ENGINE_OPENCL_KERNEL_HPP_