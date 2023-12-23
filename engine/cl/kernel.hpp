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

    void CreateBuffer(cl_context context, std::vector<uint32_t> &size) {
        cl_int err_code;
        params_->io_buffer.resize(params_->io_attri.size());
        for (uint32_t i=0; i<params_->io_attri.size(); i++) {
            KernelIOAttri *io_attri = &params_->io_attri[i];
            KernelIOBuffer *id_buffer = &params_->io_buffer[i];
            if (io_attri->mem_flag != NULL) {
                id_buffer->mem = clCreateBuffer(context, io_attri->mem_flag, size[i], NULL, &err_code);
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

    void WriteBuffer(cl_command_queue queue, cl_bool is_blocking, const void *host_ptr, uint32_t args_id) {
        if (args_id > params_->io_buffer.size() - 1) {
            PTK_LOGE("WriteBuffer -> args_id: %d is out of range.\n", args_id);
        }
        KernelIOBuffer *id_buffer = &params_->io_buffer[args_id];
        CL_CHECK(clEnqueueWriteBuffer(queue, id_buffer->mem, is_blocking, 0, id_buffer->size, host_ptr, 0, NULL, NULL));
    }

    void ReadBuffer(cl_command_queue queue, cl_bool is_blocking, void *host_ptr, uint32_t args_id) {
        if (args_id > params_->io_buffer.size() - 1) {
            PTK_LOGE("ReadBuffer -> args_id: %d is out of range.\n", args_id);
        }
        KernelIOBuffer *id_buffer = &params_->io_buffer[args_id];
        CL_CHECK(clEnqueueReadBuffer(queue, id_buffer->mem, is_blocking, 0, id_buffer->size, host_ptr, 0, NULL, NULL));
    }

    void *MapBUffer(cl_command_queue queue, cl_bool is_blocking, uint32_t args_id) {
        if (args_id > params_->io_buffer.size() - 1) {
            PTK_LOGE("MapBUffer -> args_id: %d is out of range.\n", args_id);
        }
        cl_int err_code;
        KernelIOBuffer *id_buffer = &params_->io_buffer[args_id];
        id_buffer->mapped_ptr = clEnqueueMapBuffer(queue, id_buffer->mem, is_blocking, CL_MAP_WRITE, 0, id_buffer->size, 0, NULL, NULL, &err_code);
        CL_CHECK(err_code);
        return id_buffer->mapped_ptr;
    }

    void UnmapBuffer(cl_command_queue queue, uint32_t args_id) {
        if (args_id > params_->io_buffer.size() - 1) {
            PTK_LOGE("UnmapBuffer -> args_id: %d is out of range.\n", args_id);
        }
        KernelIOBuffer *id_buffer = &params_->io_buffer[args_id];
        CL_CHECK(clEnqueueUnmapMemObject(queue, id_buffer->mem, id_buffer->mapped_ptr, 0, NULL, NULL));
    }

private:
    std::string name_;
    cl_kernel kernel_;
    KernelParams *params_;
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

class Platform {

public:
    Platform() {
        platforms_ = nullptr;
        names_ = nullptr;
        versions_ = nullptr;
    }

    ~Platform() {
        if (platforms_) {
            free(platforms_);
            platforms_ = nullptr;
        }
        if (names_) {
            for (cl_uint i = 0; i < num_; i++) {
                if (names_[i])
                    free(names_[i]);
            }
            free(names_);
        }
        if (versions_) {
            for (cl_uint i = 0; i < num_; i++) {
                if (versions_[i])
                    free(versions_[i]);
            }
            free(versions_);
        }
    }

    void GetInfos() {
        // Get an OpenCL platform.
        CL_CHECK(clGetPlatformIDs(5, NULL, &num_));
        //////////////// platforms info //////////////////
        PTK_LOGS("\n//////////////// platforms info //////////////////\n");
        PTK_LOGS("There are ( %d ) platforms that support OpenCL.\n", num_);

        platforms_ = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_);
        CL_CHECK(clGetPlatformIDs(num_, platforms_, NULL));

        names_ = (char **)malloc(num_ * sizeof(char *));
        versions_ = (char **)malloc(num_ * sizeof(char *));
        for (int i = 0; i < num_; i++) {
            size_t ext_size;
            CL_CHECK(clGetPlatformInfo(platforms_[i], CL_PLATFORM_EXTENSIONS,
                                        0, NULL, &ext_size));
            names_[i] = (char *)malloc(ext_size);
            CL_CHECK(clGetPlatformInfo(platforms_[i], CL_PLATFORM_NAME,
                                        ext_size, names_[i], NULL));

            versions_[i] = (char *)malloc(ext_size);
            CL_CHECK(clGetPlatformInfo(platforms_[i], CL_PLATFORM_VERSION,
                                        ext_size, versions_[i], NULL));

            PTK_LOGS("The name of the platform is <%s> with version <%s>.\n", names_[i], versions_[i]);
        
            if (i != 0) continue;

            cl_device_id device_id;
            clGetDeviceIDs(platforms_[i], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
                
            char* text_value = new char[200];
            size_t* arr_value = new size_t[200];
            cl_uint num_value = 0;
            size_t size_value = 0;
            cl_ulong long_value = 0;
            {
                size_t length = 0;
                // text info
                CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, 0, &length));
                CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_NAME, length, text_value, 0));
                PTK_LOGS("=========== CL_DEVICE_NAME: <%s>.\n", text_value);

                CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, 0, 0, &length));
                CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, length, text_value, 0));
                PTK_LOGS("=========== CL_DEVICE_VENDOR: <%s>.\n", text_value);

                // num info
                CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(num_value), &num_value, &length));
                PTK_LOGS("=========== CL_DEVICE_MAX_COMPUTE_UNITS: <%d>.\n", num_value);

                CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(num_value), &num_value, &length));
                PTK_LOGS("=========== CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: <%d>.\n", num_value);  

                CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_value), &size_value, &length));
                PTK_LOGS("=========== CL_DEVICE_MAX_WORK_GROUP_SIZE: <%d>.\n", size_value);
                
                CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, 0, &length));
                CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, length, arr_value, 0));
                PTK_LOGS("=========== CL_DEVICE_MAX_WORK_ITEM_SIZES: <%d, %d, %d>.\n", arr_value[0], arr_value[1], arr_value[2]);
    
                CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(num_value), &num_value, &length));
                PTK_LOGS("=========== CL_DEVICE_MAX_CLOCK_FREQUENCY: <%d>.\n", num_value);

                CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(long_value), &long_value, &length));
                PTK_LOGS("=========== CL_DEVICE_GLOBAL_MEM_CACHE_SIZE: <%llu>.\n", long_value);

                CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(long_value), &long_value, &length));
                PTK_LOGS("=========== CL_DEVICE_GLOBAL_MEM_SIZE: <%llu>.\n", long_value);

                CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(long_value), &long_value, &length));
                PTK_LOGS("=========== CL_DEVICE_LOCAL_MEM_SIZE: <%llu>.\n", long_value);
            }
            delete[] text_value;
            delete[] arr_value;
        }
        PTK_LOGS("//////////////////////////////////////////////\n\n");
    }

    inline cl_platform_id *platforms() { return platforms_; }

    // platform_name: "NVIDIA CUDA" / "Intel(R) OpenCL"
    // device_order: Serial number of the same platform.
    bool GetDeviceId(std::string platform_name, cl_device_id *device_id, int num_device = 1){
        for (int i = 0; i < num_; i++) {
            if (platform_name == std::string(names_[i])) {
                CL_CHECK(clGetDeviceIDs(platforms_[i], CL_DEVICE_TYPE_GPU, num_device, device_id, NULL));
                return true;
            }
        }
        return false;
    }

private:
    cl_platform_id *platforms_;
    cl_uint num_;

    char **names_;
    char **versions_;
};

} // namespace cl
} // namespace ptk

#endif //PTK_ENGINE_OPENCL_KERNEL_HPP_