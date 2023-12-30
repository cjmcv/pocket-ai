#ifndef PTK_ENGINE_OPENCL_PLATFORM_HPP_
#define PTK_ENGINE_OPENCL_PLATFORM_HPP_

#include "common.hpp"

namespace ptk {
namespace cl {

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
        for (uint32_t i = 0; i < num_; i++) {
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
                PTK_LOGS("=========== CL_DEVICE_MAX_WORK_GROUP_SIZE: <%zu>.\n", size_value);
                
                CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, 0, &length));
                CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, length, arr_value, 0));
                PTK_LOGS("=========== CL_DEVICE_MAX_WORK_ITEM_SIZES: <%zu, %zu, %zu>.\n", arr_value[0], arr_value[1], arr_value[2]);
    
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
    bool GetDeviceId(std::string platform_name, cl_device_id *device_id, int num_device = 1) {
        for (uint32_t i = 0; i < num_; i++) {
            if (platform_name == std::string(names_[i])) {
                CL_CHECK(clGetDeviceIDs(platforms_[i], CL_DEVICE_TYPE_GPU, num_device, device_id, NULL));
                return true;
            }
        }
        return false;
    }

    bool GetDeviceId(uint32_t platform_index, cl_device_id *device_id, int num_device = 1) {
        if (platform_index >= num_)
            PTK_LOGE("GetDeviceId -> platform_index: %d is out of range.\n", platform_index);

        CL_CHECK(clGetDeviceIDs(platforms_[platform_index], CL_DEVICE_TYPE_GPU, num_device, device_id, NULL));
        return true;
    }

private:
    cl_platform_id *platforms_;
    cl_uint num_;

    char **names_;
    char **versions_;
};

} // namespace cl
} // namespace ptk

#endif //PTK_ENGINE_OPENCL_PLATFORM_HPP_