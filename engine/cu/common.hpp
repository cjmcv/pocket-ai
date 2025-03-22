// %%cuda_group_save --group shared --name "common.h"

#ifndef POCKET_AI_ENGINE_CUDA_COMMON_HPP_
#define POCKET_AI_ENGINE_CUDA_COMMON_HPP_

#include <iostream>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

namespace pai {
namespace cu {
////////////////
// Macro.
////////////////

// #define CUDA_CHECK(condition) \
//     do {} while(0); 

template <typename T>
void cuda_check(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA_CHECK error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(val) pai::cu::cuda_check((val), #val, __FILE__, __LINE__)

#define FLOAT4(value)  *(float4*)(&(value))
#define OFFSET(i, j, ld) ((i)*(ld)+(j))
////////////////
// Structure.
////////////////

// Timer for cuda.
struct GpuTimer {
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        CUDA_CHECK(cudaEventDestroy(start_));
        CUDA_CHECK(cudaEventDestroy(stop_));
    }
    void Start() {
        CUDA_CHECK(cudaEventRecord(start_, NULL));
    }
    void Stop() {
        CUDA_CHECK(cudaEventRecord(stop_, NULL));
    }
    float ElapsedMillis() {
        float elapsed;
        CUDA_CHECK(cudaEventSynchronize(stop_));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_, stop_));
        return elapsed;
    }

    cudaEvent_t start_;
    cudaEvent_t stop_;
};

////////////////
// Function.
////////////////

// 
inline int InitEnvironment(const int dev_id) {
    CUDA_CHECK(cudaSetDevice(dev_id));
    cudaDeviceProp device_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, dev_id));
    if (device_prop.computeMode == cudaComputeModeProhibited) {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        return 1;
    }
    fprintf(stderr, "GPU Device %d: \"%s\" with compute capability %d.%d with %d multi-processors.\n\n", 
      dev_id, device_prop.name, device_prop.major, device_prop.minor, device_prop.multiProcessorCount);

    return 0;
}

inline void CleanUpEnvironment() {
    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    CUDA_CHECK(cudaDeviceReset());
}

inline int DivCeil(int a, int b) {
    return (a + b - 1) / b;
}

} // namespace cu
} // namespace pai

#endif //POCKET_AI_ENGINE_CUDA_COMMON_HPP_