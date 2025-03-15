#include <cuda_runtime.h>

#include <iostream>
#include <memory>
#include <string>

#include "pocket-ai/engine/cu/common.hpp"

// Reference: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/1_Utilities/deviceQuery/deviceQuery.cpp
//            https://github.com/NVIDIA/cuda-samples/blob/master/Samples/1_Utilities/bandwidthTest/bandwidthTest.cu
//            https://github.com/NVIDIA/cuda-samples/blob/master/Samples/1_Utilities/topologyQuery/topologyQuery.cu

// CUDART_VERSION >= 5000
class DeviceQuery {
public:
    int GetDeviceCount() {
        int device_count = 0;
        cudaError_t error_id = cudaGetDeviceCount(&device_count);
    
        if (error_id != cudaSuccess) {
            printf("cudaGetDeviceCount returned %d\n-> %s\n", static_cast<int>(error_id), cudaGetErrorString(error_id));
            printf("Result = FAIL\n");
            exit(EXIT_FAILURE);
        }
    
        // This function call returns 0 if there are no CUDA capable devices.
        if (device_count == 0) {
            printf("There are no available device(s) that support CUDA\n");
        } else {
            printf("Detected %d CUDA Capable device(s)\n", device_count);
        }

        // Console log
        int driver_version, runtime_version;
        cudaDriverGetVersion(&driver_version);
        cudaRuntimeGetVersion(&runtime_version);
        printf("CUDA Driver = CUDART, CUDA Driver Version = %d.%d, CUDA Runtime Version = %d.%d, NumDevs = %d \n", 
                driver_version / 1000, (driver_version % 100) / 10,
                runtime_version / 1000, (runtime_version % 100) / 10,
                device_count);

        return device_count;
    }

    void GetProperties(int device_id) {

        cudaSetDevice(device_id);
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, device_id);

        printf("\nDevice %d: \"%s\"\n", device_id, device_prop.name);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", device_prop.major, device_prop.minor);

        char msg[256];
        snprintf(msg, sizeof(msg),
            "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
            static_cast<float>(device_prop.totalGlobalMem / 1048576.0f), (unsigned long long)device_prop.totalGlobalMem);
        printf("%s", msg);

        printf("  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n", device_prop.multiProcessorCount,
            ConvertSMVer2Cores(device_prop.major, device_prop.minor),
            ConvertSMVer2Cores(device_prop.major, device_prop.minor) * device_prop.multiProcessorCount);
        printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f ""GHz)\n",
            device_prop.clockRate * 1e-3f, device_prop.clockRate * 1e-6f);

        // This is supported in CUDA 5.0 (runtime API device properties)
        printf("  Memory Clock rate:                             %.0f Mhz\n", device_prop.memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n", device_prop.memoryBusWidth);

        if (device_prop.l2CacheSize) {
            printf("  L2 Cache Size:                                 %d bytes\n", device_prop.l2CacheSize);
        }

        printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
        device_prop.maxTexture1D, device_prop.maxTexture2D[0],
        device_prop.maxTexture2D[1], device_prop.maxTexture3D[0],
        device_prop.maxTexture3D[1], device_prop.maxTexture3D[2]);
        printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n", device_prop.maxTexture1DLayered[0], device_prop.maxTexture1DLayered[1]);
        printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
            device_prop.maxTexture2DLayered[0], device_prop.maxTexture2DLayered[1], device_prop.maxTexture2DLayered[2]);
        printf("  Total amount of constant memory:               %zu bytes\n", device_prop.totalConstMem);
        printf("  Total amount of shared memory per block:       %zu bytes\n", device_prop.sharedMemPerBlock);
        printf("  Total shared memory per multiprocessor:        %zu bytes\n", device_prop.sharedMemPerMultiprocessor);
        printf("  Total number of registers available per block: %d\n", device_prop.regsPerBlock);
        printf("  Warp size:                                     %d\n", device_prop.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n", device_prop.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n", device_prop.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",  device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",  device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %zu bytes\n", device_prop.memPitch);
        printf("  Texture alignment:                             %zu bytes\n", device_prop.textureAlignment);
        printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (device_prop.deviceOverlap ? "Yes" : "No"), device_prop.asyncEngineCount);
        printf("  Run time limit on kernels:                     %s\n", device_prop.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n", device_prop.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n", device_prop.canMapHostMemory ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n", device_prop.surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support:                        %s\n", device_prop.ECCEnabled ? "Enabled" : "Disabled");
        printf("  Device supports Unified Addressing (UVA):      %s\n", device_prop.unifiedAddressing ? "Yes" : "No");
        printf("  Device supports Managed Memory:                %s\n", device_prop.managedMemory ? "Yes" : "No");
        printf("  Device supports Compute Preemption:            %s\n", device_prop.computePreemptionSupported ? "Yes" : "No");
        printf("  Supports Cooperative Kernel Launch:            %s\n", device_prop.cooperativeLaunch ? "Yes" : "No");
        printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n", device_prop.cooperativeMultiDeviceLaunch ? "Yes" : "No");
        printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n", device_prop.pciDomainID, device_prop.pciBusID, device_prop.pciDeviceID);

        const char *sComputeMode[] = {
            "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
            "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
            "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
            "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
            "Unknown", NULL
        };
        printf("  Compute Mode:\n");
        printf("     < %s >\n", sComputeMode[device_prop.computeMode]);
    
    }
    
    void CheckRDMA(int device_count) {
        // If there are 2 or more GPUs, query to determine whether RDMA is supported
        if (device_count >= 2) {
            cudaDeviceProp prop[64];
            int gpuid[64];  // we want to find the first two GPUs that can support P2P
            int gpu_p2p_count = 0;

            for (int i = 0; i < device_count; i++) {
                CUDA_CHECK(cudaGetDeviceProperties(&prop[i], i));

                // Only boards based on Fermi or later can support P2P
                if ((prop[i].major >= 2)) {
                    // This is an array of P2P capable GPUs
                    gpuid[gpu_p2p_count++] = i;
                }
            }

            // Show all the combinations of support P2P GPUs
            int can_access_peer;

            if (gpu_p2p_count >= 2) {
                for (int i = 0; i < gpu_p2p_count; i++) {
                    for (int j = 0; j < gpu_p2p_count; j++) {
                        if (gpuid[i] == gpuid[j]) {
                            continue;
                        }
                        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
                        printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
                                prop[gpuid[i]].name, gpuid[i], prop[gpuid[j]].name, gpuid[j],
                                can_access_peer ? "Yes" : "No");
                    }
                }
            }
        }
    }

    void CheckTopology(int device_count) {
        printf("  Topology: \n");
        // Enumerates Device <-> Device links
        for (int device1 = 0; device1 < device_count; device1++) {
            for (int device2 = 0; device2 < device_count; device2++) {
                if (device1 == device2) continue;
    
                int perf_rank = 0;
                int atomic_supported = 0;
                int access_supported = 0;
    
                CUDA_CHECK(cudaDeviceGetP2PAttribute(
                    &access_supported, cudaDevP2PAttrAccessSupported, device1, device2));
                CUDA_CHECK(cudaDeviceGetP2PAttribute(
                    &perf_rank, cudaDevP2PAttrPerformanceRank, device1, device2));
                CUDA_CHECK(cudaDeviceGetP2PAttribute(
                    &atomic_supported, cudaDevP2PAttrNativeAtomicSupported, device1, device2));
    
                if (access_supported) {
                    printf("    GPU %d <-> GPU %d : \n", device1, device2);
                    printf("      * Atomic Supported: %s\n", (atomic_supported ? "yes" : "no"));
                    printf("      * Perf Rank: %d\n", perf_rank);
                }
            }
        }
    
        // Enumerates Device <-> Host links
        for (int device = 0; device < device_count; device++) {
            int atomic_supported = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(
                &atomic_supported, cudaDevAttrHostNativeAtomicSupported, device));
            printf("    GPU %d <-> CPU: \n", device);
            printf("      * Atomic Supported: %s\n", (atomic_supported ? "yes" : "no"));
        }
    }


private:
    // Beginning of GPU Architecture definitions
    int ConvertSMVer2Cores(int major, int minor) {
        // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
        typedef struct {
            int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
            int Cores;
        } sSMtoCores;

        sSMtoCores nGpuArchCoresPerSM[] = {
            {0x30, 192},
            {0x32, 192},
            {0x35, 192},
            {0x37, 192},
            {0x50, 128},
            {0x52, 128},
            {0x53, 128},
            {0x60, 64},
            {0x61, 128},
            {0x62, 128},
            {0x70, 64},
            {0x72, 64},
            {0x75, 64},
            {0x80, 64},
            {0x86, 128},
            {0x87, 128},
            {0x89, 128},
            {0x90, 128},
            {0xa0, 128},
            {0xa1, 128},
            {0xc0, 128},
            {-1, -1}};

        int index = 0;

        while (nGpuArchCoresPerSM[index].SM != -1) {
            if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
                return nGpuArchCoresPerSM[index].Cores;
            }
            index++;
        }

        // If we don't find the values, we default use the previous one to run properly
        printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n",
            major, minor, nGpuArchCoresPerSM[index - 1].Cores);
        return nGpuArchCoresPerSM[index - 1].Cores;
    }
};