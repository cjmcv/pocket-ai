#include <cuda_runtime.h>

#include <iostream>
#include <memory>
#include <string>

// Reference: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/1_Utilities/deviceQuery/deviceQuery.cpp
//            https://github.com/NVIDIA/cuda-samples/blob/master/Samples/1_Utilities/bandwidthTest/bandwidthTest.cu
//            https://github.com/NVIDIA/cuda-samples/blob/master/Samples/1_Utilities/topologyQuery/topologyQuery.cu

// GPU Query: https://www.techpowerup.com/gpu-specs/geforce-rtx-4050-mobile.c3953

// TODO: https://github.com/sjfeng1999/gpu-arch-microbenchmark/tree/master
//       https://github.com/accel-sim/gpu-app-collection/tree/release/src/cuda/GPU_Microbenchmark

#include "pocket-ai/engine/cu/common.hpp"
#include "device_query.hpp"
#include "bandwidth_eval.hpp"
#include "max_flops.cuh"

int main(int argc, char **argv) {
    DeviceQuery query;
    BandwidthEval bw_eval;
    MaxFlopsEval mf_eval;
    int device_count = query.GetDeviceCount();
    for (int device_id=0; device_id < device_count; device_id++) {
        KeyProperties *props = query.GetProperties(device_id);
        bw_eval.Run(device_id);

        mf_eval.RunFp32(props->sm_count, props->gpu_max_clock_rate);
    }

    query.CheckRDMA(device_count);
    query.CheckTopology(device_count);
    
    // finish
    printf("\nNOTE: Results may vary when GPU Boost is enabled.\n");
    return 0;
}