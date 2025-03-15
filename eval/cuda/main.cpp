#include <cuda_runtime.h>

#include <iostream>
#include <memory>
#include <string>

#include "pocket-ai/engine/cu/common.hpp"
#include "device_query.hpp"
#include "bandwidth_eval.hpp"

int main(int argc, char **argv) {
    DeviceQuery query;
    BandwidthEval bw_eval;
    int device_count = query.GetDeviceCount();
    for (int device_id=0; device_id < device_count; device_id++) {
        query.GetProperties(device_id);
        bw_eval.Run(device_id);
    }

    query.CheckRDMA(device_count);
    query.CheckTopology(device_count);
    
    // finish
    printf("\nNOTE: Results may vary when GPU Boost is enabled.\n");
    return 0;
}