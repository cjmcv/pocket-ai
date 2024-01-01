

// opencl: local memory: __local          cuda: shared memory: __shared__
//         get_local_id(0)                      threadIdx.x
//         get_local_id(1)                      threadIdx.y
//         get_global_id(0)                     blockIdx.x * blockDim.x + threadIdx.x
//         get_global_id(1)                     blockIdx.y * blockDim.y + threadIdx.y
//         barrier(CLK_LOCAL_MEM_FENCE)         __syncthreads();
#define WORKGROUP_SIZE 256
__kernel void DotProductDevice(__global int *src1, __global int *src2, const int len, __global int *dst) {
  
    __local int shared[WORKGROUP_SIZE];
    for (int gid = get_global_id(0); gid < len; gid += get_global_size(0)) {
        
        int tid = get_local_id(0);    
        if (gid == 0)
            *dst = 0;

        // Save the intermediate result to shared. 
        //    Part of the data of the last group may not be involved 
        shared[tid] = src1[gid] * src2[gid];

        // Make sure all working groups are done.
        barrier(CLK_LOCAL_MEM_FENCE);

        // V1. 1.1ms
        // The first item of each group is responsible for
        // aggregating the group's results, and add to the dst.
        if (tid == 0) {
            int sum = 0;
            for (int i = 0; i < WORKGROUP_SIZE; i++)
                sum += shared[i];
            atomic_add(dst, sum);
        }

        //// V2. 1.8ms
        //int count = WORKGROUP_SIZE / 2;
        //while (count >= 1) {
        //    if (lid < count) {
        //        buffer[lid] += buffer[count + lid];
        //    }  
        //    barrier(CLK_LOCAL_MEM_FENCE);
        //    count = count / 2;
        //}
        //if(lid == 0)
        //    atomic_add(dst, buffer[lid]);
    }
}
