
__kernel void GemmDeviceV1(const int M, const int N, const int K,
                           __global const float *A, const int lda,
                           __global const float *B, const int ldb,
                           __global float *C, const int ldc) {

    for (int gid_x = get_global_id(0), gid_y = get_global_id(1);
        gid_x < N && gid_y < M; 
        gid_x += get_global_size(0), gid_y += get_global_size(1)) {

        float c_sub_acc = 0;
        for (int k = 0; k < K; k++) {
            c_sub_acc += A[gid_y * lda + k] * B[k * ldb + gid_x];
        }
        C[gid_y * ldc + gid_x] = c_sub_acc;
    }
}

#define BLOCK_SIDE_SIZE 16
__kernel void GemmDeviceV2(const int M, const int N, const int K,
                           __global const float *A, const int lda,
                           __global const float *B, const int ldb,
                           __global float *C, const int ldc) {

    __local float a_shared[BLOCK_SIDE_SIZE][BLOCK_SIDE_SIZE]; // cuda: __shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
    __local float b_shared[BLOCK_SIDE_SIZE][BLOCK_SIDE_SIZE]; // cuda: __shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];

    for (int gid_x = get_global_id(0), gid_y = get_global_id(1);
        gid_x < N && gid_y < M; 
        gid_x += get_global_size(0), gid_y += get_global_size(1)) {

        int tid_x = get_local_id(0);
        int tid_y = get_local_id(1);

        float c_sub_acc = 0;
        // For blocks in grid.
        for (int bk = 0; bk < K / BLOCK_SIDE_SIZE; bk++) {
            a_shared[tid_y][tid_x] = A[gid_y * lda + (bk * BLOCK_SIDE_SIZE + tid_x)];
            b_shared[tid_y][tid_x] = B[(bk * BLOCK_SIDE_SIZE + tid_y) * ldb + gid_x];
            // Wait for data to complete loading to Shared memory.
            barrier(CLK_LOCAL_MEM_FENCE); // cuda: __syncthreads()

            // For elements in a block.
            for (int k = 0; k < BLOCK_SIDE_SIZE; k++) {
                c_sub_acc += a_shared[tid_y][k] * b_shared[k][tid_x];
            }
            // To prevent the case from happening:
            // The next round of data is loaded when the data in share memory is not used up.
            barrier(CLK_LOCAL_MEM_FENCE); // cuda: __syncthreads()
        }

        C[gid_y * ldc + gid_x] += c_sub_acc;
    }
}

// 
// 一个线程处理2*2个元素，对应使用原来4倍的local memory
// local size 不变，global size 缩减到1/4，则总线程数减少到1/4。
__kernel void GemmDeviceV3(const int M, const int N, const int K,
                           __global const float *A, const int lda,
                           __global const float *B, const int ldb,
                           __global float *C, const int ldc) {
    const int STEP = 2;
    float sub_sum[STEP][STEP];
    __local float a_shared[BLOCK_SIDE_SIZE*STEP][BLOCK_SIDE_SIZE*STEP];
    __local float b_shared[BLOCK_SIDE_SIZE*STEP][BLOCK_SIDE_SIZE*STEP];

    for (int gid_x = get_global_id(0), gid_y = get_global_id(1);
        gid_x < N && gid_y < M; 
        gid_x += get_global_size(0), gid_y += get_global_size(1)) {

        int tid_x = get_local_id(0);
        int tid_y = get_local_id(1);

        sub_sum[0][0] = 0;
        sub_sum[0][1] = 0;
        sub_sum[1][0] = 0;
        sub_sum[1][1] = 0;

        // For blocks in grid.
        for (int bk = 0; bk < K / (BLOCK_SIDE_SIZE*STEP); bk++) {
            for (int si = 0; si < STEP; si++) {
                for (int sj = 0; sj < STEP; sj++) {
                    // 0->01, 1->23 => 0*2+0/0*2+1, 1*2+0/1*2+1
                    a_shared[tid_y*STEP+si][tid_x*STEP+sj] = A[(gid_y*STEP+si) * lda + (bk * BLOCK_SIDE_SIZE*STEP + tid_x*STEP+sj)];
                    b_shared[tid_y*STEP+si][tid_x*STEP+sj] = B[(bk * BLOCK_SIDE_SIZE*STEP + (tid_y*STEP+si)) * ldb + gid_x*STEP+sj];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE); 
   
            // For elements in a block.
            for (int k = 0; k < BLOCK_SIDE_SIZE*STEP; k++) {
                for (int si = 0; si < STEP; si++) {
                    for (int sj = 0; sj < STEP; sj++) {
                        sub_sum[si][sj] += a_shared[tid_y*STEP+si][k] * b_shared[k][tid_x*STEP+sj];
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE); 
        }

        C[(gid_y*STEP+0) * ldc + gid_x*STEP+0] += sub_sum[0][0];
        C[(gid_y*STEP+0) * ldc + gid_x*STEP+1] += sub_sum[0][1];
        C[(gid_y*STEP+1) * ldc + gid_x*STEP+0] += sub_sum[1][0];
        C[(gid_y*STEP+1) * ldc + gid_x*STEP+1] += sub_sum[1][1];
    }
}
