
//    K     N        N
// M     K     =  M

//// CPU版本
// for (i = 0; i < M; ++i) {
//     for (j = 0; j < N; ++j) {
//         for (k = 0; k < K; ++k) {
//             C[i*ldc + j] += A[i*lda + k] * B[k*ldb + j];
//         }
//     }
// }

// V1 从全局内存中读取所需要计算的数据到寄存器做计算。直接基于CPU版本，将i和j循环去掉。
// Kernel代码编写时省略了i和j循环，但实际i和j循环仍然是存在的，
// 因为当矩阵规模相对于硬件资源来说够大时，SM数量有限且SM内活跃warp数量也有限
// （Active Warp 数量，取决于 block 使用的资源数量），导致gpu无法一次将所有数据都读取到。
// 所以i和j两层循环中也是有一定的先后顺序的。分析访存时，可以直接弱化为三层循环进行分析。
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

// v2
// v1中，将数据从全局内部中搬到寄存器中进行计算，因为本质有三层循环，参考CPU版本，所以AB矩阵的数据会重复读取。
// 则可以将global memory 的数据一次性加载到 local memory，每个数据从全局内存读取，相当与CPU的两层循环。
// 随后计算所需数据从局部内存中读取到寄存器进行计算，
// 即将三层循环重复读取全局内存，改为两层循环一次读取全局内存和三层循环重复读取局部内存。
// 大大节省从global memory读取的次数。
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
        for (int bk = 0; bk < K; bk += BLOCK_SIDE_SIZE) {
            a_shared[tid_y][tid_x] = A[gid_y * lda + (bk + tid_x)];
            b_shared[tid_y][tid_x] = B[(bk + tid_y) * ldb + gid_x];
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

// v3 
// v2中，使用局部内存来替代全局内存的多次访问，减少全局内存的访问次数，起到加速所用。
// 但因为实际计算需要先将数据读取到寄存器，虽然局部内存访问比全局内存快不少，也仍然存在不短的耗时。
// 看内层循环可知，一次乘法运算搭配两次局部内存的读取，即计算指令占1/3，而导致访存延迟无法被隐藏。
// 
// 因为按三层循环从局部内存中读取数据，则可以使用多个寄存器作为下一层级内存。
// 即 v1 三层全局内存 -> v2 两层全局内存+三层局部内存 -> v3 两层全局内存+三层/step的局部内存+三层step寄存器（寄存器可忽略）
// 则 操作方式是将数据从局部内存读取时，本来1个线程读取K个数据，变成1个线程读取step*K个数据以及step*step的子矩阵乘法。
//
// 令一个线程处理2*2个元素，对应使用原来4倍的local memory
// local size 不变，global size 缩减到1/4，则总线程数减少到1/4
//
// note: 1个线程读取K个数据 和 1个线程读取step*k个数据 性能上无太大差异，因为数据多时前者开的线程多，则仍需要轮询，不会同一时刻全部处理完。
//       但是step*step的寄存器上矩阵乘法, 每个元素会被使用STEP次，相当于局部内存的访存少了STEP倍。
//       如step为2，则A和B矩阵从局部内存到寄存器的访存次数分别是2次，共4次，而计算也是2*2次，则计算访存比为1比1.
//       step为4，则访存4+4次，计算4*4=16次，计算访存比为16/8
__kernel void GemmDeviceV3(const int M, const int N, const int K,
                           __global const float *A, const int lda,
                           __global const float *B, const int ldb,
                           __global float *C, const int ldc) {
    const int STEP = 2;
    float a_reg[STEP] = {0};
    float b_reg[STEP] = {0};    
    float sub_sum[STEP][STEP] = {{0}};
    __local float a_shared[BLOCK_SIDE_SIZE*STEP][BLOCK_SIDE_SIZE*STEP];
    __local float b_shared[BLOCK_SIDE_SIZE*STEP][BLOCK_SIDE_SIZE*STEP];

    for (int gid_x = get_global_id(0), gid_y = get_global_id(1);
        gid_x < N && gid_y < M; 
        gid_x += get_global_size(0), gid_y += get_global_size(1)) {

        int tid_x = get_local_id(0);
        int tid_y = get_local_id(1);

        // For blocks in grid.
        for (int bk = 0; bk < K; bk += BLOCK_SIDE_SIZE*STEP) {
            for (int si = 0; si < STEP; si++) {
                for (int sj = 0; sj < STEP; sj++) {
                    // 0->01, 1->23 => 0*2+0/0*2+1, 1*2+0/1*2+1
                    a_shared[tid_y*STEP+si][tid_x*STEP+sj] = A[(gid_y*STEP+si) * lda + (bk + tid_x*STEP+sj)];
                    b_shared[tid_y*STEP+si][tid_x*STEP+sj] = B[(bk + (tid_y*STEP+si)) * ldb + gid_x*STEP+sj];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE); 
   
            // For elements in a block.
            for (int k = 0; k < BLOCK_SIDE_SIZE*STEP; k++) {
                // for (int si = 0; si < STEP; si++) {
                //     for (int sj = 0; sj < STEP; sj++) {
                //         sub_sum[si][sj] += a_shared[tid_y*STEP+si][k] * b_shared[k][tid_x*STEP+sj];
                //     }
                // }

                for (int si=0; si < STEP; si++) {
                    a_reg[si] = a_shared[tid_y*STEP+si][k];
                    b_reg[si] = b_shared[k][tid_x*STEP+si];
                }
                // Both a_reg[si] and b_reg[sj] have been used STEP times.
                for (int si = 0; si < STEP; si++) {
                    for (int sj = 0; sj < STEP; sj++) {
                        sub_sum[si][sj] += a_reg[si] * b_reg[sj]; // a_shared[tid_y*STEP+si][k] * b_shared[k][tid_x*STEP+sj];
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE); 
        }

        for (int i=0; i<STEP; i++) {
            for (int j=0; j<STEP; j++) {
                C[(gid_y*STEP+i) * ldc + gid_x*STEP+j] += sub_sum[i][j];
            }
        }
    }
}

// v4
// 基于v3，进一步扩大STEP为4，则计算访存比为(4*4)/(4+4)=16/8
// 但是测在核显中耗时略高于v3，猜测是寄存器数量不足所致。
__kernel void GemmDeviceV4(const int M, const int N, const int K,
                           __global const float *A, const int lda,
                           __global const float *B, const int ldb,
                           __global float *C, const int ldc) {

    const int STEP = 4;
    float a_reg[STEP] = {0};
    float b_reg[STEP] = {0};    
    float sub_sum[STEP][STEP] = {{0}};
    __local float a_shared[BLOCK_SIDE_SIZE*STEP][BLOCK_SIDE_SIZE*STEP];
    __local float b_shared[BLOCK_SIDE_SIZE*STEP][BLOCK_SIDE_SIZE*STEP];

    for (int gid_x = get_global_id(0), gid_y = get_global_id(1);
        gid_x < N && gid_y < M; 
        gid_x += get_global_size(0), gid_y += get_global_size(1)) {

        int tid_x = get_local_id(0);
        int tid_y = get_local_id(1);

        // For blocks in grid.
        for (int bk = 0; bk < K; bk += BLOCK_SIDE_SIZE*STEP) {
            for (int si = 0; si < STEP; si++) {
                for (int sj = 0; sj < STEP; sj++) {
                    // 0->01, 1->23 => 0*2+0/0*2+1, 1*2+0/1*2+1
                    a_shared[tid_y*STEP+si][tid_x*STEP+sj] = A[(gid_y*STEP+si) * lda + (bk + tid_x*STEP+sj)];
                    b_shared[tid_y*STEP+si][tid_x*STEP+sj] = B[(bk + (tid_y*STEP+si)) * ldb + gid_x*STEP+sj];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE); 
   
            // For elements in a block.
            for (int k = 0; k < BLOCK_SIDE_SIZE*STEP; k++) {
                for (int si=0; si < STEP; si++) {
                    a_reg[si] = a_shared[tid_y*STEP+si][k];
                    b_reg[si] = b_shared[k][tid_x*STEP+si];
                }
                // Both a_reg[si] and b_reg[sj] have been used STEP times.
                for (int si = 0; si < STEP; si++) {
                    for (int sj = 0; sj < STEP; sj++) {
                        sub_sum[si][sj] += a_reg[si] * b_reg[sj]; // a_shared[tid_y*STEP+si][k] * b_shared[k][tid_x*STEP+sj];
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE); 
        }

        for (int i=0; i<STEP; i++) {
            for (int j=0; j<STEP; j++) {
                C[(gid_y*STEP+i) * ldc + gid_x*STEP+j] += sub_sum[i][j];
            }
        }
    }
}

// https://zhuanlan.zhihu.com/p/657632577
// v5 
// 基于v3，在v3中数据从全局内存->局部内存->寄存器后才开始计算，多线程与单线程相似，
// 假设硬件资源限制只有5个线程，会同时处理0-4号数据，5-10号数据需要等0-4号的某些数据处理完了之后，才有空闲去执行新的数据。
// 就会存在如下情况，计算需要依赖读的结果，而写又依赖读的结果，造成了不必要的耗时等待。
// 读 ->      -> 写，读  ->        ->  写，读  ->        -> 写
//       计算                计算                  计算
// 而计算和访存指令可同时执行，可使用double buffer来掩盖这个耗时，变成：
// 读0 ->  读1  ->  写0，读2  -> 写1，读3  ->  写2  -> 写3
//        计算0      计算1        计算2       计算3
// 第一次计算0需要依赖读0完了后才开始计算，但计算0时可以同时读1，在计算0完了后，读1也结束了。
//
// 具体实施方式：
// 1) 设置双份局部内存，如上流程，偶数用buffer0，奇数用buffer1。在buffer0中进行计算0的同时在buffer1中读1.
// 2）第一次数据加载在主循环之前，最后一次计算在主循环之后；
// 3) 由于计算和下一次访存使用的Shared Memory不同，因此主循环中每次循环只需要一次__syncthreads()即可.
// 4) 由于GPU不能向CPU那样支持乱序执行，主循环中需要先将下一次循环计算需要的Gloabal Memory中的数据load 到寄存器，
//    然后进行本次计算，之后再将load到寄存器中的数据写到Shared Memory，这样在LDG指令向Global Memory做load时，
//    不会影响后续FFMA及其它运算指令的 launch 执行，也就达到了Double Buffering的目的。
