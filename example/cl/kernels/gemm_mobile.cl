// TODO: split-k 和 向量化类型，如float4


// GEMM 矩阵乘法例子
// 实现平台：

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

// V1 初始版本 同 用于独显的 GemmDeviceV1
__kernel void GemmMobileDeviceV1(const int M, const int N, const int K,
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

// V2 一个线程处理16个点.
// V1中的全局内存的计算访存比是1:2，最内层循环线程要读两次全局内存，然后计算一次乘加指令fma
// v2中读8次，计算4*4次fma，计算访存比是2:1
__kernel void GemmMobileDeviceV2(const int M, const int N, const int K,
                           __global const float *A, const int lda,
                           __global const float *B, const int ldb,
                           __global float *C, const int ldc) {
    
    const int STEP = 4;
    float c_sub_acc[STEP][STEP] = {0};
    for (int gid_sx = get_global_id(0)*STEP, gid_sy = get_global_id(1)*STEP;
        gid_sx < N && gid_sy < M; 
        gid_sx += get_global_size(0)*STEP, gid_sy += get_global_size(1)*STEP) {

        for (int k = 0; k < K; k++) {
            for (int si = 0; si < STEP; si++) {
                for (int sj = 0; sj < STEP; sj++) {
                    c_sub_acc[si][sj] += A[(gid_sy+si) * lda + k] * B[k * ldb + gid_sx + sj];
                }
            }

            // // 分析计算访存比, 读8次，计算4*4次fma，计算访存比是2:1
            // float Asi[STEP], Bsj[STEP];
            // Asi[0] = A[(gid_sy+0) * lda + k];
            // Asi[1] = A[(gid_sy+1) * lda + k];
            // Asi[2] = A[(gid_sy+2) * lda + k];
            // Asi[3] = A[(gid_sy+3) * lda + k];
            // Bsj[0] = B[k * ldb + gid_sx + 0];
            // Bsj[1] = B[k * ldb + gid_sx + 1];
            // Bsj[2] = B[k * ldb + gid_sx + 2];
            // Bsj[3] = B[k * ldb + gid_sx + 3];
            // for (int si = 0; si < STEP; si++) {
            //     for (int sj = 0; sj < STEP; sj++) {
            //         c_sub_acc[si][sj] += Asi[si] * Bsj[sj];
            //     }
            // }
        }

        for (int si=0; si<STEP; si++) {
            for (int sj=0; sj<STEP; sj++) {
                C[(gid_sy+si) * ldc + gid_sx+sj] += c_sub_acc[si][sj];
            }
        }
    }
}

// v3_0 向量化过渡版本0, 基于v2 (注释部分), A和B矩阵 使用向量化数据类型改写
__kernel void GemmMobileDeviceV3_0(const int M, const int N, const int K,
                           __global const float *A, const int lda,
                           __global const float *B, const int ldb,
                           __global float *C, const int ldc) {
    
    const int STEP = 4;
    float c_sub_acc[STEP][STEP] = {0};
    for (int gid_sx = get_global_id(0)*STEP, gid_sy = get_global_id(1)*STEP;
        gid_sx < N && gid_sy < M; 
        gid_sx += get_global_size(0)*STEP, gid_sy += get_global_size(1)*STEP) {

        for (int k = 0; k < K; k++) {
            float4 Asi, Bsj;
            Asi.s0 = A[(gid_sy+0) * lda + k];
            Asi.s1 = A[(gid_sy+1) * lda + k];
            Asi.s2 = A[(gid_sy+2) * lda + k];
            Asi.s3 = A[(gid_sy+3) * lda + k];
            Bsj = vload4(0, B + k * ldb + gid_sx);

            c_sub_acc[0][0] += Asi.s0 * Bsj.s0;
            c_sub_acc[0][1] += Asi.s0 * Bsj.s1;
            c_sub_acc[0][2] += Asi.s0 * Bsj.s2;
            c_sub_acc[0][3] += Asi.s0 * Bsj.s3;

            c_sub_acc[1][0] += Asi.s1 * Bsj.s0;
            c_sub_acc[1][1] += Asi.s1 * Bsj.s1;
            c_sub_acc[1][2] += Asi.s1 * Bsj.s2;
            c_sub_acc[1][3] += Asi.s1 * Bsj.s3;

            c_sub_acc[2][0] += Asi.s2 * Bsj.s0;
            c_sub_acc[2][1] += Asi.s2 * Bsj.s1;
            c_sub_acc[2][2] += Asi.s2 * Bsj.s2;
            c_sub_acc[2][3] += Asi.s2 * Bsj.s3;

            c_sub_acc[3][0] += Asi.s3 * Bsj.s0;
            c_sub_acc[3][1] += Asi.s3 * Bsj.s1;
            c_sub_acc[3][2] += Asi.s3 * Bsj.s2;
            c_sub_acc[3][3] += Asi.s3 * Bsj.s3;
        }

        for (int si=0; si<STEP; si++) {
            for (int sj=0; sj<STEP; sj++) {
                C[(gid_sy+si) * ldc + gid_sx+sj] += c_sub_acc[si][sj];
            }
        }
    }
}

// v3_1 基于v3_0, c_sub_acc也使用向量化数据类型改写
__kernel void GemmMobileDeviceV3_1(const int M, const int N, const int K,
                           __global const float *A, const int lda,
                           __global const float *B, const int ldb,
                           __global float *C, const int ldc) {
    
    const int STEP = 4;
    float4 acc[STEP] = {(float4)0, (float4)0, (float4)0, (float4)0};
    for (int gid_sx = get_global_id(0)*STEP, gid_sy = get_global_id(1)*STEP;
        gid_sx < N && gid_sy < M; 
        gid_sx += get_global_size(0)*STEP, gid_sy += get_global_size(1)*STEP) {

        for (int k = 0; k < K; k++) {
            float4 Asi, Bsj;
            Asi.s0 = A[(gid_sy+0) * lda + k];
            Asi.s1 = A[(gid_sy+1) * lda + k];
            Asi.s2 = A[(gid_sy+2) * lda + k];
            Asi.s3 = A[(gid_sy+3) * lda + k];
            Bsj = vload4(0, B + k * ldb + gid_sx);

            acc[0] += Asi.s0 * Bsj;
            acc[1] += Asi.s1 * Bsj;
            acc[2] += Asi.s2 * Bsj;
            acc[3] += Asi.s3 * Bsj;
        }

        vstore4(acc[0], 0, C + (gid_sy+0) * ldc + gid_sx);
        vstore4(acc[1], 0, C + (gid_sy+1) * ldc + gid_sx);
        vstore4(acc[2], 0, C + (gid_sy+2) * ldc + gid_sx);
        vstore4(acc[3], 0, C + (gid_sy+3) * ldc + gid_sx);
    }
}

// v3_2 基于v3_1, 将A矩阵在外部进行转置, 使A的单线程读取满足向量化加载(优化思路同cpu的缓存命中率)
// note: 独显的全局内存合并访问优化指的是warp内线程访问的地址需要连续,与线程一一对应,
//       一个线程访问一个数据.且起始地址是每个线程所存取的大小的16倍.
__kernel void GemmMobileDeviceV3_2(const int M, const int N, const int K,
                           __global const float *A, const int lda,
                           __global const float *B, const int ldb,
                           __global float *C, const int ldc) {
    
    const int STEP = 4;
    float4 acc[STEP] = {(float4)0, (float4)0, (float4)0, (float4)0};
    for (int gid_sx = get_global_id(0)*STEP, gid_sy = get_global_id(1)*STEP;
        gid_sx < N && gid_sy < M; 
        gid_sx += get_global_size(0)*STEP, gid_sy += get_global_size(1)*STEP) {

        for (int k = 0; k < K; k++) {
            float4 Asi, Bsj;
            Asi = vload4(0, A + k * lda + gid_sy);
            Bsj = vload4(0, B + k * ldb + gid_sx);

            acc[0] += Asi.s0 * Bsj;
            acc[1] += Asi.s1 * Bsj;
            acc[2] += Asi.s2 * Bsj;
            acc[3] += Asi.s3 * Bsj;
        }

        vstore4(acc[0], 0, C + (gid_sy+0) * ldc + gid_sx);
        vstore4(acc[1], 0, C + (gid_sy+1) * ldc + gid_sx);
        vstore4(acc[2], 0, C + (gid_sy+2) * ldc + gid_sx);
        vstore4(acc[3], 0, C + (gid_sy+3) * ldc + gid_sx);
    }
}

// v4 使用纹理内存
__kernel void GemmMobileDeviceV4(const int M, const int N, const int K,
                           __global const float *A, const int lda,
                           __global const float *B, const int ldb,
                           __global float *C, const int ldc) {
    
    const int STEP = 4;
    float4 acc[STEP] = {(float4)0, (float4)0, (float4)0, (float4)0};
    for (int gid_sx = get_global_id(0)*STEP, gid_sy = get_global_id(1)*STEP;
        gid_sx < N && gid_sy < M; 
        gid_sx += get_global_size(0)*STEP, gid_sy += get_global_size(1)*STEP) {

        for (int k = 0; k < K; k++) {
            float4 Asi, Bsj;
            Asi = vload4(0, A + k * lda + gid_sy);
            Bsj = vload4(0, B + k * ldb + gid_sx);

            acc[0] += Asi.s0 * Bsj;
            acc[1] += Asi.s1 * Bsj;
            acc[2] += Asi.s2 * Bsj;
            acc[3] += Asi.s3 * Bsj;
        }

        vstore4(acc[0], 0, C + (gid_sy+0) * ldc + gid_sx);
        vstore4(acc[1], 0, C + (gid_sy+1) * ldc + gid_sx);
        vstore4(acc[2], 0, C + (gid_sy+2) * ldc + gid_sx);
        vstore4(acc[3], 0, C + (gid_sy+3) * ldc + gid_sx);
    }
}
