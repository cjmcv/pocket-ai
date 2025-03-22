
#ifndef POCKET_AI_EVAL_CUDA_MAX_FLOPS_HPP_
#define POCKET_AI_EVAL_CUDA_MAX_FLOPS_HPP_

#include <stdio.h>   
#include <stdlib.h> 
#include <cuda_runtime.h>

#include "pocket-ai/engine/cu/common.hpp"

__global__ void MaxFlopsFp32CudaCore(uint32_t *start_clock, uint32_t *stop_clock, float *data1, float *data2, int repeat_times, float *res) {
	int gid = blockIdx.x*blockDim.x + threadIdx.x;
	register float s1 = data1[gid];
	register float s2 = data2[gid];

	register float result2 = 0;
	register float result3 = 0;
	register float result4 = 0;
	register float result5 = 0;
	// synchronize all threads
	asm volatile ("bar.sync 0;");

	// start timing
	uint32_t start = 0;
	asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");

	for (int j=0 ; j<repeat_times ; ++j) {
		asm volatile ("{\t\n"
			"fma.rn.f32 %2, %0, %1, %2;\n\t"
			"fma.rn.f32 %3, %0, %1, %3;\n\t"
			"fma.rn.f32 %4, %0, %1, %4;\n\t"
			"fma.rn.f32 %5, %0, %1, %5;\n\t"
			"}" : "+f"(s1),"+f"(s2), "+f"(result2), "+f"(result3), "+f"(result4), "+f"(result5)
		);
	}
	// synchronize all threads
	asm volatile("bar.sync 0;");

	// stop timing
	uint32_t stop = 0;
	asm volatile("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");

	// write time and data back to memory
	start_clock[gid] = start;
	stop_clock[gid] = stop;
	res[gid] = result2 + result3 + result4 + result5;
}

class MaxFlopsEval {
public:
	void RunFp32(int sm_count, int clock_rate) {
		const int block_num = 1;
		const int threads_per_block = 1024;
		const int data_len = block_num * threads_per_block;
		const int data_size = data_len * sizeof(float);
		const int repeat_times = 1024;

		uint32_t *start_clock = (uint32_t*) malloc(data_len*sizeof(uint32_t));
		uint32_t *stop_clock = (uint32_t*) malloc(data_len*sizeof(uint32_t));
		float *data1 = (float*) malloc(data_size);
		float *data2 = (float*) malloc(data_size);
		float *res = (float*) malloc(data_size);

		uint32_t *start_clock_d;
		uint32_t *stop_clock_d;
		float *data1_d;
		float *data2_d;
		float *res_d;

		for (uint32_t i=0; i<data_len; i++) {
			data1[i] = (float)i;
			data2[i] = (float)i;
		}

		CUDA_CHECK( cudaMalloc(&start_clock_d, data_len*sizeof(uint32_t)) );
		CUDA_CHECK( cudaMalloc(&stop_clock_d, data_len*sizeof(uint32_t)) );
		CUDA_CHECK( cudaMalloc(&data1_d, data_size) );
		CUDA_CHECK( cudaMalloc(&data2_d, data_size) );
		CUDA_CHECK( cudaMalloc(&res_d, data_size) );

		CUDA_CHECK( cudaMemcpy(data1_d, data1, data_size, cudaMemcpyHostToDevice) );
		CUDA_CHECK( cudaMemcpy(data2_d, data2, data_size, cudaMemcpyHostToDevice) );
		MaxFlopsFp32CudaCore<<<block_num, threads_per_block>>> (start_clock_d, stop_clock_d, data1_d, data2_d, repeat_times, res_d);
		CUDA_CHECK( cudaPeekAtLastError() );

		CUDA_CHECK( cudaMemcpy(start_clock, start_clock_d, data_len*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
		CUDA_CHECK( cudaMemcpy(stop_clock, stop_clock_d, data_len*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
		CUDA_CHECK( cudaMemcpy(res, res_d, data_size, cudaMemcpyDeviceToHost) );

		// one block for one sm.
		float flops = (float)(repeat_times*threads_per_block*8) / ((float)(stop_clock[0]-start_clock[0]));
		printf("  fp32 flop: %f (flop/clockk/sm) * %d (sm) * %d (KHz) = %f TFLOPS\n", flops, sm_count, clock_rate / 1024, sm_count * clock_rate * flops / 1024/1024/1024);
	}
};

#endif // POCKET_AI_EVAL_CUDA_MAX_FLOPS_HPP_