#ifndef POCKET_AI_EVAL_CUDA_BANDWIDTH_EVAL_HPP_
#define POCKET_AI_EVAL_CUDA_BANDWIDTH_EVAL_HPP_

#include <cuda.h>
#include <cuda_runtime.h>


#include <cstring>
#include <cassert>
#include <iostream>
#include <memory>

#include "pocket-ai/engine/cu/common.hpp"

class BandwidthEval {

    // CUDART_VERSION >= 2020
    enum MemcpyKind { DEVICE_TO_HOST, HOST_TO_DEVICE, DEVICE_TO_DEVICE };
    enum MemoryMode { PINNED, PAGEABLE };

public:
    BandwidthEval() {
        flush_size_ = 256 * 1024 * 1024; // CPU cache flush
        cache_clear_size_ = 16 * (1e6);  // 16 M
        default_increment_ = 4 * (1e6);  // 4 M
        default_size_ = 32 * (1e6);      // 32 M
        memory_iterations_ = 100;
    }

    void Run(int device_id, bool is_need_full_print = false) {
        printf("  BandwidthEval: \n");
        flush_buf_ = (char *)malloc(flush_size_);
        is_need_full_print_ = is_need_full_print;
        
        CheckDevice(device_id);
        // HOST_TO_DEVICE
        Dispatch(device_id, HOST_TO_DEVICE, PINNED, false);
        Dispatch(device_id, HOST_TO_DEVICE, PINNED, true);
        Dispatch(device_id, HOST_TO_DEVICE, PAGEABLE, false);
        // DEVICE_TO_HOST
        Dispatch(device_id, DEVICE_TO_HOST, PINNED, false);
        Dispatch(device_id, DEVICE_TO_HOST, PINNED, true);
        Dispatch(device_id, DEVICE_TO_HOST, PAGEABLE, false);
        // DEVICE_TO_DEVICE
        Dispatch(device_id, DEVICE_TO_DEVICE, PAGEABLE, false);
    
        // Ensure that we reset all CUDA Devices in question
        cudaSetDevice(device_id);
    
        free(flush_buf_);
    }

private:
    int CheckDevice(int device_id) {
        cudaDeviceProp deviceProp;
        cudaError_t error_id = cudaGetDeviceProperties(&deviceProp, device_id);
        if (error_id == cudaSuccess) {
            // printf(" Device %d: %s\n", device_id, deviceProp.name);

            if (deviceProp.computeMode == cudaComputeModeProhibited) {
                fprintf(stderr,
                        "Error: device is running in <Compute Mode Prohibited>, no "
                        "threads can use ::cudaSetDevice().\n");
                CUDA_CHECK(cudaSetDevice(device_id));

                exit(EXIT_FAILURE);
            }
        } else {
            printf("cudaGetDeviceProperties returned %d\n-> %s\n", (int)error_id,
                    cudaGetErrorString(error_id));
            CUDA_CHECK(cudaSetDevice(device_id));

            exit(EXIT_FAILURE);
        }
        
        return 0;
    }

    void Dispatch(int device_id, MemcpyKind kind, MemoryMode mem_mode, bool is_write_combined) {
        // count the number of copies we're going to run
        unsigned int count = 1;
    
        double *bandwidths = (double *)malloc(count * sizeof(double));
        memset(bandwidths, 0, count * sizeof(double));
        
        unsigned int *mem_sizes = (unsigned int *)malloc(count * sizeof(unsigned int));

        cudaSetDevice(device_id);

        // run each of the copies
        for (unsigned int i = 0; i < count; i++) {
            mem_sizes[i] = default_size_ + i * default_increment_;

            switch (kind) {
                case DEVICE_TO_HOST:
                bandwidths[i] += TestDeviceToHostTransfer(mem_sizes[i], mem_mode, is_write_combined);
                break;

                case HOST_TO_DEVICE:
                bandwidths[i] += TestHostToDeviceTransfer(mem_sizes[i], mem_mode, is_write_combined);
                break;

                case DEVICE_TO_DEVICE:
                bandwidths[i] += TestDeviceToDeviceTransfer(mem_sizes[i]);
                break;
            }
        }
    
        // print results
        PrintResultsCSV(mem_sizes, bandwidths, count, kind, mem_mode, is_write_combined);
        // clean up
        free(mem_sizes);
        free(bandwidths);
    }
    float TestDeviceToHostTransfer(unsigned int mem_size, MemoryMode mem_mode, bool is_write_combined){
        float elapsed_time_ms = 0.0f;
        float bandwidthGBs = 0.0f;
        unsigned char *h_idata = NULL;
        unsigned char *h_odata = NULL;
    
        // allocate host memory
        if (PINNED == mem_mode) {
        // pinned memory mode - use special function to get OS-pinned memory
            CUDA_CHECK(cudaHostAlloc((void **)&h_idata, mem_size, (is_write_combined) ? cudaHostAllocWriteCombined : 0));
            CUDA_CHECK(cudaHostAlloc((void **)&h_odata, mem_size, (is_write_combined) ? cudaHostAllocWriteCombined : 0));
        } else {
            // pageable memory mode - use malloc
            h_idata = (unsigned char *)malloc(mem_size);
            h_odata = (unsigned char *)malloc(mem_size);
    
            if (h_idata == 0 || h_odata == 0) {
                fprintf(stderr, "Not enough memory avaialable on host to run test!\n");
                exit(EXIT_FAILURE);
            }
        }
    
        // initialize the memory
        for (unsigned int i = 0; i < mem_size / sizeof(unsigned char); i++) {
            h_idata[i] = (unsigned char)(i & 0xff);
        }
    
        // allocate device memory
        unsigned char *d_idata;
        CUDA_CHECK(cudaMalloc((void **)&d_idata, mem_size));
    
        // initialize the device memory
        CUDA_CHECK(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));
    
        // copy data from GPU to Host
        if (PINNED == mem_mode) {
            timer_.Start();
            for (unsigned int i = 0; i < memory_iterations_; i++) {
                CUDA_CHECK(cudaMemcpyAsync(h_odata, d_idata, mem_size, cudaMemcpyDeviceToHost, 0));
            }
            timer_.Stop();
            elapsed_time_ms = timer_.ElapsedMillis();
        } else {
            elapsed_time_ms = 0;
            for (unsigned int i = 0; i < memory_iterations_; i++) {
                timer_.Start();
                CUDA_CHECK(cudaMemcpy(h_odata, d_idata, mem_size, cudaMemcpyDeviceToHost));
                timer_.Stop();
                elapsed_time_ms += timer_.ElapsedMillis();
                memset(flush_buf_, i, flush_size_);
            }
        }
    
        // calculate bandwidth in GB/s
        double time_s = elapsed_time_ms / 1e3;
        bandwidthGBs = (mem_size * (float)memory_iterations_) / (double)1e9;
        bandwidthGBs = bandwidthGBs / time_s;
        // clean up memory
        if (PINNED == mem_mode) {
            CUDA_CHECK(cudaFreeHost(h_idata));
            CUDA_CHECK(cudaFreeHost(h_odata));
        } else {
            free(h_idata);
            free(h_odata);
        }
    
        CUDA_CHECK(cudaFree(d_idata));
    
        return bandwidthGBs;
    }
    float TestHostToDeviceTransfer(unsigned int mem_size, MemoryMode mem_mode, bool is_write_combined) {
        float elapsed_time_ms = 0.0f;
        float bandwidthGBs = 0.0f;
    
        // allocate host memory
        unsigned char *h_odata = NULL;
    
        if (PINNED == mem_mode) {
            // pinned memory mode - use special function to get OS-pinned memory
            CUDA_CHECK(cudaHostAlloc((void **)&h_odata, mem_size,
                                        (is_write_combined) ? cudaHostAllocWriteCombined : 0));
        } else {
            // pageable memory mode - use malloc
            h_odata = (unsigned char *)malloc(mem_size);
        }
    
        unsigned char *h_cacheClear1 = (unsigned char *)malloc(cache_clear_size_);
        unsigned char *h_cacheClear2 = (unsigned char *)malloc(cache_clear_size_);
    
        if (h_odata == 0 || h_cacheClear1 == 0 || h_cacheClear2 == 0) {
            fprintf(stderr, "Not enough memory available on host to run test!\n");
            exit(EXIT_FAILURE);
        }
    
        // initialize the memory
        for (unsigned int i = 0; i < mem_size / sizeof(unsigned char); i++) {
            h_odata[i] = (unsigned char)(i & 0xff);
        }
        for (unsigned int i = 0; i < cache_clear_size_ / sizeof(unsigned char); i++) {
            h_cacheClear1[i] = (unsigned char)(i & 0xff);
            h_cacheClear2[i] = (unsigned char)(0xff - (i & 0xff));
        }
    
        // allocate device memory
        unsigned char *d_idata;
        CUDA_CHECK(cudaMalloc((void **)&d_idata, mem_size));
    
        // copy host memory to device memory
        if (PINNED == mem_mode) {
            timer_.Start();
            for (unsigned int i = 0; i < memory_iterations_; i++) {
                CUDA_CHECK(cudaMemcpyAsync(d_idata, h_odata, mem_size, cudaMemcpyHostToDevice, 0));
            }
            timer_.Stop();
            elapsed_time_ms = timer_.ElapsedMillis();
        } else {
            elapsed_time_ms = 0;
            for (unsigned int i = 0; i < memory_iterations_; i++) {
                timer_.Start();
                CUDA_CHECK(cudaMemcpy(d_idata, h_odata, mem_size, cudaMemcpyHostToDevice));
                timer_.Stop();
                elapsed_time_ms += timer_.ElapsedMillis();
                memset(flush_buf_, i, flush_size_);
            }
        }
    
        // calculate bandwidth in GB/s
        double time_s = elapsed_time_ms / 1e3;
        bandwidthGBs = (mem_size * (float)memory_iterations_) / (double)1e9;
        bandwidthGBs = bandwidthGBs / time_s;
        // clean up memory
        if (PINNED == mem_mode) {
            CUDA_CHECK(cudaFreeHost(h_odata));
        } else {
            free(h_odata);
        }
    
        free(h_cacheClear1);
        free(h_cacheClear2);
        CUDA_CHECK(cudaFree(d_idata));
    
        return bandwidthGBs;
    }
    float TestDeviceToDeviceTransfer(unsigned int mem_size) {

        float elapsed_time_ms = 0.0f;
        float bandwidthGBs = 0.0f;
        // allocate host memory
        unsigned char *h_idata = (unsigned char *)malloc(mem_size);
        if (h_idata == 0) {
            fprintf(stderr, "Not enough memory avaialable on host to run test!\n");
            exit(EXIT_FAILURE);
        }
    
        // initialize the host memory
        for (unsigned int i = 0; i < mem_size / sizeof(unsigned char); i++) {
            h_idata[i] = (unsigned char)(i & 0xff);
        }
    
        // allocate device memory
        unsigned char *d_idata;
        CUDA_CHECK(cudaMalloc((void **)&d_idata, mem_size));
        unsigned char *d_odata;
        CUDA_CHECK(cudaMalloc((void **)&d_odata, mem_size));
    
        // initialize memory
        CUDA_CHECK(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));
    
        // run the memcopy
        timer_.Start();
        for (unsigned int i = 0; i < memory_iterations_; i++) {
            CUDA_CHECK(cudaMemcpy(d_odata, d_idata, mem_size, cudaMemcpyDeviceToDevice));
        }
        timer_.Stop();
        elapsed_time_ms = timer_.ElapsedMillis();
    
        // calculate bandwidth in GB/s
        double time_s = elapsed_time_ms / 1e3;
        bandwidthGBs = (2.0f * mem_size * (float)memory_iterations_) / (double)1e9;
        bandwidthGBs = bandwidthGBs / time_s;
    
        // clean up memory
        free(h_idata);
        CUDA_CHECK(cudaFree(d_idata));
        CUDA_CHECK(cudaFree(d_odata));
    
        return bandwidthGBs;
    }
    void PrintResultsCSV(unsigned int *mem_sizes, double *bandwidths, unsigned int count, 
                         MemcpyKind kind, MemoryMode mem_mode, bool is_write_combined) {
        std::string config;
        if (kind == DEVICE_TO_DEVICE) {
            config += "D2D                     ";
        } else {
            if (kind == DEVICE_TO_HOST) {
                config += "D2H";
            } 
            else if (kind == HOST_TO_DEVICE) {
                config += "H2D";
            }
    
            if (mem_mode == PAGEABLE) {
                config += "-Paged               ";
            } 
            else if (mem_mode == PINNED) {
                config += "-Pinned";
    
                if (is_write_combined)
                    config += "-WriteCombined";
                else
                    config += "              ";
            }
        }

        if (is_need_full_print_) {
            for (int i = 0; i < count; i++) {
                double secs = (double)mem_sizes[i] / (bandwidths[i] * (double)(1e9));
                printf("    %s, BandwidthEval = %.1f GB/s, Time = %.5f s, Size = %u bytes\n",
                        config.c_str(), bandwidths[i], secs, mem_sizes[i]);
            }            
        }
        else {
            for (int i = 0; i < count; i++) {
                if (config.find("-WriteCombined") != std::string::npos || config.find("D2D") != std::string::npos)
                    printf("    %s, BandwidthEval = %.1f GB/s\n", config.c_str(), bandwidths[i]);
            }
        }
    }

private:
    pai::cu::GpuTimer timer_;
    bool is_need_full_print_;
    char *flush_buf_;
    size_t flush_size_;

    size_t memory_iterations_;
    size_t default_size_;    
    size_t default_increment_;    
    size_t cache_clear_size_;
};

#endif // POCKET_AI_EVAL_CUDA_BANDWIDTH_EVAL_HPP_