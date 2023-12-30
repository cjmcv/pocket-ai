/*!
* \brief 
*/

#include "engine/cl/engine.hpp"

using namespace ptk;

void DotProductHost(const int* src1, const int* src2, uint32_t len, int* dst) {
    *dst = 0;
    for (uint32_t i = 0; i < len; i++) {
        (*dst) += src1[i] * src2[i];
    }
}

void SetParams4DotProduct(cl::KernelParams *params) {    
    params->io_attri = {
        {CL_MEM_READ_ONLY,  sizeof(cl_mem)},
        {CL_MEM_READ_ONLY,  sizeof(cl_mem)},
        {NULL,              sizeof(uint32_t)},
        {CL_MEM_WRITE_ONLY, sizeof(cl_mem)}
    };
}

void SetParams4Gemm(cl::KernelParams *params) {    
    params->io_attri = {
        {NULL,              sizeof(uint32_t)}, // 0
        {NULL,              sizeof(uint32_t)}, // 1
        {NULL,              sizeof(uint32_t)}, // 2
        {CL_MEM_READ_ONLY,  sizeof(cl_mem)},   // 3
        {NULL,              sizeof(uint32_t)}, // 4
        {CL_MEM_READ_ONLY,  sizeof(cl_mem)},   // 5
        {NULL,              sizeof(uint32_t)}, // 6
        {CL_MEM_WRITE_ONLY, sizeof(cl_mem)},   // 7
        {NULL,              sizeof(uint32_t)}  // 8
    };
}

int main(int argc, char **argv) {
    cl_int err_code;

    std::vector<std::tuple<std::string, std::string, cl::pSetParamsFuncs>> kernels_name;
    kernels_name.push_back(std::make_tuple("dot_product", "DotProductDevice", SetParams4DotProduct));
    kernels_name.push_back(std::make_tuple("gemm", "MatrixMulDeviceV1", SetParams4Gemm));
    kernels_name.push_back(std::make_tuple("gemm", "MatrixMulDeviceV2", SetParams4Gemm));

    cl::Engine engine;
    engine.Init("./kernels", kernels_name, 0);

    // DotProductCPU
    {
        size_t num_elements = 2560000; 
        // Allocate and initialize host arrays.
        int *h_src1 = (int *)malloc(sizeof(cl_int) * num_elements);
        int *h_src2 = (int *)malloc(sizeof(cl_int) * num_elements);
        int h_dst = 0;
        for (int i = 0; i < num_elements; i++) {
            h_src1[i] = 1;
            h_src2[i] = 2;
        }
        // Compute and compare results on host.
        DotProductHost(h_src1, h_src2, num_elements, &h_dst);
        printf("DotProductHost: %d \n", h_dst);

        if (h_src1) free(h_src1);
        if (h_src2) free(h_src2);
    }

    // DotProductDevice no Map
    {
        cl::Kernel *kernel = engine.GetKernel("DotProductDevice", false);

        size_t num_elements = 2560000; 
        size_t local_work_size = 256;
        size_t global_work_size = cl::GetRoundUpMultiple(num_elements, local_work_size) * local_work_size;
        // printf("Global Work Size = %zu, Local Work Size = %zu, # of Work Groups = %zu\n\n",
        // global_work_size, local_work_size, global_work_size / local_work_size);

        // Allocate and initialize host arrays.
        int *h_src1 = (int *)malloc(sizeof(cl_int) * num_elements);
        int *h_src2 = (int *)malloc(sizeof(cl_int) * num_elements);
        int h_dst = 0;
        int h_dst4cl = 0;
        for (int i = 0; i < num_elements; i++) {
            h_src1[i] = 1;
            h_src2[i] = 2;
        }

        std::vector<size_t> size;
        size.push_back(sizeof(cl_int) * num_elements);
        size.push_back(sizeof(cl_int) * num_elements);
        size.push_back(num_elements);
        size.push_back(sizeof(cl_int));
        kernel->CreateBuffer(size);

        // Asynchronous write of data to GPU device
        kernel->WriteBuffer(CL_FALSE, h_src1, 0);
        kernel->WriteBuffer(CL_FALSE, h_src2, 1);
        // Launch kernel
        cl_event ev;
        engine.AsyncRun(kernel, 1, &global_work_size, &local_work_size, &ev);
        // Synchronous/blocking read of results, and check accumulated errors
        kernel->ReadBuffer(CL_TRUE, &h_dst4cl, 3);
        // Block until all tasks in command_queue have been completed.
        engine.FinishQueue();
        // Gets the running time of the kernel function.
        cl::PrintCommandElapsedTime(ev);

        printf("DotProductDevice: %d \n", h_dst4cl);
        // Cleanup
        if (ev) CL_CHECK(clReleaseEvent(ev));
        kernel->ReleaseBuffer();    
        
        if (h_src1) free(h_src1);
        if (h_src2) free(h_src2);
    }

    {
        cl::Kernel *kernel = engine.GetKernel("DotProductDevice", true);

        size_t num_elements = 2560000; 
        size_t local_work_size = 256;
        size_t global_work_size = cl::GetRoundUpMultiple(num_elements, local_work_size) * local_work_size;

        std::vector<size_t> size = {
            sizeof(cl_int) * num_elements,
            sizeof(cl_int) * num_elements,
            num_elements, 
            sizeof(cl_int)
        };
        kernel->CreateBuffer(size);

        cl_int *h_src1_map = (cl_int *)kernel->MapBuffer(CL_TRUE, 0);
        cl_int *h_src2_map = (cl_int *)kernel->MapBuffer(CL_TRUE, 1);
        for (int i = 0; i < num_elements; i++) {
            h_src1_map[i] = 1;
            h_src2_map[i] = 2;
        }
        kernel->UnmapBuffer(0);
        kernel->UnmapBuffer(1);

        // Launch kernel
        cl_event ev;
        engine.AsyncRun(kernel, 1, &global_work_size, &local_work_size, &ev);
        // Block until all tasks in command_queue have been completed.
        engine.FinishQueue();
        // Gets the running time of the kernel function.
        cl::PrintCommandElapsedTime(ev);

        cl_int *h_dst4cl_map = (cl_int *)kernel->MapBuffer(CL_TRUE, 3);
        printf("DotProductDeviceMap: %d \n", *h_dst4cl_map);
        kernel->UnmapBuffer(3);

        // Cleanup
        if (ev) CL_CHECK(clReleaseEvent(ev));
        kernel->ReleaseBuffer();
    }

    {
        cl::Kernel *kernel = engine.GetKernel("MatrixMulDeviceV1", true);

        uint32_t height_a = 16, width_a = 32;
        uint32_t height_b = 32, width_b = 32;
          // set and log Global and Local work size dimensions
        size_t local_work_size[2] = {16, 16}; // x, y
        // 2D var for Total # of work items - (N = height_a, M = width_b)
        size_t global_work_size[2] = 
            { cl::GetRoundUpMultiple(width_b, local_work_size[0]) * local_work_size[0],
              cl::GetRoundUpMultiple(height_a, local_work_size[1]) * local_work_size[1] };

        std::vector<size_t> size = {
            height_a, width_b, width_a,                     // 0 1 2
            sizeof(cl_float) * height_a * width_a, width_a, // A 3 4
            sizeof(cl_float) * height_b * width_b, width_b, // B 5 6
            sizeof(cl_float) * height_a * width_b, width_b  // C 7 8
        };
        kernel->CreateBuffer(size);

        cl_float *hA_map = (cl_float *)kernel->MapBuffer(CL_TRUE, 3); // A
        cl_float *hB_map = (cl_float *)kernel->MapBuffer(CL_TRUE, 5); // B
        for (uint32_t i = 0; i < height_a * width_a; i++)
            hA_map[i] = 2;
        for (uint32_t i = 0; i < height_b * width_b; i++) 
            hB_map[i] = 2;
        kernel->UnmapBuffer(3);
        kernel->UnmapBuffer(5);

        // Launch kernel
        cl_event ev;
        engine.AsyncRun(kernel, 2, global_work_size, local_work_size, &ev);
        // Block until all tasks in command_queue have been completed.
        engine.FinishQueue();
        // Gets the running time of the kernel function.
        cl::PrintCommandElapsedTime(ev);

        cl_float *hC_map = (cl_float *)kernel->MapBuffer(CL_TRUE, 7);
        printf("Gemm C out: \n");
        for (uint32_t i=0; i<height_a; i++) {
            for (uint32_t j=0; j<width_b; j++) {
                printf("%f, ", hC_map[i*width_b+j]);
            }
            printf("\n");
        }
        kernel->UnmapBuffer(7);

        // Cleanup
        if (ev) CL_CHECK(clReleaseEvent(ev));
        kernel->ReleaseBuffer();
    }
    return 0;
}