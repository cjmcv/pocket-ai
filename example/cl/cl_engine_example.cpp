/*!
* \brief Vector dot product: h_result = SUM(A * B).
*/

#include "engine/cl/kernel.hpp"

using namespace ptk;

void DotProductHost(const int* src1, const int* src2, uint32_t len, int* dst) {
    *dst = 0;
    for (uint32_t i = 0; i < len; i++) {
        (*dst) += src1[i] * src2[i];
    }
}

#define USE_MAP_DATA

void SetParams4DotProduct(cl::KernelParams *params) {    
#ifdef USE_MAP_DATA
    params->io_attri = {
        {CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(cl_mem)},
        {CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(cl_mem)},
        {NULL,                                      sizeof(uint32_t)},
        {CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(cl_mem)}
    };
#else
    params->io_attri = {
        {CL_MEM_READ_ONLY,  sizeof(cl_mem)},
        {CL_MEM_READ_ONLY,  sizeof(cl_mem)},
        {NULL,              sizeof(uint32_t)},
        {CL_MEM_WRITE_ONLY, sizeof(cl_mem)}
    };
#endif
}

int main(int argc, char **argv) {
    cl_int err_code;
    size_t num_elements = 2560000; 
    // set and log Global and Local work size dimensions
    size_t local_work_size = 256;
    // 1D var for Total # of work items
    size_t global_work_size = cl::GetRoundUpMultiple(num_elements, local_work_size) * local_work_size;

    printf("Global Work Size = %zu, Local Work Size = %zu, # of Work Groups = %zu\n\n",
      global_work_size, local_work_size, global_work_size / local_work_size);

    // Allocate and initialize host arrays.
    int *h_src1 = (int *)malloc(sizeof(cl_int) * num_elements);
    int *h_src2 = (int *)malloc(sizeof(cl_int) * num_elements);
    int h_dst = 0;
    int h_dst4cl = 0;
    for (int i = 0; i < num_elements; i++) {
        h_src1[i] = 1;
        h_src2[i] = 2;
    }

    // Load CL source.
    cl::KernelLoader *loader = new cl::KernelLoader;
    loader->Load("dot_product.cl");

    // Get an OpenCL platform.
    cl::Platform platform;
    platform.GetInfos();

    {
        // Get the devices
        cl_device_id device;
        platform.GetDeviceId("Intel(R) OpenCL Graphics", &device); // "NVIDIA CUDA" / "Intel(R) OpenCL"

        // Create the context
        cl_context context = clCreateContext(0, 1, &device, NULL, NULL, &err_code);
        CL_CHECK(err_code);

        // Get Kernel.
        loader->CreateProgram(context);
        
        cl::Kernel *kernel = loader->CreateKernel("DotProductDevice", SetParams4DotProduct);

        std::vector<uint32_t> size;
        size.push_back(sizeof(cl_int) * num_elements);
        size.push_back(sizeof(cl_int) * num_elements);
        size.push_back(num_elements);
        size.push_back(sizeof(cl_int));
        kernel->CreateBuffer(context, size);

        //--------------------------------------------------------
        // Create a command-queue
        cl_command_queue command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err_code);
        //cl_command_queue command_queue = clCreateCommandQueue(context, device, 0, &err_code);
        CL_CHECK(err_code);

    #ifdef USE_MAP_DATA
        cl_int *h_src1_map = (cl_int *)kernel->MapBUffer(command_queue, CL_TRUE, 0);
        memcpy(h_src1_map, h_src1, sizeof(cl_int) * num_elements);
        cl_int *h_src2_map = (cl_int *)kernel->MapBUffer(command_queue, CL_TRUE, 1);
        memcpy(h_src2_map, h_src2, sizeof(cl_int) * num_elements);

        kernel->UnmapBuffer(command_queue, 0);
        kernel->UnmapBuffer(command_queue, 1);
    #else
        // Asynchronous write of data to GPU device
        kernel->WriteBuffer(command_queue, CL_FALSE, h_src1, 0);
        kernel->WriteBuffer(command_queue, CL_FALSE, h_src2, 1);
    #endif

        // Launch kernel
        cl_event ev;
        CL_CHECK(clEnqueueNDRangeKernel(command_queue, kernel->kernel(), 1, NULL, &global_work_size, &local_work_size, 0, NULL, &ev));

        // Synchronous/blocking read of results, and check accumulated errors
    #ifdef USE_MAP_DATA
        cl_int *h_dst4cl = (cl_int *)kernel->MapBUffer(command_queue, CL_TRUE, 3);
    #else
        kernel->ReadBuffer(command_queue, CL_TRUE, &h_dst4cl, 3);
    #endif

        // Block until all tasks in command_queue have been completed.
        clFinish(command_queue);

        // Gets the running time of the kernel function.
        cl::PrintCommandElapsedTime(ev);
        //--------------------------------------------------------

        // Compute and compare results on host.
        DotProductHost(h_src1, h_src2, num_elements, &h_dst);
    #ifdef USE_MAP_DATA
        printf("Test: %s (%d, %d)\n \n", (*h_dst4cl == h_dst ? "PASS" : "FAILED"), *h_dst4cl, h_dst);
        kernel->UnmapBuffer(command_queue, 3);
    #else
        printf("Test: %s (%d, %d)\n \n", (h_dst4cl == h_dst ? "PASS" : "FAILED"), h_dst4cl, h_dst);
    #endif

        // Cleanup
        if (ev) CL_CHECK(clReleaseEvent(ev));
        if (command_queue) CL_CHECK(clReleaseCommandQueue(command_queue));
        if (context) CL_CHECK(clReleaseContext(context));
        kernel->ReleaseBuffer();
    }

    if (loader) {
        loader->UnLoad();
        delete loader;
    }
    if (h_src1) free(h_src1);
    if (h_src2) free(h_src2);

    return 0;
}
