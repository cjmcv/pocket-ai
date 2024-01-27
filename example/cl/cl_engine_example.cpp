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
        {0,                 sizeof(uint32_t)},
        {CL_MEM_WRITE_ONLY, sizeof(cl_mem)}
    };
}

void SetParams4Gemm(cl::KernelParams *params) {    
    params->io_attri = {
        {0,                 sizeof(uint32_t)}, // 0
        {0,                 sizeof(uint32_t)}, // 1
        {0,                 sizeof(uint32_t)}, // 2
        {CL_MEM_READ_ONLY,  sizeof(cl_mem)},   // 3
        {0,                 sizeof(uint32_t)}, // 4
        {CL_MEM_READ_ONLY,  sizeof(cl_mem)},   // 5
        {0,                 sizeof(uint32_t)}, // 6
        {CL_MEM_WRITE_ONLY, sizeof(cl_mem)},   // 7
        {0,                 sizeof(uint32_t)}  // 8
    };
}

void TestDotProductNoMap(cl::Engine *engine, std::string kernel_name) {

    cl::Kernel *kernel = engine->GetKernel("DotProductDevice", false);

    size_t num_elements = 2560000;
    size_t local_work_size = 256;
    size_t global_work_size = cl::GetRoundUpMultiple(num_elements, local_work_size) * local_work_size;
    // printf("Global Work Size = %zu, Local Work Size = %zu, # of Work Groups = %zu\n\n",
    // global_work_size, local_work_size, global_work_size / local_work_size);

    // Allocate and initialize host arrays.
    int *h_src1 = (int *)malloc(sizeof(cl_int) * num_elements);
    int *h_src2 = (int *)malloc(sizeof(cl_int) * num_elements);
    int h_dst4cl = 0;
    for (uint32_t i = 0; i < num_elements; i++) {
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
    engine->AsyncRun(kernel, 1, &global_work_size, &local_work_size, true);
    engine->FinishQueue();

    // Synchronous/blocking read of results, and check accumulated errors
    kernel->ReadBuffer(CL_TRUE, &h_dst4cl, 3);
    printf("DotProductDevice: %d \n", h_dst4cl);
    kernel->ReleaseBuffer();

    if (h_src1)
        free(h_src1);
    if (h_src2)
        free(h_src2);
}

void TestDotProduct(cl::Engine *engine, std::string kernel_name) {

    cl::Kernel *kernel = engine->GetKernel(kernel_name, true);

    size_t num_elements = 2560000;
    size_t local_work_size = 256;
    size_t global_work_size = cl::GetRoundUpMultiple(num_elements, local_work_size) * local_work_size;

    std::vector<size_t> size = {
        sizeof(cl_int) * num_elements,
        sizeof(cl_int) * num_elements,
        num_elements,
        sizeof(cl_int)};
    kernel->CreateBuffer(size);

    cl_int *h_src1_map = (cl_int *)kernel->MapBuffer(CL_TRUE, 0);
    cl_int *h_src2_map = (cl_int *)kernel->MapBuffer(CL_TRUE, 1);
    for (uint32_t i = 0; i < num_elements; i++)
    {
        h_src1_map[i] = 1;
        h_src2_map[i] = 2;
    }
    kernel->UnmapBuffer(0);
    kernel->UnmapBuffer(1);

    // Launch kernel
    engine->AsyncRun(kernel, 1, &global_work_size, &local_work_size, true);
    engine->FinishQueue();

    cl_int *h_dst4cl_map = (cl_int *)kernel->MapBuffer(CL_TRUE, 3);
    printf("DotProductDeviceMap: %d \n", *h_dst4cl_map);
    kernel->UnmapBuffer(3);
    kernel->ReleaseBuffer();
}

void TestGemm(cl::Engine *engine, std::string kernel_name, int step = 1) {
    printf(">>> %s.\n", kernel_name.c_str());

    cl::Kernel *kernel = engine->GetKernel(kernel_name, true);

    uint32_t height_a = 2560, width_a = 5120;
    uint32_t height_b = 5120, width_b = 3840;
    // set and log Global and Local work size dimensions
    size_t local_work_size[2] = {16, 16}; // x, y
    size_t global_work_size[2] =
        {cl::GetRoundUpMultiple(width_b/step, local_work_size[0]) * local_work_size[0],
         cl::GetRoundUpMultiple(height_a/step, local_work_size[1]) * local_work_size[1]};
    printf("global_work_size: (%zu, %zu).\n", global_work_size[0], global_work_size[1]);

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
        hA_map[i] = 1; // 1.23f+i%12;
    for (uint32_t i = 0; i < height_b * width_b; i++)
        hB_map[i] = 1; // 2.34f+i%22;
    kernel->UnmapBuffer(3);
    kernel->UnmapBuffer(5);

    // Launch kernel
    engine->AsyncRun(kernel, 2, global_work_size, local_work_size, true);
    engine->FinishQueue();

    cl_float *hC_map = (cl_float *)kernel->MapBuffer(CL_TRUE, 7);

    float mean = 0.f;
    for (uint32_t i = 0; i < height_a; i++) {
        // printf("\n");
        for (uint32_t j = 0; j < width_b; j++) {
            mean += hC_map[i * width_b + j];
            // printf("%f, ", hC_map[i * width_b + j]);
            // if (hC_map[i * width_b + j] != 3200)
                // printf("%f(%d, %d), ", hC_map[i * width_b + j], i, j);
        }
        
    }
    printf("GemmMap C out: %f. \n\n", mean / (height_a * width_b));
    kernel->UnmapBuffer(7);
    kernel->ReleaseBuffer();
}

int main(int argc, char **argv) {
    std::vector<std::tuple<std::string, std::string, cl::pSetParamsFuncs>> kernels_name;
    kernels_name.push_back(std::make_tuple("dot_product", "DotProductDevice", SetParams4DotProduct));
    kernels_name.push_back(std::make_tuple("gemm", "GemmDeviceV1", SetParams4Gemm));
    kernels_name.push_back(std::make_tuple("gemm", "GemmDeviceV2", SetParams4Gemm));
    kernels_name.push_back(std::make_tuple("gemm", "GemmDeviceV3", SetParams4Gemm));
    kernels_name.push_back(std::make_tuple("gemm", "GemmDeviceV4", SetParams4Gemm));
    // kernels_name.push_back(std::make_tuple("gemm", "GemmDeviceV5", SetParams4Gemm));

    cl::Engine engine;
    engine.Init("./kernels", kernels_name, 0);

    // DotProductCPU
    {
        size_t num_elements = 2560000; 
        // Allocate and initialize host arrays.
        int *h_src1 = (int *)malloc(sizeof(cl_int) * num_elements);
        int *h_src2 = (int *)malloc(sizeof(cl_int) * num_elements);
        int h_dst = 0;
        for (uint32_t i = 0; i < num_elements; i++) {
            h_src1[i] = 1;
            h_src2[i] = 2;
        }
        // Compute and compare results on host.
        DotProductHost(h_src1, h_src2, num_elements, &h_dst);
        printf("DotProductHost: %d \n", h_dst);

        if (h_src1) free(h_src1);
        if (h_src2) free(h_src2);
    }
    printf("\n##############################\n");
    TestDotProductNoMap(&engine, "DotProductDevice");
    TestDotProduct(&engine, "DotProductDevice");
    
    printf("\n##############################\n");
    TestGemm(&engine, "GemmDeviceV1");
    TestGemm(&engine, "GemmDeviceV2");
    TestGemm(&engine, "GemmDeviceV3", 2);
    TestGemm(&engine, "GemmDeviceV4", 4);
    // TestGemm(&engine, "GemmDeviceV5", 4);
    return 0;
}