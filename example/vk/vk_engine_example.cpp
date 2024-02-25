
#include <cmath>
#include "engine/vk/engine.hpp"
#include "prof/timer.hpp"

using namespace ptk;

void SetParamsEngineTest(vk::KernelParams *params) {
    params->buffer_type = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER // VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    };

    params->spec_constant = {
        {0, 32}, 
        {1, 2}, 
        {2, 2}, 
        {3, 160},
        {4, 320}};
    vk::SpecConstant c;
    c.id = 5;
    c.value.f32 = 640.123f;
    params->spec_constant.push_back(c);

    params->push_constant_num = 2; // 两个推送常量，初始化时vkCreatePipelineLayout中预设位置，在调用kernel时推送实际数据。
}

void SetParamsMatMulTiledFp32(vk::KernelParams *params) {
    params->buffer_type = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER // VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    };

    params->spec_constant = {
        {0, 16},  // 0,1,2: local_size_x = 16, local_size_y = 1, local_size_z = 1
        {1, 1}, 
        {2, 1},
        {3, 960},  // M
        {4, 640},  // N
        {5, 1280}, // k
    };
    params->push_constant_num = 0;
}

void SetParamsGemm(vk::KernelParams *params) {
    params->buffer_type = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER // VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    };

    params->spec_constant = {
        {0, 16},  // 0,1,2: local_size_x = 16, local_size_y = 1, local_size_z = 1
        {1, 1}, 
        {2, 1},
    };
    params->push_constant_num = 3;
}

void TestBaseKernel(vk::Engine *engine, std::string kernel_name) {
    PTK_LOGS(">>> %s: ", kernel_name.c_str());

    uint32_t len = 640 * 640;
    uint32_t size = sizeof(float) * len;
    vk::Buffer *input_buffer0 = engine->CreateBuffer(size);
    vk::Buffer *input_buffer1 = engine->CreateBuffer(size);
    vk::Buffer *output_buffer = engine->CreateBuffer(size);
    std::vector<vk::Buffer *> input_buffers = {input_buffer0, input_buffer1};
    std::vector<vk::Buffer *> output_buffers = {output_buffer};

    float *mapped_data0 = (float *)input_buffer0->MapMemory(0, input_buffer0->buffer_size());
    float *mapped_data1 = (float *)input_buffer1->MapMemory(0, input_buffer1->buffer_size());
    for (uint32_t i = 0; i < len; i++) {
        mapped_data0[i] = 1;
        mapped_data1[i] = 1;
    }
    input_buffer0->UnmapMemory();
    input_buffer1->UnmapMemory();

    for (uint32_t i = 0; i < 1; i++) {
        uint32_t p[2] = {12, 23};
        uint32_t num_xyz[3] = {640, 640, 1};
        engine->Run("engine_test", num_xyz, input_buffers, sizeof(uint32_t) * 2, p, output_buffers);

        {
            vk::Buffer *buffer = output_buffers[0];
            float *mapped_data = (float *)buffer->MapMemory(0, buffer->buffer_size());
            PTK_LOGS("<<< Out: (%f, %f, %f, %f) - (%f, %f, %f, %f)\n", 
                    mapped_data[0], mapped_data[1], mapped_data[2], mapped_data[3], 
                    mapped_data[4], mapped_data[5], mapped_data[6], mapped_data[7]);
            buffer->UnmapMemory();
        }
    }

    for (uint32_t i = 0; i < input_buffers.size(); i++)
        delete input_buffers[i];
    for (uint32_t i = 0; i < output_buffers.size(); i++)
        delete output_buffers[i];
}

void TestGemmBeta(vk::Engine *engine, std::string kernel_name) {
    PTK_LOGS(">>> %s: ", kernel_name.c_str());

    uint32_t height_a = 960, width_a = 1280;
    uint32_t height_b = 1280, width_b = 640;

    vk::Buffer *input_buffer0 = engine->CreateBuffer(sizeof(float) * height_a * width_a);
    vk::Buffer *input_buffer1 = engine->CreateBuffer(sizeof(float) * height_b * width_b);
    vk::Buffer *output_buffer = engine->CreateBuffer(sizeof(float) * height_a * width_b);
    std::vector<vk::Buffer *> input_buffers = {input_buffer0, input_buffer1};
    std::vector<vk::Buffer *> output_buffers = {output_buffer};

    float *mapped_data0 = (float *)input_buffer0->MapMemory(0, input_buffer0->buffer_size());
    float *mapped_data1 = (float *)input_buffer1->MapMemory(0, input_buffer1->buffer_size());
    for (uint32_t i = 0; i < height_a * width_a; i++)
        mapped_data0[i] = 1.23f + i % 12;
    for (uint32_t i = 0; i < height_b * width_b; i++)
        mapped_data1[i] = 2.34f + i % 22;

    input_buffer0->UnmapMemory();
    input_buffer1->UnmapMemory();

    prof::Timer timer(kernel_name, 10);
    for (uint32_t i = 0; i < 1; i++) {
        timer.Start();
        uint32_t num_xyz[3] = {width_b, height_a, 1};
        engine->Run(kernel_name, num_xyz, input_buffers, 0, nullptr, output_buffers);
        timer.Stop(0, "Kernel Run");
        timer.Print(0);

        {
            vk::Buffer *buffer = output_buffers[0];
            float *mapped_data = (float *)buffer->MapMemory(0, buffer->buffer_size());
            float mean = 0.f;
            for (uint32_t i = 0; i < height_a; i++) {
                // PTK_LOGS("\n");
                for (uint32_t j = 0; j < width_b; j++) {
                    mean += mapped_data[i * width_b + j];
                }
            }
            PTK_LOGS(" <<< Out: %f.\n", mean / (height_a * width_b));
            buffer->UnmapMemory();
        }
    }

    for (uint32_t i = 0; i < input_buffers.size(); i++)
        delete input_buffers[i];
    for (uint32_t i = 0; i < output_buffers.size(); i++)
        delete output_buffers[i];
}

void TestGemm(vk::Engine *engine, std::string kernel_name) {
    PTK_LOGS(">>> %s: ", kernel_name.c_str());

    uint32_t height_a = 960, width_a = 1280;
    uint32_t height_b = 1280, width_b = 640;

    vk::Buffer *input_buffer0 = engine->CreateBuffer(sizeof(float) * height_a * width_a);
    vk::Buffer *input_buffer1 = engine->CreateBuffer(sizeof(float) * height_b * width_b);
    vk::Buffer *output_buffer = engine->CreateBuffer(sizeof(float) * height_a * width_b);
    std::vector<vk::Buffer *> input_buffers = {input_buffer0, input_buffer1};
    std::vector<vk::Buffer *> output_buffers = {output_buffer};

    float *mapped_data0 = (float *)input_buffer0->MapMemory(0, input_buffer0->buffer_size());
    float *mapped_data1 = (float *)input_buffer1->MapMemory(0, input_buffer1->buffer_size());
    for (uint32_t i = 0; i < height_a * width_a; i++)
        mapped_data0[i] = 1.23f + i % 12;
    for (uint32_t i = 0; i < height_b * width_b; i++)
        mapped_data1[i] = 2.34f + i % 22;

    input_buffer0->UnmapMemory();
    input_buffer1->UnmapMemory();

    prof::Timer timer(kernel_name, 10);
    for (uint32_t i = 0; i < 1; i++) {
        timer.Start();

        uint32_t push_const[3] = {height_a, width_b, width_a}; // M N K
        uint32_t num_xyz[3] = {width_b, height_a, 1};
        engine->Run(kernel_name, num_xyz, input_buffers, 3*sizeof(uint32_t), push_const, output_buffers);

        timer.Stop(0, "Kernel Run");
        timer.Print(0);

        {
            vk::Buffer *buffer = output_buffers[0];
            float *mapped_data = (float *)buffer->MapMemory(0, buffer->buffer_size());
            float mean = 0.f;
            for (uint32_t i = 0; i < height_a; i++) {
                for (uint32_t j = 0; j < width_b; j++) {
                    mean += mapped_data[i * width_b + j];
                    // printf("%f, ", mapped_data[i * width_b + j]);
                }
            }
            PTK_LOGS(" <<< Out: %f.\n", mean / (height_a * width_b));
            buffer->UnmapMemory();
        }
    }

    for (uint32_t i = 0; i < input_buffers.size(); i++)
        delete input_buffers[i];
    for (uint32_t i = 0; i < output_buffers.size(); i++)
        delete output_buffers[i];
}

int main() {
    printf("VulkanMain Start.\n");
    vk::Engine engine;
    std::vector<std::pair<std::string, vk::SetpParamsFuncs>> shaders_name;
    shaders_name.push_back(std::make_pair("engine_test", SetParamsEngineTest));
    shaders_name.push_back(std::make_pair("gemm_beta", SetParamsMatMulTiledFp32));
    shaders_name.push_back(std::make_pair("gemm_v1", SetParamsGemm));

    engine.Init("shaders/spv", shaders_name, 0, true);

    TestBaseKernel(&engine, "engine_test");
    TestGemmBeta(&engine, "gemm_beta");
    TestGemm(&engine, "gemm_v1");

    engine.Deinit();

    printf("VulkanMain End.\n");
    return 0;
}