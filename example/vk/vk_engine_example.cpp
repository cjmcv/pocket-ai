
#include <cmath>
#include "engine/vk/engine.hpp"
#include "util/bmp_reader.hpp"

using namespace ptk;

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
        {3, 640},
        {4, 640},
        {5, 640},
    };
    params->push_constant_num = 0;
}

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

int main() {
    printf("VulkanMain Start.\n");
    vk::Engine engine;
    std::vector<std::pair<std::string, vk::SetpParamsFuncs>> shaders_name;
    shaders_name.push_back(std::make_pair("gemm", SetParamsMatMulTiledFp32));
    shaders_name.push_back(std::make_pair("engine_test", SetParamsEngineTest));

    engine.Init("shaders", shaders_name, 0, true);
    {
        uint32_t len = 640 * 640;
        uint32_t size = sizeof(float) * len;
        vk::Buffer *input_buffer0 = engine.CreateBuffer(size);
        vk::Buffer *input_buffer1 = engine.CreateBuffer(size);
        vk::Buffer *output_buffer = engine.CreateBuffer(size);
        std::vector<vk::Buffer *> input_buffers = { input_buffer0, input_buffer1 };
        std::vector<vk::Buffer *> output_buffers = { output_buffer };

        float *mapped_data0 = (float *)input_buffer0->MapMemory(0, input_buffer0->buffer_size());
        float *mapped_data1 = (float *)input_buffer1->MapMemory(0, input_buffer1->buffer_size());
        for (uint32_t i=0; i<len; i++) {
            mapped_data0[i] = 1;
            mapped_data1[i] = 1;
        }
        input_buffer0->UnmapMemory();
        input_buffer1->UnmapMemory();

        for (uint32_t i=0; i<2; i++) {
            uint32_t num_xyz[3] = {640, 640, 1};
            engine.Run("gemm", num_xyz, input_buffers, 0, nullptr, output_buffers);

            {
                vk::Buffer *buffer = output_buffers[0];
                float *mapped_data = (float *)buffer->MapMemory(0, buffer->buffer_size());
                for (uint32_t i=0; i<2; i++) { // 640
                    for (uint32_t j=0; j<2; j++) { // 640
                        printf("%f, ", mapped_data[i*640+j]);
                    }
                    printf("\n");
                }
                buffer->UnmapMemory();
            }   
        }
        
        for (uint32_t i=0; i<input_buffers.size(); i++)
            delete input_buffers[i];
        for (uint32_t i=0; i<output_buffers.size(); i++)
            delete output_buffers[i];
    }

    {
        uint32_t len = 640 * 640;
        uint32_t size = sizeof(float) * len;
        vk::Buffer *input_buffer0 = engine.CreateBuffer(size);
        vk::Buffer *input_buffer1 = engine.CreateBuffer(size);
        vk::Buffer *output_buffer = engine.CreateBuffer(size);
        std::vector<vk::Buffer *> input_buffers = { input_buffer0, input_buffer1 };
        std::vector<vk::Buffer *> output_buffers = { output_buffer };

        float *mapped_data0 = (float *)input_buffer0->MapMemory(0, input_buffer0->buffer_size());
        float *mapped_data1 = (float *)input_buffer1->MapMemory(0, input_buffer1->buffer_size());
        for (uint32_t i=0; i<len; i++) {
            mapped_data0[i] = 1;
            mapped_data1[i] = 1;
        }
        input_buffer0->UnmapMemory();
        input_buffer1->UnmapMemory();

        for (uint32_t i=0; i<1; i++) {
            uint32_t p[2] = { 12, 23 };
            uint32_t num_xyz[3] = {640, 640, 1};
            engine.Run("engine_test", num_xyz, input_buffers, sizeof(uint32_t) * 2, p, output_buffers);

            {
                vk::Buffer *buffer = output_buffers[0];
                float *mapped_data = (float *)buffer->MapMemory(0, buffer->buffer_size());
                printf("engine_test0: (%f, %f, %f, %f)\n", mapped_data[0], mapped_data[1], mapped_data[2], mapped_data[3]);
                printf("engine_test1: (%f, %f, %f, %f)\n", mapped_data[4], mapped_data[5], mapped_data[6], mapped_data[7]);
                buffer->UnmapMemory();
            }   
        }
        
        for (uint32_t i=0; i<input_buffers.size(); i++)
            delete input_buffers[i];
        for (uint32_t i=0; i<output_buffers.size(); i++)
            delete output_buffers[i];
    }

    engine.Deinit();

    printf("VulkanMain End.\n");

    return 0;
}