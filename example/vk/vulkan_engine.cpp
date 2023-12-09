
#include "vulkan/vulkan.h"

#include <cmath>
#include "engine/vk/vulkan_engine.hpp"
#include "bmp_reader.hpp"

const int WIDTH = 3200; // Size of rendered mandelbrot set.
const int HEIGHT = 2400; // Size of renderered mandelbrot set.

using namespace ptk;

// The pixels of the rendered mandelbrot set are in this format:
struct Pixel {
    float r, g, b, a;
};
void saveRenderedImage(void *mappedMemory, int idx) {
    Pixel *pmappedMemory = (Pixel *)mappedMemory;

    // Get the color data from the buffer, and cast it to bytes.
    // We save the data to a vector.
    int channels = 4;
    util::BmpReader img(WIDTH, HEIGHT, channels);
    uint8_t *img_data = img.data();
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        img_data[i * channels + 0] = (unsigned char)(255.0f * (pmappedMemory[i].b));
        img_data[i * channels + 1] = (unsigned char)(255.0f * (pmappedMemory[i].g));
        img_data[i * channels + 2] = (unsigned char)(255.0f * (pmappedMemory[i].r));
        img_data[i * channels + 3] = (unsigned char)(255.0f * (pmappedMemory[i].a));
    }
    std::string img_name = std::to_string(idx) + ".bmp";
    img.Write(img_name.c_str());
}

int main() {
    printf("VulkanMain Start.\n");

    vk::VulkanEngine engine;
    engine.Init("shaders", 0, true);

    {
        uint32_t size = sizeof(Pixel) * WIDTH * HEIGHT;
        vk::Buffer *buf = engine.CreateBuffer(size);
        std::vector<vk::Buffer *> input_buffers = { buf };
        std::vector<vk::Buffer *> output_buffers = {};
        for (uint32_t i=0; i<2; i++) {
            engine.Run("mandelbrot", input_buffers, 0, nullptr, output_buffers);
            {
                vk::Buffer *buffer = input_buffers[0];
                void *mapped_data = buffer->MapMemory(0, buffer->buffer_size());
                saveRenderedImage(mapped_data, i);
                buffer->UnmapMemory();
            }
        }
        delete buf;
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

        for (uint32_t i=0; i<2; i++) {
            engine.Run("matmul_tiled_fp32", input_buffers, 0, nullptr, output_buffers);

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
            engine.Run("engine_test", input_buffers, sizeof(uint32_t) * 2, p, output_buffers);

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