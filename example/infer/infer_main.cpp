#include "tflite_cpy_demo.hpp"

#include "resnet_q_model.h"

using namespace pai::infer;

int main() {

    // engine/infer
    resnet_q_model::Init();
    int8_t *input_data = (int8_t*)resnet_q_model::graph_input_0.data;
    for (uint32_t i=0; i<resnet_q_model::graph_input_0_size; i++)
        input_data[i] = i % 255;
    resnet_q_model::Run();

    // tflite infer
    void *tflite_out;
    std::string model_path = "./gen/tinynn/resnet_q.tflite";
    TestTfliteCpy(model_path, 
                  input_data, resnet_q_model::graph_input_0_size, 
                  resnet_q_model::graph_output_0.id, &tflite_out);

    // Check result
    bool is_pass = CheckTensr(resnet_q_model::graph_output_0, tflite_out);
    printf("Test %s.\n", is_pass==true? "Passed" : "Failed");
    resnet_q_model::Deinit();
    return 0;
}