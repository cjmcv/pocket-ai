#include "tflite_cpy_demo.hpp"

#include "resnet_q_model.h"

using namespace pai::infer;
namespace model = resnet_q_model;

int main() {

    // engine/infer
    model::Init();
    int8_t *input_data = (int8_t*)model::graph_input_0.data;
    for (uint32_t i=0; i<model::graph_input_0_size/sizeof(int8_t); i++)
        input_data[i] = i % 255;
    model::Run();

    // tflite infer
    void *tflite_out;
    std::string model_path = "./gen/tinynn/resnet_q.tflite";
    TestTfliteCpy(model_path, 
                  input_data, model::graph_input_0_size, 
                  model::graph_output_0.id, &tflite_out);

    // Check result
    bool is_pass = CheckTensr(model::graph_output_0, tflite_out);
    printf("Test %s.\n", is_pass==true? "Passed" : "Failed");
    model::Deinit();
    return 0;
}