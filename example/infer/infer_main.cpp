#include "tflite_cpy_demo.hpp"

#include "conv_test_model.h"

using namespace pai::infer;
int main() {
    conv_test_model::Init();
    
    int8_t *input_data = (int8_t*)conv_test_model::graph_input_0.data;
    for (uint32_t i=0; i<conv_test_model::graph_input_0_size; i++)
        input_data[i] = i % 255;

    void *tflite_out;
    std::string model_path = "./models/micro_speech_quantized.tflite";
    TestTfliteCpy(model_path, input_data, conv_test_model::graph_input_0_size, 9, &tflite_out);

    conv_test_model::Run();

    // PrintTensr(conv_test_model::graph_input_0);
    // PrintTensr(conv_test_model::conv_0_output);
    // PrintTensr(conv_test_model::conv_1_output);
    // PrintTensr(conv_test_model::maxpooling_2_output);
    // PrintTensr(conv_test_model::reshape_3_output);
    bool is_pass = CheckTensr(conv_test_model::graph_output_0, tflite_out);
    // PrintTensr(conv_test_model::fully_connected_2_output);

    printf("Test %s.\n", is_pass==true? "Passed" : "Failed");
    conv_test_model::Deinit();
    return 0;
}