#include "tflite_cpy_demo.hpp"

#include "conv_test_model.h"

using namespace pai::infer;
int main() {
    conv_test_model::Init();
    

    int8_t *input_data = (int8_t*)conv_test_model::graph_input_0.data;
    for (uint32_t i=0; i<conv_test_model::graph_input_0_size; i++)
        input_data[i] = i % 255;

    TestTfliteCpy(input_data, conv_test_model::graph_input_0_size);

    conv_test_model::Run();

    // PrintTensr(conv_test_model::graph_input_0);
    // PrintTensr(conv_test_model::conv_0_output);
    // PrintTensr(conv_test_model::conv_1_output);
    // PrintTensr(conv_test_model::maxpooling_2_output);
    // PrintTensr(conv_test_model::reshape_3_output);
    PrintTensr(conv_test_model::graph_output_0);

    conv_test_model::Deinit();
    return 0;
}