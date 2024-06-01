#include "tflite_cpy_demo.hpp"

#include "conv_test_model.h"

int main() {
    // TestTfliteCpy();
    
    pai::infer::conv_test_model::Init();

    // graph_input_0.data
    // graph_outnput_0.data

    pai::infer::conv_test_model::Run();
    pai::infer::conv_test_model::Deinit();
    return 0;
}