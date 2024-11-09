#include "tflite_cpy_demo.hpp"

// #include "mobilenetv3_q_model.h"
#include "trained_lstm_model.h"

// micro_speech_quantized / tf_micro_conv_test_model.int8 / resnet_q / resnet / mobilenetv3_q / mobilenetv3
// 
using namespace pai::infer;
namespace model = trained_lstm_model;

int main() {
    // engine/infer
    model::Init();
    float *input_data = (float*)model::graph_input_0.data;
    for (uint32_t i=0; i<model::graph_input_0_size/sizeof(float); i++) {
        input_data[i] = i;

        // int min = -100;
        // int max = 100;
        // input_data[i] = min + static_cast <float>(rand()) / (static_cast<float>(RAND_MAX/(max-min)));      
    }

    model::Run();

    // tflite infer
    void *tflite_out;
    std::string model_path = "./models/trained_lstm.tflite";
    TestTfliteCpy(model_path, 
                  input_data, model::graph_input_0_size, 
                  model::graph_output_0.id, &tflite_out);

    // Check result
    bool is_pass = CheckTensor(model::graph_output_0, tflite_out);
    printf("Test %s.\n", is_pass==true? "Passed" : "Failed");
    model::Deinit();
    return 0;
}