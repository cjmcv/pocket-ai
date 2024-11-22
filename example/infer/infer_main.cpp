
// #include "trained_lstm_model_test_data.h"
// #include "micro_speech_quantized_model_test_data.h"
// #include "tf_micro_conv_test_model_int8_model_test_data.h"
// #include "mobilenetv3_model_test_data.h"
#include "resnet_model_test_data.h"
// #include "simpledeconvmodel_q_model_test_data.h"
// #include "simpledeconvmodel_model_test_data.h"

// micro_speech_quantized / tf_micro_conv_test_model.int8 / resnet_q / resnet / mobilenetv3_q / mobilenetv3
// 
using namespace pai::infer;

int main() {
    // TEST_TRAINED_LSTM_MODEL();
    // TEST_MICRO_SPEECH_QUANTIZED_MODEL();
    // TEST_TF_MICRO_CONV_TEST_MODEL_INT8_MODEL();
    // TEST_MOBILENETV3_MODEL();
    TEST_RESNET_MODEL();
    // TEST_SIMPLEDECONVMODEL_Q_MODEL();
    // TEST_SIMPLEDECONVMODEL_MODEL();
    
    return 0;
}