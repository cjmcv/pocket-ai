
#include "trained_lstm_model_test_data.h"
#include "micro_speech_test_data.h"
// micro_speech_quantized / tf_micro_conv_test_model.int8 / resnet_q / resnet / mobilenetv3_q / mobilenetv3
// 
using namespace pai::infer;

int main() {
    // TEST_TRAINED_LSTM_MODEL();
    TEST_MICRO_SPEECH();
    return 0;
}