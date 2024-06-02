
#include <stdio.h>
#include "engine/infer/tools/tflite_cpy/tflite_cpy.hpp"

inline int TestTfliteCpy(void *data, uint32_t data_size) {
    pai::infer::TfliteCpy tflite_cpy;
    std::string work_space = "/home/shared_dir/PocketAI/engine/infer/tools/tflite_cpy/";
    std::string model_path = "/home/shared_dir/PocketAI/example/infer/models/tf_micro_conv_test_model.int8.tflite";
    tflite_cpy.Init(work_space, model_path);

    int8_t *input_data;
    uint32_t input_size;
    tflite_cpy.GetInputPtr(0, (void **)&input_data, &input_size);

    if (input_size != data_size) printf("Error: input_size != data_size\n");
    memcpy(input_data, data, data_size);
    
    tflite_cpy.Infer();

    tflite_cpy.PrintTensor(12);
    // tflite_cpy.Print("StatefulPartitionedCall:0");

    return 0;
}