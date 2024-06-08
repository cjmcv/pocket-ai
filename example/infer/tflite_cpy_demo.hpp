
#include <stdio.h>
#include "engine/infer/tools/tflite_cpy/tflite_cpy.hpp"

inline int TestTfliteCpy(std::string model_path, void *data, uint32_t data_size, uint32_t output_id, void **outdata) {
    pai::infer::TfliteCpy tflite_cpy;
    std::string work_space = "../../engine/infer/tools/tflite_cpy/";
    tflite_cpy.Init(work_space, model_path);

    void *input_data;
    uint32_t input_size;
    tflite_cpy.GetInputPtr(0, (void **)&input_data, &input_size);

    if (input_size != data_size) printf("Error: input_size != data_size\n");
    memcpy(input_data, data, data_size);
    
    tflite_cpy.Infer();

    tflite_cpy.PrintTensor(output_id);
    *outdata = tflite_cpy.GetTensorData(output_id);
    return 0;
}