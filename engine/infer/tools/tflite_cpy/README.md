# tflite infer

使用c++或python调用tflite进行推理，用于核对输出与pai/infer推理库是否一致

## 纯python调用：

```python

# python tflite_inference.py
infer = TfliteInference()
infer.load_model("../models/tf_micro_conv_test_model.int8.tflite")
input = infer.fill_random_inputs()
infer.run()
output = infer.get_output(0)

```

## 通过c++接口调用python：

```C++
#include <stdio.h>
#include "tools/tflite_cpy/tflite_cpy.hpp"

int main() {
    pai::infer::TfliteCpy tflite_cpy;
    std::string work_space = "/home/shared_dir/PocketAI/engine/infer/tools/tflite_cpy/";
    tflite_cpy.Init(work_space, work_space + "../models/tf_micro_conv_test_model.int8.tflite");

    int8_t *input_data;
    uint32_t input_size;
    tflite_cpy.GetInputPtr("serving_default_conv2d_input:0", (void **)&input_data, &input_size);

    for (uint32_t i=0; i<input_size/sizeof(uint8_t); i++)
        input_data[i] = i % 255;

    tflite_cpy.Infer();

    tflite_cpy.Print("StatefulPartitionedCall:0");

    return 0;
}
```