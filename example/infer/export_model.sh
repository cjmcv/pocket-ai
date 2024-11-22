#!/usr/bin/env bash

# model_path=./gen/tinynn/mobilenetv3.tflite
# model_path=./gen/tinynn/simpledeconvmodel_q.tflite
# model_path=./gen/tinynn/simpledeconvmodel.tflite
# model_path=./models/tf_micro_conv_test_model.int8.tflite
# model_path=./models/micro_speech_quantized.tflite
model_path=./gen/tinynn/resnet.tflite

# Export model
exporter_path=../../engine/infer/exporter/
python3 ${exporter_path}/tflite_export.py \
    --model_path ${model_path} \
    --output_path "./"

# Export test data
tools_path=../../engine/infer/tools
python3 ${tools_path}/generate_test_data_tflite.py --model ${model_path} --debug_all_tensors

echo 按任意键继续
read -n 1