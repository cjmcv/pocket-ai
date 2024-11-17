#!/usr/bin/env bash

model_tag=micro_speech_model
model_path=./models/micro_speech_quantized.tflite

# Export model
exporter_path=../../engine/infer/exporter/
python3 ${exporter_path}/tflite_export.py \
    --model_path ${model_path} \
    --output_path "./" \
    --model_tag ${model_tag}

# Export test data
tools_path=../../engine/infer/tools
python3 ${tools_path}/generate_test_data_tflite.py --model ${model_path} --model_tag ${model_tag}

echo 按任意键继续
read -n 1