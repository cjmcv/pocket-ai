#!/usr/bin/env bash

# micro_speech_quantized.tflite
exporter_path=../../engine/infer/exporter/
python3 ${exporter_path}/tflite_export.py \
    --model_path "./models/tf_micro_conv_test_model.int8.tflite" \
    --output_path "./" \
    --model_tag "conv_test_model"

echo 按任意键继续
read -n 1