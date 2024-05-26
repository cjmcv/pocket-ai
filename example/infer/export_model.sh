#!/usr/bin/env bash

exporter_path=../../engine/infer/tools/tflite_exporter/
python3 ${exporter_path}/tflite_export.py \
    --model_path "./models/tf_micro_conv_test_model.int8.tflite" \
    --output_path "./" \
    --model_tag "conv_test_model"

echo 按任意键继续
read -n 1