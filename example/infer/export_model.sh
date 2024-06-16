#!/usr/bin/env bash

exporter_path=../../engine/infer/exporter/
python3 ${exporter_path}/tflite_export.py \
    --model_path "./gen/tinynn/mobilenetv3_q.tflite" \
    --output_path "./" \
    --model_tag "mobilenetv3_q_model"

echo 按任意键继续
read -n 1