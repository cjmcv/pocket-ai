#!/usr/bin/env bash

exporter_path=../../engine/infer/exporter/
python3 ${exporter_path}/tflite_export.py \
    --model_path "./models/trained_lstm.tflite" \
    --output_path "./" \
    --model_tag "trained_lstm_model"

echo 按任意键继续
read -n 1