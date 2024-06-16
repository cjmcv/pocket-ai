#!/usr/bin/env bash

tools_path=../../engine/infer/tools/tflite_cpy
python3 ${tools_path}/tflite_inference.py --model "./gen/tinynn/mobilenetv3_q.tflite" --input_one --tensor_id 198 #198

echo 按任意键继续
read -n 1