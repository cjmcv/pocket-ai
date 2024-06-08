#!/usr/bin/env bash

tools_path=../../engine/infer/tools/
python3 ${tools_path}/generate_test_models_tinynn.py

python3 ${tools_path}/generate_test_models_tinynn.py --quant

echo 按任意键继续
read -n 1