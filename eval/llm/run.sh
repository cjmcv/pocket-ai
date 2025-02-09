#!/usr/bin/env bash

# sh run.sh launch sglang/vllm/lmdeploy         启动服务
# sh run.sh eval sglang/vllm/lmdeploy/local     评估在特定数据集上的模型指标
# sh run.sh bm sglang/vllm/lmdeploy             评估在sharedgpt数据集上的耗时情况
# sh run.sh task|list|nsys  task(特定数据集的具体情况) / list(列出所支持的所有数据集) / nsys(GPU运行情况分析)


MODEL_PATH="/home/cjmcv/project/llm_models/Qwen/Qwen2___5-1___5B-Instruct-AWQ" # deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B
DATASETS_PATH="/home/cjmcv/project/llm_datasets/"
# EVAL_TASK="examples/my_set_zh.txt" # my_set_mmlu / my_set_zh / my_set_others / my_set_sub_greedy / my_set_temp
EVAL_TASK="lighteval|xwinograd:zh|0|0"
EVAL_MAX_SAMPLES="100"
NSYS_PROFILER=
# NSYS_PROFILER="nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node -o sglang.out --delay 60 --duration 70"
# ncu

echo "run example: sh run.sh {launch/bm/eval/list/task} {sglang/lmdeploy/vllm/local}"

if [ "$1" = "launch" ]; then
    if [ "$2" = "sglang" ]; then
        # --grammar-backend xgrammar --disable-overlap-schedule --disable-radix-cache
        $NSYS_PROFILER python3 -m sglang.launch_server --model-path $MODEL_PATH --enable-torch-compile --enable-mixed-chunk 
    elif [ "$2" = "vllm" ]; then
        # vllm=0.6.6, AssertionError: Logits Processors are not supported in multi-step decoding, 需要去掉 --num-scheduler-steps
        $NSYS_PROFILER python3 -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --disable-log-requests --max_model_len 4096 # --num-scheduler-steps 10 
    elif [ "$2" = "lmdeploy" ]; then
        # lmdeploy=0.7.0, 不支持输出logprobs（输出token的概率值），导致LoglikelihoodResponse无法计算。
        $NSYS_PROFILER lmdeploy serve api_server $MODEL_PATH --model-name $MODEL_PATH
    else
        echo "The input parameters are incorrect."
    fi
elif [ "$1" = "bm" ]; then
    DATASETS_PATH="$DATASETS_PATH/ShareGPT_V3_unfiltered_cleaned_split.json"
    DATASET="sharegpt"
    NUM_PROMPTS=10
    REQUEST_RATE=4
    python3 main_benchmark.py --backend $2 --dataset-name $DATASET --dataset-path $DATASETS_PATH --num-prompts $NUM_PROMPTS --request-rate $REQUEST_RATE 
elif [ "$1" = "eval" ]; then
    # python3 src/lighteval/__main__.py vllm "pretrained=$MODEL_PATH,dtype=float16" "helm|quac|0|0"
    # model_args: ModelConfig
    if [ "$2" = "local" ]; then
        python3 main_eval.py --model-args "backend=$2,pretrained=$MODEL_PATH,dtype=float16" --tasks $EVAL_TASK --max-samples $EVAL_MAX_SAMPLES \
                                       --datasets-path $DATASETS_PATH # --is-eval-api-server
    else
        python3 main_eval.py --model-args "backend=$2,pretrained=$MODEL_PATH,dtype=float16" --tasks $EVAL_TASK --max-samples $EVAL_MAX_SAMPLES \
                                       --datasets-path $DATASETS_PATH --is-eval-api-server
    fi
elif [ "$1" = "list" ]; then
    # python3 src/lighteval/__main__.py tasks list
    python3 main_eval.py --tasks-list
elif [ "$1" = "inspect" ]; then
    # python3 src/lighteval/__main__.py tasks inspect $EVAL_TASK
    python3 main_eval.py --tasks-inspect --tasks $EVAL_TASK --datasets-path $DATASETS_PATH
elif [ "$1" = "nsys" ]; then
    nsys-ui profile sglang.out.nsys-rep
fi

## Setup command.
## sgLang
# pip install --upgrade pip
# pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/
#
# cd sglang
# pip install --upgrade pip
# pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/

## sglang
# curl http://localhost:30000/generate \
#   -H "Content-Type: application/json" \
#   -d '{
#     "text": "Once upon a time,",
#     "sampling_params": {
#       "max_new_tokens": 16,
#       "temperature": 0
#     }
#   }'

## vllm
# http://localhost:8000/docs
# curl http://localhost:8000/v1/completions \
# -H "Content-Type: application/json" \
# -d '{
#     "model":"/home/cjmcv/project/llm_models/Qwen/Qwen2___5-1___5B-Instruct-AWQ",
#     "prompt":"请为我生成一篇关于人工智能的短文",
#     "max_tokens":100
# }'

## lmdeploy
# # test_model与launch时的--model-name对应
# curl http://localhost:23333/v1/completions \
# -H "Content-Type: application/json" \
# -d '{
#     "model":"test_model",
#     "prompt":"请为我生成一篇关于人工智能的短文",
#     "max_tokens":100
# }'

# curl http://localhost:30000/start_profile

## vllm
# pip uninstall vllm
# pip install vllm=0.6.6

## lmdeploy
# conda create -n lmdeploy python=3.10 -y
# conda activate lmdeploy
# pip install lmdeploy

# from modelscope import snapshot_download
# model_dir = snapshot_download('Qwen/Qwen2.5-1.5B-Instruct-AWQ', cache_dir='/home/cjmcv/project/llm_models/')
# conda create -n eval-venv python=3.10 -y
# conda activate eval-venv
# pip install -e .
# lighteval vllm     "pretrained=/home/cjmcv/project/llm_models/Qwen/Qwen2___5-1___5B-Instruct-AWQ,dtype=float16"     "leaderboard|truthfulqa:mc|0|0"

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True