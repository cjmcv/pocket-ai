# eval.llm

在对推理引擎做加速优化的时候，除了要关注计算性能，还要需要确保计算结果的正确性。

而llm的输出变化多样，很多时候没有唯一答案，且覆盖面很广。使用了某些加速方法后，对于某个请求，优化前后结果不一定完全一致，人工判断费时费力。需要一个脚本帮忙去核对正确性，所以开发了这个工具。

这个工具是基于[lighteval](https://github.com/huggingface/lighteval/tree/452e031ac9f74acc3b32632d97b28f788554721d)修改而来，与lighteval的不同之处在于：

* LightEval侧重点在模型评估，而eval.llm侧重点在优化推理引擎的过程中，对引擎端到端计算结果的正确性校验。

* 小 和 快，在推理引擎优化开发期间可低成本运行。

* 主要通过api_server对各种推理引擎上进行评估，方便接入你感兴趣的开源框架，对比分析自研框架和开源框架的差距。

* 包含计算性能测试功能，可以顺便测出加速优化的效果。

# 使用例子

## 如启用sglang的api-server，确保sglang已正常安装，并对启动的api-server进行问答评估。更多例子,请看pocket-ai/eval/llm/example/run.sh

```bash
cd pocket-ai/eval/llm/
sh run.sh launch sglang
sh run.sh eval sglang
```

# 主要依赖

* Note：可先尝试在推理引擎的python环境下直接运行, 然后缺的按下面列表手动补一下即可

## 取自[lighteval](https://github.com/huggingface/lighteval/tree/452e031ac9f74acc3b32632d97b28f788554721d/pyproject.toml)

dependencies = [
    # Base dependencies
    "transformers>=4.38.0",
    "accelerate",
    "huggingface_hub>=0.23.0",
    "torch>=2.0,<2.5",
    "GitPython>=3.1.41", # for logging
    "datasets>=2.14.0",
    "numpy<2",  # pinned to avoid incompatibilities
    # Prettiness
    "typer",
    "termcolor==2.3.0",
    "pytablewriter",
    "rich",
    "colorlog",
    # Extension of metrics
    "aenum==3.1.15",
    # Base metrics
    "nltk==3.9.1",
    "scikit-learn",
    "spacy==3.7.2",
    "sacrebleu",
    "rouge_score==0.1.2",
    "sentencepiece>=0.1.99",
    "protobuf==3.20.*", # pinned for sentencepiece compat
    "pycountry",
    "fsspec>=2023.12.2",
]