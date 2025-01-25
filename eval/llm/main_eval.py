# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
from argparse import ArgumentParser

TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR: str = os.getenv("HF_HOME", "/scratch")

HELP_PANEL_NAME_1 = "Common Parameters"
HELP_PANEL_NAME_2 = "Logging Parameters"
HELP_PANEL_NAME_3 = "Debug Parameters"
HELP_PANEL_NAME_4 = "Modeling Parameters"

def inspect(
    tasks: str,
    num_samples: int = 10, #Number of samples to display
    show_config: bool = False, # Will display the full task config
    cache_dir: str = CACHE_DIR, # Cache directory used to store datasets and models
):
    """
    Inspect a tasks
    """
    from dataclasses import asdict
    from pprint import pformat
    from rich import print
    from tasks.registry import Registry, taskinfo_selector

    registry = Registry(cache_dir=cache_dir)

    # Loading task
    task_names_list, _ = taskinfo_selector(tasks, task_registry=registry)
    task_dict = registry.get_task_dict(task_names_list)
    for name, task in task_dict.items():
        print("-" * 10, name, "-" * 10)
        if show_config:
            print("-" * 10, "CONFIG")
            task.cfg.print()
        for ix, sample in enumerate(task.eval_docs()[: int(num_samples)]):
            if ix == 0:
                print("-" * 10, "SAMPLES")
            print(f"-- sample {ix} --")
            print(pformat(asdict(sample), indent=2))

def eval(args):
    from log.evaluation_tracker import EvaluationTracker
    from lighteval_model import ModelConfig
    from pipeline import EnvConfig, Pipeline, PipelineParameters

    TOKEN = os.getenv("HF_TOKEN")

    env_config = EnvConfig(token=TOKEN, cache_dir=args.cache_dir)

    evaluation_tracker = EvaluationTracker(
        output_dir=args.output_dir,
        save_details=args.save_details,
    )

    pipeline_params = PipelineParameters(
        env_config=env_config,
        dataset_loading_processes=args.dataset_loading_processes,
        override_batch_size=-1,  # Cannot override batch size when using VLLM
        num_fewshot_seeds=args.num_fewshot_seeds,
        max_samples=args.max_samples,
        use_chat_template=args.use_chat_template,
        system_prompt=args.system_prompt,
    )

    model_args_dict: dict = {k.split("=")[0]: k.split("=")[1] if "=" in k else True for k in args.model_args.split(",")}
    model_config = ModelConfig(**model_args_dict)

    pipeline = Pipeline(
        tasks=args.tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.show_results()
    results = pipeline.get_results()
    pipeline.save_and_push_results()

if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--is-eval-api-server",
        action="store_true",
        help="Whether to use the API server as the evaluation backend or use the locally executed vLLM as the backend.",
    )
    parser.add_argument(
        "--model-args",
        type=str,
        help="Model arguments in the form key1=value1,key2=value2,...",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        help="Comma-separated list of tasks to evaluate on.",
    )
    parser.add_argument(
        "--datasets-path",
        type=str,
        default="/",
        help="The path of datasets.",
    )
    # === Common parameters ===
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Use chat template for evaluation.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Use system prompt for evaluation.",
    )
    parser.add_argument(
        "--dataset-loading-processes",
        type=int,
        default=1,
        help="Number of processes to use for dataset loading.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=CACHE_DIR,
        help="Cache directory for datasets and models.",
    )
    parser.add_argument(
        "--num-fewshot-seeds",
        type=int,
        default=1,
        help="Number of seeds to use for few-shot evaluation.",
    )
    # === saving ===
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for evaluation results.",
    )
    parser.add_argument(
        "--save-details",
        action="store_true",
        help="Save detailed, sample per sample, results.",
    )
    # === debug ===
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate on.",
    )
    parser.add_argument(
        "--tasks-list",
        action="store_true",
        help="List all tasks.",
    )
    parser.add_argument(
        "--tasks-inspect",
        action="store_true",
        help="Inspect a tasks.",
    )

    args = parser.parse_args()

    # Default path: ~/.cache/huggingface/datasets
    os.environ["HF_DATASETS_CACHE"] = args.datasets_path + "/huggingface"
    os.environ["USING_API_SERVER"] = str(args.is_eval_api_server)

    if args.tasks_list:
        from tasks.registry import Registry
        registry = Registry(cache_dir=CACHE_DIR)
        registry.print_all_tasks()
        print("tasks_list done.")
    elif args.tasks_inspect:
        inspect(args.tasks)
        print("tasks_inspect done.")
    else:
        eval(args)
        print("tasks_eval done.")
