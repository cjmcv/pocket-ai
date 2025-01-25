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

import collections
import random
import shutil
from contextlib import nullcontext
from dataclasses import dataclass, field
from pytablewriter import MarkdownTableWriter

import numpy as np

from log.evaluation_tracker import EvaluationTracker
from metrics.metric_utils import MetricCategory
from lighteval_model import LightevalModel, ModelResponse #, ModelConfig
# from models.model_output import ModelResponse
from tasks.lighteval_task import LightevalTask, create_requests_from_tasks
from tasks.registry import Registry, taskinfo_selector
from tasks.requests import SampleUid

# from utils.parallelism import test_all_gather
from utils import EnvConfig # , make_results_table

import logging

logger = logging.getLogger(__name__)

@dataclass
class PipelineParameters:
    # Env parameters
    env_config: EnvConfig = field(default_factory=EnvConfig)
    dataset_loading_processes: int = 1
    # Generation parameters
    override_batch_size: int | None = None
    num_fewshot_seeds: int = 1
    max_samples: int | None = None
    use_chat_template: bool = False
    system_prompt: str | None = None

class Pipeline:
    def __init__(
        self,
        tasks: str,
        pipeline_parameters: PipelineParameters,
        evaluation_tracker: EvaluationTracker,
        model_config=None,
    ):
        self.pipeline_parameters = pipeline_parameters
        if self.pipeline_parameters.max_samples:
            logger.warning(
                "--max_samples WAS SET. THESE NUMBERS ARE ONLY PARTIAL AND SHOULD NOT BE USED FOR COMPARISON UNLESS YOU KNOW WHAT YOU ARE DOING."
            )
        self.evaluation_tracker = evaluation_tracker

        # Model.
        self.model_config = model_config
        logger.info("--- LOADING MODEL ---")
        self.model = LightevalModel(config=model_config, env_config=self.pipeline_parameters.env_config)
        self.evaluation_tracker.general_config_logger.log_model_info(self.model.model_info)

        # Get requests
        self._init_tasks_and_requests(tasks=tasks)
        self._init_random_seeds()
        # Final results
        self.final_dict: dict = None
        
    def _init_tasks_and_requests(self, tasks: str):
        with nullcontext():
            logger.info("--- LOADING TASKS ---")
            registry = Registry(cache_dir=self.pipeline_parameters.env_config.cache_dir)
            task_names_list, fewshots_dict = taskinfo_selector(tasks, registry)
            task_dict = registry.get_task_dict(task_names_list)
            # LightevalTask.load_datasets(list(task_dict.values()), self.pipeline_parameters.dataset_loading_processes)

            self.evaluation_tracker.task_config_logger.log(task_dict)

            requests, docs = create_requests_from_tasks(
                task_dict=task_dict,
                fewshot_dict=fewshots_dict,
                num_fewshot_seeds=self.pipeline_parameters.num_fewshot_seeds,
                lm=self.model,
                max_samples=self.pipeline_parameters.max_samples,
                evaluation_tracker=self.evaluation_tracker,
                use_chat_template=self.pipeline_parameters.use_chat_template,
                system_prompt=self.pipeline_parameters.system_prompt,
            )
            self.task_names_list = task_names_list
            self.task_dict = task_dict
            self.fewshot_dict = fewshots_dict
            self.requests = requests
            self.docs = docs

    def _init_random_seeds(self):
        logger.info("--- INIT SEEDS ---")
        random.seed(1234)
        np.random.seed(1234)

    def evaluate(self):
        self.evaluation_tracker.general_config_logger.log_args_info(
            num_fewshot_seeds=self.pipeline_parameters.num_fewshot_seeds,
            override_batch_size=self.pipeline_parameters.override_batch_size,
            max_samples=self.pipeline_parameters.max_samples,
            config=self.model_config,
        )

        sample_id_to_responses = self._run_model()
        self._compute_metrics(sample_id_to_responses)

        self.evaluation_tracker.general_config_logger.log_end_time()
        self.evaluation_tracker.metrics_logger.aggregate(task_dict=self.task_dict, bootstrap_iters=1000)
        self.evaluation_tracker.details_logger.aggregate()

        for weights in ["delta", "adapter"]:
            try:
                tmp_weights_dir = f"{self.evaluation_tracker.general_config_logger.model_name}-{weights}-applied"
                shutil.rmtree(tmp_weights_dir)
                logger.info(f"Removed {tmp_weights_dir}")
            except OSError:
                pass

    def _run_model(self):
        # Running all requests depending on the model call type (log likelihood, generative, ...)
        # to be able to batch them
        logger.info("--- RUNNING MODEL ---")
        sample_id_to_responses: dict[(SampleUid, MetricCategory), list[ModelResponse]] = collections.defaultdict(list)

        for request_type, requests in self.requests.items():
            logger.info(f"Running {request_type} requests")
            run_model = self.model.get_method_from_request_type(request_type=request_type)
            responses = run_model(requests, override_bs=self.pipeline_parameters.override_batch_size)

            # Storing the responses associated to the same samples together
            for response, request in zip(responses, requests):
                for metric_category in request.metric_categories:
                    sample_id = SampleUid(request.task_name, request.sample_index)
                    sample_id_to_responses[(sample_id, metric_category)].append(response)

        # Cleaning up the model before running metrics
        self.model.cleanup()

        return sample_id_to_responses

    def _compute_metrics(self, sample_id_to_responses):
        # To compute the metrics we first group the samples and task and then by metrics.
        # This way we can batch the metrics computation for each task and metric category

        # This variable will hold the samples grouped by task and metric category
        # example:
        # task_metric_category_groups = {
        #     "task_name": {
        #         "metric_category": {
        #             "ids": [sample_id1, sample_id2, ...],
        #             "responses": [[response1_1, response1_2, ...], [response2_1, response2_2, ...], ...],
        #             "docs": [doc1, doc2, ...]
        #         }
        logger.info("--- COMPUTING METRICS ---")
        task_metric_category_groups = collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(list))
        )

        for (sample_id, metric_category), sample_responses in sample_id_to_responses.items():
            task_metric_category_groups[sample_id.task_name][metric_category]["ids"].append(sample_id.doc_id_seed)
            task_metric_category_groups[sample_id.task_name][metric_category]["responses"].append(sample_responses)
            task_metric_category_groups[sample_id.task_name][metric_category]["docs"].append(self.docs[sample_id])

        for task_name, samples_per_metric in task_metric_category_groups.items():
            short_task_name = task_name.rsplit("|", 1)[0]
            task: LightevalTask = self.task_dict[short_task_name]

            for metric_category, samples in samples_per_metric.items():
                sample_ids = samples["ids"]
                responses = samples["responses"]
                docs = samples["docs"]
                metric_function = task.get_metric_method_from_category(metric_category=metric_category)
                metric_category_metrics = [metric for metric in task.metrics if metric.category == metric_category]

                outputs = metric_function(
                    sample_ids=sample_ids,
                    responses=responses,
                    formatted_docs=docs,
                    metrics=metric_category_metrics,
                )

                for output, doc, response in zip(outputs, docs, responses):
                    self.evaluation_tracker.metrics_logger.log(task_name, output)
                    self.evaluation_tracker.details_logger.log(task_name, task, doc, response, output)

    def save_and_push_results(self):
        logger.info("--- SAVING AND PUSHING RESULTS ---")
        self.evaluation_tracker.save()

    def _init_final_dict(self):
        if self.final_dict is None:
            self.final_dict = self.evaluation_tracker.generate_final_dict()

    def show_results(self):
        logger.info("--- DISPLAYING RESULTS ---")
        self._init_final_dict()

        """Generate table of results."""
        md_writer = MarkdownTableWriter()
        md_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]

        values = []

        print(self.final_dict)
        for k in sorted(self.final_dict["results"].keys()):
            dic = self.final_dict["results"][k]
            version = self.final_dict["versions"][k] if k in self.final_dict["versions"] else ""
            for m, v in dic.items():
                if m.endswith("_stderr"):
                    continue

                if m + "_stderr" in dic:
                    se = dic[m + "_stderr"]
                    values.append([k, version, m, "%.4f" % v, "±", "%.4f" % se])
                else:
                    values.append([k, version, m, "%.4f" % v, "", ""])
                k = ""
                version = ""
        md_writer.value_matrix = values
        print(md_writer.dumps())

    def get_results(self):
        self._init_final_dict()
        return self.final_dict
