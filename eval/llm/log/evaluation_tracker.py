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

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import torch
from datasets import Dataset
from fsspec import url_to_fs

from log.info_loggers import (
    DetailsLogger,
    GeneralConfigLogger,
    MetricsLogger,
    TaskConfigLogger,
    VersionsLogger,
)

logger = logging.getLogger(__name__)

# if is_nanotron_available():
#     from nanotron.config import GeneralArgs  # type: ignore


class EnhancedJSONEncoder(json.JSONEncoder):
    """
    Provides a proper json encoding for the loggers and trackers json dumps.
    Notably manages the json encoding of dataclasses.
    """

    def default(self, o):
        if is_dataclass(o):
            try:
                return asdict(o)  # type: ignore
            except Exception:
                return str(o)
        if callable(o):
            if hasattr(o, "__name__"):
                return o.__name__
            # https://stackoverflow.com/questions/20594193/dynamically-created-method-and-decorator-got-error-functools-partial-object-h
            # partial functions don't have __name__ so we have to unwrap the wrapped function
            elif hasattr(o, "func"):
                return o.func.__name__
        if isinstance(o, torch.dtype):
            return str(o)
        if isinstance(o, Enum):
            return o.name
        return super().default(o)


class EvaluationTracker:
    """Keeps track of the overall evaluation process and relevant information.

    The [`~logging.evaluation_tracker.EvaluationTracker`] contains specific loggers for experiments details
    ([`~logging.evaluation_tracker.DetailsLogger`]), metrics ([`~logging.evaluation_tracker.MetricsLogger`]), task versions
    ([`~logging.evaluation_tracker.VersionsLogger`]) as well as for the general configurations of both the
    specific task ([`~logging.evaluation_tracker.TaskConfigLogger`]) and overall evaluation run
    ([`~logging.evaluation_tracker.GeneralConfigLogger`]).  It compiles the data from these loggers and
    writes it to files, which can be published to the Hugging Face hub if
    requested.

    Args:
        output_dir (`str`): Local folder path where you want results to be saved.
        save_details (`bool`, defaults to True): If True, details are saved to the `output_dir`.

    **Attributes**:
        - **details_logger** ([`~logging.info_loggers.DetailsLogger`]) -- Logger for experiment details.
        - **metrics_logger** ([`~logging.info_loggers.MetricsLogger`]) -- Logger for experiment metrics.
        - **versions_logger** ([`~logging.info_loggers.VersionsLogger`]) -- Logger for task versions.
        - **general_config_logger** ([`~logging.info_loggers.GeneralConfigLogger`]) -- Logger for general configuration.
        - **task_config_logger** ([`~logging.info_loggers.TaskConfigLogger`]) -- Logger for task configuration.
    """

    def __init__(
        self,
        output_dir: str,
        save_details: bool = True,
    ) -> None:
        """Creates all the necessary loggers for evaluation tracking."""
        self.details_logger = DetailsLogger()
        self.metrics_logger = MetricsLogger()
        self.versions_logger = VersionsLogger()
        self.general_config_logger = GeneralConfigLogger()
        self.task_config_logger = TaskConfigLogger()

        self.fs, self.output_dir = url_to_fs(output_dir)
        self.should_save_details = save_details

    def save(self) -> None:
        """Saves the experiment information and results to files, and to the hub if requested."""
        logger.info("Saving experiment tracker")
        date_id = datetime.now().isoformat().replace(":", "-")

        # We first prepare data to save
        config_general = asdict(self.general_config_logger)
        # We remove the config from logging, which contains context/accelerator objects
        config_general.pop("config")

        results_dict = {
            "config_general": config_general,
            "results": self.metrics_logger.metric_aggregated,
            "versions": self.versions_logger.versions,
            "config_tasks": self.task_config_logger.tasks_configs,
            "summary_tasks": self.details_logger.compiled_details,
            "summary_general": asdict(self.details_logger.compiled_details_over_all_tasks),
        }

        # Create the details datasets for later upload
        details_datasets: dict[str, Dataset] = {}
        for task_name, task_details in self.details_logger.details.items():
            # Create a dataset from the dictionary - we force cast to str to avoid formatting problems for nested objects
            dataset = Dataset.from_list([{k: str(v) for k, v in asdict(detail).items()} for detail in task_details])

            # We don't keep 'id' around if it's there
            column_names = dataset.column_names
            if "id" in dataset.column_names:
                column_names = [t for t in dataset.column_names if t != "id"]

            # Sort column names to make it easier later
            dataset = dataset.select_columns(sorted(column_names))
            details_datasets[task_name] = dataset

        # We save results at every case
        self.save_results(date_id, results_dict)

        if self.should_save_details:
            self.save_details(date_id, details_datasets)

    def save_results(self, date_id: str, results_dict: dict):
        output_dir_results = Path(self.output_dir) / "results" / self.general_config_logger.model_name
        self.fs.mkdirs(output_dir_results, exist_ok=True)
        output_results_file = output_dir_results / f"results_{date_id}.json"
        logger.info(f"Saving results to {output_results_file}")
        with self.fs.open(output_results_file, "w") as f:
            f.write(json.dumps(results_dict, cls=EnhancedJSONEncoder, indent=2, ensure_ascii=False))

    def save_details(self, date_id: str, details_datasets: dict[str, Dataset]):
        output_dir_details = Path(self.output_dir) / "details" / self.general_config_logger.model_name
        output_dir_details_sub_folder = output_dir_details / date_id
        self.fs.mkdirs(output_dir_details_sub_folder, exist_ok=True)
        logger.info(f"Saving details to {output_dir_details_sub_folder}")
        for task_name, dataset in details_datasets.items():
            output_file_details = output_dir_details_sub_folder / f"details_{task_name}_{date_id}.parquet"
            with self.fs.open(str(output_file_details), "wb") as f:
                dataset.to_parquet(f)

    def generate_final_dict(self) -> dict:
        """Aggregates and returns all the logger's experiment information in a dictionary.

        This function should be used to gather and display said information at the end of an evaluation run.
        """
        to_dump = {
            "config_general": asdict(self.general_config_logger),
            "results": self.metrics_logger.metric_aggregated,
            "versions": self.versions_logger.versions,
            "config_tasks": self.task_config_logger.tasks_configs,
            "summary_tasks": self.details_logger.compiled_details,
            "summary_general": asdict(self.details_logger.compiled_details_over_all_tasks),
        }

        final_dict = {
            k: {eval_name.replace("|", ":"): eval_score for eval_name, eval_score in v.items()}
            for k, v in to_dump.items()
        }

        return final_dict