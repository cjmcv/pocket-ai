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

import logging
import math
from typing import Iterator, Tuple

import torch
from torch.utils.data import Dataset

from tasks.requests import (
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
    Request,
)


logger = logging.getLogger(__name__)


class DynamicBatchDataset(Dataset):
    def __init__(
        self,
        requests: list,
        num_dataset_splits: int,
    ):
        """
        这个类采用动态批处理来加快生成速度。
        每个请求会根据prompt长度与continuation长度之和进行排序。然后将数据集分割为 num_dataset_splits 个部分。
        第一部分将包含长度最长的请求，第二部分将包含长度第二长的请求，依此类推。
        这使得我们能够采用动态批处理，从较小的批量大小开始，对每个部分的批量大小翻倍。与对整个数据集使用固定批量大小相比，这种方式要快得多。
        This dataset class uses dynamic batching to speed up the generation.
        Each request is sorted by the length of the prompt + the length of the
        continuation. Then, the dataset is split into num_dataset_splits splits.
        The first split will contain the longest requests, the second split will
        contain the second longest requests, etc. This allows us to use dynamic
        batching by starting with a small batch size and doubling it for each
        split. This is much faster than using a fixed batch size for the whole
        dataset.

        Args:
            requests (List): A list of requests.
            num_dataset_splits (int): The number of dataset splits.
        """
        # We make sure the requests contain the tokenized versions of their values
        if any(r.tokenized_context is None for r in requests):
            raise ValueError("You passed a request for which tokenization had not happened yet.")

        # sort the requests using the collate function and save the original order
        enumerated_requests = list(enumerate(requests))
        sorted_enumerated_requests = sorted(enumerated_requests, key=lambda x: self._sorting_criteria(x[1]))

        self.sorted_data = [x[1] for x in sorted_enumerated_requests]
        self.original_order = [x[0] for x in sorted_enumerated_requests]

        self.total_size = len(self.sorted_data)

        self.num_dataset_splits, self.splits = self.init_split_limits(num_dataset_splits)

        self.split_start, self.split_end = self.splits[0]

    def init_split_limits(self, num_dataset_splits):
        if num_dataset_splits >= self.total_size:
            logger.warning(
                f"num_dataset_splits ({num_dataset_splits}) >= total_size ({self.total_size}), setting num_dataset_splits to 1"
            )
            num_dataset_splits = 1

        split_size = math.ceil(self.total_size / num_dataset_splits)
        splits_indices = [
            (ix * split_size, min((ix + 1) * split_size, self.total_size)) for ix in range(num_dataset_splits)
        ]

        return num_dataset_splits, splits_indices

    def get_original_order(self, new_arr: list) -> list:
        """
        Get the original order of the data.

        Args:
            newarr (list): Array containing any kind of data that needs to be
                reset in the original order.

        Returns:
            list: new_arr in the original order.
        """
        original_order = [None] * self.total_size

        for original_index, v in zip(self.original_order, new_arr):
            original_order[original_index] = v

        if None in original_order:
            raise RuntimeError(
                f"Some elements of the original order are None, meaning that len(new_arr) ({len(new_arr)}) != len(original_array) ({self.total_size})"
            )

        return original_order

    def get_split_start_end(self, split_id: int) -> Tuple[int, int]:
        """
        Get the start and end indices of a dataset split.

        Args:
            split_id (int): The ID of the split.

        Returns:
            tuple: A tuple containing the start and end indices of the split.
        """
        self.split_start, self.split_end = self.splits[split_id]
        return self.split_start, self.split_end

    def splits_start_end_iterator(self) -> Iterator[Tuple[int, int]]:
        """
        Iterator that yields the start and end indices of each dataset split.
        Also updates the starting batch size for each split (trying to double
        the batch every time we move to a new split).

        Yields:
            tuple: A tuple containing the start and end indices of a split.
        """
        split_range = self.num_dataset_splits
        if self.total_size == 0:
            split_range = 0
        for split_id in range(split_range):
            yield self.get_split_start_end(split_id)

    def __getitem__(self, index) -> Request:
        """
        Get an item from the dataset depending on the split we are currently in.
        For instance, if we are in split 0, we will get the item at index 0, if
        we are in split 1, we will get the item at index self.split_size, etc.
        Used for dynamic batching.

        Args:
            index (int): The index of the item.

        Returns:
            Any: The item at the specified index.
        """
        return self.sorted_data[index + self.split_start]

    def __len__(self) -> int:
        """
        Get the length of current split the dataset.
        All splits have the same length, except the last one which might be
        shorter.

        Returns:
            int: The length of the dataset.
        """
        return self.split_end - self.split_start

    def __iter__(self) -> Iterator[Request]:
        """
        Iterator that yields the items of the dataset depending on the split we
        are currently in. For instance, if we are in split 0, we will get the
        items from index 0 to self.split_size, if we are in split 1, we will get
        the items from index self.split_size to 2 * self.split_size, etc. Used
        for dynamic batching.

        Yields:
            Any: The items of the dataset.
        """
        for i in range(self.split_start, self.split_end):
            yield self.sorted_data[i]

    def _sorting_criteria(self, request) -> int:
        raise NotImplementedError()


class LoglikelihoodDataset(DynamicBatchDataset):
    def _sorting_criteria(self, request: LoglikelihoodRequest | LoglikelihoodRollingRequest) -> int:
        """
        Collates the input data for batching.

        the negative sign on len(toks) sorts descending - this has a few
        advantages:
        - time estimates will always be over not underestimates, which is
        more useful for planning
        - to know the size of a batch when going through the list, you
        know the first one is always the batch padded context length. this
        is useful to simplify the batching logic and more importantly to make
        automatic adaptive batches much much easier to implement
        - any OOMs will happen right away rather than near the end

        Args:
            x (tuple): A tuple containing the input data.

        Returns:
            tuple: A tuple containing the sorted input data.
        """
        toks = request.tokenized_context + request.tokenized_continuation
        return -len(toks)


class LoglikelihoodSingleTokenDataset(DynamicBatchDataset):
    def _sorting_criteria(self, request: LoglikelihoodSingleTokenRequest) -> int:
        """
        Collates the input data for batching.

        the negative sign on len(toks) sorts descending - this has a few # advantages:
        - time estimates will always be over not underestimates, which is
        more useful for planning
        - to know the size of a batch when going through the list, you
        know the first one is always the batch padded context length. this
        is useful to simplify the batching logic and more importantly to make
        automatic adaptive batches much much easier to implement
        - any OOMs will happen right away rather than near the end
        """
        # We take only the prompt, no need for the continuation (since it's a list of single tokens)
        toks = request.tokenized_context
        return -len(toks)


class GenerativeTaskDataset(DynamicBatchDataset):
    def init_split_limits(self, num_dataset_splits):
        """Initialises the split limits based on generation parameters.
        The splits are used to estimate time remaining when evaluating, and in the case of generative evaluations, to group similar samples together.

        For generative tasks, self._sorting_criteria outputs:
        - a boolean (whether the generation task uses logits)
        - a list (the stop sequences)
        - the item length (the actual size sorting factor).

        In the current function, we create evaluation groups by generation parameters (logits and eos), so that samples with similar properties get batched together afterwards.
        The samples will then be further organised by length in each split.

        Args:
            num_dataset_splits (_type_): _description_

        Returns:
            _type_: _description_
        """
        if num_dataset_splits is not None:
            logger.warning(
                "You cannot select the number of dataset splits for a generative evaluation at the moment. Automatically inferring."
            )

        if len(self.sorted_data) > 0:
            all_sorting_criterion = [self._sorting_criteria(self.sorted_data[0])[:-1]]
        splits_indices = [[0, None]]
        for ix, req in enumerate(self.sorted_data):
            current_sorting_criteria = self._sorting_criteria(req)
            current_key = current_sorting_criteria[:-1]
            if current_key not in all_sorting_criterion:
                all_sorting_criterion.append(current_key)
                splits_indices[-1][1] = ix
                splits_indices.append([ix, None])

        # We add the last split
        splits_indices[-1][1] = self.total_size

        num_dataset_splits = len(splits_indices)
        splits_indices = [tuple(e) for e in splits_indices]
        return num_dataset_splits, splits_indices

    def _sorting_criteria(self, request: GreedyUntilRequest) -> tuple[bool, bool, list, int, int]:
        """
        Collate function for generating batches.

        Args:
            x (Any): The input data.

        Returns:
            Any: The collated data.
        """
        toks = request.tokenized_context
        gen_length = request.generation_size
        # The generative task has no limit except the model context
        if gen_length is None:
            gen_length = 0
        return (
            request.do_sample,
            request.use_logits,
            tuple(request.stop_sequence),
            gen_length,
            -(len(toks) + gen_length),
        )

