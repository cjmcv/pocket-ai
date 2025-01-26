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

import sys
import gc
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Optional, Union
from transformers import AutoConfig
import torch
from tqdm import tqdm
import requests

from data import GenerativeTaskDataset, LoglikelihoodDataset

from transformers import BatchEncoding, PreTrainedTokenizerBase

from tasks.requests import (
    GreedyUntilMultiTurnRequest,
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
    RequestType,
)
from utils import (
    NO_VLLM_ERROR_MSG,
    is_vllm_available,
)

from utils import EnvConfig, as_list

###################
import time
import openai
from concurrent.futures import ThreadPoolExecutor
###################

logger = logging.getLogger(__name__)

@dataclass
class ModelResponse:
    result: Union[tuple, list, str]
    input_tokens: list[int] = field(default_factory=list)  # model inputs
    generated_tokens: list[int] = field(default_factory=list)  # model generations
    truncated_tokens_count: Optional[int] = 0  # How many tokens truncated
    padded_tokens_count: Optional[int] = 0  # How many tokens of padding

    def get_result_for_eval(self):
        raise NotImplementedError()

@dataclass
class LoglikelihoodResponse(ModelResponse):
    # Float: Total log prob of the continuation
    # Optional(Bool): Whether the continuation is greedy (= all the tokens in the continuation are argmax of prob)
    result: Union[tuple[float, bool], float] = field(default_factory=tuple[float, bool])

    def get_result_for_eval(self):
        return self.result

@dataclass
class LoglikelihoodSingleTokenResponse(ModelResponse):
    # Log probs of the various single token options
    result: list[float] = field(default_factory=list)

    def get_result_for_eval(self):
        return self.result

@dataclass
class GenerativeResponse(ModelResponse):
    result: list[str] = field(default_factory=str)  # generated text continuation
    logits: Optional[list[float]] = None  # Generated text logits

    def get_result_for_eval(self):
        return self.result if self.logits is None else (self.result, self.logits)

@dataclass
class GenerativeMultiturnResponse(ModelResponse):
    result: list[str] = field(default_factory=list)

    def get_result_for_eval(self):
        return self.result
    
def _get_dtype(dtype: Union[str, torch.dtype, None], config: Optional[AutoConfig] = None) -> Optional[torch.dtype]:
    """
    Get the torch dtype based on the input arguments.

    Args:
        dtype (Union[str, torch.dtype]): The desired dtype. Can be a string or a torch dtype.
        config (Optional[transformers.AutoConfig]): The model config object. Defaults to None.

    Returns:
        torch.dtype: The torch dtype based on the input arguments.
    """

    if config is not None and hasattr(config, "quantization_config"):
        # must be infered
        return None

    if dtype is not None:
        if isinstance(dtype, str) and dtype not in ["auto", "4bit", "8bit"]:
            # Convert `str` args torch dtype: `float16` -> `torch.float16`
            return getattr(torch, dtype)
        elif isinstance(dtype, torch.dtype):
            return dtype

    if config is not None:
        return config.torch_dtype

    return None


def _simplify_name(name_or_path: str) -> str:
    """
    If the model is loaded from disk, then the name will have the following format:
    /p/a/t/h/models--org--model_name/revision/model_files
    This function return the model_name as if it was loaded from the hub:
    org/model_name

    Args:
        name_or_path (str): The name or path to be simplified.

    Returns:
        str: The simplified name.
    """
    if os.path.isdir(name_or_path) or os.path.isfile(name_or_path):  # Loading from disk
        simple_name_list = name_or_path.split("/")
        # The following manages files stored on disk, loaded with the hub model format:
        # /p/a/t/h/models--org--model_name/revision/model_files
        if any("models--" in item for item in simple_name_list):  # Hub format
            simple_name = [item for item in simple_name_list if "models--" in item][0]
            simple_name = simple_name.replace("models--", "").replace("--", "/")
            return simple_name
        # This is for custom folders
        else:  # Just a folder, we don't know the shape
            return name_or_path.replace("/", "_")

    return name_or_path

@dataclass
class GenerationParameters:
    early_stopping: Optional[bool] = None  # vllm, transformers
    repetition_penalty: Optional[float] = None  # vllm, transformers, tgi
    frequency_penalty: Optional[float] = None  # vllm, tgi
    length_penalty: Optional[float] = None  # vllm, transformers
    presence_penalty: Optional[float] = None  # vllm

    max_new_tokens: Optional[int] = None  # vllm, transformers, tgi
    min_new_tokens: Optional[int] = None  # vllm, transformers

    seed: Optional[int] = None  # vllm, tgi
    stop_tokens: Optional[list[str]] = None  # vllm, transformers, tgi
    temperature: Optional[float] = None  # vllm, transformers, tgi
    top_k: Optional[int] = None  # vllm, transformers, tgi
    min_p: Optional[float] = None  # vllm, transformers
    top_p: Optional[int] = None  # vllm, transformers, tgi
    truncate_prompt: Optional[bool] = None  # vllm, tgi

    def to_vllm_openai_dict(self) -> dict:
        """Selects relevant generation and sampling parameters for vllm and openai models.
        Doc: https://docs.vllm.ai/en/v0.5.5/dev/sampling_params.html

        Returns:
            dict: The parameters to create a vllm.SamplingParams or just provide OpenAI params as such in the model config.
        """
        # Task specific sampling params to set in model: n, best_of, use_beam_search
        # Generation specific params to set in model: logprobs, prompt_logprobs
        return {k: v for k, v in asdict(self).items() if v is not None}


if is_vllm_available() and not eval(os.environ.get('USING_API_SERVER')):
    from vllm import LLM, SamplingParams
    from vllm.transformers_utils.tokenizer import get_tokenizer
    from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    logging.getLogger("vllm").propagate = True
    logging.getLogger("vllm").handlers.clear()
else:
    from transformers import AutoTokenizer
    LLM = None
    SamplingParams = None
    get_tokenizer = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"

STARTING_BATCH_SIZE = 512

TokenSequence = Union[list[int], torch.LongTensor, torch.Tensor, BatchEncoding]

@dataclass
class ModelInfo:
    model_name: str
    model_sha: Optional[str] = None
    model_dtype: Optional[str] = None
    model_size: Optional[str] = None

@dataclass
class ModelConfig:
    backend: str
    pretrained: str
    gpu_memory_utilisation: float = 0.9  # lower this if you are running out of memory
    revision: str = "main"  # revision of the model
    dtype: str | None = None
    tensor_parallel_size: int = 1  # how many GPUs to use for tensor parallelism
    pipeline_parallel_size: int = 1  # how many GPUs to use for pipeline parallelism
    max_model_length: int = 4096 # | None = None  # maximum length of the model, ussually infered automatically. reduce this if you encouter OOM issues, 4096 is usually enough
    swap_space: int = 4  # CPU swap space size (GiB) per GPU.
    seed: int = 1234
    trust_remote_code: bool = False
    use_chat_template: bool = False
    add_special_tokens: bool = True
    pairwise_tokenization: bool = False  # whether to tokenize the context and continuation separately or together.
    subfolder: Optional[str] = None

class LightevalModel():
    DATASET_SPLITS = 4

    def __init__(
        self,
        config: ModelConfig,
        env_config: EnvConfig,
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation."""
        self._config = config
        self.use_chat_template = config.use_chat_template

        self._add_special_tokens = config.add_special_tokens if config.add_special_tokens is not None else False
        self._tokenizer = self._create_auto_tokenizer(config, env_config)

        self._max_length = int(config.max_model_length) if config.max_model_length is not None else None

        self.generation_parameters = GenerationParameters() # TODO

        is_api_server = eval(os.environ.get('USING_API_SERVER'))
        if is_api_server:
            port = {
                "sglang": 30000,
                "lmdeploy": 23333,
                "vllm": 8000,
            }.get(config.backend, 30000)

            self.API_MAX_RETRY = 5
            self.API_RETRY_SLEEP = 3
            self.API_RETRY_MULTIPLIER = 2
            self.CONCURENT_CALLS = 4

            base_url=f"http://127.0.0.1:{port}"
            model_url = f"{base_url}/v1/models"
            
            # Get model name
            try:
                response = requests.get(model_url)
                model_list = response.json().get("data", [])
                self.model_api_id = model_list[0]["id"] if model_list else None
            except Exception as e:
                print(f"Failed to fetch model from {model_url}. Error: {e}")
                print(
                    "Please specify the correct host and port using `--host` and `--port`."
                )
                sys.exit(1)
        
            print("Connect to ", base_url)
            print("Model ", self.model_api_id)
            self.client = openai.Client(base_url=base_url+"/v1", api_key="EMPTY")
            self.sampling_params = self.generation_parameters.to_vllm_openai_dict()
            self.model = None
        else:
            if not is_vllm_available():
                raise ImportError(NO_VLLM_ERROR_MSG)
            # If model_parallel is not set we compare the number of processes with the number of GPUs
            self.model = self._create_auto_model(config, env_config)
            self.sampling_params = SamplingParams(**self.generation_parameters.to_vllm_openai_dict())
            self.generate_step = 10 # 10个req一起计算

        self.model_name = _simplify_name(config.pretrained)
        self.model_sha = ""  # config.get_model_sha()
        self.precision = _get_dtype(config.dtype, config=self._config)

        self.model_info = ModelInfo(model_name=self.model_name, model_sha=self.model_sha)
        self.pairwise_tokenization = config.pairwise_tokenization

    def get_method_from_request_type(self, request_type: RequestType):
        if request_type == RequestType.LOGLIKELIHOOD:
            return self.loglikelihood
        if request_type == RequestType.LOGLIKELIHOOD_SINGLE_TOKEN:
            return self.loglikelihood_single_token
        if request_type == RequestType.LOGLIKELIHOOD_ROLLING:
            return self.loglikelihood_rolling
        if request_type == RequestType.GREEDY_UNTIL:
            return self.greedy_until
        if request_type == RequestType.GREEDY_UNTIL_MULTI_TURN:
            return self.greedy_until_multi_turn
        raise NotImplementedError(f"Request type {request_type} not supported")
    
    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    # Tokenization utils
    def tok_encode(self, str_to_encode: str | list[str], add_special_tokens: Optional[bool] = None) -> TokenSequence:
        if add_special_tokens is None:
            add_special_tokens = self.add_special_tokens
        if isinstance(str_to_encode, str):
            return self.tokenizer.encode(str_to_encode, add_special_tokens=add_special_tokens)
        return self.tokenizer(
            str_to_encode,
            padding=True,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )

    # 通过处理两者之间的空格，对上下文与后续内容的组合进行编码。
    def tok_encode_pair(self, context, continuation, pairwise: bool = False):
        """Encodes a context, continuation pair by taking care of the spaces in between.
        Args:
            context (str): The context string to be encoded.
            continuation (str): The continuation string to be encoded.
            pairwise (bool):
                If True, encode context and continuation separately.
                If False, encode them together and then split.

        Returns:
            Tuple[TokenSequence, TokenSequence]: A tuple containing the encoded context and continuation.

        The advantage of pairwise is:
        1) It better aligns with how LLM predicts tokens
        2) Works in case len(tok(context,cont)) != len(tok(context)) + len(tok(continuation)).
        E.g this can happen for chinese if no space is used between context/continuation
        """

        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        if pairwise:
            # We don't add special tokens to the continuation as if bos is added
            # models tend to to completely ignore a context
            context_enc, continuation_enc = (
                self.tok_encode(context, add_special_tokens=self.add_special_tokens),
                self.tok_encode(continuation, add_special_tokens=False),
            )

            # In theory the context_enc can be ended with eos token, this would again
            # cause the model to ignore the context. We thus strip the eos token from context_enc
            if len(context_enc) > 0 and context_enc[-1] == self.tokenizer.eos_token_id:
                context_enc = context_enc[:-1]

            return context_enc, continuation_enc

        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        context_enc_len = len(context_enc)
        # In case continuation tokens merge with context tokens we use the merged token as continuation
        if len(context_enc) == len(whole_enc):
            context_enc_len = len(context_enc) - 1
            context_enc = whole_enc[:context_enc_len]

        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def tok_decode(self, tokens: torch.LongTensor) -> list[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
    
    def cleanup(self):
        if self.model is not None:
            destroy_model_parallel()
            del self.model.llm_engine.model_executor.driver_worker
            destroy_distributed_environment()

        self.model = None
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def add_special_tokens(self):
        return self._add_special_tokens

    @property
    def max_length(self) -> int:
        return self._max_length

    def _create_auto_model(self, config: ModelConfig, env_config: EnvConfig) -> Optional[LLM]:
        """
        Creates an instance of the pretrained HF model.

        Args:
            pretrained (str): The name or path of the pretrained model.
            revision (str): The revision of the model.
            subfolder (Optional[str], optional): The subfolder within the model. Defaults to None.
            max_memory (Optional[dict], optional): The maximum memory to allocate for the model per GPU. Defaults to None.
            device_map (Optional[dict], optional): The device mapping for the model. Defaults to None.
            torch_dtype (Optional[Union[str, torch.dtype]], optional): The torch data type for the model. Defaults to None.
            quantization_config (Optional[Union[BitsAndBytesConfig, GPTQConfig]], optional): The quantization configuration for the model. Defaults to None.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
            cache_dir (str, optional): The cache directory for the model. Defaults to "/scratch".

        Returns:
            transformers.PreTrainedModel: The created auto model instance.
        """
        self.model_args = {
            "model": config.pretrained,
            "gpu_memory_utilization": float(config.gpu_memory_utilisation),
            "revision": config.revision + (f"/{config.subfolder}" if config.subfolder is not None else ""),
            "dtype": config.dtype,
            "trust_remote_code": config.trust_remote_code,
            "tensor_parallel_size": int(config.tensor_parallel_size),
            "pipeline_parallel_size": int(config.pipeline_parallel_size),
            "max_model_len": self._max_length,
            "swap_space": 4,
            "seed": 1234,
        }

        model = LLM(**self.model_args)

        # If the max_length can't get extracted from the config, it will be inferred from the model
        # Inferring from the tokenizer will cause vllm to bug for models with mismatches between model
        # config and tk config, like mistralai/Mistral-7B-v0.1
        if self._max_length is None:
            self._max_length = model.llm_engine.model_config.max_seq_len_to_capture

        return model

    def _create_auto_tokenizer(self, config: ModelConfig, env_config: EnvConfig):
        if get_tokenizer is not None:
            tokenizer = get_tokenizer(
                config.pretrained,
                tokenizer_mode="auto",
                trust_remote_code=config.trust_remote_code,
                tokenizer_revision=config.revision,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                config.pretrained,
                tokenizer_mode="auto",
                trust_remote_code=config.trust_remote_code,
                tokenizer_revision=config.revision,
            )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def greedy_until_multi_turn(  # noqa: C901
        self, requests: list[GreedyUntilMultiTurnRequest], override_bs: Optional[int] = None
    ) -> GenerativeMultiturnResponse:
        """Generates responses using a greedy decoding strategy until certain ending conditions are met."""
        pass
    
    # 用于生成式任务，上游调用可搜 get_method_from_request_type 和 run_model
    def greedy_until(
        self,
        requests: list[GreedyUntilRequest],
        override_bs: Optional[int] = None,
    ) -> list[GenerativeResponse]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            requests (list[Request]): list of requests containing the context and ending conditions.
            override_bs (int, optional): Override the batch size for generation. Defaults to None.

        Returns:
            list[GenerateReturn]: list of generated responses.
        """
        for request in requests:
            request.stop_sequence = as_list(request.stop_sequence) + [self.tokenizer.eos_token] # not for api_server
            request.tokenized_context = self.tok_encode(request.context)

        dataset = GenerativeTaskDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
        results = []

        is_api_server = eval(os.environ.get('USING_API_SERVER'))
        if is_api_server:
            for _ in tqdm(
                dataset.splits_start_end_iterator(),
                total=dataset.num_dataset_splits,
                desc="Splits",
                position=0,
                disable=False,  # self.disable_tqdm,
            ):
                max_new_tokens = dataset[0].generation_size  # could be none
                return_logits = dataset[0].use_logits
                num_samples = dataset[0].num_samples
                contexts = [c.context for c in dataset]

                responses = self.__call_api_parallel(contexts, return_logits, max_new_tokens, num_samples)

                for response in responses:
                    result: list[str] = [output.message.content for output in response.choices]

                    cur_response = GenerativeResponse(
                        result=result,
                        logits=None,
                        generated_tokens=[],
                        input_tokens=[],
                    )
                    results.append(cur_response)
        else:
            for _ in tqdm(
                dataset.splits_start_end_iterator(),
                total=dataset.num_dataset_splits,
                desc="Splits",
                position=0,
                disable=False,  # self.disable_tqdm,
            ):
                # For chat models, generation stops with EOS token, so we don't need to specify stop tokens
                if self.use_chat_template:
                    stop_tokens = []
                else:
                    # NOTE: we are assuming all items in a batch behave similarly (same
                    # stop_tokens and max_tokens genrated) which is not necessarily
                    # the case! Because of that we only use batch size of 1
                    stop_tokens = dataset[0].stop_sequence

                max_new_tokens = dataset[0].generation_size  # could be none
                returns_logits = dataset[0].use_logits
                num_samples = dataset[0].num_samples

                context = [c.context for c in dataset]
                tokenized = self.tokenizer(context, add_special_tokens=self.add_special_tokens)

                # The main question for this step is the following:
                # Would we rather truncate the prompt to allow generation to go to max_new_tokens, at the risk
                # of losing some meaning, or have some generations that are exceedingly short?
                # The choice we go for here is to avoid truncating the prompt if we can, since it
                # should have been managed by the prompt creator/few shot manager if requested by the user.
                inputs = tokenized["input_ids"]
                context_size = len(inputs[0])

                # left truncate the inputs to the maximum length
                if max_new_tokens is not None:
                    if context_size + max_new_tokens > self.max_length:
                        logger.warning(
                            f"{context_size + max_new_tokens=} which is greather than {self.max_length=}. Truncating context to {self.max_length - max_new_tokens} tokens."
                        )
                        context_size = self.max_length - max_new_tokens
                        inputs = [input[-context_size:] for input in inputs]
                else:
                    if context_size > self.max_length:
                        logger.warning(
                            f"{context_size=} which is greather than {self.max_length=}. Truncating context to {self.max_length} tokens."
                        )
                        context_size = self.max_length
                        inputs = [input[-context_size:] for input in inputs]

                vllm_outputs = self._generate(
                    inputs=inputs,
                    max_new_tokens=max_new_tokens,
                    stop_tokens=stop_tokens,
                    returns_logits=returns_logits,
                    num_samples=num_samples,
                )

                for vllm_output in vllm_outputs:
                    output_token_ids = [outputs.token_ids for outputs in vllm_output.outputs]
                    logprobs = [output.logprobs for output in vllm_output.outputs] or []
                    logprobs = [logprob[token_id].logprob for token_id, logprob in zip(output_token_ids[0], logprobs[0])]
                    result = [output.text for output in vllm_output.outputs]
                    input_token_ids = vllm_output.prompt_token_ids

                    cur_response = GenerativeResponse(
                        result=result,
                        logits=logprobs,
                        generated_tokens=list(output_token_ids),
                        input_tokens=input_token_ids,
                    )
                    results.append(cur_response)

        return dataset.get_original_order(results)

    def _generate(
        self,
        inputs: list[list[int]],
        max_new_tokens: Optional[int] = None,
        stop_tokens: Optional[list[str]] = None,
        returns_logits: Optional[bool] = False,
        num_samples: int = 1,
        generate: bool = True,
    ) -> list[GenerativeResponse]:
        """Contains the actual logic of the generation."""
        sampling_params = self.sampling_params.clone() or SamplingParams()
        if generate:
            sampling_params.n = num_samples
            sampling_params.max_tokens = max_new_tokens
            sampling_params.stop = stop_tokens
            sampling_params.logprobs = 1 if returns_logits else 0
        else:
            sampling_params.temperature = 0
            sampling_params.prompt_logprobs = 1
            sampling_params.max_tokens = 1
            sampling_params.detokenize = False

        print("\n\ninputs total len: ", len(inputs))
        chunks = [inputs[i:i + self.generate_step] for i in range(0, len(inputs), self.generate_step)]

        outputs = []
        for single_chunk in chunks:
            print("<{}> ".format(chunks.index(single_chunk) * self.generate_step), end='')

            chunk_output = self.model.generate(
                prompt_token_ids=single_chunk,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            outputs.extend(chunk_output)

        # outputs = self.model.generate(
        #     prompt_token_ids=inputs,
        #     sampling_params=sampling_params,
        #     use_tqdm=True,
        # )
        return outputs

    def loglikelihood(
        self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        for request in requests:
            if request.context == "":
                request.tokenized_context = [self.tokenizer.eos_token_id]
                request.tokenized_continuation = self.tok_encode(request.choice)
            else:
                # The following line is mandatory for compatibility with the harness
                # tokenized_context是原来输入问题的内容，经过tokenize后的id序列
                # tokenized_continuation是后续拼接到输入问题内容上的部分，如对于abcd4项选择题，生成请求时会将一个问题生成4个请求。
                #    如问题：1+1等于几？选项：a/b/c/d，生成请求1）1+1等于几？a；2）1+1等于几？b；3）1+1等于几？c；4）1+1等于几？d； 
                #           以限制输出范围，便于与固定答案进行对比评估。详情搜 def construct_requests / class Request
                #    则tokenized_context对应“1+1等于几？”，四个请求都一样。tokenized_continuation 分别是a/b/c/d。
                request.tokenized_context, request.tokenized_continuation = self.tok_encode_pair(
                    request.context, request.choice, pairwise=self.pairwise_tokenization
                )
        return self._loglikelihood_tokens(requests, override_bs=override_bs)
    
    ###################################################################################

    def __call_api(self, prompt, return_logits, max_new_tokens, num_samples, logit_bias):
        # API_MAX_RETRY 是失败重新尝试的次数
        for _ in range(self.API_MAX_RETRY):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_api_id,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    response_format={"type": "text"},
                    max_tokens=max_new_tokens if max_new_tokens is not None and max_new_tokens > 0 else None,
                    logprobs=return_logits,
                    logit_bias=logit_bias,
                    n=num_samples,
                    **self.sampling_params,
                )
                return response
            except Exception as e:
                logger.warning(f"{type(e), e}")
                time.sleep(self.API_RETRY_SLEEP)
                self.API_RETRY_SLEEP = self.API_RETRY_SLEEP**self.API_RETRY_MULTIPLIER
        raise Exception("Failed to get response from the API")

    def __call_api_parallel(
        self,
        prompts,
        return_logits: bool | list[bool],
        max_new_tokens: int | list[int],
        num_samples: int | list[int],
        logit_bias: list[dict[int, float]] | None = None,
    ):
        results = []

        return_logitss = [return_logits for _ in prompts] if not isinstance(return_logits, list) else return_logits
        max_new_tokenss = [max_new_tokens for _ in prompts] if not isinstance(max_new_tokens, list) else max_new_tokens
        num_sampless = [num_samples for _ in prompts] if not isinstance(num_samples, list) else num_samples
        logit_biass = [logit_bias for _ in prompts] if logit_bias is None else logit_bias

        assert (
            len(prompts) == len(return_logitss) == len(max_new_tokenss) == len(num_sampless) == len(logit_biass)
        ), "Length of prompts, return_logitss, max_new_tokenss, num_sampless, logit_biass should be same"

        # 创建一个线程池，池内有CONCURENT_CALLS个线程，可以并发处理数据。
        # executor.map() 将多个可迭代对象(prompts/return_logitss等)作为参数，将这些可迭代对象的元素作为参数依次送入到__call_api中计算。
        # 同时executor.map() 会按照输入的顺序来收集结果，确保结果的顺序与输入的顺序一致。
        # 因为它会等待每个线程完成任务，并将结果按照输入的顺序依次返回，即使某些线程可能会比其他线程完成得更晚。
        with ThreadPoolExecutor(self.CONCURENT_CALLS) as executor:
            for entry in tqdm(
                executor.map(self.__call_api, prompts, return_logitss, max_new_tokenss, num_sampless, logit_biass),
                total=len(prompts),
            ):
                results.append(entry)

        if None in results:
            raise ValueError("Some entries are not annotated due to errors in annotate_p, please inspect and retry.")

        return results
    #######################################################################################

    def _loglikelihood_tokens(
        self,
        requests: list[LoglikelihoodRequest],
        override_bs: int = -1,
        return_bool_score: bool = True,
        rolling: bool = False,
    ) -> list[LoglikelihoodResponse]:
        # LoglikelihoodDataset是DynamicBatchDataset的派生类
        # 在LoglikelihoodDataset中会对数据顺序进行调整，以加快生成速度。用self.original_order来维护原始顺序。
        dataset = LoglikelihoodDataset(requests=requests, num_dataset_splits=1)
        res = []

        is_api_server = eval(os.environ.get('USING_API_SERVER'))
        if is_api_server:
            for _ in tqdm(dataset.splits_start_end_iterator()):
                inputs = [dataset[i].context for i in range(len(dataset))]
                logit_biass = []
                max_new_tokens = [len(dataset[i].tokenized_continuation) for i in range(len(dataset))]

                assert all(
                    new_tokens == 1 for new_tokens in max_new_tokens
                ), "Only single token continuations are supported when using openai API."

                for i in range(len(dataset)):
                    logit_bias = {tok: 100 for tok in dataset[i].tokenized_continuation}
                    logit_biass.append(logit_bias)

                outputs = self.__call_api_parallel(
                    inputs, return_logits=True, max_new_tokens=max_new_tokens, num_samples=1, logit_bias=logit_biass
                )

                for output, input in zip(outputs, dataset):
                    continuation_logprobs = [content.logprob for content in output.choices[0].logprobs.content]
                    answer = LoglikelihoodResponse(
                        input_tokens=input.tokenized_context + input.tokenized_continuation,
                        generated_tokens=input.tokenized_continuation,
                        result=(sum(continuation_logprobs), None),
                    )
                    res.append(answer)
        else:
            for _ in tqdm(dataset.splits_start_end_iterator()):
                # the last token is an eos token, so we don't need to add it
                inputs = [dataset[i].tokenized_context + dataset[i].tokenized_continuation for i in range(len(dataset))]
                # Left truncate the inputs to the maximum length
                inputs = [input[-self.max_length :] for input in inputs]
                outputs = self._generate(inputs, generate=False)

                for output, input in zip(outputs, dataset):
                    continuation_logprobs = []
                    for token, logprobs in zip(input.tokenized_continuation[::-1], output.prompt_logprobs[::-1]):
                        continuation_logprobs.append(logprobs[token])
                    bool_score = all(logprob.rank == 1 for logprob in continuation_logprobs)
                    continuation_logprobs = [logprob.logprob for logprob in continuation_logprobs]
                    answer = LoglikelihoodResponse(
                        input_tokens=input.tokenized_context + input.tokenized_continuation,
                        generated_tokens=input.tokenized_continuation,
                        result=(sum(continuation_logprobs), bool_score if return_bool_score else None),
                    )
                    res.append(answer)

        # 计算结束后，根据self.original_order把顺序调整回去。
        return dataset.get_original_order(res)

    def loglikelihood_rolling(
        self, requests: list[LoglikelihoodRollingRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        pass
    
    def loglikelihood_single_token(
        self, requests: list[LoglikelihoodSingleTokenRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodSingleTokenResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        pass