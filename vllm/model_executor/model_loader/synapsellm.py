from typing import List, Optional

import torch
from torch import nn

from neural_speed import Model

import vllm.envs as envs
from vllm.config import (ModelConfig,
                         SchedulerConfig,
                         CacheConfig,
                         DeviceConfig,
                         PretrainedConfig)
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.platforms import current_platform

logger = init_logger(__name__)

_SYNAPSE_SUPPORTED_MODELS: List [str] = [
    "LlamaForCausalLM",
    "GPTJForCausalLM",
    "PhiForCausalLM",
    "Phi3ForCausalLM",
    "Qwen2ForCausalLM",
    "BertForMaskedLM",
]

def _valid_model_architecture(config: PretrainedConfig) -> None:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch not in _SYNAPSE_SUPPORTED_MODELS:
            raise ValueError(
                f"Model architectures {architectures} are not supported on SynapseLLM "
                f"for now. Supported architectures: "
                f"{list(_SYNAPSE_SUPPORTED_MODELS.keys())}")

# TODO default config func and add synapsellm quant_config in vllm
def _get_model_quant_config():
    quant_kwargs = {}
    quant_kwargs["weight_dtype"] = str(envs.VLLM_SYNAPSELLM_WEIGHT_DTYPE).lower()
    quant_kwargs["use_quant"] = False if quant_kwargs["weight_dtype"] == "auto" else True
    quant_kwargs["scale_dtype"] = str(envs.VLLM_SYNAPSELLM_SCALE_DTYPE).lower()
    quant_kwargs["compute_dtype"] = str(envs.VLLM_SYNAPSELLM_COMPUTE_DTYPE).lower()
    quant_kwargs["group_size"] = int(envs.VLLM_SYNAPSELLM_GROUP_SIZE)
    quant_kwargs["alg"] = str(envs.VLLM_SYNAPSELLM_QUANT_ALGORITHM).lower()

def _get_cache_config(cache_config: CacheConfig, scheduler_config: SchedulerConfig):
    cache_kwargs = {}
    cache_kwargs["memory_dtype"] = cache_config.cache_dtype
    cache_kwargs["n_ctx"] = scheduler_config.max_model_len
    cache_kwargs["n_stream"] = scheduler_config.max_num_seqs

    return cache_kwargs

def _get_execute_config(scheduler_config: SchedulerConfig):
    execute_kwargs = {}
    execute_kwargs["threads"] = int(envs.VLLM_SYNAPSELLM_NUM_THREADS)
    execute_kwargs["n_chunk"] = scheduler_config.max_num_batched_tokens

    return execute_kwargs

class SynapseLLMCausalLM(nn.Module):

    def __init__(
        self,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> None:
        super().__init__()
        _valid_model_architecture(model_config.hf_config)

        self.logits_processor = LogitsProcessor(
            model_config.hf_config.vocab_size, logits_as_input=True)
        self.sampler = Sampler()

        # args for SynapseLLM model creation
        assert device_config.device_type == "synapsellm"
        self.device = "cpu" if current_platform.is_synapsellm_cpu() else "hpu"
        self.interm_dtype = "f16"
        self.cache_dir = "synapsellm_executable_model"
        self.delete_after_load = False

        self.model = Model(model_config.model,
                      device=self.device,
                      interm_dtype=self.interm_dtype,
                      cache_dir=self.cache_dir,
                      delete_after_load=self.delete_after_load,
                      )

    def init_model(self, **kwargs) -> None:
        logger.debug(f"SynapseLLM model init: {kwargs}")
        self.model.init(**kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        input_block_ids: torch.Tensor = None,
    ) -> torch.Tensor:

        logits = self.model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            stream_ids=input_block_ids)

        return logits

    # kv cache operations
    def get_running_kv_cache_block_ids(self) -> torch.Tensor:
        return self.model.get_streams()

    def free_block_id_kv_cache(self, block_id) -> None:
        self.model.free_stream(block_id)

    def free_block_ids_kv_cache(self, block_ids) -> None:
        self.model.free_stream(block_ids)

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(None, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens


def get_model(
    model_config: ModelConfig,
    scheduler_config: SchedulerConfig,
    cache_config: CacheConfig,
    device_config: DeviceConfig,
    **kwargs,
) -> torch.nn.Module:

    lora_config = kwargs.get("lora_config")
    if lora_config:
        raise ValueError(
            "SynapseLLM backend does not support LoRA, "
            "but LoRA is enabled.")

    speculative_config = kwargs.get("speculative_config")
    if speculative_config:
        raise ValueError(
            "SynapseLLM backend does not support speculative decoding, "
            "but speculative decoding is enabled.")

    if scheduler_config.enable_chunked_prefill:
        raise ValueError(
            "SynapseLLM backend does not support chunked_prefill, "
            "but it is enabled."
        )

    # create model
    synapsellm_model = SynapseLLMCausalLM(model_config, device_config) 

    # create config
    quant_kwargs = _get_model_quant_config()
    cache_kwargs = _get_cache_config(cache_config, scheduler_config)
    execute_kwargs = _get_execute_config(scheduler_config)

    # init model
    synapsellm_model.init_model(**quant_kwargs, **cache_kwargs, **execute_kwargs)

    return synapsellm_model
