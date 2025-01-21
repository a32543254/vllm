import os
import itertools
from dataclasses import dataclass
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Callable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.synapsellm import get_model
from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensorInputs,
                             MultiModalKwargs)
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata
from vllm.utils import is_pin_memory_available, make_tensor_with_pad
from vllm.worker.model_runner_base import ModelRunnerBase, ModelRunnerInputBase

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)

_SAMPLING_EPS = 1e-5

@dataclass(frozen=True)
class ModelInputForSynapseLLM(ModelRunnerInputBase):
    """
    Used by the SynapseLLMModelRunner.
    """
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    token_type_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    input_block_ids: Optional[torch.Tensor] = None,
    sampling_metadata: Optional["SamplingMetadata"] = None
    multi_modal_kwargs: Optional[BatchedTensorInputs] = None
    is_prompt: Optional[str] = True
    kv_cache_block_ids_freed: Optional[torch.Tensor] = None
    logits_all: Optional[bool] = False
    async_callback: Optional[Callable] = None

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        raise NotImplementedError("ModelInputForSynapseLLM cannot be broadcast.")

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "ModelInputForSynapseLLM":
        assert attn_backend is None, "SynapseLLM backend does not upport PagedAttention."
        return cls.from_broadcasted_tensor_dict(tensor_dict)


class SynapseLLMModelRunner(ModelRunnerBase[ModelInputForSynapseLLM]):

    def __init__(
        self,
        vllm_config: VllmConfig,
    ):

        ModelRunnerBase.__init__(self, vllm_config)
        model_config = self.model_config
        if model_config is not None and model_config.get_sliding_window():
            logger.warning("Sliding window is not supported on SynapseLLM. "
                           "The model will run without sliding window.")
        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.multi_modal_input_mapper = self.mm_registry \
            .create_input_mapper(self.model_config)

        # Lazy initialization.
        self.model: nn.Module  # initialize after load_model.

        # Once SYNAPSELLM_ON_DEVICE_SAMPLING_DISABLED is set to a non-zero value,
        # turn off on-device sampling.
        self._on_device_sampling_disabled = int(
            os.getenv("SYNAPSELLM_ON_DEVICE_SAMPLING_DISABLED", "0")) > 0

        # SynapseLLM needs to update sampling parameters when request IDs change
        # across batches. This variable stores the previous batch's request IDs
        # to determine if an update is needed.
        self._previous_batch_request_ids: List[str] = []

        self.on_device_sampling_params = None
        if not self._on_device_sampling_disabled:
            logger.warning(
                "On-device sampling is turned on in SynapseLLM by default, only "
                "top_k, and temperature are current supported sampling "
                "parameters. To turn off the on-device sampling, please set "
                "the environment variable SYNAPSELLM_ON_DEVICE_SAMPLING_DISABLED=1."
            )
            self.on_device_sampling_params = {"max_new_tokens": 1,
                                              "top_k": 1,
                                              "temperature": 1.0,
                                              "ignore_eos": False,
                                              }

    def load_model(self) -> None:
        if find_spec("neural_speed") is not None:
            self.model = get_model(
                self.model_config,
                self.scheduler_config,
                self.cache_config,
                self.device_config,
                on_device_sampling_disabled=self._on_device_sampling_disabled,
            )
        else:
            raise NotImplementedError(
                "Supports only Neural-Speed based models.")

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int],
               BatchedTensorInputs]:

        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        token_type_ids = None
        attention_mask: List[List[int]] = []
        input_block_ids: List[int] = []
        kv_cache_block_ids_freed: List[int] = []

        seq_lens: List[int] = []
        multi_modal_kwargs_list: List[MultiModalKwargs] = []
        occupied_block_ids = []
        if self.model is not None:
            occupied_block_ids = self.model.get_occupied_kv_cache_block_ids()

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            seq_len = len(prompt_tokens)
            seq_lens.append(seq_len)

            input_tokens.append(prompt_tokens)
            input_positions.append(list(range(seq_len)))
            attention_mask.append([1]*seq_len)

            assert seq_group_metadata.block_tables is not None
            block_table = seq_group_metadata.block_tables[seq_id]
            assert len(block_table) == 1
            input_block_ids.append(block_table[0])
            if block_table[0] in occupied_block_ids:
                kv_cache_block_ids_freed.append(block_table[0])

            mm_data = seq_group_metadata.multi_modal_data
            if mm_data:
                if self.mm_registry.has_processor(self.model_config):
                    mm_kwargs = mm_data
                else:
                    mm_kwargs = self.multi_modal_input_mapper(
                        mm_data,
                        seq_group_metadata.mm_processor_kwargs,
                    )

                multi_modal_kwargs_list.append(mm_kwargs)

        max_seq_len = max(seq_lens)
        assert max_seq_len > 0
        input_tokens = make_tensor_with_pad(input_tokens,
                                            pad=0,
                                            max_len=max_seq_len,
                                            dtype=torch.long,
                                            device=self.device)
        input_positions = make_tensor_with_pad(input_positions,
                                               pad=0,
                                               max_len=max_seq_len,
                                               dtype=torch.long,
                                               device=self.device)
        attention_mask = make_tensor_with_pad(attention_mask,
                                              pad=0,
                                              max_len=max_seq_len,
                                              dtype=torch.long,
                                              device=self.device)
        input_block_ids = torch.tensor(input_block_ids,
                                       dtype=torch.long,
                                       device=self.device)

        multi_modal_kwargs = MultiModalKwargs.batch(multi_modal_kwargs_list)

        if len(kv_cache_block_ids_freed) == 0:
            kv_cache_block_ids_freed = None

        return (input_tokens, input_positions, token_type_ids, attention_mask, input_block_ids,
                seq_lens, multi_modal_kwargs, kv_cache_block_ids_freed)

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        token_type_ids = None
        attention_mask = None
        input_block_ids: List[int] = []
        context_lens: List[int] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt

            seq_ids = list(seq_group_metadata.seq_data.keys())

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append([generation_token])

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append([position])
                context_lens.append(seq_len)

                assert seq_group_metadata.block_tables is not None
                block_table = seq_group_metadata.block_tables[seq_id]
                assert len(block_table) == 1
                input_block_ids.append(block_table[0])

        if self._on_device_sampling_disabled:
            input_tokens = make_tensor_with_pad(input_tokens,
                                                pad=0,
                                                max_len=1,
                                                dtype=torch.long,
                                                device=self.device)
        else:
            # synapsellm sampling will hanld next_tokens by itself
            input_tokens = None
        input_positions = make_tensor_with_pad(input_positions,
                                               pad=0,
                                               max_len=1,
                                               dtype=torch.long,
                                               device=self.device)
        context_lens = torch.tensor(context_lens,
                                    dtype=torch.int,
                                    device=self.device)
        input_block_ids = torch.tensor(input_block_ids,
                                       dtype=torch.long,
                                       device=self.device)

        return (input_tokens, input_positions, token_type_ids, attention_mask, input_block_ids)

    def make_model_input_from_broadcasted_tensor_dict(
            self, tensor_dict: Dict[str, Any]) -> ModelInputForSynapseLLM:

        return ModelInputForSynapseLLM.from_broadcasted_tensor_dict(tensor_dict)

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForSynapseLLM:

        multi_modal_kwargs = None
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt
        # Prepare input tensors.
        if is_prompt:
            (input_tokens, input_positions, token_type_ids, attention_mask,
             input_block_ids, seq_lens, multi_modal_kwargs, kv_cache_block_ids_freed
             ) = self._prepare_prompt(seq_group_metadata_list)
        else:
            (input_tokens, input_positions, token_type_ids, attention_mask,
             input_block_ids) = self._prepare_decode(seq_group_metadata_list)
            # TODO chunk_prefill
            seq_lens = None
            # decoding should not free kv cache
            kv_cache_block_ids_freed = None

        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            seq_lens,
            # query_lens is not needed if chunked prefill is not
            # supported. Since SynapseLLM worker doesn't support chunked prefill
            # just use seq_lens instead.
            seq_lens,
            self.device,
            self.pin_memory,
            generators=self.get_generators(finished_requests_ids))

        # FIXME: We need to adjust selected_token_indices to accommodate
        # for padding
        logits_all = False
        if is_prompt:
            max_len = input_tokens.size(1)
            paddings = [max_len - s for s in seq_lens]
            paddings = [0] + paddings[:-1]
            paddings = list(itertools.accumulate(paddings))
            paddings_prompt_logprobs = []
            for i, seq_group_metadata in enumerate(seq_group_metadata_list):
                if seq_group_metadata.sampling_params.prompt_logprobs is not None \
                                and seq_group_metadata.is_prompt:
                    paddings_prompt_logprobs += ([paddings[i]] * seq_lens[i])
                    logits_all = True
            paddings = torch.tensor(
                paddings_prompt_logprobs if paddings_prompt_logprobs else paddings,
                dtype=sampling_metadata.selected_token_indices.dtype,
                device=sampling_metadata.selected_token_indices.device)
            sampling_metadata.selected_token_indices.add_(paddings)

        if not self._on_device_sampling_disabled:
            # Once the request IDs are changed in current iteration, we will
            # update the on-device sampling parameters.
            current_batch_request_ids = [
                seq_group_meta_data.request_id
                for seq_group_meta_data in seq_group_metadata_list
            ]

            if current_batch_request_ids != self._previous_batch_request_ids:
                self._update_synapsellm_sampling_params(sampling_metadata)
                self._previous_batch_request_ids = current_batch_request_ids

        return ModelInputForSynapseLLM(
                                    input_tokens=input_tokens,
                                    input_positions=input_positions,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    input_block_ids=input_block_ids,
                                    sampling_metadata=sampling_metadata,
                                    multi_modal_kwargs=multi_modal_kwargs,
                                    is_prompt=is_prompt,
                                    kv_cache_block_ids_freed=kv_cache_block_ids_freed,
                                    logits_all=logits_all,
                                )

    def _reset_block_ids_kv_cache(self, block_ids: torch.Tensor) -> None:
        if self.model is not None:
            self.model.free_block_ids_kv_cache(block_ids)

    def _get_valid_positive_top_k(self, top_k: int) -> int:
        if top_k < 0:
            return self.model_config.get_vocab_size()
        return top_k

    def _maybe_convert_to_synapsellm_greedy_sampling(self, temp, top_k):
        # Zero temperature means greedy sampling in vllm
        # convert to SynapseLLM sampling params
        if temp < _SAMPLING_EPS:
            return (1.0, 1)
        else:
            return (temp, self._get_valid_positive_top_k(top_k))

    def _update_synapsellm_sampling_params(self, sampling_metadata: SamplingMetadata):
        # Update SynapseLLM sampling parameters
        assert self.on_device_sampling_params is not None, (
            f"Failed to update sampling_params, "
            f"current sampling params is {self.on_device_sampling_params}")
        top_k: List[int] = []
        top_p: List[float] = []
        temperature: List[float] = []
        ignore_eos: List[bool] = []
        for index, sequence_group_to_sample in enumerate(
                sampling_metadata.seq_groups):
            top_p.append(sequence_group_to_sample.sampling_params.top_p)
            assert top_p[-1] == 1.0, "Unsupport top_p sampling in SynapseLLM"
            cur_temp = sequence_group_to_sample.sampling_params.temperature
            cur_top_k = sequence_group_to_sample.sampling_params.top_k
            valid_temp, valid_top_k = self._maybe_convert_to_synapsellm_greedy_sampling(cur_temp,
                                                                                        cur_top_k)
            temperature.append(valid_temp)
            top_k.append(valid_top_k)
            ignore_eos.append(sequence_group_to_sample.sampling_params.ignore_eos)

        self.on_device_sampling_params["top_k"] = top_k
        self.on_device_sampling_params["temperature"] = temperature
        self.on_device_sampling_params["ignore_eos"] = ignore_eos

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForSynapseLLM,
        kv_caches: Optional[List[torch.Tensor]] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:

        if num_steps > 1:
            raise ValueError(
                "SynapseLLMModelRunner does not support multi-step execution.")

        # free previous kv cache
        if model_input.kv_cache_block_ids_freed is not None:
            self._reset_block_ids_kv_cache(model_input.kv_cache_block_ids_freed)

        execute_model_kwargs = {
            "input_ids": model_input.input_tokens,
            "token_type_ids": model_input.token_type_ids,
            "attention_mask": model_input.attention_mask,
            "input_block_ids": model_input.input_block_ids,
            **MultiModalKwargs.as_kwargs(model_input.multi_modal_kwargs or {},
                                         device=self.device),
        }
        if not self._on_device_sampling_disabled:
            execute_model_kwargs["generate_config"] = self.on_device_sampling_params
        logger.debug(f"SynapseLLM input_block_ids: {model_input.input_block_ids}")

        # SynapseLLM does not support emit hidden_states directly
        # HPU -> CPU Sync
        logits = self.model(**execute_model_kwargs)
        # default: [bs, vocab_size] (last one token before padding tokens)
        # logits_all: [bs*max_seq_len, vocab_size] (contains both real and padding tokens)
        logits = logits.view(-1, logits.shape[-1])

        if model_input.logits_all:
            if int(os.environ.get("NS_LOGITS_ALL", "0")) <= 0:
                logger.fatal(f"Return prompt_logprobs needing set env var `NS_LOGITS_ALL=1` with "
                             f"SynapseLLM backend. We will fix it in later release.")
            if not self._on_device_sampling_disabled:
                logger.fatal(f"Return prompt_logprobs can not use device sampling.")
            selected_token_indices = model_input.sampling_metadata.selected_token_indices
            logits = logits.index_select(0, selected_token_indices)

        # update occupied_kv_cache_block_ids
        if model_input.is_prompt:
            self.model.update_occupied_kv_cache_block_ids()

        if model_input.async_callback is not None:
            model_input.async_callback()

        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        return [output]
