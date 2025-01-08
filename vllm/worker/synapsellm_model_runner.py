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

    def load_model(self) -> None:
        if find_spec("neural_speed") is not None:
            self.model = get_model(
                self.model_config,
                self.scheduler_config,
                self.cache_config,
                self.device_config,
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

        input_tokens = make_tensor_with_pad(input_tokens,
                                            pad=0,
                                            max_len=1,
                                            dtype=torch.long,
                                            device=self.device)
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
        logger.debug(f"SynapseLLM input_block_ids: {model_input.input_block_ids}")

        # SynapseLLM does not support emit hidden_states directly
        logits = self.model(**execute_model_kwargs)
        # default: [bs, vocab_size] (last one token before padding tokens)
        # logits_all: [bs*max_seq_len, vocab_size] (contains both real and padding tokens)
        logits = logits.view(-1, logits.shape[-1])

        if model_input.logits_all:
            if int(os.environ.get("NS_LOGITS_ALL", "0")) <= 0:
                logger.fatal(f"Return prompt_logprobs needing set env var `NS_LOGITS_ALL=1` with "
                             f"SynapseLLM backend. We will fix it in later release.")
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
