from typing import TYPE_CHECKING, Optional

import torch

import vllm.envs as envs
from vllm.logger import init_logger

from .interface import Platform, PlatformEnum, _Backend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)

try:
    from neural_speed import Model
except ImportError as e:
    logger.warning("Failed to import SynapseLLM with %r", e)


class SynapseLLMPlatform(Platform):
    _enum = PlatformEnum.SYNAPSELLM
    device_name: str = "synapsellm"
    device_type: str = "synapsellm"
    dispatch_key: str = "CPU"

    @classmethod
    def get_default_attn_backend(cls, selected_backend: _Backend) -> _Backend:
        if selected_backend != _Backend.SYNAPSELLM:
            logger.info("Cannot use %s backend on SynapseLLM.", selected_backend)
        return _Backend.SYNAPSELLM

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "synapsellm"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return False

    @classmethod
    def inference_mode(cls):
        return torch.inference_mode(mode=True)

    @classmethod
    def is_synapsellm_cpu(cls) -> bool:
        return "CPU" in envs.VLLM_SYNAPSELLM_DEVICE

    @classmethod
    def is_synapsellm_hpu(cls) -> bool:
        return "HPU" in envs.VLLM_SYNAPSELLM_DEVICE

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        logger.warning("Pin memory is not supported on SynapseLLM.")
        return False

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:

        scheduler_config = vllm_config.scheduler_config
        if scheduler_config.is_multi_step:
            raise NotImplementedError(
                "Multi-step execution is not implemented for SynapseLLM")

        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "Speculative decoding is not implemented for SynapseLLM")

        parallel_config = vllm_config.parallel_config
        assert (
            parallel_config.world_size == 1
        ), "SynapseLLMExecutor only supports single CPU socket currently."

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = \
                "vllm.worker.synapsellm_worker.SynapseLLMWorker"

        # check and update model config
        model_config = vllm_config.model_config
        # TODO remove this
        if model_config.dtype == torch.fp8e4m3:
            logger.warning(
                f"Only float32 dtype is supported on SynapseLLM, casting from {model_config.dtype}."  # noqa: G004, E501
            )
            model_config.dtype = torch.float32
        if model_config.enforce_eager:
            logger.warning(
                "eager_mode is not supported on SynapseLLM backend, fallback to "
                "the graph mode.")
            model_config.enforce_eager = False

        # check and update cache config
        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 128
            # cache_config.cache_dtype = "f16"
