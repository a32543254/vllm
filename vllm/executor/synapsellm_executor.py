from typing import List, Set, Tuple

from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        make_async)
from vllm.worker.worker_base import WorkerWrapperBase
from vllm.platforms import current_platform
import vllm.envs as envs

logger = init_logger(__name__)


class SynapseLLMExecutor(ExecutorBase):

    uses_ray: bool = False

    def _init_executor(self) -> None:
        assert self.device_config.device_type == "synapsellm"
        assert (self.lora_config is
                None), "LoRA is not supported for SynapseLLM backend."
        assert (not self.speculative_config
                ), "Speculative decoding is not supported for SynapseLLM backend."
        assert current_platform.is_synapsellm_cpu() or \
               current_platform.is_synapsellm_hpu(), \
            "SynapseLLM backend only supports CPU and HPU devices for now."

        # Instantiate the worker and load the model to the device.
        self._init_worker()

    def _init_worker(self):
        wrapper = WorkerWrapperBase(vllm_config=self.vllm_config)

        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())
        wrapper.init_worker(
            vllm_config=self.vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method=distributed_init_method,
        )
        self.driver_worker = wrapper.worker
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks by invoking the
        underlying worker.
        """
        return self.driver_worker.determine_num_available_blocks()

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache by invoking the underlying worker.
        """
        # NOTE: In case of a CPU device, `cpu block` for SynapseLLM backend
        # is located on CPU memory but is referred as `gpu block`.
        # Because we want to reuse the existing block management procedure.
        # NOTE: num_gpu_blocks = max_num_reqs, block_size = max_model_len
        logger.info(f"SynapseLLM on {envs.VLLM_SYNAPSELLM_DEVICE}: "
                    f"# device blocks: {num_gpu_blocks}; "
                    f"# swap blocks: {num_cpu_blocks}",
                    )
        self.driver_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        assert (not execute_model_req.blocks_to_swap_in
                and not execute_model_req.blocks_to_swap_out
                and not execute_model_req.blocks_to_copy), (
                    "Cache operations are not supported for SynapseLLM backend.")
        assert execute_model_req.num_lookahead_slots == 0, (
            "lookahead not supported for SynapseLLM backend.")

        output = self.driver_worker.execute_model(execute_model_req)
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.driver_worker.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.driver_worker.remove_lora(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        return self.driver_worker.pin_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.driver_worker.list_loras()

    def add_prompt_adapter(self, prompt_adapter_request) -> bool:
        raise NotImplementedError(
            "Soft prompt is currently not supported by the SynapseLLM backend.")

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise NotImplementedError(
            "Soft prompt is currently not supported by the SynapseLLM backend.")

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise NotImplementedError(
            "Soft prompt is currently not supported by the SynapseLLM backend.")

    def list_prompt_adapters(self) -> Set[int]:
        raise NotImplementedError(
            "Soft prompt is currently not supported by the SynapseLLM backend.")

    def check_health(self) -> None:
        # SynapseLLMExecutor will always be healthy as long as
        # it's running.
        return


class SynapseLLMExecutorAsync(SynapseLLMExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[SamplerOutput]:
        output = await make_async(self.driver_worker.execute_model
                                  )(execute_model_req=execute_model_req, )
        return output

    async def check_health_async(self) -> None:
        # SynapseLLMExecutor will always be healthy as long as
        # it's running.
        return