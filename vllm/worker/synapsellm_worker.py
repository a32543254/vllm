"""A SynapseLLM worker class."""
from typing import List, Optional, Tuple

import torch
import torch.distributed

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.model_executor import set_random_seed
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.worker.synapsellm_model_runner import SynapseLLMModelRunner
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase,
                                     LoraNotSupportedWorkerBase, WorkerBase,
                                     WorkerInput)

logger = init_logger(__name__)


class SynapseLLMWorker(LoraNotSupportedWorkerBase, LocalOrDistributedWorkerBase):
    """A worker class that executes the model on SynapseLLM backend.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = True,
    ) -> None:
        WorkerBase.__init__(self, vllm_config=vllm_config)
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        self.model_runner: SynapseLLMModelRunner = SynapseLLMModelRunner(
            vllm_config=vllm_config)
        # TODO (check this) TP will maintain broadcast inside SynapseLLM
        self.is_driver_worker = is_driver_worker

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                ],
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True))
        else:
            self.profiler = None

    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> Optional[List[SamplerOutput]]:
        assert execute_model_req is not None
        assert (not execute_model_req.blocks_to_swap_in
                and not execute_model_req.blocks_to_swap_out
                and not execute_model_req.blocks_to_copy), (
                    "Cache operations are not supported for SynapseLLM backend.")
        assert execute_model_req.num_lookahead_slots == 0, (
            "lookahead not supported for SynapseLLM backend.")
        output = LocalOrDistributedWorkerBase.execute_model(
            self, execute_model_req)
        return output

    def start_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.start()

    def stop_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.stop()

    def init_device(self) -> None:
        self.init_distributed_environment()
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks.

        SynapseLLM supports both cpu and hpu devices.

        Swapping is not yet supported when run SynapseLLM in hpu devices,
        so always return num_swap_blocks=0.

        We configure num_device_blocks to be equal to max_num_seqs since SynapseLLM
        allocates static  kv cache memory inside.
        """
        # Set the number of device blocks to be the same as the maximum number of
        # sequences that can be processed in a single batch. This is equivalent
        # to schedule without PagedAttention.
        num_device_blocks = self.scheduler_config.max_num_seqs

        # Swap not yet supported with SynapseLLM backend in hpu devices.
        num_swap_blocks = 0

        return num_device_blocks, num_swap_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache.

        For CPU, we use the num_gpu_blocks to
        determine how many non-swappable CPU blocks to schedule.

        For HPU, swappable CPU memory is not supported.
        """

        num_device_blocks = num_gpu_blocks
        num_swap_blocks = num_cpu_blocks

        # Different values are not tested.
        assert num_swap_blocks == 0
        assert num_device_blocks == self.scheduler_config.max_num_seqs

        self.cache_config.num_gpu_blocks = num_device_blocks
        self.cache_config.num_cpu_blocks = num_swap_blocks

    @property
    def do_metadata_broadcast(self) -> bool:
        return False

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        # kv cache memory will be maintained insdise SynapseLLM.
        return None

    @torch.inference_mode()
    def prepare_worker_input(
            self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
        return WorkerInput(num_seq_groups=len(
            execute_model_req.seq_group_metadata_list), )

    def execute_worker(self, worker_input: WorkerInput) -> None:
        pass

    def get_cache_block_size_bytes(self) -> int:
        """Determine the size in bytes of a cache block.

        This is required for speculative decoding; it is not yet implemented.
        """
        raise NotImplementedError

    # TODO (check this)
    def init_distributed_environment(self):
        """SynapseLLM uses its own implementation for tensor parallelism.
        It has only one process to control multiple devices.
        vLLM still needs the environment initialized when TP/PP > 1,
        so we initialize a distributed environment with one process.
        """
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=self.distributed_init_method,
            backend="gloo",
        )
        ensure_model_parallel_initialized(
            1,
            1,
        )
