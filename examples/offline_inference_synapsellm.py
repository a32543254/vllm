"""vLLM offline inference with SynapseLLM backend.

Benchmark latency:
VLLM_SYNAPSELLM_NUM_THREADS=32 python ../benchmarks/benchmark_latency.py \
       --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
       --max-num-seqs 10 --batch-size 8 --max-model-len 1024 --max-num-batched-tokens 1024 \
       --input-len 512 --output-len 32 --device synapsellm

Benchmark throughput:
VLLM_SYNAPSELLM_NUM_THREADS=32 python ../benchmarks/benchmark_throughput.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
       --max-num-seqs 10 --num-prompts 100 --max-model-len 1024 --max-num-batched-tokens 1024 \
       --input-len 512 --output-len 32 --device synapsellm
"""

import os

from vllm import LLM, SamplingParams

# CPU or HPU
os.environ["VLLM_SYNAPSELLM_DEVICE"] = "HPU"
os.environ['VLLM_SYNAPSELLM_NUM_THREADS'] = "32"

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=32)

# Create an LLM.
llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dtype="bfloat16",
    max_num_seqs=6,
    # the max_num_batched_tokens will determine the chunk_size for prefill
    # which is max_num_batched_tokens // max_num_seqs
    max_num_batched_tokens=512*6,
    # The max_model_len and block_size arguments are required to be same as
    # max sequence length when using SynapseLLM backend (no PagedAttention).
    max_model_len=1024,
    block_size=1024,
    # The device can be automatically detected when setting `VLLM_TARGET_DEVICE=synapsellm`
    # before vllm  installing process.
    # The device argument can be either unspecified for automated detection,
    # or explicitly assigned.
    device="synapsellm",
    tensor_parallel_size=1,
)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
