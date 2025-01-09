(installation-synapsellm)=

# Installation with SynapseLLM (TODO)

SynapseLLM is a lightweight and high-performance LLM inference engine on Intel platforms (Xeon, Gaudi-2).

Limitations (unspported features):
- PagedAttention
- Chunked prefill
- Prefix caching
- Multi-step

**Table of contents**:

- [Requirements](#synapsellm-backend-requirements)
- [Quick start using Dockerfile](#synapsellm-backend-quick-start-dockerfile)
- [Build from source](#build-synapsellm-backend-from-source)
- [Related runtime environment variables](#env-intro)
- [Performance tips](#synapsellm-backend-performance-tips)
- [LM_eval accuracy](#synapsellm-backend-accuracy-validation)


(synapsellm-backend-requirements)=

## Requirements (TODO)

- OS: Linux
- Accelerator (Optional): Gaudi-2
- Habana driver (1.19.0)


(synapsellm-backend-quick-start-dockerfile)=

## Quick start using Dockerfile (TODO)


(build-from-source-synapsellm)=

## Build from source

```bash
# install synapsellm by following the related instruction
git clone https://github.com/luoyu-intel/synapse_llm.git
...

# install vllm
git clone https://github.com/a32543254/vllm.git vllm_synapsellm
cd vllm_synapsellm && git switch yzt/synapse_llm_backend
VLLM_TARGET_DEVICE=synapsellm pip install -v -e .
```


(env-intro)=

## Related runtime environment variables

- `VLLM_SYNAPSELLM_DEVICE`: specify SynapseLLM device (CPU, HPU). Default is CPU.
- `VLLM_SYNAPSELLM_NUM_THREADS`: specify the CPU cores, 4 by default.


(synapsellm-backend-performance-tips)=

## Performance tips (TODO)

### CPU performance tips

- offline example

```bash
cd vllm_synapsellm/examples
python offline_inference_synapsellm.py
```

- benchmark latency

```bash
cd vllm_synapsellm/benchmarks
VLLM_SYNAPSELLM_NUM_THREADS=32 numactl -l -C 0-31 python benchmark_latency.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --max-num-seqs 10 --batch-size 8 --max-model-len 1024 --max-num-batched-tokens 1024 \
        --input-len 512 --output-len 32 --device synapsellm
```

- benchmark throughput

```bash
cd vllm_synapsellm/benchmarks
VLLM_SYNAPSELLM_NUM_THREADS=32 numactl -l -C 0-31 python benchmark_throughput.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --max-num-seqs 10 --num-prompts 100 --max-model-len 1024 --max-num-batched-tokens 1024 \
        --input-len 512 --output-len 32 --device synapsellm
```

- server example (contains serving bechmark)

```bash
cd vllm_synapsellm/examples
bash server_synapsellm.sh
```


### HPU performance tips


(synapsellm-backend-accuracy-validation)=

## LM_eval accuracy

Please install `lm_eval` first by following the [related document](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#install).

> Note:
> 1. `lm_eval` offically supports vLLM CUDA backend. However, we can use it to test the accuracy of SynapseLLM backend without specifying the `device` arg.
> 2. Please run `pip install ray` if you meet `NameError: name 'LLM' is not defined.`. It's a workaround of this [issue](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/vllm_causallms.py#L27).

Using the below command to test accuracy of Habana Gaudi-2 devices.

```bash
VLLM_SYNAPSELLM_NUM_THREADS=32 VLLM_SYNAPSELLM_DEVICE=HPU lm_eval --model vllm --model_args pretrained="Qwen/Qwen2.5-1.5B-Instruct",tensor_parallel_size=1,dtype=bfloat16,max_model_len=4096,max_num_seqs=10,max_num_batched_tokens=10240 --tasks lambada_openai --batch_size 8
```
