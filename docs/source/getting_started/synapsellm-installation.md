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