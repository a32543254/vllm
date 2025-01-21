#!/bin/bash
# serving example for SynapseLLM inference backend

export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

# # a function that waits vLLM server to start
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/chat/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'cleanup' INT

# Cleanup function
cleanup() {
    echo "Caught Ctrl+C, cleaning up..."
    # Cleanup commands
    pgrep python | xargs kill -9
    pkill -f python
    echo "Cleanup complete. Exiting."
    exit 0
}

# avoid refusing requests
export http_proxy=""

export VLLM_SYNAPSELLM_DEVICE="HPU"
export VLLM_SYNAPSELLM_NUM_THREADS=32
# turn off synapsellm sampling
# export SYNAPSELLM_ON_DEVICE_SAMPLING_DISABLED=1

MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_DTYPE="bfloat16"

# benchmark config
MAX_INPUT_LENGTH=512
MAX_PREFIX_LENGTH=50
MAX_OUTPUT_LENGTH=32
NUM_PROMPTS=100
QPS=2

# synapsellm model config
MAX_NUM_STREAMS=16
MAX_CONTEXT_LENGTH=1024
PREFILL_CHUNK_SIZE=640
MAX_NUM_BATCHED_TOKENS=$[${PREFILL_CHUNK_SIZE} * ${MAX_NUM_STREAMS}]
VLLM_DEVICE_TYPE="synapsellm"

SERVER_PORT=8000

# launch synapsellm server
vllm serve ${MODEL_NAME} \
    --port ${SERVER_PORT} \
    --dtype ${MODEL_DTYPE} \
    --max-num-seqs ${MAX_NUM_STREAMS} \
    --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
    --max-model-len ${MAX_CONTEXT_LENGTH} \
    --device ${VLLM_DEVICE_TYPE} &

# wait until server instances is ready
wait_for_server ${SERVER_PORT}


output1=$(curl -X POST -s http://localhost:${SERVER_PORT}/v1/chat/completions \
-H "Content-Type: application/json" \
-d @- <<EOF
{
"model": "${MODEL_NAME}",
"messages": [{"role": "user", "content": "Tell me something about Intel"}],
"max_completion_tokens": 128,
"temperature": 0,
"ignore_eos": true
}
EOF
)

output2=$(curl -X POST -s http://localhost:${SERVER_PORT}/v1/chat/completions \
-H "Content-Type: application/json" \
-d @- <<EOF
{
"model": "${MODEL_NAME}",
"messages": [{"role": "user", "content": "什么是量子力学？"}],
"max_completion_tokens": 128,
"temperature": 0,
"ignore_eos": true
}
EOF
)

echo ""
echo "Output of first request: $output1"
echo ""
echo "Output of second request: $output2"


echo ""
echo "Start to benchmark synapsellm serving..."

# Benchmark serving
rm -rf synapsellm_benchmark_serving
mkdir synapsellm_benchmark_serving

python3 ../benchmarks/benchmark_serving.py \
          --backend vllm \
          --model ${MODEL_NAME} \
          --dataset-name sonnet \
          --dataset-path ../benchmarks/sonnet.txt \
          --sonnet-input-len ${MAX_INPUT_LENGTH} \
          --sonnet-output-len ${MAX_OUTPUT_LENGTH} \
          --sonnet-prefix-len ${MAX_PREFIX_LENGTH} \
          --num-prompts ${NUM_PROMPTS} \
          --port ${SERVER_PORT} \
          --save-result \
          --result-dir "synapsellm_benchmark_serving" \
          --result-filename "synapsellm-qps-${QPS}.json" \
          --request-rate ${QPS} \
          --ignore-eos

echo ""
echo "Finish benchmarking synapsellm serving..."


# clean up
pgrep python | xargs kill -9
pkill -f python
