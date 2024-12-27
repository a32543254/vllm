import os

from vllm import LLM, SamplingParams

os.environ["VLLM_SYNAPSELLM_DEVICE"] = "CPU"
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
    max_num_seqs=6,
    max_num_batched_tokens=2048,
    # The max_model_len and block_size arguments are required to be same as
    # max sequence length when using SynapseLLM backend (no PagedAttention).
    max_model_len=1024,
    block_size=1024,
    # The device can be automatically detected when AWS Neuron SDK is installed.
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
