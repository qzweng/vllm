# %%
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="internlm/internlm2-chat-1_8b-sft", trust_remote_code=True)

# %%
tokenizer = llm.get_tokenizer()
input_ids = tokenizer.encode(prompts[0])
output_ids = llm.generate(prompt_token_ids=[input_ids], sampling_params=sampling_params)

# %%
tokenizer.apply_chat_template(prompts[0])

# %%
outputs = llm.generate(prompts[0], sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
# %%
