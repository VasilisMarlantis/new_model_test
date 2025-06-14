

# 🔒 Replace with your Hugging Face token — for testing only
hf_token = "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"  # Your actual token here



import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B"

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)
pipeline("Hey how are you doing today?")
