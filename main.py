from transformers import AutoTokenizer, AutoModelForCausalLM

# 🔒 Replace with your Hugging Face token — for testing only
hf_token = "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"  # Your actual token here

model_id = "meta-llama/Llama-3.2-1B"

# Load tokenizer and model using token
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=hf_token)

# Run a test prompt
prompt = "Once upon a time,"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=30)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated text:")
print(result)
