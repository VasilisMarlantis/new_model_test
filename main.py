from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "meta-llama/Llama-3.2-1B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Create text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Run inference
prompt = "Tell me a fun fact about the universe."
outputs = generator(prompt, max_new_tokens=50)

print(outputs[0]["generated_text"])
