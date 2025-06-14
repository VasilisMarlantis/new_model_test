from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def paraphrase(text, model_name="tuner007/pegasus_paraphrase", max_length=60):
    """Self-contained paraphrasing with embedded input text"""
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Format input - TEXT NOW EMBEDDED IN SCRIPT
    input_text = "The quick brown fox jumps over the lazy dog"  # <- Your text here
    prompt = f"paraphrase: {input_text}"
    
    # Generate paraphrases
    inputs = tokenizer(
        [prompt],
        truncation=True,
        padding="longest",
        max_length=max_length,
        return_tensors="pt"
    )
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=10,
        temperature=0.7,
        do_sample=True,
        early_stopping=True
    )
    
    # Return cleaned results
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

if __name__ == "__main__":
    print("Generating paraphrases...\n")
    for i, result in enumerate(paraphrase(""), 1):  # Text already embedded above
        print(f"{i}. {result}")
