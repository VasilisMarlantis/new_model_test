from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", message="Some weights of PegasusForConditionalGeneration")
warnings.filterwarnings("ignore", category=DeprecationWarning)

def paraphrase(text, model_name="tuner007/pegasus_paraphrase", max_length=60, num_return_sequences=5):
    """Improved Pegasus paraphrasing with better generation parameters"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Pegasus-specific formatting
    input_text = f"paraphrase: {text}"
    
    inputs = tokenizer(
        [input_text],
        truncation=True,
        padding="longest",
        max_length=max_length,
        return_tensors="pt"
    )
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        num_beams=15,  # Increased from default 5
        temperature=0.7,  # Added for more creativity
        do_sample=True,  # Enables temperature sampling
        top_k=50,  # Increased diversity
        early_stopping=True
    )
    
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="Rihanna’s Fenty Beauty opens first Mainland China concept store")
    parser.add_argument("--max_length", type=int, default=60)
    parser.add_argument("--num_sequences", type=int, default=5)
    args = parser.parse_args()

    print(f"Original: {args.text}\n")
    print("Generated Paraphrases:")
    for i, res in enumerate(paraphrase(args.text, args.max_length, args.num_sequences), 1):
        print(f"{i}. {res}")
