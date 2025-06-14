from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

def paraphrase(text, model_name="tuner007/pegasus_paraphrase", max_length=50, num_return_sequences=3):
    """
    Proper Pegasus paraphrasing implementation with:
    - Explicit tokenizer/model loading
    - Device management
    - Correct prompt formatting
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Format input text (Pegasus requires "paraphrase: " prefix)
    input_text = f"paraphrase: {text}"
    
    # Tokenize and generate
    inputs = tokenizer(
        [input_text],
        truncation=True,
        padding="longest",
        max_length=max_length,
        return_tensors="pt"
    )
    
    outputs = model.generate(
    **inputs,
    max_length=100,  # Increased from 50
    num_return_sequences=num_return_sequences,
    num_beams=10,    # More beams = better diversity (but slower)
    early_stopping=True,
    length_penalty=2.0,  # Penalizes short outputs
    no_repeat_ngram_size=2,  # Avoids word repetition
)
    
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="Rihanna’s Fenty Beauty opens first Mainland China concept store")
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--num_sequences", type=int, default=3)
    args = parser.parse_args()

    results = paraphrase(args.text, max_length=args.max_length, num_return_sequences=args.num_sequences)
    
    print("\nGenerated Paraphrases:")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res}")
