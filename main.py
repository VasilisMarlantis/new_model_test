from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

def paraphrase(text, model_name="tuner007/pegasus_paraphrase", max_length=50, num_return_sequences=3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

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
        num_beams=5,
        early_stopping=True
    )
    
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--num_sequences", type=int, default=3)
    args = parser.parse_args()

    # Hardcoded input text here
    input_text = "The quick brown fox jumps over the lazy dog"

    results = paraphrase(input_text, max_length=args.max_length, num_return_sequences=args.num_sequences)
    
    print("\nGenerated Paraphrases:")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res}")
