from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

def paraphrase(text, model_name="tuner007/pegasus_paraphrase", max_length=128, num_return_sequences=3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Split into rough sentences using full stops
    sentences = [s.strip() for s in text.split('.') if s.strip()]

    all_outputs = []

    for sentence in sentences:
        input_text = f"paraphrase: {sentence}"
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
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_outputs.append(decoded)

    return all_outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_sequences", type=int, default=3)
    args = parser.parse_args()

    input_text = """Your long paragraph here..."""

    results = paraphrase(input_text, max_length=args.max_length, num_return_sequences=args.num_sequences)

    print("\nGenerated Paraphrases:")
    for i, paraphrases in enumerate(results, 1):
        print(f"\nOriginal sentence {i}:")
        for j, p in enumerate(paraphrases, 1):
            print(f"  {j}. {p}")
