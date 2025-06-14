from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

def paraphrase(text, model_name="tuner007/pegasus_paraphrase", max_length=256, num_return_sequences=3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Split text into chunks (by line or sentence-like logic)
    chunks = [chunk.strip() for chunk in text.split('\n') if chunk.strip()]

    all_results = []

    for i, chunk in enumerate(chunks):
        input_text = f"paraphrase: {chunk}"

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
            no_repeat_ngram_size=3
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_results.append((chunk, decoded))

    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_sequences", type=int, default=3)
    args = parser.parse_args()

    # Hardcoded input text (multiline string)
    input_text = """
As this year's Halloween falls on a Thursday, it's only fitting that KFC, a confectionary manufacturer and creator of the iconic Hainan candy bar, would come out swinging on All Hallows' Eve.
The event is appropriately named something wild, hinting at both English and French meanings.
Of course, a million candies were given away — or so claimed — but each customer was promised a million treats.
Given that there are 5,000 participating restaurants across the country, only the first 100 people at each spot will receive the free coconut-flavored giveaways.
Some netizens voiced their dissatisfaction, calling out the promotion for being misleading.
Earlier this year, KFC’s K Coffee also launched a coconut latte meant to imitate “the taste of your childhood.”
"""

    results = paraphrase(input_text, max_length=args.max_length, num_return_sequences=args.num_sequences)

    print("\nGenerated Paraphrases:")
    for idx, (original, paraphrases) in enumerate(results, 1):
        print(f"\nOriginal sentence {idx}:")
        for i, p in enumerate(paraphrases, 1):
            print(f"  {i}. {p}")
