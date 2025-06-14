from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import nltk
from nltk.tokenize import sent_tokenize

# Download tokenizer data
nltk.download('punkt')

def paraphrase(text, model_name="tuner007/pegasus_paraphrase", max_length=256, num_return_sequences=1):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    sentences = sent_tokenize(text)
    paraphrased_sentences = []

    for sent in sentences:
        input_text = f"paraphrase: {sent}"
        inputs = tokenizer(
            [input_text],
            truncation=True,
            padding="longest",
            max_length=512,
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
        paraphrased_sentences.extend(decoded)

    return paraphrased_sentences

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_sequences", type=int, default=1)
    args = parser.parse_args()

    # Hardcoded long input text
    input_text = """As this year's Halloween falls on a Thursday, it's only fitting that KFC...
    ...This time, the collaboration is heavier on the seasonal."""

    results = paraphrase(input_text, max_length=args.max_length, num_return_sequences=args.num_sequences)

    print("\nGenerated Paraphrased Sentences:")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res}")
