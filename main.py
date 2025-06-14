from transformers import pipeline

def paraphrase(text, model_name="tuner007/pegasus_paraphrase", max_length=50, num_return_sequences=3):
    paraphraser = pipeline("text2text-generation", model=model_name)
    paraphrases = paraphraser(
        f"paraphrase: {text}",  # Pegasus needs this prefix!
        max_length=max_length,
        num_return_sequences=num_return_sequences,
    )
    return [p["generated_text"] for p in paraphrases]

if __name__ == "__main__":
    input_text = "The quick brown fox jumps over the lazy dog"
    results = paraphrase(input_text)
    for i, res in enumerate(results, 1):
        print(f"{i}. {res}")
