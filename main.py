from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

def paraphrase(text, model_name="tuner007/pegasus_paraphrase", max_length=300, num_return_sequences=3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_text = f"paraphrase: {text}"

    inputs = tokenizer(
        [input_text],
        truncation=True,
        padding="longest",
        max_length=512,  # for long paragraph input
        return_tensors="pt"
    )

    outputs = model.generate(
        **inputs,
        max_length=5000,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=300)
    parser.add_argument("--num_sequences", type=int, default=3)
    args = parser.parse_args()

    # Hardcoded input text
    input_text = """As this year's Halloween falls on a Thursday, it's only fitting that KFC, a confectionary manufacturer and manufacturer of the iconic Hainan candy bar, would come out swinging on All Hallows' Day eve.

The event is appropriately named "" (lit.

Going wild and giving out candy, playing on the fact that "" is the verb for both English and French.

Of course, one million candies was given away, but KFC also stated in the T&Cs that each customer will be given a million candies.

Given that there are 5,000 participating restaurants around the country, only the first 100 people to arrive at each restaurant will receive the free coconut-flavoured treats.

Some netizens have voiced their dissatisfactions.

This is not the first time KFC has played "I thought you were doing April Fools" in the meagre number, and the joke "I thought you were doing 2 candies" was not the first time KFC has appeared on the internet.

KFC's K Coffee launched a co-branded roasted coconut latte earlier this year, aiming to imitate "the taste of your childhood." This time, the collaboration is heavier on the seasonal.

However, no one likes free candy, particularly because it is less about the partners' nostalgic "emotional value."""

    results = paraphrase(input_text, max_length=args.max_length, num_return_sequences=args.num_sequences)

    print("\nGenerated Paraphrases:")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res}")
