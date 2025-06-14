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
    max_length=3000,           # You can bump this up if needed (e.g. 100 or 150)
    num_return_sequences=num_return_sequences,
    do_sample=True,                  # Enables sampling (instead of beam search)
    temperature=0.9,                 # Lower = more deterministic, Higher = more creative (suggest 0.8–1.2)
    top_k=50,                        # Limits to top-k likely next tokens (e.g. 50 or 100)
    top_p=0.95,                      # Nucleus sampling: choose from top 95% probability mass
    repetition_penalty=1.2,         # Penalize repeating phrases
    no_repeat_ngram_size=3,         # Avoid repeated 3-grams (optional)
)

    
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--num_sequences", type=int, default=3)
    args = parser.parse_args()

    # Hardcoded input text here
    input_text = """ As this year's halloween falls on a thursday, it's only fitting that kfc, a confectionary manufacturer and manufacturer of the iconic hainan candy bar, would come out swinging on all hallows' day eve.


The event is appropriately named "" (lit.


Going wild and giving out candy, playing on the fact that "" is the verb for both english and french.


Of course, one million candies was given away, but kfc also stated in the t&cs that each customer will be given a million candies.


Given that there are 5,000 participating restaurants around the country, only the first 100 people to arrive at each restaurant will receive the free coconut-flavoured treats.


Some netizens have voiced their dissatisfactions.


This is not the first time kfc has played "i thought you were doing april fools" in the meagre number, and the joke "i thought you were doing 2 candies" was not the first time kfc has appeared on the internet.


Kfc's k coffee launched a co-branded roasted coconut latte earlier this year, aiming to imitate "the taste of your childhood." this time, the collaboration is heavier on the seasonal.


However, no one likes free candy, particularly because it is less about the partners' nostalgic "emotional value.". """

    results = paraphrase(input_text, max_length=args.max_length, num_return_sequences=args.num_sequences)
    
    print("\nGenerated Paraphrases:")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res}")
