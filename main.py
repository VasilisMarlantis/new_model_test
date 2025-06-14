

# 🔒 Replace with your Hugging Face token — for testing only
hf_token = "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"  # Your actual token here


from transformers import pipeline
import argparse

def paraphrase(text, model_name="facebook/bart-large-cnn", max_length=50, num_return_sequences=3):
    """
    Paraphrase input text using Hugging Face's BART model.
    
    Args:
        text (str): Input text to paraphrase.
        model_name (str): Model to use (default: facebook/bart-large-cnn).
        max_length (int): Max length of output text.
        num_return_sequences (int): Number of paraphrases to generate.
    
    Returns:
        List of paraphrased texts.
    """
    # Load the paraphrasing model
    paraphraser = pipeline("text2text-generation", model=model_name)
    
    # Generate paraphrases
    paraphrases = paraphraser(
        text,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
    )
    
    return [p["generated_text"] for p in paraphrases]

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Text Paraphraser using BART")
    parser.add_argument("--text", type=str, required=True, help="Input text to paraphrase")
    parser.add_argument("--max_length", type=int, default=50, help="Max output length")
    parser.add_argument("--num_sequences", type=int, default=3, help="Number of paraphrases")
    args = parser.parse_args()

    # Run paraphrasing
    results = paraphrase(args.text, max_length=args.max_length, num_return_sequences=args.num_sequences)
    
    # Print results
    print("\nGenerated Paraphrases:")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res}")
