from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
from typing import List

# Configuration
MODEL_NAME = "tuner007/pegasus_paraphrase"
INPUT_TEXT = "The quick brown fox jumps over the lazy dog"  # Change this as needed
MAX_LENGTH = 60
NUM_BEAMS = 15

def initialize_model():
    """Initialize model with proper settings to suppress warnings"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        ignore_mismatched_sizes=True  # Suppresses weight initialization warnings
    )
    return tokenizer, model

def generate_paraphrases(text: str) -> List[str]:
    """Generate high-quality paraphrases with complete sentences"""
    tokenizer, model = initialize_model()
    
    inputs = tokenizer(
        f"paraphrase: {text}",
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding="longest"
    )
    
    outputs = model.generate(
        **inputs,
        max_length=MAX_LENGTH,
        num_beams=NUM_BEAMS,
        temperature=0.9,  # Increased for more creativity
        do_sample=True,
        early_stopping=True,
        length_penalty=1.5,
        no_repeat_ngram_size=2
    )
    
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def format_output(paraphrases: List[str]) -> str:
    """Ensure proper formatting of results"""
    output = ["Generated Paraphrases:"]
    for i, p in enumerate(paraphrases, 1):
        # Ensure sentences end with punctuation
        if p[-1] not in {'.', '!', '?'}:
            p += '.'
        output.append(f"{i}. {p}")
    return '\n'.join(output)

if __name__ == "__main__":
    # Suppress all warnings
    warnings.filterwarnings("ignore")
    
    print("Running paraphrase generation...\n")
    paraphrases = generate_paraphrases(INPUT_TEXT)
    print(format_output(paraphrases))
