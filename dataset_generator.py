"""
dataset_generator.py

Generates training data for the interleaving task.
Samples fragment pairs from source texts and creates prompt + expected output.
"""

import json
import random
from pathlib import Path
from datasets import Dataset

# ============================================================================
# CONFIG - CHANGE THESE
# ============================================================================

# Paths
SOURCE_TEXTS_PATH = "source_texts.json"

# Dataset size
NUM_SAMPLES = 1000

# Fragment length (difficulty dial - start small, increase as model improves)
NUM_WORDS = 10

# Random seed for reproducibility
SEED = 42

# Output format: "space" for "word1 word2 word3", "newline" for "word1\nword2\nword3"
OUTPUT_FORMAT = "newline"

# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

PROMPT_TEMPLATE = """You will run two processes in parallel, each outputting words from a text.

Process A outputs words from:
{fragment_a}

Process B outputs words from:
{fragment_b}

Each process maintains its own internal position.
Each process remembers where it is in its own text.
Each process advances only when it is that process's turn.

Output one word from Process A, then one word from Process B.
Continue alternating Process A, Process B, Process A, Process B.
When one process reaches the end of its text, continue outputting only from the remaining process until it also ends.

Do not add commentary, explanation, labels, or metadata.
Output one word per line, including any attached punctuation (e.g., "be," not "be").
Do not wait for additional input.
Do this all in one turn.
Begin now and continue until complete."""

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def load_texts(path: str = SOURCE_TEXTS_PATH) -> list[dict]:
    """Load source texts from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data["texts"]


def extract_words(text: str) -> list[str]:
    """Split text into words, preserving punctuation."""
    return text.split()


def sample_fragment(text: str, num_words: int, start_pos: int = None) -> tuple[list[str], int]:
    """
    Sample a fragment of num_words from text.
    Returns (words, start_position).
    """
    words = extract_words(text)
    
    if len(words) <= num_words:
        return words, 0
    
    max_start = len(words) - num_words
    if start_pos is None:
        start_pos = random.randint(0, max_start)
    else:
        start_pos = min(start_pos, max_start)
    
    return words[start_pos:start_pos + num_words], start_pos


def interleave_words(words_a: list[str], words_b: list[str]) -> list[str]:
    """
    Interleave two word lists: A1, B1, A2, B2, ...
    Continues with remaining words when one list ends.
    """
    result = []
    max_len = max(len(words_a), len(words_b))
    
    for i in range(max_len):
        if i < len(words_a):
            result.append(words_a[i])
        if i < len(words_b):
            result.append(words_b[i])
    
    return result


def format_expected(words: list[str], output_format: str = OUTPUT_FORMAT) -> str:
    """Format expected output according to config."""
    if output_format == "newline":
        return "\n".join(words)
    else:
        return " ".join(words)


def create_sample(
    texts: list[dict],
    num_words: int = NUM_WORDS,
    text_a_idx: int = None,
    text_b_idx: int = None,
) -> dict:
    """
    Create a single training sample.
    
    Returns dict with:
        - prompt: formatted chat messages
        - fragment_a: source fragment A as string
        - fragment_b: source fragment B as string  
        - expected: list of interleaved words
        - expected_str: interleaved words as formatted string
        - text_a_id: source text A identifier
        - text_b_id: source text B identifier
    """
    # Pick two different texts
    if text_a_idx is None or text_b_idx is None:
        indices = random.sample(range(len(texts)), 2)
        text_a_idx, text_b_idx = indices
    
    text_a = texts[text_a_idx]
    text_b = texts[text_b_idx]
    
    # Sample fragments
    words_a, _ = sample_fragment(text_a["text"], num_words)
    words_b, _ = sample_fragment(text_b["text"], num_words)
    
    fragment_a = " ".join(words_a)
    fragment_b = " ".join(words_b)
    
    # Build expected output
    expected = interleave_words(words_a, words_b)
    expected_str = format_expected(expected)
    
    # Build prompt
    user_content = PROMPT_TEMPLATE.format(
        fragment_a=fragment_a,
        fragment_b=fragment_b
    )
    
    prompt = [
        {"role": "user", "content": user_content},
    ]
    
    return {
        "prompt": prompt,
        "fragment_a": fragment_a,
        "fragment_b": fragment_b,
        "expected": expected,
        "expected_str": expected_str,
        "text_a_id": text_a["id"],
        "text_b_id": text_b["id"],
    }


def generate_dataset(
    texts_path: str = SOURCE_TEXTS_PATH,
    num_samples: int = NUM_SAMPLES,
    num_words: int = NUM_WORDS,
    seed: int = SEED,
) -> Dataset:
    """
    Generate a HuggingFace Dataset for training.
    
    Args:
        texts_path: path to source_texts.json
        num_samples: number of training samples to generate
        num_words: words per fragment (controls difficulty)
        seed: random seed for reproducibility
    
    Returns:
        HuggingFace Dataset with columns:
            - prompt, fragment_a, fragment_b, expected, expected_str, text_a_id, text_b_id
    """
    random.seed(seed)
    texts = load_texts(texts_path)
    
    samples = [create_sample(texts, num_words) for _ in range(num_samples)]
    
    # Convert to Dataset format
    dataset_dict = {
        "prompt": [s["prompt"] for s in samples],
        "fragment_a": [s["fragment_a"] for s in samples],
        "fragment_b": [s["fragment_b"] for s in samples],
        "expected": [s["expected"] for s in samples],
        "expected_str": [s["expected_str"] for s in samples],
        "text_a_id": [s["text_a_id"] for s in samples],
        "text_b_id": [s["text_b_id"] for s in samples],
    }
    
    return Dataset.from_dict(dataset_dict)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate interleaving dataset")
    parser.add_argument("--texts", default=SOURCE_TEXTS_PATH, help="Path to source texts JSON")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES, help="Number of samples")
    parser.add_argument("--num-words", type=int, default=NUM_WORDS, help="Words per fragment")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--output", default=None, help="Output path (optional, saves to disk)")
    parser.add_argument("--preview", type=int, default=2, help="Number of samples to preview")
    
    args = parser.parse_args()
    
    print(f"Generating {args.num_samples} samples with {args.num_words} words per fragment...")
    print(f"Output format: {OUTPUT_FORMAT}")
    print()
    
    dataset = generate_dataset(
        texts_path=args.texts,
        num_samples=args.num_samples,
        num_words=args.num_words,
        seed=args.seed,
    )
    
    print(f"Generated {len(dataset)} samples")
    print()
    
    # Preview
    for i in range(min(args.preview, len(dataset))):
        sample = dataset[i]
        print(f"=== Sample {i+1} ===")
        print(f"Text A ({sample['text_a_id']}): {sample['fragment_a']}")
        print(f"Text B ({sample['text_b_id']}): {sample['fragment_b']}")
        print(f"Expected:\n{sample['expected_str']}")
        print()
    
    # Save if requested
    if args.output:
        dataset.save_to_disk(args.output)
        print(f"Saved to {args.output}")