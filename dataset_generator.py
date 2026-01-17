"""
dataset_generator.py

Generates training data for the interleaving task.
Samples fragment pairs from source texts and creates prompt + expected output.

Usage:
    # Generate and save a dataset
    python dataset_generator.py --num-words 10 --save datasets/10words.jsonl
    
    # Generate curriculum (10, 25, 50, 100, 200, 500 words)
    python dataset_generator.py --curriculum --output-dir datasets/
    
    # Preview without saving
    python dataset_generator.py --num-words 25 --preview 3
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
NUM_SAMPLES = 5000

# Train/val/test split ratios
VAL_SPLIT = 0.1   # 10% for validation during training
TEST_SPLIT = 0.1  # 10% held out for evaluate.py

# Fragment length (difficulty dial - start small, increase as model improves)
NUM_WORDS = 10  # Default to easiest level

# Random seed for reproducibility
SEED = 42

# Output format: "space" for "word1 word2 word3", "newline" for "word1\nword2\nword3"
OUTPUT_FORMAT = "newline"

# Curriculum stages
CURRICULUM_STAGES = [10, 25, 50, 100, 200, 500]

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
        "num_words": num_words,  # Track difficulty level
    }


# ============================================================================
# SAVE / LOAD (JSONL format for easy inspection)
# ============================================================================

def save_jsonl(samples: list[dict], path: str):
    """Save samples to JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    print(f"Saved {len(samples)} samples to {path}")


def load_jsonl(path: str) -> list[dict]:
    """Load samples from JSONL file."""
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def load_dataset_from_jsonl(path: str) -> Dataset:
    """Load JSONL file into HuggingFace Dataset."""
    samples = load_jsonl(path)
    
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
# DATASET GENERATION
# ============================================================================

def generate_samples(
    texts_path: str = SOURCE_TEXTS_PATH,
    num_samples: int = NUM_SAMPLES,
    num_words: int = NUM_WORDS,
    seed: int = SEED,
) -> list[dict]:
    """Generate raw samples as list of dicts."""
    random.seed(seed)
    texts = load_texts(texts_path)
    return [create_sample(texts, num_words) for _ in range(num_samples)]


def generate_dataset(
    texts_path: str = SOURCE_TEXTS_PATH,
    num_samples: int = NUM_SAMPLES,
    num_words: int = NUM_WORDS,
    seed: int = SEED,
    val_split: float = VAL_SPLIT,
    test_split: float = TEST_SPLIT,
    dataset_path: str = None,  # If provided, load from this JSONL instead of generating
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Generate HuggingFace Datasets for training, validation, and testing.
    
    Args:
        texts_path: path to source_texts.json
        num_samples: total number of samples to generate
        num_words: words per fragment (controls difficulty)
        seed: random seed for reproducibility
        val_split: fraction for validation (used during training)
        test_split: fraction for test (held out for evaluate.py)
        dataset_path: if provided, load from this JSONL file instead of generating
    
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    # Load or generate
    if dataset_path and Path(dataset_path).exists():
        print(f"Loading dataset from {dataset_path}")
        samples = load_jsonl(dataset_path)
        num_words = samples[0].get("num_words", "unknown") if samples else "unknown"
        print(f"  Loaded {len(samples)} samples ({num_words} words/fragment)")
    else:
        print(f"Generating {num_samples} samples with {num_words} words/fragment...")
        samples = generate_samples(texts_path, num_samples, num_words, seed)
    
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
    
    full_dataset = Dataset.from_dict(dataset_dict)
    
    # Split: first separate test, then split remainder into train/val
    temp_test = full_dataset.train_test_split(test_size=test_split, seed=seed)
    test_dataset = temp_test["test"]
    
    train_val = temp_test["train"].train_test_split(
        test_size=val_split / (1 - test_split), 
        seed=seed
    )
    train_dataset = train_val["train"]
    val_dataset = train_val["test"]
    
    return train_dataset, val_dataset, test_dataset


def generate_curriculum(
    output_dir: str,
    texts_path: str = SOURCE_TEXTS_PATH,
    num_samples: int = NUM_SAMPLES,
    seed: int = SEED,
    stages: list[int] = None,
):
    """Generate datasets for all curriculum stages."""
    if stages is None:
        stages = CURRICULUM_STAGES
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating curriculum datasets in {output_dir}/")
    print(f"Stages: {stages}")
    print()
    
    for num_words in stages:
        filename = f"{num_words}words.jsonl"
        path = output_dir / filename
        
        samples = generate_samples(texts_path, num_samples, num_words, seed)
        save_jsonl(samples, path)
        
        # Show sample info
        sample = samples[0]
        prompt_len = len(sample['prompt'][0]['content'])
        expected_len = len(sample['expected'])
        print(f"  {filename}: prompt~{prompt_len} chars, expected~{expected_len} tokens")
    
    print()
    print("Done! Use with: python interleave_grpo.py --dataset datasets/10words.jsonl")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate interleaving dataset")
    parser.add_argument("--texts", default=SOURCE_TEXTS_PATH, help="Path to source texts JSON")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES, help="Total number of samples")
    parser.add_argument("--num-words", type=int, default=NUM_WORDS, help="Words per fragment")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT, help="Validation split ratio")
    parser.add_argument("--test-split", type=float, default=TEST_SPLIT, help="Test split ratio")
    parser.add_argument("--preview", type=int, default=2, help="Number of samples to preview")
    
    # Save/load options
    parser.add_argument("--save", default=None, help="Save to JSONL file")
    parser.add_argument("--load", default=None, help="Load from JSONL file (ignores generation params)")
    
    # Curriculum generation
    parser.add_argument("--curriculum", action="store_true", help="Generate all curriculum stages")
    parser.add_argument("--output-dir", default="datasets", help="Output directory for curriculum")
    
    args = parser.parse_args()
    
    # Curriculum mode
    if args.curriculum:
        generate_curriculum(
            output_dir=args.output_dir,
            texts_path=args.texts,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        exit(0)
    
    # Load mode
    if args.load:
        samples = load_jsonl(args.load)
        print(f"Loaded {len(samples)} samples from {args.load}")
        num_words = samples[0].get("num_words", "unknown")
        print(f"Fragment size: {num_words} words")
    else:
        # Generate mode
        print(f"Generating {args.num_samples} samples with {args.num_words} words per fragment...")
        print(f"Splits: train={1-args.val_split-args.test_split:.0%}, val={args.val_split:.0%}, test={args.test_split:.0%}")
        print(f"Output format: {OUTPUT_FORMAT}")
        print()
        
        samples = generate_samples(
            texts_path=args.texts,
            num_samples=args.num_samples,
            num_words=args.num_words,
            seed=args.seed,
        )
    
    # Save if requested
    if args.save:
        save_jsonl(samples, args.save)
    
    # Preview
    print()
    print(f"=== Preview ({min(args.preview, len(samples))} samples) ===")
    for i in range(min(args.preview, len(samples))):
        sample = samples[i]
        print(f"\nSample {i+1}:")
        print(f"  Text A ({sample['text_a_id']}): {sample['fragment_a'][:80]}...")
        print(f"  Text B ({sample['text_b_id']}): {sample['fragment_b'][:80]}...")
        print(f"  Expected ({len(sample['expected'])} words): {' '.join(sample['expected'][:10])}...")
        
        # Show prompt length (important for checking truncation)
        prompt_len = len(sample['prompt'][0]['content'])
        print(f"  Prompt length: {prompt_len} chars")
    
    # Summary stats
    if samples:
        prompt_lens = [len(s['prompt'][0]['content']) for s in samples]
        expected_lens = [len(s['expected']) for s in samples]
        print(f"\n=== Stats ===")
        print(f"  Prompt length: {min(prompt_lens)}-{max(prompt_lens)} chars (avg {sum(prompt_lens)//len(prompt_lens)})")
        print(f"  Expected length: {min(expected_lens)}-{max(expected_lens)} tokens (avg {sum(expected_lens)//len(expected_lens)})")