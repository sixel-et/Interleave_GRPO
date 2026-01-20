"""
dataset_generator.py

Generates training data for the interleaving task.
Samples fragment pairs from source texts and creates prompt + expected output.

IMPORTANT: Source texts must have a "split" field (train/val/test) to ensure
zero text leakage between splits. Use add_splits_to_corpus.py to add splits
to your corpus first.

Usage:
    # Generate and save datasets (respects corpus split tags)
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

# Dataset size (per split - train will get more, val/test fewer)
NUM_TRAIN_SAMPLES = 4000
NUM_VAL_SAMPLES = 500
NUM_TEST_SAMPLES = 500

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

def load_texts(path: str = SOURCE_TEXTS_PATH, split: str = None) -> list[dict]:
    """
    Load source texts from JSON file, optionally filtering by split.
    
    Args:
        path: Path to corpus JSON file
        split: If provided, only return texts with this split tag ("train", "val", "test")
    
    Returns:
        List of text records (dicts with 'id', 'text', 'source', and optionally 'split')
    """
    with open(path) as f:
        data = json.load(f)
    
    texts = data["texts"]
    
    if split is not None:
        # Filter by split tag
        filtered = [t for t in texts if t.get("split") == split]
        if not filtered:
            available_splits = set(t.get("split", "NONE") for t in texts)
            raise ValueError(
                f"No texts found with split='{split}'. "
                f"Available splits: {available_splits}. "
                f"Run add_splits_to_corpus.py first if corpus lacks split tags."
            )
        return filtered
    
    return texts


def load_texts_by_split(path: str = SOURCE_TEXTS_PATH) -> dict[str, list[dict]]:
    """
    Load corpus and return texts organized by split.
    
    Returns:
        Dict with keys 'train', 'val', 'test', each containing list of texts
    """
    with open(path) as f:
        data = json.load(f)
    
    texts = data["texts"]
    
    # Check if corpus has split tags
    if not any("split" in t for t in texts):
        raise ValueError(
            "Corpus has no split tags! Run add_splits_to_corpus.py first:\n"
            "  python add_splits_to_corpus.py source_texts.json source_texts_split.json"
        )
    
    splits = {"train": [], "val": [], "test": []}
    untagged = []
    
    for t in texts:
        split = t.get("split")
        if split in splits:
            splits[split].append(t)
        else:
            untagged.append(t)
    
    if untagged:
        print(f"WARNING: {len(untagged)} texts have no/invalid split tag and will be ignored")
    
    print(f"Corpus loaded: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])} texts")
    
    return splits


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
    Create a single training sample from a list of texts.
    
    IMPORTANT: Pass only texts from the same split to ensure zero leakage.
    
    Args:
        texts: List of text records (should be pre-filtered by split)
        num_words: Words per fragment
        text_a_idx: Optional specific index for text A
        text_b_idx: Optional specific index for text B
    
    Returns dict with:
        - prompt: formatted chat messages
        - fragment_a: source fragment A as string
        - fragment_b: source fragment B as string  
        - expected: list of interleaved words
        - expected_str: interleaved words as formatted string
        - text_a_id: source text A identifier
        - text_b_id: source text B identifier
    """
    if len(texts) < 2:
        raise ValueError(f"Need at least 2 texts to create sample, got {len(texts)}")
    
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
    return samples_to_dataset(samples)


def samples_to_dataset(samples: list[dict]) -> Dataset:
    """Convert list of sample dicts to HuggingFace Dataset."""
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
# DATASET GENERATION (SPLIT-AWARE)
# ============================================================================

def generate_samples_for_split(
    texts: list[dict],
    num_samples: int,
    num_words: int = NUM_WORDS,
    seed: int = SEED,
) -> list[dict]:
    """
    Generate samples using only the provided texts (should be pre-filtered by split).
    
    Args:
        texts: List of text records (pre-filtered to a single split)
        num_samples: Number of samples to generate
        num_words: Words per fragment
        seed: Random seed
    
    Returns:
        List of sample dicts
    """
    random.seed(seed)
    return [create_sample(texts, num_words) for _ in range(num_samples)]


def generate_all_splits(
    texts_path: str = SOURCE_TEXTS_PATH,
    num_train: int = NUM_TRAIN_SAMPLES,
    num_val: int = NUM_VAL_SAMPLES,
    num_test: int = NUM_TEST_SAMPLES,
    num_words: int = NUM_WORDS,
    seed: int = SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Generate train/val/test samples with zero text leakage.
    
    Each split's samples are generated only from texts tagged with that split.
    
    Args:
        texts_path: Path to corpus JSON with split tags
        num_train: Number of training samples
        num_val: Number of validation samples
        num_test: Number of test samples
        num_words: Words per fragment
        seed: Random seed
    
    Returns:
        (train_samples, val_samples, test_samples) as lists of dicts
    """
    # Load texts organized by split
    texts_by_split = load_texts_by_split(texts_path)
    
    print(f"\nGenerating samples with {num_words} words/fragment...")
    
    # Generate each split independently (different seed offsets for variety)
    train_samples = generate_samples_for_split(
        texts_by_split["train"], num_train, num_words, seed
    )
    val_samples = generate_samples_for_split(
        texts_by_split["val"], num_val, num_words, seed + 1000
    )
    test_samples = generate_samples_for_split(
        texts_by_split["test"], num_test, num_words, seed + 2000
    )
    
    print(f"Generated: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    
    return train_samples, val_samples, test_samples


def save_all_splits(
    train_samples: list[dict],
    val_samples: list[dict],
    test_samples: list[dict],
    base_path: str,
    seed: int = SEED,
) -> dict:
    """
    Save train/val/test samples to separate JSONL files.
    
    Args:
        train_samples, val_samples, test_samples: Sample lists
        base_path: Path like 'datasets/10words.jsonl'
                   Creates: 10words_train.jsonl, 10words_val.jsonl, 10words_test.jsonl
        seed: Random seed used (for metadata)
    
    Returns:
        Dict with file paths
    """
    base = Path(base_path)
    stem = base.stem
    suffix = base.suffix
    parent = base.parent
    
    train_path = parent / f"{stem}_train{suffix}"
    val_path = parent / f"{stem}_val{suffix}"
    test_path = parent / f"{stem}_test{suffix}"
    
    save_jsonl(train_samples, train_path)
    save_jsonl(val_samples, val_path)
    save_jsonl(test_samples, test_path)
    
    # Save metadata
    metadata = {
        "seed": seed,
        "num_words": train_samples[0].get("num_words") if train_samples else None,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "zero_text_leakage": True,  # Mark that this was generated properly
        "files": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        }
    }
    metadata_path = parent / f"{stem}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    return {
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path),
        "metadata": str(metadata_path),
    }


def load_splits(base_path: str) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Load train/val/test samples from separate JSONL files.
    
    Args:
        base_path: Path like 'datasets/10words.jsonl'
                   Looks for: 10words_train.jsonl, 10words_val.jsonl, 10words_test.jsonl
    
    Returns:
        (train_samples, val_samples, test_samples) as lists of dicts
    """
    base = Path(base_path)
    stem = base.stem
    suffix = base.suffix
    parent = base.parent
    
    train_path = parent / f"{stem}_train{suffix}"
    val_path = parent / f"{stem}_val{suffix}"
    test_path = parent / f"{stem}_test{suffix}"
    
    if not train_path.exists():
        raise FileNotFoundError(
            f"Split files not found. Expected: {train_path}\n"
            f"Generate with: python dataset_generator.py --save {base_path}"
        )
    
    train_samples = load_jsonl(train_path)
    val_samples = load_jsonl(val_path)
    test_samples = load_jsonl(test_path)
    
    print(f"Loaded: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    
    return train_samples, val_samples, test_samples


def generate_dataset(
    texts_path: str = SOURCE_TEXTS_PATH,
    num_train: int = NUM_TRAIN_SAMPLES,
    num_val: int = NUM_VAL_SAMPLES,
    num_test: int = NUM_TEST_SAMPLES,
    num_words: int = NUM_WORDS,
    seed: int = SEED,
    dataset_path: str = None,  # If provided, load from pre-saved files
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Generate or load HuggingFace Datasets for training, validation, and testing.
    
    Ensures zero text leakage by using corpus-level split tags.
    
    Args:
        texts_path: Path to corpus JSON with split tags
        num_train: Number of training samples (ignored if loading)
        num_val: Number of validation samples (ignored if loading)
        num_test: Number of test samples (ignored if loading)
        num_words: Words per fragment (ignored if loading)
        seed: Random seed (ignored if loading)
        dataset_path: If provided, load from pre-saved split files
    
    Returns:
        (train_dataset, val_dataset, test_dataset) as HuggingFace Datasets
    """
    if dataset_path:
        # Load from pre-saved files
        train_samples, val_samples, test_samples = load_splits(dataset_path)
    else:
        # Generate fresh
        train_samples, val_samples, test_samples = generate_all_splits(
            texts_path, num_train, num_val, num_test, num_words, seed
        )
    
    return (
        samples_to_dataset(train_samples),
        samples_to_dataset(val_samples),
        samples_to_dataset(test_samples),
    )


def generate_curriculum(
    output_dir: str,
    texts_path: str = SOURCE_TEXTS_PATH,
    num_train: int = NUM_TRAIN_SAMPLES,
    num_val: int = NUM_VAL_SAMPLES,
    num_test: int = NUM_TEST_SAMPLES,
    seed: int = SEED,
    stages: list[int] = None,
):
    """Generate datasets for all curriculum stages with zero text leakage."""
    if stages is None:
        stages = CURRICULUM_STAGES
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load texts once (verifies split tags exist)
    texts_by_split = load_texts_by_split(texts_path)
    
    print(f"\nGenerating curriculum datasets in {output_dir}/")
    print(f"Stages: {stages}")
    print(f"Samples per stage: train={num_train}, val={num_val}, test={num_test}")
    print()
    
    for num_words in stages:
        print(f"\n--- {num_words} words ---")
        
        # Generate samples for each split
        train = generate_samples_for_split(texts_by_split["train"], num_train, num_words, seed)
        val = generate_samples_for_split(texts_by_split["val"], num_val, num_words, seed + 1000)
        test = generate_samples_for_split(texts_by_split["test"], num_test, num_words, seed + 2000)
        
        # Save
        base_path = output_dir / f"{num_words}words.jsonl"
        save_all_splits(train, val, test, base_path, seed)
        
        # Show sample info
        sample = train[0]
        prompt_len = len(sample['prompt'][0]['content'])
        expected_len = len(sample['expected'])
        print(f"  prompt~{prompt_len} chars, expected~{expected_len} tokens")
    
    print()
    print("="*60)
    print("Done! Use with:")
    print(f"  python interleave_grpo.py --dataset {output_dir}/10words.jsonl")
    print()
    print("Zero text leakage verified: train/val/test use separate source texts")
    print("="*60)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate interleaving dataset (split-aware)")
    parser.add_argument("--texts", default=SOURCE_TEXTS_PATH, help="Path to corpus JSON with split tags")
    parser.add_argument("--num-train", type=int, default=NUM_TRAIN_SAMPLES, help="Training samples")
    parser.add_argument("--num-val", type=int, default=NUM_VAL_SAMPLES, help="Validation samples")
    parser.add_argument("--num-test", type=int, default=NUM_TEST_SAMPLES, help="Test samples")
    parser.add_argument("--num-words", type=int, default=NUM_WORDS, help="Words per fragment")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--preview", type=int, default=2, help="Number of samples to preview")
    
    # Save/load options
    parser.add_argument("--save", default=None, help="Save to JSONL (creates _train/_val/_test files)")
    parser.add_argument("--load", default=None, help="Load from pre-saved split files")
    
    # Curriculum generation
    parser.add_argument("--curriculum", action="store_true", help="Generate all curriculum stages")
    parser.add_argument("--output-dir", default="datasets", help="Output directory for curriculum")
    
    args = parser.parse_args()
    
    # Curriculum mode
    if args.curriculum:
        generate_curriculum(
            output_dir=args.output_dir,
            texts_path=args.texts,
            num_train=args.num_train,
            num_val=args.num_val,
            num_test=args.num_test,
            seed=args.seed,
        )
        exit(0)
    
    # Load mode
    if args.load:
        train_samples, val_samples, test_samples = load_splits(args.load)
        samples = train_samples  # Preview train by default
        num_words = samples[0].get("num_words", "unknown") if samples else "unknown"
        print(f"Fragment size: {num_words} words")
    else:
        # Generate mode
        print(f"Generating samples with {args.num_words} words per fragment...")
        print(f"  Train: {args.num_train}")
        print(f"  Val:   {args.num_val}")
        print(f"  Test:  {args.num_test}")
        print(f"Output format: {OUTPUT_FORMAT}")
        print()
        
        train_samples, val_samples, test_samples = generate_all_splits(
            texts_path=args.texts,
            num_train=args.num_train,
            num_val=args.num_val,
            num_test=args.num_test,
            num_words=args.num_words,
            seed=args.seed,
        )
        samples = train_samples  # Preview train
    
    # Save if requested
    if args.save:
        save_all_splits(train_samples, val_samples, test_samples, args.save, args.seed)
    
    # Preview
    print()
    print(f"=== Preview ({min(args.preview, len(samples))} train samples) ===")
    for i in range(min(args.preview, len(samples))):
        sample = samples[i]
        print(f"\nSample {i+1}:")
        print(f"  Text A ({sample['text_a_id']}): {sample['fragment_a'][:80]}...")
        print(f"  Text B ({sample['text_b_id']}): {sample['fragment_b'][:80]}...")
        print(f"  Expected ({len(sample['expected'])} words): {' '.join(sample['expected'][:10])}...")
        
        # Show prompt length
        prompt_len = len(sample['prompt'][0]['content'])
        print(f"  Prompt length: {prompt_len} chars")
    
    # Summary stats
    all_samples = train_samples + val_samples + test_samples
    if all_samples:
        prompt_lens = [len(s['prompt'][0]['content']) for s in all_samples]
        expected_lens = [len(s['expected']) for s in all_samples]
        print(f"\n=== Stats (all splits) ===")
        print(f"  Total samples: {len(all_samples)}")
        print(f"  Prompt length: {min(prompt_lens)}-{max(prompt_lens)} chars")
        print(f"  Expected length: {min(expected_lens)}-{max(expected_lens)} tokens")