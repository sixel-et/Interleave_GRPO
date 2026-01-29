"""
dataset_generator_unified.py

Generates training data for interleaving or sequential recitation tasks.
Samples fragment pairs from source texts and creates prompt + expected output.
Supports both interleaved output (A1 B1 A2 B2...) and sequential output (A1 A2... B1 B2...).

IMPORTANT: Source texts must have a "split" field (train/val/test) to ensure
zero text leakage between splits. Use add_splits_to_corpus.py to add splits
to your corpus first.

Usage:
    # Generate interleaving datasets (original behavior)
    python dataset_generator_unified.py --mode interleave --num-words 10 --save datasets/10words.jsonl

    # Generate sequential recitation datasets
    python dataset_generator_unified.py --mode sequential --num-words 10 --save datasets/10words_sequential.jsonl

    # Generate only test split for sequential (for evaluation)
    python dataset_generator_unified.py --mode sequential --only-split test --num-words 100 --save datasets/100words_sequential_test.jsonl

    # Generate curriculum for interleaving (10, 25, 50, 100, 200, 500 words)
    python dataset_generator_unified.py --mode interleave --curriculum --output-dir datasets/

    # Generate curriculum for sequential
    python dataset_generator_unified.py --mode sequential --curriculum --output-dir datasets_sequential/

    # Preview samples during generation (still saves files)
    python dataset_generator_unified.py --mode interleave --num-words 25 --preview 3 --save datasets/preview.jsonl

    # Print samples to stdout instead of saving files
    python dataset_generator_unified.py --mode sequential --num-words 10 --print-samples | head -5

    # Print curriculum samples to stdout
    python dataset_generator_unified.py --mode sequential --curriculum --print-samples | head -10
"""

import json
import random
from pathlib import Path

# Optional import for datasets library
try:
    from datasets import Dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    Dataset = None

# ============================================================================
# CONFIG - CHANGE THESE
# ============================================================================

# Paths
SOURCE_TEXTS_PATH = "source_texts_split.json"

# Dataset size (per split - train will get more, val/test fewer)
NUM_TRAIN_SAMPLES = 4000
NUM_VAL_SAMPLES = 500
NUM_TEST_SAMPLES = 500

# Special defaults for print mode
PRINT_TRAIN_SAMPLES = 0
PRINT_VAL_SAMPLES = 0
PRINT_TEST_SAMPLES = 0

# Fragment length (difficulty dial - start small, increase as model improves)
NUM_WORDS = 10  # Default to easiest level

# Random seed for reproducibility
SEED = 42

# Output format: "space" for "word1 word2 word3", "newline" for "word1\nword2\nword3"
OUTPUT_FORMAT = "newline"

# Curriculum stages
CURRICULUM_STAGES = [10, 25, 50, 100, 200, 500]

# Task mode
MODE = "interleave"  # Default to original interleaving behavior

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

PROMPT_TEMPLATE_INTERLEAVE = """You will run two processes in parallel, each outputting words from a text.

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

PROMPT_TEMPLATE_SEQUENTIAL = """You will run two processes in sequence.

First, output all words from:
{fragment_a}

Then, output all words from:
{fragment_b}

Do not add commentary, explanation, labels, or metadata.
Output one word per line, including any attached punctuation (e.g., "be," not "be").
Do not wait for additional input.
Output the first text completely, then the second text.
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
    mode: str = MODE,
) -> dict:
    """
    Create a single training sample from a list of texts.

    IMPORTANT: Pass only texts from the same split to ensure zero leakage.

    Args:
        texts: List of text records (should be pre-filtered by split)
        num_words: Words per fragment
        text_a_idx: Optional specific index for text A
        text_b_idx: Optional specific index for text B
        mode: "interleave" or "sequential"

    Returns dict with:
        - prompt: formatted chat messages
        - fragment_a: source fragment A as string
        - fragment_b: source fragment B as string
        - expected: list of words (interleaved or sequential)
        - expected_str: words as formatted string
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

    # Build expected output based on mode
    if mode == "interleave":
        expected = interleave_words(words_a, words_b)
        prompt_template = PROMPT_TEMPLATE_INTERLEAVE
    elif mode == "sequential":
        expected = words_a + words_b  # Concatenate for sequential
        prompt_template = PROMPT_TEMPLATE_SEQUENTIAL
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'interleave' or 'sequential'.")

    expected_str = format_expected(expected)

    # Build prompt
    user_content = prompt_template.format(
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
        "mode": mode,  # Track which task type this is
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
    if not HAS_DATASETS:
        raise ImportError("HuggingFace datasets library not available. Use --print-samples to test sample generation.")
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
    mode: str = MODE,
    interleave_ratio: int = 1,
    sequential_ratio: int = 1,
    pattern: str = "random",
) -> list[dict]:
    """
    Generate samples using only the provided texts (should be pre-filtered by split).

    Args:
        texts: List of text records (pre-filtered to a single split)
        num_samples: Number of samples to generate
        num_words: Words per fragment
        seed: Random seed
        mode: "interleave", "sequential", or "mixed"
        interleave_ratio: For mixed mode, ratio of interleave samples
        sequential_ratio: For mixed mode, ratio of sequential samples
        pattern: For mixed mode, "alternating" or "random"

    Returns:
        List of sample dicts
    """
    random.seed(seed)

    if mode == "mixed":
        # Generate sequence of modes
        total_ratio = interleave_ratio + sequential_ratio
        modes_sequence = []
        if pattern == "alternating":
            # Alternate blocks
            block_size = interleave_ratio + sequential_ratio
            full_blocks = num_samples // block_size
            remainder = num_samples % block_size

            for _ in range(full_blocks):
                modes_sequence.extend(["interleave"] * interleave_ratio)
                modes_sequence.extend(["sequential"] * sequential_ratio)

            # Handle remainder
            remaining_modes = (["interleave"] * interleave_ratio + ["sequential"] * sequential_ratio)[:remainder]
            modes_sequence.extend(remaining_modes)
        else:  # random
            # Random assignment based on ratio
            for _ in range(num_samples):
                r = random.random() * total_ratio
                if r < interleave_ratio:
                    modes_sequence.append("interleave")
                else:
                    modes_sequence.append("sequential")

        return [create_sample(texts, num_words, mode=m) for m in modes_sequence]
    else:
        return [create_sample(texts, num_words, mode=mode) for _ in range(num_samples)]


def generate_all_splits(
    texts_path: str = SOURCE_TEXTS_PATH,
    num_train: int = NUM_TRAIN_SAMPLES,
    num_val: int = NUM_VAL_SAMPLES,
    num_test: int = NUM_TEST_SAMPLES,
    num_words: int = NUM_WORDS,
    seed: int = SEED,
    mode: str = MODE,
    only_split: str = None,
    interleave_ratio: int = 1,
    sequential_ratio: int = 1,
    pattern: str = "random",
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
        mode: "interleave" or "sequential"
        only_split: If specified, generate only this split ("train", "val", or "test")

    Returns:
        (train_samples, val_samples, test_samples) as lists of dicts
        Empty lists for splits not generated if only_split is used
    """
    # Load texts organized by split
    texts_by_split = load_texts_by_split(texts_path)

    print(f"\nGenerating {mode} samples with {num_words} words/fragment...")
    if only_split:
        print(f"Generating only {only_split} split")

    # Generate splits (only the specified one if only_split is set)
    train_samples = []
    val_samples = []
    test_samples = []

    if not only_split or only_split == "train":
        train_samples = generate_samples_for_split(
            texts_by_split["train"], num_train, num_words, seed, mode,
            interleave_ratio, sequential_ratio, pattern
        )
    if not only_split or only_split == "val":
        val_samples = generate_samples_for_split(
            texts_by_split["val"], num_val, num_words, seed + 1000, mode,
            interleave_ratio, sequential_ratio, pattern
        )
    if not only_split or only_split == "test":
        test_samples = generate_samples_for_split(
            texts_by_split["test"], num_test, num_words, seed + 2000, mode,
            interleave_ratio, sequential_ratio, pattern
        )

    generated_splits = []
    if train_samples: generated_splits.append(f"train={len(train_samples)}")
    if val_samples: generated_splits.append(f"val={len(val_samples)}")
    if test_samples: generated_splits.append(f"test={len(test_samples)}")
    print(f"Generated: {', '.join(generated_splits)}")

    return train_samples, val_samples, test_samples


def save_all_splits(
    train_samples: list[dict],
    val_samples: list[dict],
    test_samples: list[dict],
    base_path: str,
    seed: int = SEED,
    mode: str = MODE,
) -> dict:
    """
    Save train/val/test samples to separate JSONL files.

    Args:
        train_samples, val_samples, test_samples: Sample lists (empty lists are skipped)
        base_path: Path like 'datasets/10words.jsonl'
                   Creates: 10words_train.jsonl, 10words_val.jsonl, 10words_test.jsonl (only for non-empty)
        seed: Random seed used (for metadata)
        mode: Task mode (for metadata)

    Returns:
        Dict with file paths for saved splits
    """
    base = Path(base_path)
    stem = base.stem
    suffix = base.suffix
    parent = base.parent

    saved_files = {}

    if train_samples:
        train_path = parent / f"{stem}_train{suffix}"
        save_jsonl(train_samples, train_path)
        saved_files["train"] = str(train_path)

    if val_samples:
        val_path = parent / f"{stem}_val{suffix}"
        save_jsonl(val_samples, val_path)
        saved_files["val"] = str(val_path)

    if test_samples:
        test_path = parent / f"{stem}_test{suffix}"
        save_jsonl(test_samples, test_path)
        saved_files["test"] = str(test_path)

    # Save metadata
    metadata = {
        "seed": seed,
        "num_words": (train_samples[0] if train_samples else val_samples[0] if val_samples else test_samples[0]).get("num_words") if any([train_samples, val_samples, test_samples]) else None,
        "mode": mode,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "zero_text_leakage": True,  # Mark that this was generated properly
        "files": saved_files,
    }
    metadata_path = parent / f"{stem}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    saved_files["metadata"] = str(metadata_path)
    return saved_files


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
            f"Generate with: python dataset_generator_unified.py --mode interleave --save {base_path}"
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
    mode: str = MODE,
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
        mode: "interleave" or "sequential"

    Returns:
        (train_dataset, val_dataset, test_dataset) as HuggingFace Datasets
    """
    if dataset_path:
        # Load from pre-saved files
        train_samples, val_samples, test_samples = load_splits(dataset_path)
    else:
        # Generate fresh
        train_samples, val_samples, test_samples = generate_all_splits(
            texts_path, num_train, num_val, num_test, num_words, seed, mode
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
    stages: list[int] = CURRICULUM_STAGES,
    mode: str = MODE,
    return_samples: bool = False,
    only_split: str = None,
    interleave_ratio: int = 1,
    sequential_ratio: int = 1,
    pattern: str = "random",
):
    """Generate datasets for all curriculum stages with zero text leakage."""
    if stages is None:
        stages = CURRICULUM_STAGES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load texts once (verifies split tags exist)
    texts_by_split = load_texts_by_split(texts_path)

    mode_suffix = "_sequential" if mode == "sequential" else ""
    if return_samples:
        print(f"\nGenerating {mode} curriculum samples...")
    else:
        print(f"\nGenerating {mode} curriculum datasets in {output_dir}/")
    print(f"Stages: {stages}")
    if only_split:
        split_count = {"train": num_train, "val": num_val, "test": num_test}[only_split]
        print(f"Samples per stage: {only_split}={split_count}")
    else:
        print(f"Samples per stage: train={num_train}, val={num_val}, test={num_test}")
    print()

    all_samples = []
    for num_words in stages:
        print(f"\n--- {num_words} words ---")

        # Generate samples for each split
        train, val, test = generate_all_splits(
            texts_path=texts_path,
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            num_words=num_words,
            seed=seed,
            mode=mode,
            only_split=only_split,
            interleave_ratio=interleave_ratio,
            sequential_ratio=sequential_ratio,
            pattern=pattern,
        )

        if return_samples:
            # Only collect the requested split(s)
            if only_split:
                if only_split == "train":
                    all_samples.extend(train)
                elif only_split == "val":
                    all_samples.extend(val)
                elif only_split == "test":
                    all_samples.extend(test)
            else:
                all_samples.extend(train + val + test)
        else:
            # Save
            base_path = output_dir / f"{num_words}words{mode_suffix}.jsonl"
            save_all_splits(train, val, test, base_path, seed, mode)

        # Show sample info
        sample_list = train or val or test
        if sample_list:
            sample = sample_list[0]
            prompt_len = len(sample['prompt'][0]['content'])
            expected_len = len(sample['expected'])
            print(f"  prompt~{prompt_len} chars, expected~{expected_len} tokens")

    if return_samples:
        return all_samples

    print()
    print("="*60)
    print("Done! Use with:")
    first_stage = stages[0] if stages else 10
    print(f"  python interleave_grpo.py --dataset {output_dir}/{first_stage}words{mode_suffix}.jsonl")
    print()
    print("Zero text leakage verified: train/val/test use separate source texts")
    print("="*60)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate dataset (interleave or sequential)")
    parser.add_argument("--texts", default=SOURCE_TEXTS_PATH, help="Path to corpus JSON with split tags")
    parser.add_argument("--num-train", type=int, default=None, help="Training samples")
    parser.add_argument("--num-val", type=int, default=None, help="Validation samples")
    parser.add_argument("--num-test", type=int, default=None, help="Test samples")
    parser.add_argument("--num-words", type=int, default=NUM_WORDS, help="Words per fragment")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--preview", type=int, default=2, help="Number of samples to preview")
    parser.add_argument("--mode", default=MODE, choices=["interleave", "sequential", "mixed"],
                        help="Task type: interleave, sequential, or mixed (interleave+sequential)")
    parser.add_argument("--interleave-ratio", type=int, default=1, help="For mixed mode: ratio of interleave samples (e.g., 3 for 3:1 interleave:sequential)")
    parser.add_argument("--sequential-ratio", type=int, default=1, help="For mixed mode: ratio of sequential samples (e.g., 1 for 3:1 interleave:sequential)")
    parser.add_argument("--pattern", choices=["alternating", "random"], default="random", help="For mixed mode: alternating blocks or random shuffle")
    parser.add_argument("--only-split", choices=["train", "val", "test"], default=None,
                        help="Generate only this split (default: generate all splits)")
    parser.add_argument("--print-samples", action="store_true",
                        help="Print generated samples to stdout as JSONL instead of creating Dataset objects")

    # Save/load options
    parser.add_argument("--save", default=None, help="Save to JSONL (creates _train/_val/_test files)")
    parser.add_argument("--load", default=None, help="Load from pre-saved split files")

    # Curriculum generation
    parser.add_argument("--curriculum", nargs='*', type=int, help="Generate curriculum stages (default: 10,25,50,100,200,500). With --print-samples, defaults to test split only")
    parser.add_argument("--output-dir", default="datasets", help="Output directory for curriculum")

    args = parser.parse_args()

    # Set defaults based on context
    if args.print_samples and args.curriculum is not None:
        # For print curriculum mode, default to 0 for val/test, 1 for test if not specified
        if args.num_train is None:
            args.num_train = 0
        if args.num_val is None:
            args.num_val = 0
        if args.num_test is None:
            args.num_test = PRINT_TEST_SAMPLES
    else:
        # Normal defaults
        if args.num_train is None:
            args.num_train = NUM_TRAIN_SAMPLES
        if args.num_val is None:
            args.num_val = NUM_VAL_SAMPLES
        if args.num_test is None:
            args.num_test = NUM_TEST_SAMPLES

    # Curriculum mode
    if args.curriculum is not None:
        # Use provided stages or default
        if args.curriculum:
            stages = args.curriculum
        else:
            stages = CURRICULUM_STAGES

        # No default split override for print-samples in curriculum mode

        # When printing samples in curriculum mode, default to the split(s) with non-zero counts
        if args.print_samples and args.curriculum is not None and not args.only_split:
            non_zero_splits = []
            if args.num_train > 0:
                non_zero_splits.append("train")
            if args.num_val > 0:
                non_zero_splits.append("val")
            if args.num_test > 0:
                non_zero_splits.append("test")
            if len(non_zero_splits) == 1:
                args.only_split = non_zero_splits[0]

        # When printing samples, limit to the specified num_words stage only if no stages provided
        if args.print_samples and not args.curriculum:
            stages = [args.num_words]

        if args.print_samples:
            # Generate and return all samples for printing
            all_samples = generate_curriculum(
                output_dir=args.output_dir,
                texts_path=args.texts,
                num_train=args.num_train,
                num_val=args.num_val,
                num_test=args.num_test,
                seed=args.seed,
                mode=args.mode,
                stages=stages,
                return_samples=True,
                only_split=args.only_split,
                interleave_ratio=args.interleave_ratio,
                sequential_ratio=args.sequential_ratio,
                pattern=args.pattern,
            )
            # Print samples to stdout
            import sys
            import json
            print("=== PRINTING CURRICULUM SAMPLES TO STDOUT (JSONL format) ===", file=sys.stderr)
            for sample in all_samples:
                print(json.dumps(sample))
        else:
            generate_curriculum(
                output_dir=args.output_dir,
                texts_path=args.texts,
                num_train=args.num_train,
                num_val=args.num_val,
                num_test=args.num_test,
                seed=args.seed,
                mode=args.mode,
                stages=stages,
                only_split=args.only_split,
                interleave_ratio=args.interleave_ratio,
                sequential_ratio=args.sequential_ratio,
                pattern=args.pattern,
            )
        exit(0)

    # Load mode
    if args.load:
        train_samples, val_samples, test_samples = load_splits(args.load)
        samples = train_samples  # Preview train by default
        num_words = samples[0].get("num_words", "unknown") if samples else "unknown"
        mode = samples[0].get("mode", "unknown") if samples else "unknown"
        print(f"Fragment size: {num_words} words")
        print(f"Mode: {mode}")

        # TEST FEATURE: Print samples to stdout instead of creating Dataset
        # This block can be easily removed later
        if args.print_samples:
            import sys
            import json
            print("=== PRINTING SAMPLES TO STDOUT (JSONL format) ===", file=sys.stderr)
            all_print_samples = []
            if train_samples:
                all_print_samples.extend(train_samples)
            if val_samples:
                all_print_samples.extend(val_samples)
            if test_samples:
                all_print_samples.extend(test_samples)
            for sample in all_print_samples:
                print(json.dumps(sample))
            exit(0)  # Skip the rest
    else:
        # Generate mode
        print(f"Generating {args.mode} samples with {args.num_words} words per fragment...")
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
            mode=args.mode,
            only_split=args.only_split,
            interleave_ratio=args.interleave_ratio,
            sequential_ratio=args.sequential_ratio,
            pattern=args.pattern,
        )
        # Preview the generated samples (prefer train, then val, then test)
        samples = train_samples or val_samples or test_samples

    # TEST FEATURE: Print samples to stdout instead of creating Dataset
    # This block can be easily removed later
    if args.print_samples:
        import sys
        import json
        print("=== PRINTING SAMPLES TO STDOUT (JSONL format) ===", file=sys.stderr)
        all_print_samples = []
        if train_samples:
            all_print_samples.extend(train_samples)
        if val_samples:
            all_print_samples.extend(val_samples)
        if test_samples:
            all_print_samples.extend(test_samples)
        for sample in all_print_samples:
            print(json.dumps(sample))
        exit(0)  # Skip the rest

    # Save if requested
    if args.save:
        save_all_splits(train_samples, val_samples, test_samples, args.save, args.seed, args.mode)

    # Preview
    print()
    # Determine which split we're previewing
    split_name = "train"
    if not train_samples and val_samples:
        split_name = "val"
    elif not train_samples and not val_samples and test_samples:
        split_name = "test"
    elif args.only_split:
        split_name = args.only_split

    print(f"=== Preview ({min(args.preview, len(samples))} {split_name} samples) ===")
    for i in range(min(args.preview, len(samples))):
        sample = samples[i]
        print(f"\nSample {i+1}:")
        print(f"  Text A ({sample['text_a_id']}): {sample['fragment_a'][:80]}...")
        print(f"  Text B ({sample['text_b_id']}): {sample['fragment_b'][:80]}...")
        print(f"  Expected ({len(sample['expected'])} words): {' '.join(sample['expected'][:10])}...")

        # Show prompt length
        prompt_len = len(sample['prompt'][0]['content'])
        print(f"  Prompt length: {prompt_len} chars")
        print(f"  Mode: {sample.get('mode', args.mode)}")

    # Summary stats
    all_samples = train_samples + val_samples + test_samples
    if all_samples:
        prompt_lens = [len(s['prompt'][0]['content']) for s in all_samples]
        expected_lens = [len(s['expected']) for s in all_samples]
        print(f"\n=== Stats (all splits) ===")
        print(f"  Total samples: {len(all_samples)}")
        print(f"  Prompt length: {min(prompt_lens)}-{max(prompt_lens)} chars")
        print(f"  Expected length: {min(expected_lens)}-{max(expected_lens)} tokens")