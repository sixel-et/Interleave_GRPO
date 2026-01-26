#!/usr/bin/env python3
"""
Add train/val/test split field to each text record in corpus.

This permanently assigns each text to a split, ensuring:
- Zero text leakage between splits
- Reproducible splits across all dataset generations
- Explicit split membership for each text

Usage: python add_splits_to_corpus.py input.json output.json [--seed 42]
"""

import json
import random
import sys
from pathlib import Path


def add_splits_to_corpus(
    input_path: str,
    output_path: str,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
):
    """
    Add 'split' field to each text in corpus.
    
    Args:
        input_path: Path to corpus JSON without splits
        output_path: Path to save corpus JSON with splits
        train_split: Fraction for train (default 0.8)
        val_split: Fraction for val (default 0.1)
        test_split: Fraction for test (default 0.1)
        seed: Random seed for reproducible splits
    """
    print("="*80)
    print("ADD TRAIN/VAL/TEST SPLITS TO CORPUS")
    print("="*80)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Splits: train={train_split:.0%}, val={val_split:.0%}, test={test_split:.0%}")
    print(f"Seed:   {seed}")
    print()
    
    # Validate splits sum to 1.0
    if abs(train_split + val_split + test_split - 1.0) > 0.001:
        raise ValueError(f"Splits must sum to 1.0, got {train_split + val_split + test_split}")
    
    # Load corpus
    print("Loading corpus...")
    with open(input_path) as f:
        data = json.load(f)
    
    texts = data["texts"]
    print(f"Loaded {len(texts)} texts")
    
    # Check if splits already exist
    if any("split" in t for t in texts):
        print("\n⚠ WARNING: Some texts already have 'split' field!")
        response = input("Overwrite existing splits? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            return
    
    # Shuffle texts deterministically
    print(f"\nShuffling texts with seed={seed}...")
    random.seed(seed)
    indices = list(range(len(texts)))
    random.shuffle(indices)
    
    # Calculate split boundaries
    n_train = int(len(texts) * train_split)
    n_val = int(len(texts) * val_split)
    
    train_indices = set(indices[:n_train])
    val_indices = set(indices[n_train:n_train + n_val])
    test_indices = set(indices[n_train + n_val:])
    
    print(f"Split boundaries:")
    print(f"  Train: {len(train_indices)} texts (indices 0-{n_train-1})")
    print(f"  Val:   {len(val_indices)} texts (indices {n_train}-{n_train+n_val-1})")
    print(f"  Test:  {len(test_indices)} texts (indices {n_train+n_val}-{len(texts)-1})")
    
    # Assign splits
    print("\nAssigning splits...")
    for i, text in enumerate(texts):
        if i in train_indices:
            text["split"] = "train"
        elif i in val_indices:
            text["split"] = "val"
        elif i in test_indices:
            text["split"] = "test"
        else:
            raise ValueError(f"Text {i} not assigned to any split!")
    
    # Verify
    train_count = sum(1 for t in texts if t["split"] == "train")
    val_count = sum(1 for t in texts if t["split"] == "val")
    test_count = sum(1 for t in texts if t["split"] == "test")
    
    print(f"\nVerification:")
    print(f"  Train: {train_count} texts ({train_count/len(texts)*100:.1f}%)")
    print(f"  Val:   {val_count} texts ({val_count/len(texts)*100:.1f}%)")
    print(f"  Test:  {test_count} texts ({test_count/len(texts)*100:.1f}%)")
    print(f"  Total: {train_count + val_count + test_count} texts")
    
    if train_count + val_count + test_count != len(texts):
        raise ValueError("Split counts don't match total texts!")
    
    # Save
    print(f"\nSaving to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved {len(texts)} texts with splits ({file_size_mb:.2f} MB)")
    
    # Show examples
    print("\nExample text records:")
    for split_type in ["train", "val", "test"]:
        example = next(t for t in texts if t["split"] == split_type)
        print(f"\n  {split_type.upper()}:")
        print(f"    ID: {example['id']}")
        print(f"    Source: {example['source'][:60]}...")
        print(f"    Split: {example['split']}")
        word_count = len(example['text'].split())
        print(f"    Words: {word_count}")
    
    print("\n" + "="*80)
    print("✓ SPLITS ADDED SUCCESSFULLY")
    print("="*80)
    print()
    print("Next step: Generate datasets using split-filtered texts")
    print("  python dataset_generator.py --texts corpus_with_splits.json --curriculum")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add train/val/test splits to corpus")
    parser.add_argument("input", help="Input corpus JSON (without splits)")
    parser.add_argument("output", help="Output corpus JSON (with splits)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--train", type=float, default=0.8, help="Train fraction (default: 0.8)")
    parser.add_argument("--val", type=float, default=0.1, help="Val fraction (default: 0.1)")
    parser.add_argument("--test", type=float, default=0.1, help="Test fraction (default: 0.1)")
    
    args = parser.parse_args()
    
    add_splits_to_corpus(
        input_path=args.input,
        output_path=args.output,
        train_split=args.train,
        val_split=args.val,
        test_split=args.test,
        seed=args.seed,
    )