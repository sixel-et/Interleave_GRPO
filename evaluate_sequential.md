# evaluate_sequential.py

Evaluation script for sequential recitation performance.

## Overview

Evaluates model performance on sequential text tasks (A1 A2... B1 B2...) using Needleman-Wunsch alignment scoring. Designed to test whether interleaving training degrades sequential abilities.

## Command Line Arguments

### Required
- `--dataset PATH`: Path to JSONL dataset file

### Model
- `--model PATH`: Model name or path (default: meta-llama/Llama-3.2-3B-Instruct)

### Evaluation
- `--samples N`: Number of samples to evaluate (default: 100)
- `--verbose`: Show individual completions
- `--verbose-rate N`: Show verbose output every N samples (default: 10)
- `--truncate N`: Truncate verbose output to N characters

### Export
- `--export-csv PATH`: Export per-sample results to CSV
- `--export-json PATH`: Export per-sample results to JSON
- `--export-summary PATH`: Export summary statistics to JSON
- `--export-distribution PATH`: Export score distribution to JSON
- `--export-all DIR`: Export all formats to directory

### Examples

```bash
# Quick evaluation
python evaluate_sequential.py --dataset datasets/100words_sequential_test.jsonl --samples 100

# Evaluate trained model
python evaluate_sequential.py --dataset datasets/100words_sequential_test.jsonl --model checkpoint-3900 --samples 500

# Verbose evaluation with truncation
python evaluate_sequential.py --dataset datasets/100words_sequential_test.jsonl --model checkpoint-3900 --samples 50 --verbose --truncate 100

# Export all results
python evaluate_sequential.py --dataset datasets/100words_sequential_test.jsonl --model checkpoint-3900 --samples 500 --export-all results/sequential/
```

## Output Metrics

### Summary Statistics
- **Mean/Median/Min/Max Score**: Alignment scores (0-1, 1.0 = perfect)
- **Performance Buckets**:
  - Perfect: ≥0.999
  - High: ≥0.9
  - Medium: 0.5-0.9
  - Low: <0.5

### Alignment Details
- **Matches/Mismatches/Gaps**: Per-sample alignment statistics
- **Expected/Output Lengths**: Sequence length comparisons

## Scoring

Uses Needleman-Wunsch alignment with:
- Match score: +2.0
- Mismatch penalty: -1.0
- Gap open: -5.0
- Gap extend: -1.0

Scores normalized to 0-1 range for RL compatibility.

## Dependencies

- `transformers`
- `torch`
- Optional: `numba` (falls back to pure Python if unavailable)