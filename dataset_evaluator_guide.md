# Dataset Generator and Evaluator Guide

This guide explains the new dataset generation and evaluation tools created to test sequential recitation performance alongside interleaving tasks.

## Overview

The original `dataset_generator.py` and `evaluate.py` were designed only for interleaving tasks (A1 B1 A2 B2...). To test whether interleaving training degrades sequential performance (A1 A2... B1 B2...), we created:

1. **`dataset_generator_unified.py`** - Unified generator for both interleaving and sequential tasks
2. **`evaluate_sequential.py`** - Evaluator specifically for sequential recitation tasks

## dataset_generator_unified.py

### Purpose
Generate datasets for both interleaving and sequential tasks using the same codebase and corpus split logic.

### Key Features

#### Dual Task Support
- **`--mode interleave`**: Original behavior (A1 B1 A2 B2...)
- **`--mode sequential`**: Sequential output (A1 A2... B1 B2...)

#### Flexible Generation
- **`--only-split train/val/test`**: Generate from single split only (for evaluation datasets)
- **`--num-words N`**: Control fragment length per text
- **`--print-samples`**: Output JSONL to stdout instead of creating Dataset objects (for testing without `datasets` library)

#### Safe Defaults
- Corpus: `source_texts_split.json` (prevents train/val/test leakage)
- Mode: `interleave` (backward compatible with original)
- Generates all splits by default (can be overridden with `--only-split`)

### Usage Examples

```bash
# Generate interleaving dataset (original behavior)
python dataset_generator_unified.py --mode interleave --num-words 10 --save datasets/10words.jsonl

# Generate sequential dataset for evaluation
python dataset_generator_unified.py --mode sequential --only-split test --num-words 100 --save datasets/100words_sequential_test.jsonl

# Test generation without datasets library
python dataset_generator_unified.py --mode sequential --num-words 5 --print-samples | head -3

# Generate curriculum for sequential task
python dataset_generator_unified.py --mode sequential --curriculum --output-dir datasets_sequential/
```

### Output Format

Each sample contains:
- `prompt`: Chat-formatted prompt for the model
- `fragment_a`/`fragment_b`: Input text fragments
- `expected`: List of expected output words
- `expected_str`: Formatted expected output
- `text_a_id`/`text_b_id`: Source text identifiers
- `mode`: Task type ("interleave" or "sequential")

## evaluate_sequential.py

### Purpose
Evaluate model performance on sequential recitation tasks using the same alignment scoring as the original evaluator.

### Key Features

#### Sequential-Specific Evaluation
- Optimized prompts and descriptions for sequential tasks
- Uses identical scoring logic (Needleman-Wunsch alignment) for fair comparison
- Generates detailed performance reports

#### Flexible Input
- Accepts any JSONL dataset (works with single-split datasets from `--only-split`)
- Command-line evaluation without dataset creation dependencies

#### Comprehensive Reporting
- Per-sample results with alignment details
- Summary statistics (mean, std, performance buckets)
- Export options (CSV, JSON, summary, distribution plots)

### Usage Examples

```bash
# Quick evaluation of sequential performance
python evaluate_sequential.py --dataset datasets/100words_sequential_test.jsonl --model your_checkpoint --samples 100

# Compare sequential vs interleaving performance
python evaluate_sequential.py --dataset datasets/100words_sequential_test.jsonl --model checkpoint_interleave --samples 500 --export-all results/sequential_results/

# Evaluate baseline (untrained) model on sequential task
python evaluate_sequential.py --dataset datasets/100words_sequential_test.jsonl --model meta-llama/Llama-3.2-3B-Instruct --samples 100
```

### Output Metrics

- **Score**: 0-1 alignment score (1.0 = perfect)
- **Performance Buckets**:
  - Perfect: ≥0.999
  - High: ≥0.9
  - Medium: 0.5-0.9
  - Low: <0.5
- **Alignment Details**: Matches, mismatches, gaps
- **Length Statistics**: Expected vs actual output lengths

## Testing the Components

Use `test_components.py` to verify core functionality without external dependencies:

```bash
python test_components.py
```

Tests cover:
- Text processing (`extract_words`, `interleave_words`, `format_expected`)
- Sample generation (`create_sample` for both modes)
- Output parsing (`parse_output`)
- Alignment scoring (`compute_alignment_score`)

## Workflow for Performance Comparison

1. **Train on Interleaving**:
   ```bash
   python interleave_grpo.py --dataset datasets/10words.jsonl
   ```

2. **Generate Sequential Test Data**:
   ```bash
   python dataset_generator_unified.py --mode sequential --only-split test --num-words 100 --save datasets/100words_sequential_test.jsonl
   ```

3. **Evaluate Sequential Performance**:
   ```bash
   python evaluate_sequential.py --dataset datasets/100words_sequential_test.jsonl --model your_checkpoint --samples 500 --export-all results/sequential_eval/
   ```

4. **Compare to Interleaving Baseline**:
   ```bash
   python evaluate.py --dataset datasets/100words_test.jsonl --model your_checkpoint --samples 500 --export-all results/interleave_eval/
   ```

If sequential scores are significantly lower than interleaving scores, it suggests performance degradation.

## Corpus and Split Management

- **Default Corpus**: `source_texts_split.json` with pre-assigned train/val/test splits
- **Leakage Prevention**: `--only-split test` ensures evaluation data comes only from test-tagged texts
- **Split Assignment**: Run `add_splits_to_corpus.py` if corpus lacks splits

## Dependencies

- **Required**: `transformers`, `torch`
- **Optional**: `datasets` (for Dataset objects), `numba` (for fast alignment)
- **Testing**: No external dependencies required for `test_components.py`