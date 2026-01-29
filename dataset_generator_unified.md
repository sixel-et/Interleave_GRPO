# dataset_generator_unified.py

Unified dataset generator for interleaving and sequential text tasks.

## Overview

This script generates training and evaluation datasets for:
- **Interleaving**: A1 B1 A2 B2... (original task)
- **Sequential**: A1 A2... B1 B2... (new task for testing degradation)
- **Mixed**: Combination of interleaving and sequential tasks with configurable ratios and patterns

## Command Line Arguments

### Core Arguments
- `--mode {interleave,sequential,mixed}`: Task type (default: interleave)
- `--interleave-ratio N`: For mixed mode, ratio of interleave samples (default: 1)
- `--sequential-ratio N`: For mixed mode, ratio of sequential samples (default: 1)
- `--pattern {alternating,random}`: For mixed mode, alternating blocks or random shuffle (default: random)
- `--num-words N`: Words per text fragment (default: 10)
- `--texts PATH`: Path to corpus JSON with split tags (default: source_texts_split.json)

### Sample Control
- `--num-train N`: Training samples per split (default: 4000)
- `--num-val N`: Validation samples per split (default: 500)
- `--num-test N`: Test samples per split (default: 500)
- `--seed N`: Random seed for reproducibility (default: 42)

### Split Control
- `--only-split {train,val,test}`: Generate only this split (default: generate all splits)

### Output Control
- `--save PATH`: Save datasets to JSONL files (creates _train/_val/_test files)
- `--load PATH`: Load from existing split files
- `--curriculum [STAGES]`: Generate curriculum datasets (default stages: 10, 25, 50, 100, 200, 500)
- `--output-dir DIR`: Output directory for curriculum (default: datasets)

### Testing/Development
- `--preview N`: Show N sample previews during generation (can be combined with --save, default: 2)
- `--print-samples`: Print samples as JSONL to stdout instead of saving to files

### Examples

```bash
# Generate interleaving dataset
python dataset_generator_unified.py --mode interleave --num-words 10 --save datasets/10words.jsonl

# Generate sequential test-only dataset
python dataset_generator_unified.py --mode sequential --only-split test --num-words 100 --save datasets/100words_sequential_test.jsonl

# Preview samples during generation (and save)
python dataset_generator_unified.py --mode sequential --num-words 5 --preview 3 --save datasets/preview.jsonl

# Generate curriculum for sequential (default stages)
python dataset_generator_unified.py --mode sequential --curriculum --output-dir datasets_sequential/

# Generate custom curriculum stages
python dataset_generator_unified.py --mode interleave --curriculum 25 50 100 --output-dir datasets_custom/

# Print samples to stdout (no files created)
python dataset_generator_unified.py --mode sequential --num-words 10 --print-samples > test.jsonl

# Print curriculum samples (defaults to test split with --print-samples)
python dataset_generator_unified.py --mode sequential --curriculum 25 50 --print-samples | head -5

# Print mixed curriculum samples (8 prompts: 4 per stage with 3 interleave + 1 sequential alternating pattern)
python dataset_generator_unified.py --mode mixed --interleave-ratio 3 --sequential-ratio 1 --pattern alternating --curriculum 1 2 --print-samples --num-test 4

# Print single test sample
python dataset_generator_unified.py --mode interleave --num-words 25 --print-samples --only-split test --num-test 1

# Generate mixed mode dataset (3:1 interleave:sequential, alternating)
python dataset_generator_unified.py --mode mixed --interleave-ratio 3 --sequential-ratio 1 --pattern alternating --save datasets/mixed.jsonl
```

## Output Format

Each sample is a JSON object with:
- `prompt`: Chat-formatted prompt array
- `fragment_a`/`fragment_b`: Input text fragments
- `expected`: List of expected output words
- `expected_str`: Formatted expected output
- `text_a_id`/`text_b_id`: Source text identifiers
- `mode`: Task type ("interleave" or "sequential")
- `num_words`: Words per fragment

## Corpus Requirements

Corpus JSON must have:
- `texts` array with objects containing:
  - `id`: Unique identifier
  - `text`: Full text content
  - `split`: "train", "val", or "test"

Use `add_splits_to_corpus.py` to add split tags if missing.