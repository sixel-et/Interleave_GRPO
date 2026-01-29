# test_components.py

Unit testing framework for dataset generation and evaluation components.

## Overview

Tests core functions from `dataset_generator_unified.py` and `evaluate_sequential.py` (via `reward_for_new_evaluate.py`) independently of external files and dependencies.

## Command Line Arguments

This script has no command line arguments. Run it directly:

```bash
python test_components.py
```

## Tests Covered

### Text Processing
- `extract_words()`: Splits text into words preserving punctuation
- `interleave_words()`: Interleaves two word lists with continuation
- `format_expected()`: Formats word lists to expected output strings

### Sample Generation
- `create_sample()`: Generates complete training samples for both interleaving and sequential modes

### Output Processing
- `parse_output()`: Parses model outputs, handles formats, filters metadata

### Alignment Scoring
- `compute_alignment_score()`: Scores sequence alignment using Needleman-Wunsch

## Test Data

Uses synthetic mock data:
- Mock texts with train/val/test splits
- Controlled word sequences for predictable outputs
- Edge cases (empty inputs, mismatched lengths, punctuation)

## Output

Shows detailed test results:
- Function purpose and inputs
- Expected vs actual outputs
- Pass/fail status for each test

Example output:
```
Testing extract_words...
  Purpose: Split text into words, keeping punctuation attached
  Input: 'The quick brown fox jumps'
  Expected: ['The', 'quick', 'brown', 'fox', 'jumps']
  Actual: ['The', 'quick', 'brown', 'fox', 'jumps']
✓ extract_words tests passed
```

## Dependencies

- `dataset_generator_unified.py`
- `reward_for_new_evaluate.py`
- No external libraries required

## Usage

Run all tests:
```bash
python test_components.py
```

Tests should pass in any environment with the required Python files.