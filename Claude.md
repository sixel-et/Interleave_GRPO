# Interleave GRPO Project

## What This Is

Training Llama-3.2-3B to interleave two texts word-by-word using GRPO (Group Relative Policy Optimization). Given fragments from Text A and Text B, the model outputs: A1 B1 A2 B2 A3 B3...

This is a capability evaluation and potential training intervention for parallel cognitive state maintenance in LLMs.

## Current State

**Working:**
- GRPO training pipeline functional
- 10-word interleaving: baseline 0.486 → trained 0.992 alignment score
- Needleman-Wunsch sequence alignment for evaluation
- Sanity check logging (best/worst completions every N steps)
- Resume from checkpoint

**Problem:**
- 500-word training collapsed (all 16 generations identical, no gradient signal)
- Root cause: difficulty jump too large (10 → 500 words)
- Solution: curriculum learning (10 → 25 → 50 → 100 → 200 → 500)

**Also needed:**
- Source texts are too short - many < 500 words. Need to merge in longer texts from Grok.

## Key Files

```
interleave_grpo.py    # Main training script
dataset_generator.py  # Creates JSONL datasets, supports --curriculum
reward.py             # Needleman-Wunsch alignment scoring (nw_align)
evaluate.py           # Eval on held-out test set
source_texts.json     # Text corpus (needs longer texts)
```

## Usage

```bash
# Generate curriculum datasets
python dataset_generator.py --curriculum --output-dir datasets/

# Train stage 1
python interleave_grpo.py --dataset datasets/10words.jsonl

# Continue to stage 2 (loads checkpoint)
python interleave_grpo.py --dataset datasets/25words.jsonl --resume

# Evaluate
python evaluate.py --model outputs/Llama-3B-interleave/checkpoint-XXX
```

## Architecture Notes

- GRPO needs variance in generation scores. If all 16 completions are identical → zero gradient → collapse
- `sample_outputs.log` shows best/worst completions for debugging
- MAX_COMPLETION_LENGTH must fit output (~2x words per fragment)
- Generation time scales linearly with output length (500 words = 50x slower than 10 words)

## Prompt Template

Uses "process" framing (two parallel processes maintaining state). Key instruction: "Do this all in one turn" - without this, model waits for more input.

## Evaluation

Needleman-Wunsch alignment score: normalized 0-1, measures sequence similarity to ground truth interleaving. Current models get ~15-20% on novel text pairs without training.