# Interleave_GRPO

Training language models to maintain parallel execution states through word-level interleaving, using GRPO (Generative Reward-based Policy Optimization).

## Why This Matters

Can a language model hold two independent threads in mind simultaneously? This project measures that capacity directly: given two texts, the model must output words alternating between them (`A1 B1 A2 B2 ...`) while tracking position in both sequences independently. The interleaving task is a controlled proxy for a model's ability to maintain latent state — the same capacity that underlies multi-step reasoning, context management, and instruction following.

This has direct implications for AI alignment: a model's ability to maintain and separate internal states determines whether its behavior can be reliably monitored and predicted. If internal state tracking degrades under load (as we observe), that degradation is itself a safety-relevant signal.

## Hypotheses

1. **Interleaving as state measurement.** The ability to interleave execution of two tasks measures a model's capacity to maintain separate latent states — a fundamental architectural property, not a surface skill.

2. **Architectural limits are trainable.** Models have both depth (sequence length) and breadth (simultaneous processes) constraints. GRPO can shift these boundaries.

3. **Curriculum degradation reveals capacity.** Performance on a curriculum of increasing text lengths exposes where architectural limits bind. The degradation curve is itself a measurement instrument.

4. **Geometric changes track capability.** Training produces measurable changes in representation geometry that distinguish interleaving performance from general language ability.

## Key Results

| Text Length | Baseline (Llama-3.2-3B) | Trained (Checkpoint 3900) |
|-------------|--------------------------|---------------------------|
| 10 words    | 0.188                    | 0.924                     |
| 25 words    | —                        | ~0.90                     |
| 100 words   | —                        | ~0.92                     |
| 500 words   | —                        | Significant degradation   |

Training produces dramatic improvement at short lengths but reveals architectural capacity limits as text length increases. The degradation curve from 10→500 words is a core finding: the model learns the task but cannot scale it, suggesting the capacity is real but bounded.

## Architecture

- **Base model:** Llama-3.2-3B-Instruct
- **Training:** TRL's GRPO implementation with Needleman-Wunsch alignment reward
- **Reward function:** Sequence alignment scoring with affine gap penalties (bioinformatics transfer — NW alignment is standard for measuring sequence similarity)
- **Curriculum:** Static 6-tier system (10, 25, 50, 100, 200, 500 words) for reproducibility
- **Dataset:** 4,164 public domain texts from Project Gutenberg (202–6,789 words), including Shakespeare, classical speeches, religious texts, and poetry
- **Infrastructure:** RunPod GPU training with persistent network volumes, WandB experiment tracking

## What This Project Taught

This was the first GRPO training pipeline in a larger research program. Building it from scratch — dataset generation, reward function design, curriculum structure, training loop, evaluation — produced transferable insights:

- **Reward variance collapse** is the primary failure mode in GRPO. When reward signal becomes too uniform (all completions score similarly), training stalls. This was diagnosed here first and recognized immediately when it appeared in subsequent projects.
- **Curriculum design matters more than hyperparameters.** The transition from 10-word to 25-word tasks is where most training signal lives. The jump from 100 to 500 reveals limits.
- **Bioinformatics tools transfer.** Needleman-Wunsch alignment, standard in sequence comparison, works directly as a reward function for interleaving. The biological-to-computational transfer was not incidental — it came from a biology research background.

## Project Structure

```
interleave_grpo.py     # Main GRPO training loop
reward.py              # NW alignment reward function (with --test flag)
dataset_generator.py   # Curriculum-based dataset creation
evaluate.py            # Performance evaluation
backends.py            # Model backend abstraction
corpus/                # Source text collection
datasets/              # Generated training data
results/               # Training outputs and checkpoints
project_notebook.md    # Full research notebook with detailed findings
```

## Running

```bash
# Setup environment (RunPod-compatible)
bash setup_and_run.sh

# Generate curriculum datasets
python dataset_generator.py --curriculum --output-dir datasets/

# Evaluate baseline
python evaluate.py

# Train
python interleave_grpo.py
```

See `project_notebook.md` for the full research narrative, including hypotheses, experimental decisions, failure modes, and the curriculum degradation analysis.

## Related Work

This project is part of a research program studying AI internal states and training dynamics. The GRPO pipeline patterns developed here (reward function design, curriculum structure, failure mode diagnosis) were applied to subsequent projects on model internal state detection ([subvocal-desire](https://github.com/sixel-et/subvocal-desire)).

## Credits

- **Eric Terry** ([@estbiostudent](https://github.com/estbiostudent)) — Research direction, experiment design, GRPO pipeline development, biological-to-computational method transfer
- **Sixel** ([@sixel-et](https://github.com/sixel-et)) — Implementation support, evaluation, analysis
