"""
evaluate_sequential.py

Standalone evaluation script for sequential recitation tasks.
Uses same reward logic as training.

Features for publication:
- Per-sample detailed results with export to CSV/JSON
- Alignment statistics and error analysis
- Score distribution visualization data
- Reproducible test set evaluation

Usage Examples:
    # Generate test-only dataset for evaluation
    python dataset_generator_unified.py --mode sequential --only-split test --num-words 100 --save datasets/100words_sequential_test.jsonl

    # Or generate samples on-the-fly for testing (if datasets library unavailable)
    python dataset_generator_unified.py --mode sequential --only-split test --num-words 100 --print-samples > datasets/100words_sequential_test.jsonl

    # Quick eval for sequential recitation - just print summary to stdout
    python evaluate_sequential.py --dataset datasets/10words_sequential_test.jsonl --model meta-llama/Llama-3.2-3B-Instruct --samples 100

    # Eval trained checkpoint on sequential task
    python evaluate_sequential.py --dataset datasets/100words_sequential_test.jsonl --model outputs/Llama-3B-interleave/checkpoint-3900 --samples 500

    # Verbose mode - show individual samples for sequential
    python evaluate_sequential.py --dataset datasets/100words_sequential_test.jsonl --model checkpoint-3900 --samples 50 --verbose --verbose-rate 10

    # Verbose with truncation for quick sanity check on sequential
    python evaluate_sequential.py --dataset datasets/100words_sequential_test.jsonl --model checkpoint-3900 --samples 20 --verbose --verbose-rate 5 --truncate 100

    # Export all formats for publication on sequential results
    python evaluate_sequential.py --dataset datasets/10words_sequential_test.jsonl --model checkpoint-3900 --samples 500 --export-all results/sequential/

    # Export just JSON (per-sample details for sequential)
    python evaluate_sequential.py --dataset datasets/10words_sequential_test.jsonl --model checkpoint-3900 --samples 500 --export-json results/sequential_results.json

    # Export just summary stats for sequential
    python evaluate_sequential.py --dataset datasets/10words_sequential_test.jsonl --model checkpoint-3900 --samples 500 --export-summary results/sequential_summary.json

    # Full test set evaluation with all exports for sequential
    python evaluate_sequential.py --dataset datasets/10words_sequential_test.jsonl --model checkpoint-3900 --samples 500 --export-all results/sequential_baseline/ --verbose --verbose-rate 100

    # Compare baseline vs trained on sequential (run separately, different output dirs)
    python evaluate_sequential.py --dataset datasets/100words_sequential_test.jsonl --model meta-llama/Llama-3.2-3B-Instruct --samples 500 --export-all results/sequential_baseline/
    python evaluate_sequential.py --dataset datasets/100words_sequential_test.jsonl --model checkpoint-3900 --samples 500 --export-all results/sequential_trained/

    # Curriculum evaluation across difficulties for sequential
    for words in 10 25 50 100 200 500; do
        python dataset_generator_unified.py --mode sequential --only-split test --num-words $words --save datasets/${words}words_sequential_test.jsonl
        python evaluate_sequential.py --dataset datasets/${words}words_sequential_test.jsonl --model checkpoint-3900 --samples 500 --export-all results/sequential_${words}words/
    done
"""

import torch
import json
import csv
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset_generator_unified import load_jsonl, samples_to_dataset
from reward import compute_alignment_score, parse_output, evaluate_single

# ============================================================================
# CONFIG
# ============================================================================

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
NUM_EVAL_SAMPLES = 100  # subset of test set
MAX_NEW_TOKENS = 2500

# ============================================================================
# RESULT DATA STRUCTURES
# ============================================================================

@dataclass
class SampleResult:
    """Detailed result for a single evaluation sample."""
    sample_id: int
    score: float
    raw_score: float
    expected_len: int
    output_len: int
    matches: int
    mismatches: int
    gaps: int
    text_a_id: str
    text_b_id: str
    fragment_a: str
    fragment_b: str
    prompt_text: str        # full prompt sent to model
    expected_str: str       # ground truth sequential output
    raw_completion: str     # raw model output before parsing
    output_str: str         # parsed model output used for scoring
    # Optional detailed alignment (for verbose mode)
    aligned_expected: Optional[list] = None
    aligned_output: Optional[list] = None


@dataclass
class EvalSummary:
    """Summary statistics for an evaluation run."""
    model_name: str
    dataset_path: str
    num_samples: int
    timestamp: str
    # Score statistics
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    median_score: float
    # Performance buckets
    perfect_count: int  # score == 1.0
    high_count: int     # score >= 0.9
    medium_count: int   # 0.5 <= score < 0.9
    low_count: int      # score < 0.5
    # Length statistics
    mean_expected_len: float
    mean_output_len: float
    # Error analysis
    total_matches: int
    total_mismatches: int
    total_gaps: int


# ============================================================================
# EVALUATION
# ============================================================================

def generate_completion(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    """Generate a single completion."""
    inputs = tokenizer.apply_chat_template(
        prompt,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    completion = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return completion


def evaluate_sample(
    model,
    tokenizer,
    sample: dict,
    sample_id: int,
    verbose: bool = False
) -> SampleResult:
    """Evaluate a single sample and return detailed results."""
    completion = generate_completion(model, tokenizer, sample["prompt"])
    output_words = parse_output(completion)

    # Get detailed alignment info
    result = evaluate_single(sample["expected"], completion, verbose=verbose)

    # Extract prompt text
    prompt = sample["prompt"]
    if isinstance(prompt, list) and len(prompt) > 0:
        prompt_text = prompt[0].get("content", str(prompt))
    else:
        prompt_text = str(prompt)

    return SampleResult(
        sample_id=sample_id,
        score=result["score"],
        raw_score=result["raw_score"],
        expected_len=result["expected_len"],
        output_len=result["output_len"],
        matches=result["matches"],
        mismatches=result["mismatches"],
        gaps=result["gaps"],
        text_a_id=sample.get("text_a_id", ""),
        text_b_id=sample.get("text_b_id", ""),
        fragment_a=sample["fragment_a"],
        fragment_b=sample["fragment_b"],
        prompt_text=prompt_text,
        expected_str=" ".join(sample["expected"]),
        raw_completion=completion,
        output_str=" ".join(output_words),
        aligned_expected=result.get("aligned_expected"),
        aligned_output=result.get("aligned_output"),
    )


def run_eval(
    model,
    tokenizer,
    dataset,
    num_samples: int = NUM_EVAL_SAMPLES,
    verbose: bool = False,
    verbose_rate: int = 10,
    truncate: Optional[int] = None,
    collect_alignments: bool = False,
) -> list[SampleResult]:
    """
    Run evaluation on dataset, return detailed per-sample results.

    Args:
        model: loaded model
        tokenizer: loaded tokenizer
        dataset: HuggingFace dataset with test samples
        num_samples: number of samples to evaluate
        verbose: print progress to stdout
        verbose_rate: how often to print verbose output
        truncate: truncate verbose output to N characters
        collect_alignments: include full alignments in results (memory intensive)

    Returns:
        List of SampleResult objects
    """
    num_samples = min(num_samples, len(dataset))
    results = []

    for i in range(num_samples):
        sample = dataset[i]
        result = evaluate_sample(
            model, tokenizer, sample, i,
            verbose=collect_alignments
        )
        results.append(result)

        # Show verbose output for every Nth sample
        if verbose and (i % verbose_rate == 0):
            print()
            print(f"{'='*60}")
            print(f"Sample {i+1}/{num_samples} - Score: {result.score:.3f}")
            print(f"{'='*60}")

            if truncate:
                # Truncated mode - for quick sanity check
                trunc_len = truncate
                print(f"Fragment A: {result.fragment_a[:trunc_len]}{'...' if len(result.fragment_a) > trunc_len else ''}")
                print(f"Fragment B: {result.fragment_b[:trunc_len]}{'...' if len(result.fragment_b) > trunc_len else ''}")
                print()
                print(f"Expected (sequential, {result.expected_len} words):")
                print(f"  {result.expected_str[:trunc_len]}{'...' if len(result.expected_str) > trunc_len else ''}")
                print()
                print(f"Model Output ({result.output_len} words):")
                print(f"  {result.output_str[:trunc_len]}{'...' if len(result.output_str) > trunc_len else ''}")
            else:
                # Full mode
                print(f"Fragment A:")
                print(result.fragment_a)
                print()
                print(f"Fragment B:")
                print(result.fragment_b)
                print()
                print(f"Expected (sequential, {result.expected_len} words):")
                print(result.expected_str)
                print()
                print(f"Model Output ({result.output_len} words):")
                print(result.output_str)

            print(f"\nAlignment: matches={result.matches}, mismatches={result.mismatches}, gaps={result.gaps}")
            print()

        if (i + 1) % 10 == 0:
            scores_so_far = [r.score for r in results]
            print(f"  {i+1}/{num_samples} - running avg: {sum(scores_so_far)/len(scores_so_far):.3f}")

    return results


def compute_summary(
    results: list[SampleResult],
    model_name: str,
    dataset_path: str,
) -> EvalSummary:
    """Compute summary statistics from detailed results."""
    import statistics

    scores = [r.score for r in results]

    return EvalSummary(
        model_name=model_name,
        dataset_path=dataset_path or "generated",
        num_samples=len(results),
        timestamp=datetime.now().isoformat(),
        # Score statistics
        mean_score=statistics.mean(scores),
        std_score=statistics.stdev(scores) if len(scores) > 1 else 0.0,
        min_score=min(scores),
        max_score=max(scores),
        median_score=statistics.median(scores),
        # Performance buckets
        perfect_count=sum(1 for s in scores if s >= 0.999),
        high_count=sum(1 for s in scores if s >= 0.9),
        medium_count=sum(1 for s in scores if 0.5 <= s < 0.9),
        low_count=sum(1 for s in scores if s < 0.5),
        # Length statistics
        mean_expected_len=statistics.mean(r.expected_len for r in results),
        mean_output_len=statistics.mean(r.output_len for r in results),
        # Error analysis
        total_matches=sum(r.matches for r in results),
        total_mismatches=sum(r.mismatches for r in results),
        total_gaps=sum(r.gaps for r in results),
    )


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_results_csv(results: list[SampleResult], path: str):
    """Export per-sample results to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Exclude alignment fields (too large for CSV)
    fieldnames = [
        'sample_id', 'score', 'raw_score', 'expected_len', 'output_len',
        'matches', 'mismatches', 'gaps', 'text_a_id', 'text_b_id',
        'fragment_a', 'fragment_b', 'prompt_text', 'expected_str',
        'raw_completion', 'output_str'
    ]

    with open(path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: getattr(r, k) for k in fieldnames}
            writer.writerow(row)

    print(f"Exported {len(results)} results to {path}")


def export_results_json(results: list[SampleResult], path: str, include_alignments: bool = False):
    """Export per-sample results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for r in results:
        d = asdict(r)
        if not include_alignments:
            d.pop('aligned_expected', None)
            d.pop('aligned_output', None)
        data.append(d)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"Exported {len(results)} results to {path}")


def export_summary_json(summary: EvalSummary, path: str):
    """Export summary statistics to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(asdict(summary), f, indent=2)

    print(f"Exported summary to {path}")


def export_score_distribution(results: list[SampleResult], path: str, bins: int = 20):
    """Export score distribution data for plotting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    scores = [r.score for r in results]

    # Create histogram data
    import numpy as np
    counts, bin_edges = np.histogram(scores, bins=bins, range=(0, 1))

    data = {
        "scores": scores,
        "histogram": {
            "counts": counts.tolist(),
            "bin_edges": bin_edges.tolist(),
            "bins": bins,
        },
        "percentiles": {
            "p5": float(np.percentile(scores, 5)),
            "p25": float(np.percentile(scores, 25)),
            "p50": float(np.percentile(scores, 50)),
            "p75": float(np.percentile(scores, 75)),
            "p95": float(np.percentile(scores, 95)),
        }
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Exported distribution data to {path}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate sequential recitation performance")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name or path")
    parser.add_argument("--dataset", default=None, help="Path to JSONL dataset file")
    parser.add_argument("--samples", type=int, default=NUM_EVAL_SAMPLES, help="Number of samples")
    parser.add_argument("--verbose", action="store_true", help="Show individual completions")
    parser.add_argument("--verbose-rate", type=int, default=10,
                        help="Show verbose output every N samples (default: 10)")
    parser.add_argument("--truncate", nargs='?', const=100, type=int, default=None,
                        help="Truncate verbose output to N characters")

    # Export options
    parser.add_argument("--export-csv", default=None, help="Export per-sample results to CSV")
    parser.add_argument("--export-json", default=None, help="Export per-sample results to JSON")
    parser.add_argument("--export-summary", default=None, help="Export summary statistics to JSON")
    parser.add_argument("--export-distribution", default=None, help="Export score distribution to JSON")
    parser.add_argument("--export-all", default=None,
                        help="Export all outputs to directory (creates results.csv, results.json, summary.json, distribution.json)")

    args = parser.parse_args()

    print(f"Loading model: {args.model}")

    # For local checkpoints, load tokenizer from base model
    import os
    if os.path.isdir(args.model):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load dataset
    if args.dataset:
        print(f"Loading dataset: {args.dataset}")
        samples = load_jsonl(args.dataset)
        dataset = samples_to_dataset(samples)
        print(f"Loaded {len(dataset)} samples")
    else:
        print("ERROR: No dataset specified. Use --dataset path/to/file.jsonl")
        exit(1)

    print(f"Evaluating on {args.samples} samples...")
    results = run_eval(
        model, tokenizer, dataset, args.samples,
        verbose=args.verbose,
        verbose_rate=args.verbose_rate,
        truncate=args.truncate
    )

    # Compute summary
    summary = compute_summary(results, args.model, args.dataset)

    # Print summary
    print()
    print("=" * 60)
    print(f"SEQUENTIAL RECITATION RESULTS ({summary.num_samples} samples)")
    print("=" * 60)
    print(f"Model: {summary.model_name}")
    print(f"Dataset: {summary.dataset_path}")
    print()
    print("Score Statistics:")
    print(f"  Mean:   {summary.mean_score:.4f} ± {summary.std_score:.4f}")
    print(f"  Median: {summary.median_score:.4f}")
    print(f"  Min:    {summary.min_score:.4f}")
    print(f"  Max:    {summary.max_score:.4f}")
    print()
    print("Performance Buckets:")
    print(f"  Perfect (≥0.999): {summary.perfect_count:4d} ({100*summary.perfect_count/summary.num_samples:.1f}%)")
    print(f"  High (≥0.9):      {summary.high_count:4d} ({100*summary.high_count/summary.num_samples:.1f}%)")
    print(f"  Medium (0.5-0.9): {summary.medium_count:4d} ({100*summary.medium_count/summary.num_samples:.1f}%)")
    print(f"  Low (<0.5):       {summary.low_count:4d} ({100*summary.low_count/summary.num_samples:.1f}%)")
    print()
    print("Alignment Analysis:")
    print(f"  Avg expected length: {summary.mean_expected_len:.1f} words")
    print(f"  Avg output length:   {summary.mean_output_len:.1f} words")
    print(f"  Total matches:       {summary.total_matches}")
    print(f"  Total mismatches:     {summary.total_mismatches}")
    print(f"  Total gaps:          {summary.total_gaps}")
    print("=" * 60)

    # Handle exports
    if args.export_all:
        export_dir = Path(args.export_all)
        export_results_csv(results, export_dir / "results.csv")
        export_results_json(results, export_dir / "results.json")
        export_summary_json(summary, export_dir / "summary.json")
        export_score_distribution(results, export_dir / "distribution.json")
    else:
        if args.export_csv:
            export_results_csv(results, args.export_csv)
        if args.export_json:
            export_results_json(results, args.export_json)
        if args.export_summary:
            export_summary_json(summary, args.export_summary)
        if args.export_distribution:
            export_score_distribution(results, args.export_distribution)