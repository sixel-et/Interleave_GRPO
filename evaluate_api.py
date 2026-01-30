"""
evaluate_api.py

API-based evaluation script for frontier models on the interleave task.
Supports evaluation of Claude, GPT-4, and other API-based models.

Reuses existing infrastructure from evaluate.py:
- SampleResult and EvalSummary dataclasses
- compute_summary() function
- All export functions (CSV, JSON, distribution)
- Verbose output formatting

New features:
- Pluggable backend system for different APIs
- Cost tracking and --max-cost budget control
- Rate limit handling with automatic retries

Usage Examples:
    # Quick test (5 samples, cost-limited)
    python evaluate_api.py \\
        --backend anthropic \\
        --model claude-sonnet-4-20250514 \\
        --dataset datasets/10words_test.jsonl \\
        --samples 5 \\
        --max-cost 1.00 \\
        --verbose

    # Full evaluation with Sonnet
    python evaluate_api.py \\
        --backend anthropic \\
        --model claude-sonnet-4-20250514 \\
        --dataset datasets/10words_test.jsonl \\
        --samples 100 \\
        --export-all results/sonnet_10w/

    # Curriculum evaluation
    for words in 10 25 50 100 200 500; do
        python evaluate_api.py \\
            --backend anthropic \\
            --model claude-sonnet-4-20250514 \\
            --dataset datasets/${words}words_test.jsonl \\
            --samples 100 \\
            --max-cost 20.00 \\
            --export-all results/sonnet_${words}w/
    done
"""

import asyncio
import json
import csv
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

import tiktoken

from backends import get_backend, CompletionBackend, PRICING


# ============================================================================
# TOKENIZER
# ============================================================================

# Use cl100k_base (GPT-4/Claude-approximate) for token estimation
_tokenizer = tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str) -> int:
    """Estimate token count for a string."""
    return len(_tokenizer.encode(text))
from dataset_generator import load_jsonl, samples_to_dataset
from reward import parse_output, evaluate_single, evaluate_single_lines

# ============================================================================
# CONFIG
# ============================================================================

MAX_NEW_TOKENS = 35000  # Increased for 6k tasks (~27600 tokens needed)
NUM_EVAL_SAMPLES = 100


# ============================================================================
# COLLAPSE DETECTION
# ============================================================================

def detect_repetition(words: list[str], threshold: int = 5) -> bool:
    """
    Detect if the model output has collapsed into repetitive text.

    Args:
        words: List of words from model output
        threshold: Number of consecutive identical words to trigger detection

    Returns:
        True if same word appears threshold+ times consecutively
    """
    if len(words) < threshold:
        return False
    for i in range(len(words) - threshold + 1):
        if len(set(words[i:i + threshold])) == 1:
            return True
    return False

# ============================================================================
# RESULT DATA STRUCTURES (imported from evaluate.py pattern)
# ============================================================================

@dataclass
class SampleResult:
    """Detailed result for a single evaluation sample."""
    sample_id: int
    score: float           # composite: word_score * format_score
    word_score: float      # NW alignment of words only
    format_score: float    # proportion of correctly formatted lines
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
    expected_str: str       # ground truth interleaved output
    raw_completion: str     # raw model output before parsing
    output_str: str         # parsed model output used for scoring
    # Optional detailed alignment (for verbose mode)
    aligned_expected: Optional[list] = None
    aligned_output: Optional[list] = None
    # Diagnostic fields for profiling
    api_time: float = 0.0           # Time spent on API call (seconds)
    score_time: float = 0.0         # Time spent on scoring (seconds)
    stop_reason: str = ""           # "end_turn" or "max_tokens"
    has_repetition: bool = False    # Collapse detection flag
    # Line-level format metrics
    empty_lines: int = 0            # Count of empty lines in output
    multiword_lines: int = 0        # Count of lines with >1 word


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
    # Cost tracking (new for API evaluation)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    # Timing statistics
    total_api_time: float = 0.0
    avg_api_time: float = 0.0
    min_api_time: float = 0.0
    max_api_time: float = 0.0
    # Diagnostic statistics
    max_tokens_count: int = 0       # samples that hit max_tokens
    repetition_count: int = 0       # samples with detected repetition


# ============================================================================
# API COMPLETION
# ============================================================================

def generate_completion(
    backend: CompletionBackend,
    prompt: list[dict],
    max_new_tokens: int = MAX_NEW_TOKENS
) -> tuple[str, dict]:
    """
    Generate a completion using the API backend.

    Args:
        backend: CompletionBackend instance
        prompt: List of message dicts with 'role' and 'content'
        max_new_tokens: Maximum tokens to generate

    Returns:
        Tuple of (text, metadata) where metadata includes timing and stop_reason
    """
    # Convert prompt format if needed (ensure it's a list of dicts)
    if isinstance(prompt, list) and len(prompt) > 0:
        if isinstance(prompt[0], dict):
            messages = prompt
        else:
            # Assume it's a list of strings, convert to user message
            messages = [{"role": "user", "content": " ".join(str(p) for p in prompt)}]
    else:
        messages = [{"role": "user", "content": str(prompt)}]

    return backend.generate(messages, max_new_tokens)


def evaluate_sample(
    backend: CompletionBackend,
    sample: dict,
    sample_id: int,
    verbose: bool = False
) -> SampleResult:
    """Evaluate a single sample and return detailed results."""
    import time

    # Dynamic max_tokens: tokenize expected output format + 30% overhead
    expected_output = "\n".join(sample["expected"])
    base_tokens = estimate_tokens(expected_output)
    max_tokens = int(base_tokens * 1.3)

    # Generate completion with timing
    completion, api_metadata = generate_completion(backend, sample["prompt"], max_tokens)

    # Parse and score with timing (line-level alignment)
    score_start = time.perf_counter()
    output_words = parse_output(completion)
    result = evaluate_single_lines(expected_output, completion, verbose=verbose)
    score_time = time.perf_counter() - score_start

    # Detect repetition/collapse
    has_repetition = detect_repetition(output_words)

    # Extract prompt text
    prompt = sample["prompt"]
    if isinstance(prompt, list) and len(prompt) > 0:
        prompt_text = prompt[0].get("content", str(prompt))
    else:
        prompt_text = str(prompt)

    return SampleResult(
        sample_id=sample_id,
        score=result["score"],
        word_score=result.get("word_score", result["score"]),
        format_score=result.get("format_score", 1.0),
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
        expected_str=sample.get("expected_str", "\n".join(sample["expected"])),
        raw_completion=completion,
        output_str=" ".join(output_words),
        aligned_expected=result.get("aligned_expected"),
        aligned_output=result.get("aligned_output"),
        # Diagnostic fields
        api_time=api_metadata.get("api_time", 0.0),
        score_time=score_time,
        stop_reason=api_metadata.get("stop_reason", ""),
        has_repetition=has_repetition,
        # Line-level format metrics
        empty_lines=result.get("empty_lines", 0),
        multiword_lines=result.get("multiword_lines", 0),
    )


async def async_evaluate_sample(
    backend: CompletionBackend,
    sample: dict,
    sample_id: int,
    verbose: bool = False
) -> SampleResult:
    """Async version of evaluate_sample for parallel evaluation."""
    import time

    # Dynamic max_tokens: tokenize expected output format + 30% overhead
    expected_output = "\n".join(sample["expected"])
    base_tokens = estimate_tokens(expected_output)
    max_tokens = int(base_tokens * 1.3)

    # Convert prompt format
    prompt = sample["prompt"]
    if isinstance(prompt, list) and len(prompt) > 0:
        if isinstance(prompt[0], dict):
            messages = prompt
        else:
            messages = [{"role": "user", "content": " ".join(str(p) for p in prompt)}]
    else:
        messages = [{"role": "user", "content": str(prompt)}]

    # Generate completion with timing (async)
    completion, api_metadata = await backend.async_generate(messages, max_tokens)

    # Parse and score with timing (line-level alignment)
    score_start = time.perf_counter()
    output_words = parse_output(completion)
    result = evaluate_single_lines(expected_output, completion, verbose=verbose)
    score_time = time.perf_counter() - score_start

    # Detect repetition/collapse
    has_repetition = detect_repetition(output_words)

    # Extract prompt text
    if isinstance(prompt, list) and len(prompt) > 0:
        prompt_text = prompt[0].get("content", str(prompt))
    else:
        prompt_text = str(prompt)

    return SampleResult(
        sample_id=sample_id,
        score=result["score"],
        word_score=result.get("word_score", result["score"]),
        format_score=result.get("format_score", 1.0),
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
        expected_str=sample.get("expected_str", "\n".join(sample["expected"])),
        raw_completion=completion,
        output_str=" ".join(output_words),
        aligned_expected=result.get("aligned_expected"),
        aligned_output=result.get("aligned_output"),
        # Diagnostic fields
        api_time=api_metadata.get("api_time", 0.0),
        score_time=score_time,
        stop_reason=api_metadata.get("stop_reason", ""),
        has_repetition=has_repetition,
        # Line-level format metrics
        empty_lines=result.get("empty_lines", 0),
        multiword_lines=result.get("multiword_lines", 0),
    )


# ============================================================================
# EVALUATION LOOP
# ============================================================================

def load_results_json(path: str) -> list[dict]:
    """Load existing results from JSON file for resumption."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_eval(
    backend: CompletionBackend,
    dataset,
    num_samples: int = NUM_EVAL_SAMPLES,
    max_cost: Optional[float] = None,
    verbose: bool = False,
    verbose_rate: int = 10,
    truncate: Optional[int] = None,
    collect_alignments: bool = False,
    resume_path: Optional[str] = None,
    export_json_path: Optional[str] = None,
) -> list[SampleResult]:
    """
    Run evaluation on dataset, return detailed per-sample results.

    Args:
        backend: CompletionBackend instance
        dataset: HuggingFace dataset with test samples
        num_samples: number of samples to evaluate
        max_cost: stop if cost exceeds this threshold (USD)
        verbose: print progress to stdout
        verbose_rate: how often to print verbose output
        truncate: truncate verbose output to N characters
        collect_alignments: include full alignments in results
        resume_path: path to partial results JSON to resume from
        export_json_path: path to save results after each sample (for auto-save)

    Returns:
        List of SampleResult objects
    """
    num_samples = min(num_samples, len(dataset))

    # Handle resumption
    if resume_path and Path(resume_path).exists():
        existing = load_results_json(resume_path)
        # Reconstruct SampleResult objects from dicts
        results = []
        for r in existing:
            # Handle optional fields that might not exist in older exports
            r.setdefault("api_time", 0.0)
            r.setdefault("score_time", 0.0)
            r.setdefault("stop_reason", "")
            r.setdefault("has_repetition", False)
            r.setdefault("aligned_expected", None)
            r.setdefault("aligned_output", None)
            results.append(SampleResult(**r))
        start_idx = len(results)
        print(f"Resuming from {resume_path}: {start_idx} samples already completed")
    else:
        results = []
        start_idx = 0

    for i in range(start_idx, num_samples):
        # Check cost budget before each sample
        if max_cost and backend.get_cost() >= max_cost:
            print(f"\nStopping: cost ${backend.get_cost():.2f} reached --max-cost ${max_cost:.2f}")
            break

        sample = dataset[i]
        result = evaluate_sample(
            backend, sample, i,
            verbose=collect_alignments
        )
        results.append(result)

        # Auto-save after each sample if export path provided
        if export_json_path:
            export_results_json(results, export_json_path, quiet=True)

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
                print(f"Expected ({result.expected_len} words):")
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
                print(f"Expected ({result.expected_len} words):")
                print(result.expected_str)
                print()
                print(f"Model Output ({result.output_len} words):")
                print(result.output_str)

            print(f"\nAlignment: matches={result.matches}, mismatches={result.mismatches}, gaps={result.gaps}")
            print()

        # Progress output with timing info
        if (i + 1) % 10 == 0 or i == start_idx:
            scores_so_far = [r.score for r in results]
            cost_so_far = backend.get_cost()
            # Format timing and stop_reason
            timing_str = f"{result.api_time:.1f}s"
            stop_str = result.stop_reason or "unknown"
            warning = " ⚠️" if result.stop_reason == "max_tokens" or result.has_repetition else ""
            print(f"  {i+1}/{num_samples} - avg: {sum(scores_so_far)/len(scores_so_far):.2f} - cost: ${cost_so_far:.2f} - last: {timing_str} ({stop_str}){warning}")

    return results


async def async_run_eval(
    backend: CompletionBackend,
    dataset,
    num_samples: int = NUM_EVAL_SAMPLES,
    max_concurrent: int = 3,
    max_cost: Optional[float] = None,
    resume_path: Optional[str] = None,
    export_json_path: Optional[str] = None,
) -> list[SampleResult]:
    """
    Async parallel evaluation on dataset.

    Args:
        backend: CompletionBackend instance (must support async_generate)
        dataset: HuggingFace dataset with test samples
        num_samples: number of samples to evaluate
        max_concurrent: maximum concurrent API calls
        max_cost: stop if cost exceeds this threshold (USD)
        resume_path: path to partial results JSON to resume from
        export_json_path: path to save results periodically

    Returns:
        List of SampleResult objects
    """
    num_samples = min(num_samples, len(dataset))

    # Handle resumption
    if resume_path and Path(resume_path).exists():
        existing = load_results_json(resume_path)
        results_dict = {}
        for r in existing:
            r.setdefault("api_time", 0.0)
            r.setdefault("score_time", 0.0)
            r.setdefault("stop_reason", "")
            r.setdefault("has_repetition", False)
            r.setdefault("aligned_expected", None)
            r.setdefault("aligned_output", None)
            results_dict[r["sample_id"]] = SampleResult(**r)
        print(f"Resuming from {resume_path}: {len(results_dict)} samples already completed")
    else:
        results_dict = {}

    # Determine which samples still need to be evaluated
    pending_indices = [i for i in range(num_samples) if i not in results_dict]
    print(f"Evaluating {len(pending_indices)} samples with {max_concurrent} concurrent calls...")

    semaphore = asyncio.Semaphore(max_concurrent)
    results_lock = asyncio.Lock()
    completed = [0]  # Use list for mutable counter in closure

    async def eval_one(idx: int) -> Optional[SampleResult]:
        # Check cost budget
        if max_cost and backend.get_cost() >= max_cost:
            return None

        async with semaphore:
            sample = dataset[idx]
            result = await async_evaluate_sample(backend, sample, idx)

            # Thread-safe update
            async with results_lock:
                results_dict[idx] = result
                completed[0] += 1

                # Progress update
                scores = [r.score for r in results_dict.values()]
                avg_score = sum(scores) / len(scores)
                cost = backend.get_cost()
                warning = " ⚠️" if result.stop_reason == "max_tokens" or result.has_repetition else ""
                print(f"  {completed[0]}/{len(pending_indices)} - avg: {avg_score:.2f} - cost: ${cost:.2f} - "
                      f"sample {idx}: {result.score:.2f} in {result.api_time:.1f}s ({result.stop_reason}){warning}")

                # Periodic save
                if export_json_path and completed[0] % max_concurrent == 0:
                    sorted_results = [results_dict[i] for i in sorted(results_dict.keys())]
                    export_results_json(sorted_results, export_json_path, quiet=True)

            return result

    # Run all pending evaluations
    await asyncio.gather(*[eval_one(i) for i in pending_indices])

    # Final save
    if export_json_path:
        sorted_results = [results_dict[i] for i in sorted(results_dict.keys())]
        export_results_json(sorted_results, export_json_path, quiet=True)

    # Return results in order
    return [results_dict[i] for i in sorted(results_dict.keys())]


# ============================================================================
# SUMMARY COMPUTATION
# ============================================================================

def compute_summary(
    results: list[SampleResult],
    model_name: str,
    dataset_path: str,
    backend: Optional[CompletionBackend] = None,
) -> EvalSummary:
    """Compute summary statistics from detailed results."""
    import statistics

    scores = [r.score for r in results]

    # Get cost info from backend if available
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost_usd = 0.0
    if backend:
        total_input_tokens = getattr(backend, 'total_input_tokens', 0)
        total_output_tokens = getattr(backend, 'total_output_tokens', 0)
        total_cost_usd = backend.get_cost()

    # Timing statistics
    api_times = [r.api_time for r in results]
    total_api_time = sum(api_times)
    avg_api_time = statistics.mean(api_times) if api_times else 0.0
    min_api_time = min(api_times) if api_times else 0.0
    max_api_time = max(api_times) if api_times else 0.0

    # Diagnostic statistics
    max_tokens_count = sum(1 for r in results if r.stop_reason == "max_tokens")
    repetition_count = sum(1 for r in results if r.has_repetition)

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
        # Cost tracking
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cost_usd=total_cost_usd,
        # Timing statistics
        total_api_time=total_api_time,
        avg_api_time=avg_api_time,
        min_api_time=min_api_time,
        max_api_time=max_api_time,
        # Diagnostic statistics
        max_tokens_count=max_tokens_count,
        repetition_count=repetition_count,
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
        'raw_completion', 'output_str',
        # Diagnostic fields
        'api_time', 'score_time', 'stop_reason', 'has_repetition'
    ]

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: getattr(r, k) for k in fieldnames}
            writer.writerow(row)

    print(f"Exported {len(results)} results to {path}")


def export_results_json(results: list[SampleResult], path: str, include_alignments: bool = False, quiet: bool = False):
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

    if not quiet:
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

    parser = argparse.ArgumentParser(
        description="Evaluate frontier models on interleaving task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with Claude Sonnet
  python evaluate_api.py --backend anthropic --model claude-sonnet-4-20250514 \\
      --dataset datasets/10words_test.jsonl --samples 5 --verbose

  # Full evaluation with cost limit
  python evaluate_api.py --backend anthropic --model claude-sonnet-4-20250514 \\
      --dataset datasets/100words_test.jsonl --samples 100 --max-cost 10.00 \\
      --export-all results/sonnet_100w/
        """
    )

    # Backend options
    parser.add_argument("--backend", default="anthropic",
                        help="Backend to use (anthropic, openai, google, xai, deepseek)")
    parser.add_argument("--model", default=None,
                        help="Model ID (uses backend default if not specified)")

    # Dataset options
    parser.add_argument("--dataset", required=True,
                        help="Path to JSONL dataset file")
    parser.add_argument("--samples", type=int, default=NUM_EVAL_SAMPLES,
                        help="Number of samples to evaluate")

    # Cost control
    parser.add_argument("--max-cost", type=float, default=None,
                        help="Stop if cost exceeds this threshold (USD)")

    # Parallelism
    parser.add_argument("--parallel", type=int, default=None,
                        help="Number of concurrent API calls (default: sequential)")

    # Output options
    parser.add_argument("--verbose", action="store_true",
                        help="Show individual completions")
    parser.add_argument("--verbose-rate", type=int, default=10,
                        help="Show verbose output every N samples (default: 10)")
    parser.add_argument("--truncate", nargs='?', const=100, type=int, default=None,
                        help="Truncate verbose output to N characters")

    # Resumption
    parser.add_argument("--resume", default=None,
                        help="Resume from partial results JSON file")

    # Export options
    parser.add_argument("--export-csv", default=None,
                        help="Export per-sample results to CSV")
    parser.add_argument("--export-json", default=None,
                        help="Export per-sample results to JSON")
    parser.add_argument("--export-summary", default=None,
                        help="Export summary statistics to JSON")
    parser.add_argument("--export-distribution", default=None,
                        help="Export score distribution to JSON")
    parser.add_argument("--export-all", default=None,
                        help="Export all outputs to directory")

    args = parser.parse_args()

    # Initialize backend
    print(f"Initializing {args.backend} backend...")
    backend = get_backend(args.backend, args.model)
    print(f"Model: {backend.name}")

    # Show pricing info
    model_id = args.model or backend.model if hasattr(backend, 'model') else 'unknown'
    if model_id in PRICING:
        pricing = PRICING[model_id]
        print(f"Pricing: ${pricing['input']}/1M input, ${pricing['output']}/1M output")

    if args.max_cost:
        print(f"Cost limit: ${args.max_cost:.2f}")

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    samples = load_jsonl(args.dataset)
    dataset = samples_to_dataset(samples)
    print(f"Loaded {len(dataset)} samples")

    # Determine export JSON path for auto-save
    export_json_path = args.export_json
    if args.export_all:
        export_json_path = str(Path(args.export_all) / "results.json")

    # Run evaluation
    print(f"\nEvaluating on {args.samples} samples...")
    if args.parallel:
        print(f"Running with {args.parallel} concurrent calls")
        results = asyncio.run(async_run_eval(
            backend, dataset, args.samples,
            max_concurrent=args.parallel,
            max_cost=args.max_cost,
            resume_path=args.resume,
            export_json_path=export_json_path,
        ))
    else:
        results = run_eval(
            backend, dataset, args.samples,
            max_cost=args.max_cost,
            verbose=args.verbose,
            verbose_rate=args.verbose_rate,
            truncate=args.truncate,
            resume_path=args.resume,
            export_json_path=export_json_path,
        )

    # Compute summary
    summary = compute_summary(results, backend.name, args.dataset, backend)

    # Print summary
    print()
    print("=" * 60)
    print(f"RESULTS ({summary.num_samples} samples)")
    print("=" * 60)
    print(f"Model: {summary.model_name}")
    print(f"Dataset: {summary.dataset_path}")
    print()
    print("Score Statistics:")
    print(f"  Mean:   {summary.mean_score:.4f} +/- {summary.std_score:.4f}")
    print(f"  Median: {summary.median_score:.4f}")
    print(f"  Min:    {summary.min_score:.4f}")
    print(f"  Max:    {summary.max_score:.4f}")
    print()
    print("Performance Buckets:")
    print(f"  Perfect (>=0.999): {summary.perfect_count:4d} ({100*summary.perfect_count/summary.num_samples:.1f}%)")
    print(f"  High (>=0.9):      {summary.high_count:4d} ({100*summary.high_count/summary.num_samples:.1f}%)")
    print(f"  Medium (0.5-0.9):  {summary.medium_count:4d} ({100*summary.medium_count/summary.num_samples:.1f}%)")
    print(f"  Low (<0.5):        {summary.low_count:4d} ({100*summary.low_count/summary.num_samples:.1f}%)")
    print()
    print("Alignment Analysis:")
    print(f"  Avg expected length: {summary.mean_expected_len:.1f} words")
    print(f"  Avg output length:   {summary.mean_output_len:.1f} words")
    print(f"  Total matches:       {summary.total_matches}")
    print(f"  Total mismatches:    {summary.total_mismatches}")
    print(f"  Total gaps:          {summary.total_gaps}")
    print()
    print("Cost Summary:")
    print(f"  Input tokens:  {summary.total_input_tokens:,}")
    print(f"  Output tokens: {summary.total_output_tokens:,}")
    print(f"  Total cost:    ${summary.total_cost_usd:.2f}")
    if args.max_cost:
        remaining = args.max_cost - summary.total_cost_usd
        print(f"  Remaining budget: ${remaining:.2f}")
    print()
    print("Timing Summary:")
    print(f"  Total API time:  {summary.total_api_time:.1f}s")
    print(f"  Avg per call:    {summary.avg_api_time:.2f}s")
    print(f"  Min/Max call:    {summary.min_api_time:.2f}s / {summary.max_api_time:.2f}s")
    print()
    print("Diagnostics:")
    pct_max_tokens = 100 * summary.max_tokens_count / summary.num_samples if summary.num_samples > 0 else 0
    pct_repetition = 100 * summary.repetition_count / summary.num_samples if summary.num_samples > 0 else 0
    print(f"  Samples hitting max_tokens: {summary.max_tokens_count} ({pct_max_tokens:.0f}%)")
    print(f"  Samples with repetition:    {summary.repetition_count} ({pct_repetition:.0f}%)")
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
