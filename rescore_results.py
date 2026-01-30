"""
rescore_results.py

Rescore existing evaluation results using line-level alignment.
This preserves the raw_completion and expected_str but recalculates scores
to account for formatting (newlines).

Usage:
    python rescore_results.py results/sonnet_500w_50samples.json
    python rescore_results.py --all  # Rescore all result files
"""

import json
import sys
from pathlib import Path
from reward import evaluate_single_lines, tokenize_with_newlines


def rescore_result(result: dict, expected_words: list = None) -> dict:
    """
    Rescore a single result using line-level alignment.

    Args:
        result: Original result dict with raw_completion and expected_str
        expected_words: Optional list of expected words (if expected_str not reliable)

    Returns:
        Updated result dict with new scores
    """
    raw_completion = result['raw_completion']

    # Build expected_str with newlines if needed
    if expected_words:
        expected_str = '\n'.join(expected_words)
    else:
        expected_str = result.get('expected_str', '')
        # Check if it's space-separated (old format) and convert
        if '\n' not in expected_str and ' ' in expected_str:
            expected_str = '\n'.join(expected_str.split())

    # Rescore with line-level alignment
    new_result = evaluate_single_lines(expected_str, raw_completion)

    # Update result with new scores, keeping original data
    result['score'] = new_result['score']
    result['word_score'] = new_result['word_score']
    result['format_score'] = new_result['format_score']
    result['raw_score'] = new_result['raw_score']
    result['expected_len'] = new_result['expected_len']
    result['output_len'] = new_result['output_len']
    result['matches'] = new_result['matches']
    result['mismatches'] = new_result['mismatches']
    result['gaps'] = new_result['gaps']
    result['empty_lines'] = new_result['empty_lines']
    result['multiword_lines'] = new_result['multiword_lines']

    # Store the properly formatted expected_str
    result['expected_str'] = expected_str

    return result


def rescore_file(filepath: str, dataset_path: str = None) -> None:
    """
    Rescore all results in a file.

    Args:
        filepath: Path to results JSON file
        dataset_path: Optional path to dataset JSONL for expected words
    """
    filepath = Path(filepath)
    print(f"Rescoring {filepath}...")

    # Load results
    with open(filepath) as f:
        results = json.load(f)

    # Load dataset if provided (for accurate expected words)
    dataset = None
    if dataset_path:
        with open(dataset_path) as f:
            dataset = [json.loads(line) for line in f]

    # Rescore each result
    for i, result in enumerate(results):
        expected_words = None
        if dataset and i < len(dataset):
            expected_words = dataset[i].get('expected')

        rescore_result(result, expected_words)

    # Save to new file
    output_path = filepath.parent / f"{filepath.stem}_linescored{filepath.suffix}"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved to {output_path}")

    # Print summary
    print_summary(results)


def print_summary(results: list) -> None:
    """Print summary statistics."""
    scores = [r['score'] for r in results]
    word_scores = [r.get('word_score', r['score']) for r in results]
    format_scores = [r.get('format_score', 1.0) for r in results]
    empty_lines = [r.get('empty_lines', 0) for r in results]
    multiword_lines = [r.get('multiword_lines', 0) for r in results]

    print(f"\n  Samples: {len(results)}")
    print(f"  Composite score: {sum(scores)/len(scores):.3f}")
    print(f"  Word score:      {sum(word_scores)/len(word_scores):.3f}")
    print(f"  Format score:    {sum(format_scores)/len(format_scores):.3f}")
    print(f"  High (>=0.9): {sum(1 for s in scores if s >= 0.9)}/{len(scores)}")
    print(f"  Low (<0.5): {sum(1 for s in scores if s < 0.5)}/{len(scores)}")
    print(f"  Avg empty lines: {sum(empty_lines)/len(empty_lines):.1f}")
    print(f"  Avg multiword lines: {sum(multiword_lines)/len(multiword_lines):.1f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python rescore_results.py <results.json> [dataset.jsonl]")
        print("       python rescore_results.py --all")
        sys.exit(1)

    if sys.argv[1] == '--all':
        # Rescore all result files
        result_files = [
            ('results/sonnet_200w_50samples.json', 'api_datasets/200words_test.jsonl'),
            ('results/sonnet_500w_50samples.json', 'api_datasets/500words_test.jsonl'),
            ('results/sonnet_750w_50samples.json', 'api_datasets/750words_test.jsonl'),
            ('results/sonnet_1000w_50samples.json', 'api_datasets/1000words_test.jsonl'),
        ]
        for result_path, dataset_path in result_files:
            if Path(result_path).exists():
                rescore_file(result_path, dataset_path)
                print()
    else:
        result_path = sys.argv[1]
        dataset_path = sys.argv[2] if len(sys.argv) > 2 else None
        rescore_file(result_path, dataset_path)


if __name__ == '__main__':
    main()
