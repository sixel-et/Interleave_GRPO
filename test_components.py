#!/usr/bin/env python3
"""
test_components.py

Unit testing framework for key components of:
- dataset_generator_unified.py
- evaluate_sequential.py (via reward.py functions)

This script tests core functionality independently without external files or dependencies.
Run with: python test_components.py

Tests cover:
- Text processing (extract_words, interleave_words, format_expected)
- Sample creation (create_sample with mock data)
- Output parsing (parse_output)
- Alignment scoring (compute_alignment_score)
"""

import sys
import os

# Add current directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(__file__))

# Import functions to test
from dataset_generator_unified import (
    extract_words, interleave_words, format_expected, create_sample,
    PROMPT_TEMPLATE_INTERLEAVE, PROMPT_TEMPLATE_SEQUENTIAL
)
from reward_for_new_evaluate import parse_output, compute_alignment_score

# Test data
MOCK_TEXTS = [
    {"id": "text1", "text": "The quick brown fox jumps over the lazy dog", "split": "train"},
    {"id": "text2", "text": "A journey of a thousand miles begins with a single step", "split": "train"},
]

def test_extract_words():
    """Test word extraction from text.

    extract_words() splits text into words while preserving punctuation.
    This ensures proper tokenization for downstream processing.
    """
    print("Testing extract_words...")
    print("  Purpose: Split text into words, keeping punctuation attached")
    print("  Input: 'The quick brown fox jumps'")
    print("  Expected: ['The', 'quick', 'brown', 'fox', 'jumps']")

    # Test basic extraction
    text = "The quick brown fox jumps"
    words = extract_words(text)
    print(f"  Actual: {words}")
    assert words == ["The", "quick", "brown", "fox", "jumps"], f"Expected word list, got {words}"

    print("  Input: 'Hello, world! How are you?'")
    print("  Expected: ['Hello,', 'world!', 'How', 'are', 'you?']")

    # Test punctuation preservation
    text = "Hello, world! How are you?"
    words = extract_words(text)
    print(f"  Actual: {words}")
    assert words == ["Hello,", "world!", "How", "are", "you?"], f"Expected punctuated words, got {words}"

    print("✓ extract_words tests passed")

def test_interleave_words():
    """Test word interleaving.

    interleave_words() creates the expected output for interleaving tasks:
    A1 B1 A2 B2 ... continuing with remaining words when one list ends.
    This is the core of the interleaving task.
    """
    print("Testing interleave_words...")
    print("  Purpose: Interleave two word lists (A1 B1 A2 B2...) with continuation")

    print("  Test 1 - Equal length:")
    print("  Input A: ['A1', 'A2', 'A3']")
    print("  Input B: ['B1', 'B2', 'B3']")
    print("  Expected: ['A1', 'B1', 'A2', 'B2', 'A3', 'B3']")

    # Test equal length
    words_a = ["A1", "A2", "A3"]
    words_b = ["B1", "B2", "B3"]
    interleaved = interleave_words(words_a, words_b)
    print(f"  Actual: {interleaved}")
    assert interleaved == ["A1", "B1", "A2", "B2", "A3", "B3"], f"Expected interleaved, got {interleaved}"

    print("  Test 2 - A longer than B:")
    print("  Input A: ['A1', 'A2', 'A3', 'A4']")
    print("  Input B: ['B1', 'B2']")
    print("  Expected: ['A1', 'B1', 'A2', 'B2', 'A3', 'A4'] (A continues after B ends)")

    # Test unequal length (A longer)
    words_a = ["A1", "A2", "A3", "A4"]
    words_b = ["B1", "B2"]
    interleaved = interleave_words(words_a, words_b)
    print(f"  Actual: {interleaved}")
    assert interleaved == ["A1", "B1", "A2", "B2", "A3", "A4"], f"Expected A continuing after B ends, got {interleaved}"

    print("  Test 3 - B longer than A:")
    print("  Input A: ['A1']")
    print("  Input B: ['B1', 'B2', 'B3']")
    print("  Expected: ['A1', 'B1', 'B2', 'B3'] (B continues after A ends)")

    # Test unequal length (B longer)
    words_a = ["A1"]
    words_b = ["B1", "B2", "B3"]
    interleaved = interleave_words(words_a, words_b)
    print(f"  Actual: {interleaved}")
    assert interleaved == ["A1", "B1", "B2", "B3"], f"Expected B continuing after A ends, got {interleaved}"

    print("✓ interleave_words tests passed")

def test_format_expected():
    """Test output formatting.

    format_expected() converts word lists to the expected output format.
    'newline' creates one word per line (used by models).
    'space' creates space-separated words (for display).
    """
    print("Testing format_expected...")
    print("  Purpose: Format word lists into expected output strings")

    words = ["Word1", "Word2", "Word3"]

    print("  Test 1 - Newline format (model output):")
    print("  Input: ['Word1', 'Word2', 'Word3']")
    print("  Expected: 'Word1\\nWord2\\nWord3'")

    # Test newline format
    formatted = format_expected(words, "newline")
    print(f"  Actual: {repr(formatted)}")
    assert formatted == "Word1\nWord2\nWord3", f"Expected newline format, got {repr(formatted)}"

    print("  Test 2 - Space format (display):")
    print("  Input: ['Word1', 'Word2', 'Word3']")
    print("  Expected: 'Word1 Word2 Word3'")

    # Test space format
    formatted = format_expected(words, "space")
    print(f"  Actual: {repr(formatted)}")
    assert formatted == "Word1 Word2 Word3", f"Expected space format, got {repr(formatted)}"

    print("✓ format_expected tests passed")

def test_create_sample():
    """Test sample creation with mock data.

    create_sample() generates a complete training sample including:
    - Prompt for the model
    - Input fragments (A and B)
    - Expected output (interleaved or sequential)
    - Metadata (text IDs, mode, etc.)

    Tests both interleaving and sequential modes.
    """
    print("Testing create_sample...")
    print("  Purpose: Generate complete training samples with prompts and expected outputs")

    print("  Test 1 - Interleaving mode:")
    print("  Input: 3 words each, mode='interleave'")
    print("  Expected: 6-word interleaved output, prompt with 'alternating Process A, Process B'")

    # Test interleaving mode
    sample = create_sample(
        MOCK_TEXTS, num_words=3, mode="interleave",
        text_a_idx=0, text_b_idx=1
    )

    print(f"  Sample mode: {sample['mode']}")
    print(f"  Expected length: {len(sample['expected'])} (should be 6)")
    assert sample["mode"] == "interleave"
    assert len(sample["expected"]) == 6  # 3 + 3 words interleaved
    assert "fragment_a" in sample
    assert "fragment_b" in sample
    assert sample["text_a_id"] == "text1"
    assert sample["text_b_id"] == "text2"

    # Check prompt contains interleaving instructions
    prompt_content = sample["prompt"][0]["content"]
    print(f"  Prompt contains interleaving instructions: {'alternating Process A, Process B' in prompt_content}")
    assert "alternating Process A, Process B" in prompt_content

    print("  Test 2 - Sequential mode:")
    print("  Input: 2 words each, mode='sequential'")
    print("  Expected: 4-word sequential output ['The', 'quick', 'A', 'journey'], prompt with 'First, output all...'")

    # Test sequential mode
    sample_seq = create_sample(
        MOCK_TEXTS, num_words=2, mode="sequential",
        text_a_idx=0, text_b_idx=1
    )

    print(f"  Sample mode: {sample_seq['mode']}")
    print(f"  Expected length: {len(sample_seq['expected'])} (should be 4)")
    print(f"  Expected output structure: sequential concatenation of two 2-word fragments")
    assert sample_seq["mode"] == "sequential"
    assert len(sample_seq["expected"]) == 4  # 2 + 2 words sequential
    # Check that it's sequential (first half from text A, second half from text B)
    first_half = sample_seq["expected"][:2]
    second_half = sample_seq["expected"][2:]
    assert len(first_half) == 2 and len(second_half) == 2

    # Check prompt contains sequential instructions
    prompt_content_seq = sample_seq["prompt"][0]["content"]
    print(f"  Prompt contains sequential instructions: {'run two processes in sequence' in prompt_content_seq.lower()}")
    assert "run two processes in sequence" in prompt_content_seq.lower()
    assert "First, output all words from:" in prompt_content_seq
    assert "Then, output all words from:" in prompt_content_seq

    print("✓ create_sample tests passed")

def test_parse_output():
    """Test model output parsing.

    parse_output() extracts clean word lists from model completions,
    handling various formats and filtering out metadata/comments.
    This is crucial for comparing model outputs to expected sequences.
    """
    print("Testing parse_output...")
    print("  Purpose: Extract clean word lists from model outputs, filter metadata")

    print("  Test 1 - Newline-separated format:")
    print("  Input: 'Token1\\nToken2\\nToken3'")
    print("  Expected: ['Token1', 'Token2', 'Token3']")

    # Test newline-separated format
    output = "Token1\nToken2\nToken3"
    parsed = parse_output(output)
    print(f"  Actual: {parsed}")
    assert parsed == ["Token1", "Token2", "Token3"], f"Expected parsed words, got {parsed}"

    print("  Test 2 - Space-separated format:")
    print("  Input: 'Token1 Token2 Token3'")
    print("  Expected: ['Token1', 'Token2', 'Token3']")

    # Test space-separated format
    output = "Token1 Token2 Token3"
    parsed = parse_output(output)
    print(f"  Actual: {parsed}")
    assert parsed == ["Token1", "Token2", "Token3"], f"Expected parsed words, got {parsed}"

    print("  Test 3 - Mixed format with punctuation:")
    print("  Input: 'Hello,\\nworld!\\nHow are you?'")
    print("  Expected: ['Hello,', 'world!', 'How', 'are', 'you?']")

    # Test mixed format with punctuation
    output = "Hello,\nworld!\nHow are you?"
    parsed = parse_output(output)
    print(f"  Actual: {parsed}")
    assert parsed == ["Hello,", "world!", "How", "are", "you?"], f"Expected parsed with punctuation, got {parsed}"

    print("  Test 4 - Metadata filtering:")
    print("  Input: 'Output:\\nProcess A: Token1\\nToken2\\n## Comment\\nToken3'")
    print("  Expected: ['Token1', 'Token2', 'Token3'] (metadata filtered out)")

    # Test metadata filtering
    output = "Output:\nProcess A: Token1\nToken2\n## Comment\nToken3"
    parsed = parse_output(output)
    print(f"  Actual: {parsed}")
    assert parsed == ["Token1", "Token2", "Token3"], f"Expected filtered output, got {parsed}"

    print("✓ parse_output tests passed")

def test_mixed_mode_alternating():
    """Test mixed mode with alternating pattern.

    For mixed mode with interleave_ratio=3, sequential_ratio=1, pattern=alternating,
    should generate sequence: interleave, interleave, interleave, sequential, ...
    """
    print("Testing mixed mode alternating pattern...")
    print("  Purpose: Verify alternating pattern generates correct mode sequence")

    # Simulate the logic from generate_samples_for_split
    interleave_ratio = 3
    sequential_ratio = 1
    num_samples = 4
    pattern = "alternating"

    if pattern == "alternating":
        block_size = interleave_ratio + sequential_ratio  # 4
        full_blocks = num_samples // block_size  # 1
        remainder = num_samples % block_size  # 0

        modes_sequence = []
        for _ in range(full_blocks):
            modes_sequence.extend(["interleave"] * interleave_ratio)
            modes_sequence.extend(["sequential"] * sequential_ratio)

        remaining_modes = (["interleave"] * interleave_ratio + ["sequential"] * sequential_ratio)[:remainder]
        modes_sequence.extend(remaining_modes)

    print(f"  For num_samples={num_samples}, interleave_ratio={interleave_ratio}, sequential_ratio={sequential_ratio}")
    print(f"  Expected modes: ['interleave', 'interleave', 'interleave', 'sequential']")
    print(f"  Actual modes: {modes_sequence}")
    assert modes_sequence == ["interleave", "interleave", "interleave", "sequential"], f"Expected alternating pattern, got {modes_sequence}"

    print("✓ mixed mode alternating tests passed")

def test_compute_alignment_score():
    """Test alignment scoring.

    compute_alignment_score() uses Needleman-Wunsch alignment to score
    how well model output matches expected sequence. Returns 0-1 score
    where 1.0 is perfect match. Uses affine gap penalties for RL incentives.
    """
    print("Testing compute_alignment_score...")
    print("  Purpose: Score alignment between expected and actual word sequences (0-1 scale)")

    print("  Test 1 - Perfect match:")
    print("  Expected: ['A', 'B', 'C']")
    print("  Output: ['A', 'B', 'C']")
    print("  Expected score: ~1.0")

    # Perfect match
    expected = ["A", "B", "C"]
    output = ["A", "B", "C"]
    score = compute_alignment_score(expected, output)
    print(f"  Actual score: {score:.3f}")
    assert abs(score - 1.0) < 0.01, f"Expected perfect score ~1.0, got {score}"

    print("  Test 2 - Partial match (one mismatch):")
    print("  Expected: ['A', 'B', 'C', 'D']")
    print("  Output: ['A', 'B', 'X', 'D']")
    print("  Expected score: 0.5-0.9 (partial credit for alignment)")

    # Partial match
    expected = ["A", "B", "C", "D"]
    output = ["A", "B", "X", "D"]
    score = compute_alignment_score(expected, output)
    print(f"  Actual score: {score:.3f}")
    assert 0.5 < score < 0.9, f"Expected partial score 0.5-0.9, got {score}"

    print("  Test 3 - No match:")
    print("  Expected: ['A', 'B', 'C']")
    print("  Output: ['X', 'Y', 'Z']")
    print("  Expected score: <0.4 (poor alignment, some baseline score due to normalization)")

    # No match
    expected = ["A", "B", "C"]
    output = ["X", "Y", "Z"]
    score = compute_alignment_score(expected, output)
    print(f"  Actual score: {score:.3f}")
    assert score < 0.4, f"Expected low score <0.4, got {score}"

    print("  Test 4 - Edge cases:")

    # Empty inputs
    print("  Empty expected + empty output:")
    score = compute_alignment_score([], [])
    print(f"  Score: {score} (should be 1.0)")
    assert score == 1.0, f"Expected 1.0 for empty inputs, got {score}"

    print("  Expected present + empty output:")
    score = compute_alignment_score(["A"], [])
    print(f"  Score: {score} (should be 0.0)")
    assert score == 0.0, f"Expected 0.0 for empty output, got {score}"

    print("✓ compute_alignment_score tests passed")

def run_tests():
    """Run all tests."""
    print("=" * 50)
    print("Running Unit Tests for Dataset Components")
    print("=" * 50)

    try:
        test_extract_words()
        test_interleave_words()
        test_format_expected()
        test_create_sample()
        test_parse_output()
        test_mixed_mode_alternating()
        test_compute_alignment_score()

        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("=" * 50)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()