"""
reward.py

Reward function for the interleaving task.
Uses Needleman-Wunsch alignment to score model outputs against expected sequences.

Can be used:
1. As reward function during GRPO training
2. Standalone for pre/post training evaluation
"""

# ============================================================================
# CONFIG
# ============================================================================

# Needleman-Wunsch scoring parameters
MATCH_SCORE = 1.0
MISMATCH_PENALTY = -1.0
GAP_PENALTY = -0.5

# Reward scaling (NW scores can vary widely, this normalizes to ~0-1 range)
NORMALIZE_BY_LENGTH = True

# ============================================================================
# OUTPUT PARSING
# ============================================================================

def parse_output(text: str) -> list[str]:
    """
    Parse model output into list of words.
    Handles both newline-separated and space-separated formats.
    Strips any commentary/metadata the model might add.
    """
    if text is None:
        return []
    
    # Try newline-separated first (expected format)
    lines = text.strip().split('\n')
    
    # Filter out empty lines and obvious metadata
    words = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Handle "Process A: word" format - extract the word
        if ':' in line and len(line.split(':')[0].split()) <= 2:
            line = line.split(':', 1)[-1].strip()
            if line:
                words.extend(line.split())
            continue
        
        # Skip lines that are purely metadata/commentary (no word content)
        if line.startswith(('Output:', 'Word', '##', '---', '***')):
            continue
        
        # If line has content, use it
        if line:
            words.extend(line.split())
    
    return words


# ============================================================================
# NEEDLEMAN-WUNSCH ALIGNMENT
# ============================================================================

def nw_align(seq_a: list[str], seq_b: list[str]) -> tuple[float, list, list]:
    """
    Needleman-Wunsch global alignment algorithm.
    Treats words as alignment units (like amino acids in bioinformatics).
    
    Args:
        seq_a: expected sequence (ground truth)
        seq_b: model output sequence
    
    Returns:
        (score, aligned_a, aligned_b)
    """
    n, m = len(seq_a), len(seq_b)
    
    # Initialize scoring matrix
    score_matrix = [[0.0] * (m + 1) for _ in range(n + 1)]
    
    # Initialize gap penalties
    for i in range(n + 1):
        score_matrix[i][0] = i * GAP_PENALTY
    for j in range(m + 1):
        score_matrix[0][j] = j * GAP_PENALTY
    
    # Fill matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = score_matrix[i-1][j-1] + (
                MATCH_SCORE if seq_a[i-1] == seq_b[j-1] else MISMATCH_PENALTY
            )
            delete = score_matrix[i-1][j] + GAP_PENALTY
            insert = score_matrix[i][j-1] + GAP_PENALTY
            score_matrix[i][j] = max(match, delete, insert)
    
    # Traceback
    aligned_a, aligned_b = [], []
    i, j = n, m
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            current = score_matrix[i][j]
            diag = score_matrix[i-1][j-1]
            match_score = MATCH_SCORE if seq_a[i-1] == seq_b[j-1] else MISMATCH_PENALTY
            
            if current == diag + match_score:
                aligned_a.insert(0, seq_a[i-1])
                aligned_b.insert(0, seq_b[j-1])
                i -= 1
                j -= 1
                continue
        
        if i > 0 and score_matrix[i][j] == score_matrix[i-1][j] + GAP_PENALTY:
            aligned_a.insert(0, seq_a[i-1])
            aligned_b.insert(0, '-')
            i -= 1
        elif j > 0:
            aligned_a.insert(0, '-')
            aligned_b.insert(0, seq_b[j-1])
            j -= 1
        else:
            break
    
    return score_matrix[n][m], aligned_a, aligned_b


def compute_alignment_score(expected: list[str], output: list[str]) -> float:
    """
    Compute normalized alignment score between expected and output.
    
    Returns:
        Score in range [0, 1] where 1 is perfect match
    """
    if not expected:
        return 1.0 if not output else 0.0
    
    raw_score, _, _ = nw_align(expected, output)
    
    if NORMALIZE_BY_LENGTH:
        # Normalize by max possible score (all matches)
        max_score = len(expected) * MATCH_SCORE
        # Handle edge case where max_score could be 0
        if max_score <= 0:
            return 0.0
        # Shift and scale to [0, 1]
        # Min possible score is all gaps/mismatches
        min_score = max(len(expected), len(output)) * min(GAP_PENALTY, MISMATCH_PENALTY)
        normalized = (raw_score - min_score) / (max_score - min_score)
        return max(0.0, min(1.0, normalized))
    else:
        return raw_score


# ============================================================================
# GRPO REWARD FUNCTION
# ============================================================================

def interleave_reward_func(completions, expected, **kwargs) -> list[float]:
    """
    Reward function compatible with TRL's GRPOTrainer.
    
    Args:
        completions: list of completion dicts from model
        expected: list of expected word sequences (from dataset)
        **kwargs: additional dataset columns (ignored)
    
    Returns:
        List of reward scores, one per completion
    """
    rewards = []
    
    for completion, exp in zip(completions, expected):
        # Extract text from completion
        if isinstance(completion, list):
            text = completion[0].get("content", "") if completion else ""
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)
        
        # Parse output and score
        output_words = parse_output(text)
        score = compute_alignment_score(exp, output_words)
        rewards.append(score)
    
    return rewards


# ============================================================================
# EVALUATION UTILITIES
# ============================================================================

def evaluate_single(expected: list[str], output_text: str, verbose: bool = False) -> dict:
    """
    Evaluate a single model output against expected.
    
    Returns dict with score and alignment details.
    """
    output_words = parse_output(output_text)
    raw_score, aligned_exp, aligned_out = nw_align(expected, output_words)
    normalized_score = compute_alignment_score(expected, output_words)
    
    result = {
        "score": normalized_score,
        "raw_score": raw_score,
        "expected_len": len(expected),
        "output_len": len(output_words),
        "matches": sum(1 for a, b in zip(aligned_exp, aligned_out) if a == b and a != '-'),
        "mismatches": sum(1 for a, b in zip(aligned_exp, aligned_out) if a != b and a != '-' and b != '-'),
        "gaps": sum(1 for a, b in zip(aligned_exp, aligned_out) if a == '-' or b == '-'),
    }
    
    if verbose:
        result["aligned_expected"] = aligned_exp
        result["aligned_output"] = aligned_out
        result["output_words"] = output_words
    
    return result


def print_alignment(expected: list[str], output_text: str):
    """Pretty print alignment for debugging."""
    result = evaluate_single(expected, output_text, verbose=True)
    
    print(f"Score: {result['score']:.3f} (raw: {result['raw_score']:.1f})")
    print(f"Lengths: expected={result['expected_len']}, output={result['output_len']}")
    print(f"Matches: {result['matches']}, Mismatches: {result['mismatches']}, Gaps: {result['gaps']}")
    print()
    print("Alignment:")
    
    aligned_exp = result["aligned_expected"]
    aligned_out = result["aligned_output"]
    
    # Print in chunks
    chunk_size = 5
    for i in range(0, len(aligned_exp), chunk_size):
        exp_chunk = aligned_exp[i:i+chunk_size]
        out_chunk = aligned_out[i:i+chunk_size]
        
        # Format with fixed width
        width = max(max(len(str(w)) for w in exp_chunk + out_chunk), 8)
        
        exp_str = " ".join(f"{w:<{width}}" for w in exp_chunk)
        out_str = " ".join(f"{w:<{width}}" for w in out_chunk)
        match_str = " ".join(
            f"{'✓':<{width}}" if e == o else f"{'✗':<{width}}" 
            for e, o in zip(exp_chunk, out_chunk)
        )
        
        print(f"Exp: {exp_str}")
        print(f"Out: {out_str}")
        print(f"     {match_str}")
        print()


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test reward function")
    parser.add_argument("--test", action="store_true", help="Run test cases")
    args = parser.parse_args()
    
    if args.test:
        print("=== Test Cases ===\n")
        
        # Perfect match
        expected = ["To", "Four", "be", "score", "or", "and"]
        output = "To\nFour\nbe\nscore\nor\nand"
        print("Test 1: Perfect match")
        print_alignment(expected, output)
        
        # One error
        expected = ["To", "Four", "be", "score", "or", "and"]
        output = "To\nFour\nbe\nXXX\nor\nand"
        print("Test 2: One mismatch")
        print_alignment(expected, output)
        
        # Missing word
        expected = ["To", "Four", "be", "score", "or", "and"]
        output = "To\nFour\nbe\nor\nand"
        print("Test 3: Missing word (gap)")
        print_alignment(expected, output)
        
        # Extra word
        expected = ["To", "Four", "be", "score"]
        output = "To\nFour\nbe\nEXTRA\nscore"
        print("Test 4: Extra word (insertion)")
        print_alignment(expected, output)
        
        # With metadata
        expected = ["To", "Four", "be", "score"]
        output = "Process A: To\nProcess B: Four\nbe\nscore"
        print("Test 5: With metadata prefix")
        print_alignment(expected, output)