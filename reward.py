"""
reward.py

Reward function for the interleaving task.
Uses Needleman-Wunsch alignment to score model outputs against expected sequences.

Can be used:
1. As reward function during GRPO training
2. Standalone for pre/post training evaluation

Performance:
- Uses numba JIT compilation for ~100x speedup
- Vocabulary mapping converts words to integers for fast comparison
- Affine gap penalties for better RL incentives
"""

import numpy as np
from numba import njit
from typing import List, Tuple, Dict

# ============================================================================
# CONFIG
# ============================================================================

# Needleman-Wunsch scoring parameters (affine gap penalties)
MATCH_SCORE = 2.0
MISMATCH_PENALTY = -1.0
GAP_OPEN = -5.0      # Cost to open a new gap
GAP_EXTEND = -1.0    # Cost to extend an existing gap

# Reward scaling (NW scores can vary widely, this normalizes to ~0-1 range)
NORMALIZE_BY_LENGTH = True


# ============================================================================
# VOCABULARY MANAGEMENT
# ============================================================================

class Vocabulary:
    """Maps words to integers for fast numba comparison."""
    
    def __init__(self):
        self.word2id: Dict[str, int] = {}
        self.id2word: Dict[int, str] = {}
        self.next_id = 0
    
    def add(self, word: str) -> int:
        """Add word to vocab, return its ID."""
        if word not in self.word2id:
            self.word2id[word] = self.next_id
            self.id2word[self.next_id] = word
            self.next_id += 1
        return self.word2id[word]
    
    def encode(self, words: List[str]) -> np.ndarray:
        """Convert word list to numpy array of IDs."""
        return np.array([self.add(w) for w in words], dtype=np.int32)
    
    def decode(self, ids: np.ndarray) -> List[str]:
        """Convert ID array back to words."""
        return [self.id2word[i] for i in ids]


# Global vocabulary for training (persists across calls)
_vocab = Vocabulary()


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
# NUMBA-OPTIMIZED NEEDLEMAN-WUNSCH (AFFINE GAP PENALTIES)
# ============================================================================

@njit(cache=True)
def _nw_score_only_affine(
    seq1: np.ndarray,
    seq2: np.ndarray,
    match: float,
    mismatch: float,
    gap_open: float,
    gap_extend: float
) -> float:
    """
    Affine gap penalty Needleman-Wunsch, score only (no traceback).
    Uses O(m) memory by keeping only current and previous rows.
    
    Uses three matrices:
    - M[i,j]: best score ending with seq1[i] aligned to seq2[j]
    - X[i,j]: best score ending with gap in seq2 (deletion from seq1)
    - Y[i,j]: best score ending with gap in seq1 (insertion from seq2)
    
    Args:
        seq1: reference sequence as integer IDs
        seq2: query sequence as integer IDs
        match: score for matching elements
        mismatch: score for mismatching elements
        gap_open: penalty for opening a gap
        gap_extend: penalty for extending a gap
    
    Returns:
        Final alignment score
    """
    n, m = len(seq1), len(seq2)
    
    if n == 0 or m == 0:
        return 0.0
    
    NEG_INF = -1e9
    
    # Only keep two rows (previous and current)
    M_prev = np.full(m + 1, NEG_INF, dtype=np.float64)
    M_curr = np.full(m + 1, NEG_INF, dtype=np.float64)
    X_prev = np.full(m + 1, NEG_INF, dtype=np.float64)
    X_curr = np.full(m + 1, NEG_INF, dtype=np.float64)
    Y_prev = np.full(m + 1, NEG_INF, dtype=np.float64)
    Y_curr = np.full(m + 1, NEG_INF, dtype=np.float64)
    
    # Base cases
    M_prev[0] = 0.0
    for j in range(1, m + 1):
        Y_prev[j] = gap_open + (j - 1) * gap_extend
    
    # Fill matrices row by row
    for i in range(1, n + 1):
        # Initialize current row
        M_curr[0] = NEG_INF
        X_curr[0] = gap_open + (i - 1) * gap_extend
        Y_curr[0] = NEG_INF
        
        for j in range(1, m + 1):
            # Match/mismatch score
            s = match if seq1[i-1] == seq2[j-1] else mismatch
            
            # M[i,j]: align seq1[i] with seq2[j]
            M_curr[j] = s + max(M_prev[j-1], X_prev[j-1], Y_prev[j-1])
            
            # X[i,j]: gap in seq2 (consume seq1[i])
            X_curr[j] = max(
                gap_open + M_prev[j],    # open new gap
                gap_extend + X_prev[j]   # extend existing gap
            )
            
            # Y[i,j]: gap in seq1 (consume seq2[j])
            Y_curr[j] = max(
                gap_open + M_curr[j-1],  # open new gap
                gap_extend + Y_curr[j-1] # extend existing gap
            )
        
        # Swap rows (previous becomes current for next iteration)
        M_prev, M_curr = M_curr, M_prev
        X_prev, X_curr = X_curr, X_prev
        Y_prev, Y_curr = Y_curr, Y_prev
    
    # Final score is max of all three matrices at bottom-right
    return max(M_prev[m], X_prev[m], Y_prev[m])


@njit(cache=True)
def _nw_full_affine(
    seq1: np.ndarray,
    seq2: np.ndarray,
    match: float,
    mismatch: float,
    gap_open: float,
    gap_extend: float
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full affine gap NW with traceback matrices.
    Returns score and all three matrices for traceback.
    """
    n, m = len(seq1), len(seq2)
    
    if n == 0 or m == 0:
        return 0.0, np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))
    
    NEG_INF = -1e9
    
    # Initialize all three matrices
    M = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)
    X = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)
    Y = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)
    
    # Base cases
    M[0, 0] = 0.0
    for i in range(1, n + 1):
        X[i, 0] = gap_open + (i - 1) * gap_extend
    for j in range(1, m + 1):
        Y[0, j] = gap_open + (j - 1) * gap_extend
    
    # Fill matrices
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if seq1[i-1] == seq2[j-1] else mismatch
            
            M[i, j] = s + max(M[i-1, j-1], X[i-1, j-1], Y[i-1, j-1])
            X[i, j] = max(gap_open + M[i-1, j], gap_extend + X[i-1, j])
            Y[i, j] = max(gap_open + M[i, j-1], gap_extend + Y[i, j-1])
    
    score = max(M[n, m], X[n, m], Y[n, m])
    return score, M, X, Y


def traceback_affine(
    seq1: np.ndarray,
    seq2: np.ndarray,
    M: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    match: float,
    mismatch: float,
    gap_open: float,
    gap_extend: float
) -> Tuple[List[int], List[int]]:
    """
    Traceback through affine gap matrices to get alignment.
    
    Returns:
        (aligned_seq1, aligned_seq2) where gaps are represented as -1
    """
    n, m = len(seq1), len(seq2)
    i, j = n, m
    
    # Determine which matrix we're ending in
    final_score = max(M[n, m], X[n, m], Y[n, m])
    if final_score == M[n, m]:
        current_matrix = 'M'
    elif final_score == X[n, m]:
        current_matrix = 'X'
    else:
        current_matrix = 'Y'
    
    aligned_1 = []
    aligned_2 = []
    
    while i > 0 or j > 0:
        if current_matrix == 'M':
            if i == 0 or j == 0:
                break
            
            s = match if seq1[i-1] == seq2[j-1] else mismatch
            prev_score = M[i, j] - s
            
            # Came from M, X, or Y at (i-1, j-1)
            if abs(prev_score - M[i-1, j-1]) < 1e-9:
                current_matrix = 'M'
            elif abs(prev_score - X[i-1, j-1]) < 1e-9:
                current_matrix = 'X'
            else:
                current_matrix = 'Y'
            
            aligned_1.insert(0, int(seq1[i-1]))
            aligned_2.insert(0, int(seq2[j-1]))
            i -= 1
            j -= 1
        
        elif current_matrix == 'X':
            # Gap in seq2, consume seq1[i]
            if abs(X[i, j] - (gap_open + M[i-1, j])) < 1e-9:
                current_matrix = 'M'
            else:
                current_matrix = 'X'
            
            aligned_1.insert(0, int(seq1[i-1]))
            aligned_2.insert(0, -1)  # gap marker
            i -= 1
        
        else:  # current_matrix == 'Y'
            # Gap in seq1, consume seq2[j]
            if abs(Y[i, j] - (gap_open + M[i, j-1])) < 1e-9:
                current_matrix = 'M'
            else:
                current_matrix = 'Y'
            
            aligned_1.insert(0, -1)  # gap marker
            aligned_2.insert(0, int(seq2[j-1]))
            j -= 1
    
    return aligned_1, aligned_2


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

def nw_align(
    seq_a: List[str],
    seq_b: List[str],
    vocab: Vocabulary = None
) -> Tuple[float, List, List]:
    """
    Needleman-Wunsch alignment with affine gap penalties.
    
    Args:
        seq_a: expected sequence (ground truth)
        seq_b: model output sequence
        vocab: optional vocabulary (uses global if not provided)
    
    Returns:
        (score, aligned_a, aligned_b)
    """
    if vocab is None:
        vocab = _vocab
    
    # Encode to integers
    seq1_ids = vocab.encode(seq_a)
    seq2_ids = vocab.encode(seq_b)
    
    # Run alignment with traceback
    score, M, X, Y = _nw_full_affine(
        seq1_ids, seq2_ids,
        MATCH_SCORE, MISMATCH_PENALTY,
        GAP_OPEN, GAP_EXTEND
    )
    
    # Traceback to get alignment
    aligned_1_ids, aligned_2_ids = traceback_affine(
        seq1_ids, seq2_ids, M, X, Y,
        MATCH_SCORE, MISMATCH_PENALTY,
        GAP_OPEN, GAP_EXTEND
    )
    
    # Convert back to words (using '-' for gaps)
    aligned_a = [vocab.id2word[i] if i >= 0 else '-' for i in aligned_1_ids]
    aligned_b = [vocab.id2word[i] if i >= 0 else '-' for i in aligned_2_ids]
    
    return score, aligned_a, aligned_b


def compute_alignment_score(
    expected: List[str],
    output: List[str],
    vocab: Vocabulary = None
) -> float:
    """
    Compute normalized alignment score between expected and output.
    Fast score-only version (no traceback).
    
    Returns:
        Score in range [0, 1] where 1 is perfect match
    """
    if not expected:
        return 1.0 if not output else 0.0
    
    if vocab is None:
        vocab = _vocab
    
    # Encode to integers
    seq1_ids = vocab.encode(expected)
    seq2_ids = vocab.encode(output)
    
    # Fast score-only computation
    raw_score = _nw_score_only_affine(
        seq1_ids, seq2_ids,
        MATCH_SCORE, MISMATCH_PENALTY,
        GAP_OPEN, GAP_EXTEND
    )
    
    if NORMALIZE_BY_LENGTH:
        # Normalize by max possible score (all matches)
        max_score = len(expected) * MATCH_SCORE
        if max_score <= 0:
            return 0.0
        
        # Min score: all gaps (longer sequence determines gap cost)
        max_len = max(len(expected), len(output))
        min_score = GAP_OPEN + (max_len - 1) * GAP_EXTEND
        
        # Normalize to [0, 1]
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

def evaluate_single(expected: List[str], output_text: str, verbose: bool = False) -> dict:
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


def print_alignment(expected: List[str], output_text: str):
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
    parser.add_argument("--benchmark", action="store_true", help="Run speed benchmark")
    args = parser.parse_args()
    
    if args.benchmark:
        import time
        
        print("=== Speed Benchmark ===\n")
        
        # Generate test sequences
        expected = ["word" + str(i) for i in range(500)]
        output_words = expected.copy()
        # Add some errors
        for i in range(0, 500, 10):
            output_words[i] = "ERROR"
        
        vocab = Vocabulary()
        seq1_ids = vocab.encode(expected)
        seq2_ids = vocab.encode(output_words)
        
        # Warm up numba
        _ = _nw_score_only_affine(seq1_ids, seq2_ids, MATCH_SCORE, MISMATCH_PENALTY, GAP_OPEN, GAP_EXTEND)
        
        # Benchmark
        n_runs = 100
        start = time.time()
        for _ in range(n_runs):
            score = _nw_score_only_affine(seq1_ids, seq2_ids, MATCH_SCORE, MISMATCH_PENALTY, GAP_OPEN, GAP_EXTEND)
        elapsed = time.time() - start
        
        print(f"500-word sequences, {n_runs} runs:")
        print(f"Total time: {elapsed:.3f}s")
        print(f"Time per alignment: {elapsed/n_runs*1000:.3f}ms")
        print(f"Score: {score:.1f}")
    
    elif args.test:
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
    
    else:
        parser.print_help()