# Changelog

## 2026-01-28 Claude Code

### Scoring Changes
- **Composite scoring**: Implemented three-tier scoring system:
  - `word_score`: NW alignment of words only (ignores formatting)
  - `format_score`: 1 - (violations / output_words) where violations = empty_lines + extra_words_on_multiword_lines
  - `score` (composite): word_score * format_score
- Format can only penalize, never boost - ensures wrong words with good formatting still score low
- Added `word_score`, `format_score`, `empty_lines`, `multiword_lines` fields to SampleResult
- Updated `evaluate_single_lines()` in reward.py with composite approach

### Dataset Changes
- Extended curriculum: added 750w, 1000w, 1500w datasets
- Created `1500words_test_clean.jsonl` (97 samples with correct expected length)
- Identified issue: some source texts shorter than requested word count, causing short expected lengths in 1000w (4 samples) and 1500w (3 samples)

### Evaluation Changes
- Fixed `expected_str` to use newline-separated format from dataset (was incorrectly space-joining)
- Added content-based refusal detection (checking first 20 words for "i understand you" / "i appreciate your")
- Created `rescore_results.py` for rescoring existing results without re-running API calls

### Data Collected (Sonnet, composite scoring)
| Dataset | Genuine | Word | Format | Composite | High>=0.9 |
|---------|---------|------|--------|-----------|-----------|
| 200w | 41 | 100.0% | 98.8% | 98.8% | 98% |
| 500w | 37 | 84.0% | 98.3% | 84.0% | 78% |
| 750w | 36 | 80.3% | 100.0% | 80.3% | 72% |
| 1000w | 39 | 63.1% | 97.2% | 62.7% | 51% |
| 1500w | 37 | 44.5% | 94.5% | 44.5% | 32% |

Key finding: Format compliance stays ~94-100% even as word accuracy drops significantly (100% → 45%)

### Token Budget Analysis (updated)
- Tier 2 upgrade: 90k tokens/min, 1k requests/min, no per-response limit
- Token estimation: ~2.3 tokens per word with newlines (7800 words ≈ 18k tokens)
- MAX_NEW_TOKENS increased to 25000 to support 3900w tasks
- Added streaming for requests >4096 tokens (required by Anthropic SDK for long requests)

### Extended Evaluation: 3900w
| Dataset | n | Genuine | Word | Format | Completion |
|---------|---|---------|------|--------|------------|
| 3900w | 30 | 28 | 66.0% | 86.1% | 85% |

⚠️ Earlier 1500w results (44.5%) had max_tokens issue - only 55% completion rate.
Re-running 1500w with updated settings for fair comparison.

### Infrastructure
- Added async/parallel evaluation with semaphore-based concurrency
- Added auto-save during evaluation runs
- Added `--resume` flag for interrupted runs
- Added streaming support in backends.py for long requests
- Tier 2: can now run 5 concurrent 3900w evaluations

### 1500w Rerun Results
| Dataset | n | Genuine | Word | Format | Completion |
|---------|---|---------|------|--------|------------|
| 1500w | 30 | 23 | 55.4% | 98.4% | 80.9% |
| 3900w | 30 | 27 | 68.5% | 95.3% | 94.4% |

### Analysis: Apparent Non-Monotonicity
- 3900w outperformed 1500w on mean score (66.5% vs 55.4%)
- Investigation revealed different failure modes:
  - 1500w: fast bail (6-42s API time) on some samples
  - 3900w: slow incomplete (150-275s) when failing
- **Key finding**: When filtered to >=95% completion, both score ~77%
- Max score is 100% at both lengths - capability exists
- Hypothesis: Success depends on triage → internal tool/strategy selection
  - Longer tasks may be easier to classify correctly
  - Predicts 6k could outperform 3900w

### 6k Dataset Preparation
- Created `source_texts_split_6k.json` with 39 concatenated text entries
  - Combined consecutive chunks from same works (e.g., chunk1 + chunk2)
  - All entries 9400-9900 words (sufficient for 6k fragments)
- Generated `api_datasets/6000words_test.jsonl` (100 samples)
  - 12000 expected words per sample
  - ~27600 estimated output tokens
  - 94 unique text pairs from 39 source texts
- Created `used_test_texts.json` tracking 164 texts used in prior testing
- MAX_NEW_TOKENS will need increase to ~35000 for 6k evaluation
