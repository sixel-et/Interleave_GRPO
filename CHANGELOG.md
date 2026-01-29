# Interleave GRPO Project Changelog

## Project Structure Overview

```
/testbed/Interleave_GRPO/
├── add_splits_to_corpus.py                    # Utility to add train/val/test splits to corpus
├── chat.py                                   # Chat interface or utilities
├── Claude.md                                 # Documentation or notes related to Claude
├── clean_corpus.py                           # Corpus cleaning utilities
├── corpus/                                   # Corpus data directory
│   ├── backup_of_json/                       # Backup JSON files
│   │   ├── corpus_metadata.json
│   │   └── url_list.json
│   ├── corpus_metadata.json                  # Metadata for corpus
│   ├── existing_texts.json                   # Existing text data
│   └── url_list.json                         # URL lists for sources
├── dataset_generator.py                      # Original dataset generator (interleaving only)
├── dataset_generator_unified.py              # **NEW** Unified generator (interleave/sequential modes, created by AI agent)
├── datasets/                                 # Generated dataset files
│   ├── 10words.jsonl
│   └── 10words_alt.jsonl
├── DISCORD_SETUP.md                          # Discord setup documentation
├── evaluate.py                               # Original evaluation script (interleaving only)
├── evaluate_sequential.py                    # **NEW** Sequential evaluation script (created by AI agent)
├── gutenberg_text_getter.py                  # Gutenberg text fetching utilities
├── interleave_grpo.py                        # Main GRPO training script
├── pg_collect_titles_and_urls.py             # Project Gutenberg title/URL collection
├── process_from_list.py                      # List processing utilities
├── README.md                                 # Project README
├── reward.py                                 # Reward function for alignment scoring
├── setup_and_run.sh                          # Setup and run script
├── source_texts.json                         # Original source texts
├── source_texts.json.backup_pre_grok_add     # Backup before Grok additions
├── source_texts.json.backup_pre_grok_add.json# Another backup
├── source_texts_before_claude_cleanup.json   # Pre-Claude cleanup backup
├── source_texts_old.json                     # Old source texts
├── source_texts_split.json                   # Split-tagged source texts (default corpus)
├── spectral_analysis.py                      # Spectral analysis utilities
├── test.md                                   # Test documentation or notes
├── CHANGELOG.md                              # **NEW** This changelog file
└── test_components.py                        # **NEW** Unit testing framework (created by AI agent)
```

## Files Touched by AI Agent

### New Files Created

1. **dataset_generator_unified.py** (Created: Recent)
   - Unified dataset generator supporting both interleaving and sequential modes
   - Features:
     - `--mode` flag for interleave vs sequential
     - `--only-split` flag to generate from single split (train/val/test)
     - `--print-samples` flag for testing without datasets library
     - Optional datasets import with graceful fallback
     - Updated corpus default to `source_texts_split.json`

2. **evaluate_sequential.py** (Created: Recent)
   - Evaluation script specifically for sequential recitation tasks
   - Based on original evaluate.py but focused on sequential performance
   - Includes updated usage examples for single-split generation
   - Uses reward_for_new_evaluate.py for optional numba support

4. **reward_for_new_evaluate.py** (Created: Recent)
   - Modified version of reward.py with optional numba support
   - Falls back to pure Python implementations when numba unavailable
   - Maintains same API and functionality as original reward.py

3. **CHANGELOG.md** (Created: Now)
   - This changelog file documenting project structure and changes

4. **test_components.py** (Created: Now)
   - Unit testing framework for key components of dataset_generator_unified.py and evaluate_sequential.py
   - Tests functions independently without requiring external files
   - Includes mock data and assertions for core functionality

5. **dataset_generator_unified.md** (Created: Now)
   - Documentation for dataset_generator_unified.py
   - Includes detailed command line argument explanations
   - Usage examples and output format description

6. **evaluate_sequential.md** (Created: Now)
   - Documentation for evaluate_sequential.py
   - Command line arguments, metrics, and scoring explanation

7. **test_components.md** (Created: Now)
   - Documentation for test_components.py
   - Test coverage and usage instructions

### Modifications to Existing Files

- **evaluate_sequential.py**: Updated usage examples to include `--print-samples` workflow

## Recent Changes Summary

- **2026-01-29**: Created unified dataset generator and sequential evaluator to test for performance degradation in sequential tasks vs interleaving training
- **2026-01-29**: Added optional datasets import and print-samples feature for environments without HuggingFace datasets library
- **2026-01-29**: Implemented single-split generation capability (`--only-split` flag)
- **2026-01-29**: Created unit testing framework for key components with iterative debugging and fixes for edge cases
- **2026-01-29**: Fixed `--print-samples` to work with `--curriculum` mode through multiple iterations
- **2026-01-29**: Reverted `--preview` to allow file saving while showing snippets during generation
- **2026-01-29**: Made `--curriculum` accept custom stage specifications at runtime
- **2026-01-29**: Implemented smart split selection for `--curriculum --print-samples` to avoid unwanted sample generation
- **2026-01-29**: Added mixed mode combining interleave and sequential tasks with configurable ratios and patterns (alternating/random)
- **2026-01-29**: Fixed multiple bugs in curriculum and print-samples logic through back-and-forth testing and refinement
- **2026-01-29**: Created comprehensive markdown documentation for all Python files with CLI argument explanations and use cases

## Notes

- All new files are independent and don't modify existing functionality
- Corpus defaults updated to use `source_texts_split.json` (split-tagged)
- Testing framework provides basic unit tests for core functions without external dependencies
- Original files (`dataset_generator.py`, `evaluate.py`) remain unchanged for backward compatibility

## Questions for Unit Testing

1. **Evaluation Testing**: The evaluation functions depend on model loading and generation, which requires actual models and hardware. How would you like to handle testing these? Mock the model responses, or focus only on the parsing/scoring parts that can be tested independently?

2. **Integration Testing**: Would you want tests that verify the end-to-end workflow (generate dataset + evaluate), or keep it purely unit-level?

3. **Test Data**: Should I use synthetic/mock text data for tests, or do you have specific test cases in mind?

4. **Coverage**: Which specific functions should have priority for testing? (e.g., sample creation, alignment scoring, parsing)

Let me know how you'd like to proceed with the testing approach!