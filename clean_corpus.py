#!/usr/bin/env python3
"""
Post-process corpus to clean Gutenberg artifacts.

Handles:
- \r\n line endings
- Unicode → ASCII conversion (ALL unicode removed)
- Markdown formatting
- Gutenberg boilerplate
- Excessive blank lines
- Leading/trailing whitespace

Usage: python clean_corpus.py input.json output.json

Requires: pip install unidecode
"""

import json
import re
import sys
from pathlib import Path
from unidecode import unidecode


def clean_gutenberg_text(text: str) -> str:
    """
    Clean Project Gutenberg artifacts while preserving punctuation.
    
    Strategy:
    1. Remove Gutenberg headers/boilerplate
    2. Convert ALL unicode to ASCII (removes accents, special characters)
    3. Remove markdown formatting
    4. Normalize line endings and whitespace
    """
    
    # Remove Project Gutenberg boilerplate
    # Common header markers (START)
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THE PROJECT GUTENBERG",
        "START OF THIS PROJECT GUTENBERG EBOOK",
    ]
    
    for marker in start_markers:
        if marker in text:
            parts = text.split(marker, 1)
            if len(parts) > 1:
                # Take everything after the marker, skip the next 2 lines (title info)
                remainder = parts[1]
                lines = remainder.split('\n', 3)
                text = lines[-1] if len(lines) > 2 else remainder
    
    # Common footer markers (END) - remove everything after
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THE PROJECT GUTENBERG",
        "END OF THIS PROJECT GUTENBERG EBOOK",
        "END OF THE PROJECT GUTENBERG EBOOK",
    ]
    
    for marker in end_markers:
        if marker in text:
            parts = text.split(marker, 1)
            # Keep only everything before the marker
            text = parts[0]
            break
    
    # Remove common boilerplate phrases
    boilerplate_phrases = [
        "The Project Gutenberg EBook of",
        "This eBook is for the use of anyone anywhere",
        "Project Gutenberg License",
        "Title:",
        "Author:",
        "Release Date:",
        "Language:",
        "Character set encoding:",
        "Produced by",
        "www.gutenberg.org",
    ]
    
    for phrase in boilerplate_phrases:
        # Remove lines containing these phrases
        lines = text.split('\n')
        text = '\n'.join(line for line in lines if phrase not in line)
    
    # Convert ALL unicode to ASCII (removes accents, converts special chars)
    # Château → Chateau, Honoré → Honore, — → -, é → e, etc.
    text = unidecode(text)
    
    # Remove markdown formatting
    import re
    text = re.sub(r'#{2,}', '', text)      # Headers (##, ###, etc)
    text = re.sub(r'\*{2,}', '', text)     # Bold/emphasis (**, ***)
    text = re.sub(r'\|', ' ', text)        # Table borders → space (prevent word smashing)
    text = re.sub(r'={3,}', ' ', text)     # Horizontal rules (===)
    text = re.sub(r'-{3,}', ' ', text)     # Long dashes (---) → space (prevent word smashing)
    
    # Normalize line endings
    text = text.replace('\r\n', '\n')
    text = text.replace('\r', '\n')
    
    # Collapse excessive newlines (3+) to paragraph break
    import re
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Normalize spaces (but preserve newlines)
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs → single space
    text = re.sub(r' {2,}', ' ', text)   # Collapse any remaining multiple spaces
    text = re.sub(r' \n', '\n', text)     # Remove trailing spaces before newlines
    text = re.sub(r'\n ', '\n', text)     # Remove leading spaces after newlines
    
    # Trim
    text = text.strip()
    
    return text


def validate_text(text: str, text_id: str, min_words: int = 200) -> dict:
    """
    Validate cleaned text meets requirements.
    
    Only removes truly broken/tiny texts (< 200 words).
    Short texts (200-5000w) are kept - they're useful for lower curriculum stages.
    
    Returns dict with:
    - valid: bool
    - word_count: int
    - issues: list of strings
    """
    issues = []
    
    if not text.strip():
        issues.append("Empty text")
        return {"valid": False, "word_count": 0, "issues": issues}
    
    words = text.split()
    word_count = len(words)
    
    if word_count < min_words:
        issues.append(f"Too short: {word_count} words (threshold: {min_words})")
    
    if word_count < 20:
        issues.append(f"Extremely short: {word_count} words - likely broken")
    
    # Check for obvious artifacts
    if '\\u' in text:
        issues.append("Contains unprocessed unicode escapes")
    
    if '\r' in text:
        issues.append("Contains carriage returns (\\r)")
    
    # Check for remaining boilerplate
    if 'Project Gutenberg' in text or 'www.gutenberg.org' in text:
        issues.append("Still contains Gutenberg boilerplate")
    
    # Check word quality (sample first 10)
    for word in words[:10]:
        if len(word) > 50:
            issues.append(f"Suspiciously long word: {word[:30]}...")
            break
    
    # Valid if no issues OR just a warning about being shorter than 5000w
    # (short texts are fine, they just can't be used for 500w stage)
    critical_issues = [i for i in issues if "Too short" not in i or word_count < min_words]
    valid = len(critical_issues) == 0
    
    return {
        "valid": valid,
        "word_count": word_count,
        "issues": issues
    }


def clean_corpus(input_path: str, output_path: str, min_words: int = 200):
    """
    Clean entire corpus file.
    
    Args:
        input_path: Path to raw corpus JSON
        output_path: Path to save cleaned corpus JSON
        min_words: Minimum words to keep a text (default 200 to preserve short but important texts)
    """
    print("="*80)
    print("CORPUS POST-PROCESSOR")
    print("="*80)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")
    print()
    
    # Load
    print("Loading corpus...")
    with open(input_path) as f:
        data = json.load(f)
    
    texts = data["texts"]
    print(f"Loaded {len(texts)} texts")
    
    # Clean
    print("\nCleaning texts...")
    cleaned_texts = []
    removed_texts = []
    stats = {
        "original_count": len(texts),
        "cleaned_count": 0,
        "removed_count": 0,
        "total_issues": 0
    }
    
    for i, text_entry in enumerate(texts):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(texts)}...")
        
        text_id = text_entry["id"]
        original_text = text_entry["text"]
        
        # Clean
        cleaned_text = clean_gutenberg_text(original_text)
        
        # Validate
        validation = validate_text(cleaned_text, text_id, min_words)
        
        if validation["valid"]:
            cleaned_texts.append({
                "id": text_id,
                "text": cleaned_text,
                "source": text_entry["source"],
                "word_count": validation["word_count"]
            })
            stats["cleaned_count"] += 1
        else:
            removed_texts.append({
                "id": text_id,
                "issues": validation["issues"],
                "word_count": validation["word_count"]
            })
            stats["removed_count"] += 1
            stats["total_issues"] += len(validation["issues"])
    
    print(f"  Processed {len(texts)}/{len(texts)}")
    
    # Report
    print("\n" + "="*80)
    print("CLEANING REPORT")
    print("="*80)
    print(f"Original texts: {stats['original_count']}")
    print(f"Cleaned texts:  {stats['cleaned_count']}")
    print(f"Removed texts:  {stats['removed_count']}")
    
    if removed_texts:
        print(f"\nRemoved {len(removed_texts)} texts with issues:")
        for item in removed_texts[:10]:
            print(f"  {item['id']}: {item['word_count']}w - {', '.join(item['issues'])}")
        if len(removed_texts) > 10:
            print(f"  ... and {len(removed_texts) - 10} more")
    
    # Word count stats
    word_counts = [t["word_count"] for t in cleaned_texts]
    if word_counts:
        print(f"\nWord count distribution:")
        print(f"  Min:  {min(word_counts)}")
        print(f"  Max:  {max(word_counts)}")
        print(f"  Mean: {sum(word_counts)/len(word_counts):.0f}")
        print(f"  Median: {sorted(word_counts)[len(word_counts)//2]}")
        
        print(f"\nCurriculum stage availability:")
        for threshold in [10, 25, 50, 100, 200, 500]:
            qualified = sum(1 for wc in word_counts if wc >= threshold)
            max_pairs = qualified * (qualified - 1) // 2
            print(f"  {threshold:3}w: {qualified:4} texts → {max_pairs:,} max pairs")
        
        print(f"\nGeneral text availability:")
        for threshold in [300, 500, 1000, 2000, 5000]:
            qualified = sum(1 for wc in word_counts if wc >= threshold)
            print(f"  ≥{threshold:4}w: {qualified}/{len(word_counts)} ({qualified/len(word_counts)*100:.1f}%)")
    
    # Save
    print(f"\nSaving to {output_path}...")
    output_data = {"texts": cleaned_texts}
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved {len(cleaned_texts)} texts ({output_size_mb:.2f} MB)")
    
    print("\n" + "="*80)
    print("✓ CLEANING COMPLETE")
    print("="*80)
    
    return stats


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_corpus.py input.json output.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    stats = clean_corpus(input_file, output_file)
    
    if stats["removed_count"] > stats["original_count"] * 0.1:
        print(f"\n⚠ WARNING: Removed {stats['removed_count']/stats['original_count']*100:.0f}% of texts")
        print("  Review removed texts to check if cleaning is too aggressive")