# File: process_from_list.py

import requests
import json
import random
import re
import os
import time  # Added for profiling

# Config Section: User-defined variables for easy customization
CHUNK_SIZE = 5000  # Fixed number of words per excerpt for non-overlapping chunks
MAX_CHUNKS_PER_BOOK = 8  # Maximum number of chunks to extract from each book
OUTPUT_DIR = "corpus"  # Directory to save excerpts and metadata
URL_LIST_FILE = f"{OUTPUT_DIR}/url_list.json"  # Input file from Step 1
RESUME_STATE_FILE = f"{OUTPUT_DIR}/resume_state.json"  # File for resuming progress
RANDOM_SEED = 42*sum(ord(c) for c in "et")  # Optional random seed for reproducibility (e.g., 42); set to None for default unseeded behavior

# Create output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to clean Gutenberg text (remove headers and footers)
def clean_gutenberg_text(text):
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    if start_idx != -1 and end_idx != -1:
        return text[start_idx + len(start_marker):end_idx].strip()
    return text.strip()

# Function to extract multiple non-overlapping random chunks
def extract_chunks(words, chunk_size, max_chunks):
    total_words = len(words)
    n = total_words // chunk_size  # Number of possible non-overlapping chunks
    if n == 0:
        return []  # Skip if too short for even one chunk
    
    k = min(n, max_chunks)
    if k == 0:
        return []
    
    # Select k unique random indices from 0 to n-1
    indices = random.sample(range(n), k)
    
    chunks = []
    for idx in indices:
        start_pos = idx * chunk_size
        chunk = ' '.join(words[start_pos:start_pos + chunk_size])
        chunks.append(chunk)
    
    return chunks

# Function to process from the collected URL list with resumability and profiling
def process_from_list(url_list_file, chunk_size, max_chunks_per_book, random_seed):
    if random_seed is not None:
        random.seed(random_seed)  # Apply seed for reproducibility
    
    with open(url_list_file, "r", encoding='utf-8') as f:
        data = json.load(f)
        items = data.get("items", [])  # Extract the list of items correctly
    
    texts = []
    processed_indices = set()  # Track completed book indices
    start_time = time.time()  # Start overall timer
    
    # Load resume state if exists
    if os.path.exists(RESUME_STATE_FILE):
        with open(RESUME_STATE_FILE, "r", encoding='utf-8') as f:
            resume_data = json.load(f)
            texts = resume_data.get("texts", [])
            processed_indices = set(resume_data.get("processed_indices", []))
        print(f"Resuming from {len(texts)} excerpts across {len(processed_indices)} books.")
    
    global_idx = len(texts) + 1  # Continue global counter for file naming
    
    for idx, item in enumerate(items):
        if idx in processed_indices:
            continue  # Skip already processed books
        
        title = item["title"]
        author = item["author"]
        download_url = item["download_url"]
        
        book_start_time = time.time()  # Start per-book timer
        
        # Download text
        text_response = requests.get(download_url)
        if text_response.status_code == 200:
            raw_text = text_response.text
            cleaned_text = clean_gutenberg_text(raw_text)
            words = re.findall(r'\b\w+\b', cleaned_text)  # Simple word split
            
            chunks = extract_chunks(words, chunk_size, max_chunks_per_book)
            for chunk_num, chunk in enumerate(chunks, start=1):
                # Save chunk to file
                file_name = f"{OUTPUT_DIR}/{global_idx}_{title.replace(' ', '_')[:40]}_chunk{chunk_num}.txt"
                with open(file_name, 'w', encoding='utf-8') as f:
                    f.write(chunk)
                texts.append({
                    "title": title,
                    "author": author,
                    "chunk_number": chunk_num,
                    "file": file_name
                })
                print(f"Processed chunk {chunk_num} from {title} (global index {global_idx})")
                global_idx += 1
        
        book_end_time = time.time()
        book_duration = book_end_time - book_start_time
        print(f"Processed book {idx + 1} ({title}): {book_duration:.2f} seconds")
        
        # Mark as processed and save progress
        processed_indices.add(idx)
        save_resume_data = {
            "texts": texts,
            "processed_indices": list(processed_indices)
        }
        with open(RESUME_STATE_FILE, "w", encoding='utf-8') as f:
            json.dump(save_resume_data, f, indent=4)
        with open(f"{OUTPUT_DIR}/corpus_metadata.json", "w", encoding='utf-8') as f:
            json.dump(texts, f, indent=4)
        print(f"Progress saved after processing book {idx + 1}.")
    
    # Final overall timing
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Processed {len(texts)} excerpts in total: {total_duration:.2f} seconds.")
    
    # Clean up resume file upon completion (optional)
    if os.path.exists(RESUME_STATE_FILE):
        os.remove(RESUME_STATE_FILE)

# Run the function with user-defined config
process_from_list(
    url_list_file=URL_LIST_FILE,
    chunk_size=CHUNK_SIZE,
    max_chunks_per_book=MAX_CHUNKS_PER_BOOK,
    random_seed=RANDOM_SEED
)