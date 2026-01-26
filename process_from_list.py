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
OUTPUT_DIR = "corpus"  # Directory to save metadata and resume state
JSON_OUTPUT_FILE = f"source_texts.json"  # Replace with the actual name of your existing JSON file
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

# Function to extract multiple non-overlapping random chunks, preserving original formatting
def extract_chunks(text, chunk_size, max_chunks):
    # Find all word matches with their positions
    word_matches = list(re.finditer(r'\b\w+\b', text))
    total_words = len(word_matches)
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
        start_word_idx = idx * chunk_size
        end_word_idx = start_word_idx + chunk_size - 1
        if end_word_idx >= total_words:
            end_word_idx = total_words - 1  # Safety check
        start_pos = word_matches[start_word_idx].start()
        end_pos = word_matches[end_word_idx].end()
        chunk = text[start_pos:end_pos].strip()
        chunks.append(chunk)
    
    return chunks

# Function to load existing JSON data
def load_existing_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("texts", [])
    return []

# Function to save updated JSON data
def save_to_json(file_path, texts_array):
    data = {"texts": texts_array}
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

# Function to process from the collected URL list with resumability and profiling
def process_from_list(url_list_file, chunk_size, max_chunks_per_book, json_output_file, random_seed):
    if random_seed is not None:
        random.seed(random_seed)  # Apply seed for reproducibility
    
    with open(url_list_file, "r", encoding='utf-8') as f:
        data = json.load(f)
        items = data.get("items", [])  # Extract the list of items correctly
    
    # Load existing texts array from the JSON file
    texts_array = load_existing_json(json_output_file)
    
    processed_indices = set()  # Track completed book indices
    start_time = time.time()  # Start overall timer
    
    # Load resume state if exists
    if os.path.exists(RESUME_STATE_FILE):
        with open(RESUME_STATE_FILE, "r", encoding='utf-8') as f:
            resume_data = json.load(f)
            processed_indices = set(resume_data.get("processed_indices", []))
        print(f"Resuming from {len(processed_indices)} processed books.")
    
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
            
            chunks = extract_chunks(cleaned_text, chunk_size, max_chunks_per_book)
            for chunk_num, chunk in enumerate(chunks, start=1):
                # Create entry and append to texts_array
                entry = {
                    "id": f"{title.lower().replace(' ', '_')}_chunk{chunk_num}",
                    "name": f"{title} (chunk {chunk_num})",
                    "source": f"{author}, from Project Gutenberg",
                    "text": chunk
                }
                texts_array.append(entry)
                print(f"Appended chunk {chunk_num} from {title} to texts array")
        
        book_end_time = time.time()
        book_duration = book_end_time - book_start_time
        print(f"Processed book {idx + 1} ({title}): {book_duration:.2f} seconds")
        
        # Mark as processed and save progress
        processed_indices.add(idx)
        save_resume_data = {
            "processed_indices": list(processed_indices)
        }
        with open(RESUME_STATE_FILE, "w", encoding='utf-8') as f:
            json.dump(save_resume_data, f, indent=4)
        # Save updated JSON file after each book
        save_to_json(json_output_file, texts_array)
        print(f"Progress saved to {json_output_file} after processing book {idx + 1}.")
    
    # Final overall timing
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Processed {len(items) - len(processed_indices)} new books, added {len(texts_array) - len(load_existing_json(json_output_file))} new excerpts in total: {total_duration:.2f} seconds.")
    
    # Clean up resume file upon completion (optional)
    if os.path.exists(RESUME_STATE_FILE):
        os.remove(RESUME_STATE_FILE)

# Run the function with user-defined config
process_from_list(
    url_list_file=URL_LIST_FILE,
    chunk_size=CHUNK_SIZE,
    max_chunks_per_book=MAX_CHUNKS_PER_BOOK,
    json_output_file=JSON_OUTPUT_FILE,
    random_seed=RANDOM_SEED
)