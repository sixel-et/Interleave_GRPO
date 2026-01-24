# File: collect_titles_and_urls.py

import requests
import json
import os

# Config Section: User-defined variables for easy customization
TARGET_COUNT = 2000  # Number of texts to collect
TOPIC = 'fiction'  # API topic filter (e.g., 'fiction', 'novel', 'history')
LANGUAGES = 'en'  # Language code (e.g., 'en' for English)
MIME_TYPES = 'text/plain'  # Preferred file format
SORT = 'popular'  # Sorting method (e.g., 'popular', 'downloads')
EXCLUDED_KEYWORDS = ["drama", "play", "theater", "dialogue", "tragedy", "comedy (drama)"]  # Keywords to exclude (e.g., for avoiding plays)
OUTPUT_DIR = "corpus"  # Directory to save URL list
URL_LIST_FILE = f"{OUTPUT_DIR}/url_list.json"  # File for storing titles, URLs, and resume state

# Create output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to collect titles and URLs with resumability
def collect_titles_and_urls(target_count, topic, languages, mime_types, sort, excluded_keywords):
    base_url = "https://gutendex.com/books"
    params = {
        "languages": languages,
        "topic": topic,
        "mime_types": mime_types,
        "sort": sort
    }
    
    # Load existing data if file exists for resuming
    items = []
    next_url = base_url + "?" + "&".join([f"{k}={v}" for k, v in params.items()])
    if os.path.exists(URL_LIST_FILE):
        with open(URL_LIST_FILE, "r", encoding='utf-8') as f:
            data = json.load(f)
            items = data.get("items", [])
            saved_next_url = data.get("next_url")
            if saved_next_url:
                next_url = saved_next_url
        print(f"Resuming from {len(items)} collected items.")

    while next_url and len(items) < target_count:
        response = requests.get(next_url)
        if response.status_code != 200:
            print(f"Error fetching {next_url}: {response.status_code}")
            break
        data = response.json()
        
        for book in data["results"]:
            # Check subjects and bookshelves for exclusion
            subjects = [s.lower() for s in book.get("subjects", [])]
            bookshelves = [b.lower() for b in book.get("bookshelves", [])]
            if any(keyword in ' '.join(subjects + bookshelves) for keyword in excluded_keywords):
                continue  # Skip excluded categories
            
            title = book["title"]
            author = book["authors"][0]["name"] if book["authors"] else "Unknown"
            download_url = book["formats"].get("text/plain; charset=utf-8") or book["formats"].get("text/plain")
            
            if download_url and len(items) < target_count:
                items.append({"title": title, "author": author, "download_url": download_url})
                print(f"Collected {len(items)}: {title}")
        
        next_url = data["next"]
        
        # Save progress after each page (including next_url for resume)
        save_data = {
            "items": items,
            "next_url": next_url
        }
        with open(URL_LIST_FILE, "w", encoding='utf-8') as f:
            json.dump(save_data, f, indent=4)
        print(f"Progress saved to {URL_LIST_FILE} after processing page.")

    # Final save without next_url if complete
    if len(items) >= target_count or not next_url:
        save_data = {"items": items}
        with open(URL_LIST_FILE, "w", encoding='utf-8') as f:
            json.dump(save_data, f, indent=4)
    
    print(f"Collected {len(items)} titles and URLs, saved to {URL_LIST_FILE}.")

# Run the function with user-defined config
collect_titles_and_urls(
    target_count=TARGET_COUNT,
    topic=TOPIC,
    languages=LANGUAGES,
    mime_types=MIME_TYPES,
    sort=SORT,
    excluded_keywords=EXCLUDED_KEYWORDS
)