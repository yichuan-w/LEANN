#!/usr/bin/env python3
import os
import json
import argparse
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    from datasets import load_dataset, Dataset, IterableDataset
    from tqdm import tqdm
except ImportError:
    print("Error: Required libraries 'datasets' or 'tqdm' not found.")
    print("Please install them using: pip install datasets tqdm")
    sys.exit(1)

from demo.config import TASK_CONFIGS, get_example_path

# Color constants for output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# --- Dataset Specific Loading Configurations --- 
# You might need to adjust these based on the exact dataset structure on Hugging Face
DATASET_LOAD_INFO = {
    "nq": {
        "hf_name": "nq_open",
        "split": "validation", # NQ Open doesn't have a standard HF dataset, usually custom splits are used.
                               # This entry assumes a custom formatted source or might need manual creation.
                               # Let's mark it as needing manual setup for now.
        "query_key": "question",
        "needs_manual_setup": True,
        "manual_setup_instructions": "nq_open requires a pre-formatted file. Please ensure it exists at the target path."
    },
    "trivia": {
        "hf_name": "trivia_qa",
        "subset": "rc.nocontext",  # Use rc.nocontext as a valid config
        "split": "validation",
        "query_key": "question",
        "needs_manual_setup": False
    },
    "hotpot": {
        "hf_name": "hotpot_qa",
        "subset": "distractor",   # Explicitly choose the 'distractor' config
        "split": "validation",
        "query_key": "question",
        "needs_manual_setup": False
    },
    "gpqa": {
        "hf_name": "Idavidrein/gpqa", # Corrected HF identifier
        "subset": "gpqa_main",  # Use subset (name) for the configuration
        "split": "train",          # Align with evaluation_demo
        "query_key": "Question", # CORRECTED: Use uppercase 'Q' as found in the dataset item
        "needs_manual_setup": False # Assuming this config loads correctly now
    },
    "retrievalqa": {
        "hf_name": "aialt/RetrievalQA",
        "split": "train",
        "query_key": "question",
        "needs_manual_setup": False,
        "custom_loading": True  # Flag to use custom loading logic
    }
}
# --- End Dataset Specific Loading Configurations ---

def format_query(original_query: str) -> str:
    # """Formats the query string according to the NQ example."""
    # # Basic check to prevent double formatting if somehow the prefix is already there
    # if original_query.startswith("Answer these questions:"):
    #     return original_query
    # return f"Answer these questions:\n\nQ: {original_query}?\nA:"
    return original_query

def load_retrievalqa():
    """Custom function to load the RetrievalQA dataset with its complex structure.
    Downloads the JSONL file directly and processes it line by line to avoid schema issues.
    """
    import requests
    import json
    import tempfile
    
    try:
        print(f"{YELLOW}Attempting to directly download and parse RetrievalQA dataset...{RESET}")
        url = "https://huggingface.co/datasets/aialt/RetrievalQA/resolve/main/retrievalqa.jsonl"
        
        # Create a temp file to store the downloaded data
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_file:
            # Download the file
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Ensure we got a successful response
            
            # Process the dataset line by line to avoid schema issues
            data = []
            line_count = 0
            
            for line in response.iter_lines(decode_unicode=True):
                if line:  # Skip empty lines
                    try:
                        item = json.loads(line)
                        # Extract just what we need - the question
                        if "question" in item:
                            data.append({"question": item["question"]})
                        line_count += 1
                        if line_count % 500 == 0:
                            print(f"{YELLOW}Processed {line_count} lines...{RESET}")
                    except json.JSONDecodeError as e:
                        print(f"{RED}Error parsing JSON at line {line_count}: {e}{RESET}")
                        continue
            
            print(f"{GREEN}Successfully parsed {len(data)} questions from RetrievalQA dataset.{RESET}")
            return data
    except Exception as e:
        print(f"{RED}Custom loading for RetrievalQA failed: {e}{RESET}")
        raise

def prepare_file(task: str, force_overwrite: bool = False):
    """Loads, formats, and saves the query file for a specific task."""
    print(f"--- Processing task: {task} ---")

    if task not in TASK_CONFIGS:
        print(f"{RED}Error: Task '{task}' not found in TASK_CONFIGS in config.py.{RESET}")
        return False

    if task not in DATASET_LOAD_INFO:
        print(f"{RED}Error: Loading configuration for task '{task}' not defined in DATASET_LOAD_INFO.{RESET}")
        return False

    config = TASK_CONFIGS[task]
    load_info = DATASET_LOAD_INFO[task]
    target_path = Path(config.query_path) # Use the path from config.py

    if target_path.exists() and not force_overwrite:
        print(f"{YELLOW}Target file already exists: {target_path}. Skipping.{RESET}")
        print(f"Use --force to overwrite.")
        return True

    # Initialize query_key before the try block uses it in except
    query_key: Optional[str] = None 
    try:
        # Use custom loading for retrievalqa
        if task == "retrievalqa" and load_info.get('custom_loading', False):
            raw_dataset = load_retrievalqa()
            
            # Custom handling for retrievalqa data format (list of dicts)
            print(f"Formatting and saving queries to {target_path}...")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            count = 0
            with open(target_path, "w", encoding="utf-8") as f_out:
                for item in tqdm(raw_dataset, desc=f"Formatting {task}"):
                    if "question" in item:
                        formatted_query = format_query(item["question"])
                        f_out.write(json.dumps({"query": formatted_query}) + "\n")
                        count += 1
            
            print(f"{GREEN}Successfully generated query file for {task} with {count} queries: {target_path}{RESET}")
            return True
        else:
            print(f"Loading raw dataset: {load_info['hf_name']} (subset: {load_info.get('subset')}, split: {load_info['split']}) ...")
            raw_dataset = load_dataset(
                load_info['hf_name'],
                name=load_info.get('subset'), # Pass the config name via 'name' (subset)
                split=load_info['split']
            )

            query_key = load_info['query_key']
            print(f"Formatting and saving queries to {target_path}...")
            target_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

            count = 0
            with open(target_path, "w", encoding="utf-8") as f_out:
                for item in tqdm(raw_dataset, desc=f"Formatting {task}"):
                    if not isinstance(item, dict) or query_key not in item:
                         print(f"{YELLOW}Warning: Skipping item due to unexpected format or missing key '{query_key}'. Item: {item}{RESET}")
                         continue
                    original_query = item[query_key]
                    formatted_query = format_query(original_query)
                    f_out.write(json.dumps({"query": formatted_query}) + "\n")
                    count += 1

            print(f"{GREEN}Successfully generated query file for {task} with {count} queries: {target_path}{RESET}")
            return True

    except Exception as e:
        print(f"{RED}Error processing task '{task}': {e}{RESET}")
        key_info = f"query key ('{query_key}')" if query_key else "query key (not assigned)"
        print(f"Check dataset name ('{load_info['hf_name']}'), subset ('{load_info.get('subset')}'), split ('{load_info['split']}'), and {key_info}.")
        print(f"Target path was: {target_path}")
        # Attempt to clean up potentially incomplete file
        if target_path.exists():
            try:
                target_path.unlink()
                print(f"{YELLOW}Cleaned up potentially incomplete file: {target_path}{RESET}")
            except OSError as unlink_e:
                print(f"{RED}Error cleaning up file {target_path}: {unlink_e}{RESET}")
        return False # Indicate failure

def main():
    parser = argparse.ArgumentParser(description="Prepare formatted query files for datasets.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["nq", "trivia", "hotpot", "gpqa", "retrievalqa"],
        choices=list(TASK_CONFIGS.keys()),
        help="Which tasks to prepare query files for."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing query files."
    )
    args = parser.parse_args()

    print(f"Starting query file preparation for tasks: {', '.join(args.tasks)}")
    if args.force:
        print(f"{YELLOW}Force overwrite enabled.{RESET}")

    success_count = 0
    fail_count = 0
    for task in args.tasks:
        if prepare_file(task, args.force):
            success_count += 1
        else:
            fail_count += 1

    print("\n--- Preparation Summary ---")
    print(f"Tasks processed: {len(args.tasks)}")
    print(f"Successful: {success_count}")
    print(f"Failed/Skipped due to errors or manual setup needed: {fail_count}")
    if fail_count > 0:
         print(f"{RED}Some tasks requires manual intervention or failed. Please check the logs above.{RESET}")
         sys.exit(1)
    else:
        print(f"{GREEN}All specified tasks prepared successfully.{RESET}")


if __name__ == "__main__":
    main() 