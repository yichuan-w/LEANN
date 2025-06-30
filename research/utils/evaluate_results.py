# Filename: evaluate_results_xai_line_sync.py
import openai
import json
import os
import time
from dotenv import load_dotenv
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures
from typing import List, Dict, Any, Tuple

# --- Configuration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY in your .env file")

try:
    client = openai.OpenAI(
        api_key=OPENAI_API_KEY,
    )
except ImportError:
    print("Please install the latest OpenAI library: pip install --upgrade openai")
    exit()
except openai.AuthenticationError:
     print("OpenAI library reported an AuthenticationError. Ensure OPENAI_API_KEY is correct.")
     exit()

LLM_MODEL = "gpt-3.5-turbo"  # Using OpenAI's standard model
MAX_RETRIES = 5
INITIAL_RETRY_DELAY_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 90
MAX_WORKERS = 10  # Number of parallel workers

# --- File Paths (Adjust as needed) ---
# User provided paths
QUERIES_FILE_PATH = "/opt/dlami/nvme/scaling_out/examples/enron_eval_retrieval.jsonl"
RAW_PASSAGES_FILE_PATH = "/opt/dlami/nvme/scaling_out/passages/enron_emails/1-shards/raw_passages-0-of-1.jsonl"
RESULTS_FILE_PATH = "search_results_top10_bm25.jsonl" # This file's Nth line corresponds to QUERIES_FILE_PATH's Nth line
OUTPUT_EVALUATION_FILE = "llm_containment_evaluations_xai_line_sync.jsonl"

# --- LLM Prompt Definitions for Containment (Same as before) ---
CONTAINMENT_SYSTEM_PROMPT = """You are an AI evaluator. Your task is to determine if the core information presented in the 'Retrieved Passage' is directly contained within *any* of the text snippets provided in the 'Ground Truth Email Snippets' list."""
CONTAINMENT_USER_TEMPLATE = """Retrieved Passage:
"{retrieved_passage_text}"

---
Ground Truth Email Snippets (Parts of the correct source email):
{ground_truth_snippets_formatted_list}
---

Is the core information of the 'Retrieved Passage' directly present or fully contained within *any* of the 'Ground Truth Email Snippets' listed above?
- Focus on whether the specific facts or statements in the 'Retrieved Passage' can be found within the ground truth snippets.
- Ignore minor formatting differences. If the retrieved passage is a direct quote or a very close paraphrase of content within the ground truth snippets, answer YES.
- Respond YES if the Retrieved Passage's content is clearly represented in one or more of the ground truth snippets.
- Respond NO if the Retrieved Passage's content is not found, is contradictory, or introduces significant information not present in the ground truth snippets.

Your response must be a single word: YES or NO.
"""

# --- Data Loading Functions ---

def load_queries_as_list(file_path):
    """
    Loads queries from a jsonl file into a list, preserving order.
    Each item in the list is a dict containing original_id, query_text, and ground_truth_message_ids.
    """
    queries_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line)
                    required_keys = ["id", "query", "ground_truth_message_ids"]
                    if not all(key in data for key in required_keys):
                        print(f"Warning: Skipping line {line_num + 1} in query file due to missing keys: {line.strip()}")
                        continue
                    if not isinstance(data["ground_truth_message_ids"], list):
                        print(f"Warning: 'ground_truth_message_ids' is not a list in line {line_num + 1}. Skipping: {line.strip()}")
                        continue
                    queries_list.append({
                        "original_id": data["id"], # Store the original ID from the file
                        "query_text": data["query"],
                        "ground_truth_message_ids": data["ground_truth_message_ids"]
                    })
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line {line_num + 1} in query file: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: Queries file not found at {file_path}")
        exit()
    print(f"Loaded {len(queries_list)} queries (as a list) from {file_path}")
    return queries_list

def load_all_passages_by_message_id(raw_passages_file_path):
    """Loads all raw passages into memory, grouped by message_id. (Same as before)"""
    passages_dict = defaultdict(list)
    # ... (implementation from previous script, no changes needed here) ...
    print(f"Loading all raw passages from {raw_passages_file_path} into memory...")
    try:
        with open(raw_passages_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line)
                    if "message_id" in data and "text" in data:
                        passages_dict[data["message_id"]].append(data["text"])
                    else:
                         print(f"Warning: Skipping line {line_num+1} in raw passages file due to missing 'message_id' or 'text'.")
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line {line_num + 1} in raw passages file: {line.strip()}")
        print(f"Finished loading raw passages. Found {len(passages_dict)} unique message IDs.")
    except FileNotFoundError:
        print(f"Error: Raw passages file not found at {raw_passages_file_path}")
        exit()
    except MemoryError:
        print("Error: Ran out of memory loading all raw passages. Consider an indexed approach.")
        exit()
    return dict(passages_dict)

def load_search_results_as_list(file_path):
    """Loads search results from a jsonl file into a list, preserving order."""
    results_list = []
    # ... (implementation similar to load_queries_as_list, parsing each line as a dict) ...
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line)
                    # We expect "query_id" (though not used for matching) and "passages"
                    if "passages" not in data: # query_id might be implicitly by order
                        print(f"Warning: Skipping line {line_num + 1} in search results file due to missing 'passages' key: {line.strip()}")
                        continue
                    results_list.append(data)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line {line_num + 1} in search results file: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: Search results file not found at {file_path}")
        exit()
    print(f"Loaded {len(results_list)} search result sets (as a list) from {file_path}")
    return results_list


def format_ground_truth_snippets(snippet_list):
    """Formats the list of ground truth snippets for the prompt. (Same as before)"""
    # ... (implementation from previous script) ...
    if not snippet_list:
        return "  [No ground truth snippets found for the target message ID(s)]"
    formatted = []
    for i, snippet in enumerate(snippet_list):
        display_snippet = (snippet[:500] + '...') if len(snippet) > 500 else snippet
        formatted.append(f"  {i+1}. {display_snippet}")
    return "\n".join(formatted)

# --- LLM API Call Function ---
def get_llm_containment_evaluation(retrieved_passage_text: str, ground_truth_snippets_list: List[str], query_id_for_log: str, passage_identifier_info: str, query_text_for_context: str = "") -> str:
    """Calls the OpenAI API with retry logic."""
    formatted_gt_snippets = format_ground_truth_snippets(ground_truth_snippets_list)
    # max_gt_chars_in_prompt = 5000 # Arbitrary limit, adjust as needed
    # if len(formatted_gt_snippets) > max_gt_chars_in_prompt:
    #     print(f"Warning: Ground truth snippets for Q_log_id:{query_id_for_log} are too long ({len(formatted_gt_snippets)} chars), truncating for LLM prompt.")
    #     formatted_gt_snippets = formatted_gt_snippets[:max_gt_chars_in_prompt] + "\n  [... Snippets Truncated ...]"

    user_prompt = CONTAINMENT_USER_TEMPLATE.format(
        retrieved_passage_text=retrieved_passage_text,
        ground_truth_snippets_formatted_list=formatted_gt_snippets
    )
    messages = [
        {"role": "system", "content": CONTAINMENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    current_retry_delay = INITIAL_RETRY_DELAY_SECONDS
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=10,
                timeout=REQUEST_TIMEOUT_SECONDS
            )
            answer = response.choices[0].message.content.strip().upper()
            if answer in ["YES", "NO"]:
                return answer
            else:
                print(f"Warning: Unexpected LLM response content '{answer[:100]}' for Q_log_id:{query_id_for_log} P:{passage_identifier_info}. Defaulting to NO.")
                return "NO"
        except openai.APIConnectionError as e:
            error_message = f"API Connection Error (Attempt {attempt + 1}/{MAX_RETRIES}): {e}"
        except openai.RateLimitError as e:
            error_message = f"API Rate Limit Error (Attempt {attempt + 1}/{MAX_RETRIES}): {e}"
        except openai.APIStatusError as e:
            error_message = f"API Status Error (Attempt {attempt + 1}/{MAX_RETRIES}): {e.status_code} - {e.response}"
            if e.status_code == 401:
                return "ERROR_AUTH"
            if e.status_code == 500:
                pass
            else:
                return "ERROR_API_CLIENT"
        except Exception as e:
            error_message = f"Unexpected error with OpenAI lib (Attempt {attempt + 1}/{MAX_RETRIES}): {type(e).__name__} - {e}"

        print(f"{error_message}. Query Log ID: {query_id_for_log}, Passage: {passage_identifier_info}")
        if "ERROR_AUTH" in error_message or "ERROR_API_CLIENT" in error_message:
            break

        if attempt < MAX_RETRIES - 1:
            print(f"Retrying in {current_retry_delay} seconds...")
            time.sleep(current_retry_delay)
            current_retry_delay = min(current_retry_delay * 2, 60)
        else:
            print(f"Max retries ({MAX_RETRIES}) reached for Q_log_id:{query_id_for_log} P:{passage_identifier_info}. Skipping.")
            return "ERROR_MAX_RETRIES"
    return "ERROR_MAX_RETRIES"

def process_query_passage_pair(args: Tuple[Dict[str, Any], Dict[str, Any], Dict[str, List[str]], set]) -> List[Dict[str, Any]]:
    """Process a single query-passage pair for parallel execution."""
    query_info, result_item, passages_lookup, already_evaluated = args
    evaluations = []
    
    query_original_id = query_info["original_id"]
    query_text = query_info["query_text"]
    target_message_ids = query_info.get("ground_truth_message_ids", [])
    
    if not target_message_ids:
        return evaluations

    ground_truth_snippets = []
    for msg_id_in_query_file in target_message_ids:
        msg_id_to_lookup = msg_id_in_query_file
        if msg_id_in_query_file.startswith("<") and msg_id_in_query_file.endswith(">"):
            msg_id_to_lookup = msg_id_in_query_file[1:-1]
        
        snippets = passages_lookup.get(msg_id_to_lookup)
        if snippets:
            ground_truth_snippets.extend(snippets)

    if not ground_truth_snippets:
        return evaluations

    retrieved_passages = result_item.get("passages", [])
    if not retrieved_passages:
        return evaluations

    for passage_idx, passage_obj in enumerate(retrieved_passages):
        if not isinstance(passage_obj, dict):
            print(f"Warning: Invalid passage format for Q_original_id:{query_original_id}, passage index {passage_idx}. Skipping passage.")
            continue

        retrieved_passage_text = passage_obj.get("text", "").strip()
        passage_identifier = passage_obj.get("passage_id", passage_obj.get("id", f"retrieved_idx_{passage_idx}"))
        
        evaluation_key = (query_original_id, passage_identifier)
        if evaluation_key in already_evaluated:
            continue

        passage_text_preview = (retrieved_passage_text[:75] + '...') if len(retrieved_passage_text) > 75 else retrieved_passage_text

        if not retrieved_passage_text:
            evaluation = "NO"
        else:
            evaluation = get_llm_containment_evaluation(
                retrieved_passage_text,
                ground_truth_snippets,
                query_original_id,
                passage_identifier,
                query_text
            )
            if evaluation == "ERROR_AUTH":
                print("Authentication error with OpenAI API. Stopping script.")
                return evaluations

        evaluation_record = {
            "query_original_id": query_original_id,
            "passage_identifier": passage_identifier,
            "passage_text_preview": passage_text_preview,
            "evaluation": evaluation,
            "model_used": LLM_MODEL,
            "ground_truth_message_ids_checked": target_message_ids
        }
        evaluations.append(evaluation_record)
    
    return evaluations

# --- Resume Logic ---
def load_existing_evaluations(output_file):
    """Loads already evaluated query-passage pairs using 'passage_identifier' and 'query_original_id'. (Same as before, but keying with original_id)"""
    # ... (implementation from previous script, ensure it uses the correct ID for keys) ...
    evaluated_pairs = set()
    if os.path.exists(output_file):
        print(f"Loading existing containment evaluations from {output_file}...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line)
                    # Key for resuming should be based on the logged original query ID
                    query_original_id = data.get('query_original_id')
                    passage_identifier = data.get('passage_identifier')
                    if query_original_id is not None and passage_identifier is not None:
                         evaluated_pairs.add((query_original_id, passage_identifier))
                    else:
                         print(f"Warning: Could not identify query_original_id/passage_identifier in existing file line {line_num + 1}.")
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line {line_num + 1} in existing file: {line.strip()}")
                except KeyError as e:
                     print(f"Warning: Skipping line {line_num + 1} with missing key '{e}' in existing file: {line.strip()}")
        print(f"Loaded {len(evaluated_pairs)} existing evaluation records.")
    else:
         print(f"No existing evaluation file found at {output_file}. Starting fresh.")
    return evaluated_pairs

# --- Main Execution Logic ---

def main():
    """Main function to run the containment evaluation process using parallel processing."""
    print(f"Starting containment evaluation using OpenAI model: {LLM_MODEL} via OpenAI library interface.")

    # Load data as lists
    queries_list = load_queries_as_list(QUERIES_FILE_PATH)
    passages_lookup = load_all_passages_by_message_id(RAW_PASSAGES_FILE_PATH)
    search_results_list = load_search_results_as_list(RESULTS_FILE_PATH)

    if not queries_list or not search_results_list or not passages_lookup:
        print("Error loading one or more input files or raw passages. Exiting.")
        return

    # Determine the number of items to process
    num_items_to_process = min(len(queries_list), len(search_results_list))
    print(f"Will process {num_items_to_process} query-result pairs.")

    already_evaluated = load_existing_evaluations(OUTPUT_EVALUATION_FILE)

    try:
        with open(OUTPUT_EVALUATION_FILE, 'a', encoding='utf-8') as outfile:
            # Prepare arguments for parallel processing
            process_args = [
                (queries_list[i], search_results_list[i], passages_lookup, already_evaluated)
                for i in range(num_items_to_process)
            ]

            # Use ThreadPoolExecutor for parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all tasks and get futures
                futures = [executor.submit(process_query_passage_pair, args) for args in process_args]
                
                # Process results as they complete
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing query-result pairs"):
                    try:
                        evaluations = future.result()
                        for evaluation in evaluations:
                            outfile.write(json.dumps(evaluation) + "\n")
                            outfile.flush()
                            # Update already_evaluated set
                            already_evaluated.add((evaluation["query_original_id"], evaluation["passage_identifier"]))
                    except Exception as e:
                        print(f"Error processing query-result pair: {e}")

    except IOError as e:
        print(f"Error writing to output file {OUTPUT_EVALUATION_FILE}: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during the main processing loop: {e}")
        return

    print("\n--- Containment Evaluation Script Finished ---")

    # --- Final Summary Calculation ---
    print(f"Calculating final summary statistics from: {OUTPUT_EVALUATION_FILE}")
    final_query_containment_found = {}
    total_evaluated_pairs = 0
    error_count = 0
    evaluated_query_original_ids = set()

    try:
        with open(OUTPUT_EVALUATION_FILE, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                total_evaluated_pairs += 1
                try:
                    data = json.loads(line)
                    q_original_id = data['query_original_id']
                    eval_result = data['evaluation']
                    evaluated_query_original_ids.add(q_original_id)

                    if eval_result == "YES":
                        final_query_containment_found[q_original_id] = True
                    elif q_original_id not in final_query_containment_found:
                        final_query_containment_found[q_original_id] = False
                    if eval_result not in ["YES", "NO"]:
                        error_count += 1
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error reading line {line_num + 1} during summary: {e} - Line: {line.strip()}")
                    error_count += 1

        num_queries_with_any_contained = sum(1 for contained in final_query_containment_found.values() if contained)
        total_unique_queries_evaluated = len(evaluated_query_original_ids)

        if total_unique_queries_evaluated > 0:
            containment_rate_at_10 = num_queries_with_any_contained / total_unique_queries_evaluated
            print(f"\n--- Final Statistics (Containment Check) ---")
            print(f"Total unique queries processed (based on output file entries): {total_unique_queries_evaluated}")
            print(f"Number of queries with at least one contained passage (YES): {num_queries_with_any_contained}")
            print(f"Containment Match Rate @ Top 10 (Any YES): {containment_rate_at_10:.4f}")
            print(f"Total query-passage pairs processed (lines in output file): {total_evaluated_pairs}")
            if error_count > 0:
                print(f"Number of evaluation errors or non-YES/NO results: {error_count}")
        else:
            print("No evaluation results found to summarize.")
    except FileNotFoundError:
        print(f"Error: Output file {OUTPUT_EVALUATION_FILE} not found for summary.")
    except Exception as e:
        print(f"An unexpected error occurred during summary calculation: {e}")

    print(f"\nDetailed containment evaluations saved to: {OUTPUT_EVALUATION_FILE}")


if __name__ == "__main__":
    # Dummy files for testing the line sync logic
    if not os.path.exists(QUERIES_FILE_PATH):
        print(f"Warning: {QUERIES_FILE_PATH} not found. Creating dummy file.")
        with open(QUERIES_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump({"id": "q_alpha", "query": "Query Alpha Text", "ground_truth_message_ids": ["<msg_A>"]}, f); f.write("\n") # Line 0
            json.dump({"id": "q_beta", "query": "Query Beta Text", "ground_truth_message_ids": ["<msg_B>"]}, f); f.write("\n")  # Line 1
            json.dump({"id": "q_gamma", "query": "Query Gamma Text", "ground_truth_message_ids": ["<msg_C>"]}, f); f.write("\n")# Line 2

    if not os.path.exists(RAW_PASSAGES_FILE_PATH):
         print(f"Warning: {RAW_PASSAGES_FILE_PATH} not found. Creating dummy file.")
         with open(RAW_PASSAGES_FILE_PATH, 'w', encoding='utf-8') as f:
              json.dump({"text": "Content from message A snippet 1.", "id": 100, "message_id": "<msg_A>"}, f); f.write("\n")
              json.dump({"text": "Content from message A snippet 2.", "id": 101, "message_id": "<msg_A>"}, f); f.write("\n")
              json.dump({"text": "Content from message B.", "id": 200, "message_id": "<msg_B>"}, f); f.write("\n")
              json.dump({"text": "Content from message D (unrelated).", "id": 300, "message_id": "<msg_D>"}, f); f.write("\n")

    # RESULTS_FILE_PATH should have results corresponding line-by-line to QUERIES_FILE_PATH
    if not os.path.exists(RESULTS_FILE_PATH):
        print(f"Warning: {RESULTS_FILE_PATH} not found. Creating dummy file (2 entries).")
        with open(RESULTS_FILE_PATH, 'w', encoding='utf-8') as f:
            # Result for query "q_alpha" (line 0 in queries file)
            json.dump({"query_id": "this_can_be_ignored_if_line_sync", "passages": [{"id": 101, "text": "Content from message A snippet 2."}, {"id": 300, "text": "Content from message D (unrelated)."}]}, f); f.write("\n")
            # Result for query "q_beta" (line 1 in queries file)
            json.dump({"query_id": "this_too", "passages": [{"id": 999, "text": "Some other text."}, {"id": 200, "text": "Content from message B."}]}, f); f.write("\n")
            # Note: Only 2 result sets, but 3 queries in dummy QUERIES_FILE_PATH.
            # The script will process min(len(queries_list), len(search_results_list)) if you uncomment that logic,
            # or just len(search_results_list) as it's currently written for tqdm.

    main()