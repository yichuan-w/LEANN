#!/usr/bin/env python3
"""
This script runs a recall evaluation on a given LEANN index.
It correctly compares results by fetching the text content for both the new search
results and the golden standard results, making the comparison robust to ID changes.
"""

import json
import argparse
import time
from pathlib import Path
import sys
import numpy as np
from typing import List

from leann.api import LeannSearcher


def download_data_if_needed(data_root: Path):
    """Checks if the data directory exists, and if not, downloads it from HF Hub."""
    if not data_root.exists():
        print(f"Data directory '{data_root}' not found.")
        print(
            "Downloading evaluation data from Hugging Face Hub... (this may take a moment)"
        )
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id="LEANN-RAG/leann-rag-evaluation-data",
                repo_type="dataset",
                local_dir=data_root,
                local_dir_use_symlinks=False,  # Recommended for Windows compatibility and simpler structure
            )
            print("Data download complete!")
        except ImportError:
            print(
                "Error: huggingface_hub is not installed. Please install it to download the data:"
            )
            print("uv pip install -e '.[dev]'")
            sys.exit(1)
        except Exception as e:
            print(f"An error occurred during data download: {e}")
            sys.exit(1)


# --- Helper Function to get Golden Passages ---
def get_golden_texts(searcher: LeannSearcher, golden_ids: List[int]) -> set:
    """
    Retrieves the text for golden passage IDs directly from the LeannSearcher's
    passage manager.
    """
    golden_texts = set()
    for gid in golden_ids:
        try:
            # PassageManager uses string IDs
            passage_data = searcher.passage_manager.get_passage(str(gid))
            golden_texts.add(passage_data["text"])
        except KeyError:
            print(
                f"Warning: Golden passage ID '{gid}' not found in the index's passage data."
            )
    return golden_texts


def load_queries(file_path: Path) -> List[str]:
    queries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            queries.append(data["query"])
    return queries


def main():
    parser = argparse.ArgumentParser(
        description="Run recall evaluation on a LEANN index."
    )
    parser.add_argument(
        "index_path", type=str, help="Path to the LEANN index to evaluate."
    )
    parser.add_argument(
        "--num-queries", type=int, default=10, help="Number of queries to evaluate."
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="The 'k' value for recall@k."
    )
    parser.add_argument(
        "--ef-search", type=int, default=120, help="The 'efSearch' parameter for HNSW."
    )
    args = parser.parse_args()

    # --- Path Configuration ---
    # Assumes a project structure where the script is in 'examples/'
    # and data is in 'data/' at the project root.
    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "data"

    # Automatically download data if it doesn't exist
    download_data_if_needed(data_root)

    # Detect dataset type from index path to select the correct ground truth
    index_path_str = str(args.index_path)
    if "rpj_wiki" in index_path_str:
        dataset_type = "rpj_wiki"
    elif "dpr" in index_path_str:
        dataset_type = "dpr"
    else:
        # Fallback: try to infer from the index directory name
        dataset_type = Path(args.index_path).name
        print(
            f"WARNING: Could not detect dataset type from path, inferred '{dataset_type}'."
        )

    queries_file = data_root / "queries" / "nq_open.jsonl"
    golden_results_file = (
        data_root / "ground_truth" / dataset_type / "flat_results_nq_k3.json"
    )

    print(f"INFO: Detected dataset type: {dataset_type}")
    print(f"INFO: Using queries file: {queries_file}")
    print(f"INFO: Using ground truth file: {golden_results_file}")

    try:
        searcher = LeannSearcher(args.index_path)
        queries = load_queries(queries_file)

        with open(golden_results_file, "r") as f:
            golden_results_data = json.load(f)

        num_eval_queries = min(args.num_queries, len(queries))
        queries = queries[:num_eval_queries]

        print(f"\nRunning evaluation on {num_eval_queries} queries...")
        recall_scores = []
        search_times = []

        for i in range(num_eval_queries):
            start_time = time.time()
            new_results = searcher.search(
                queries[i], top_k=args.top_k, ef=args.ef_search
            )
            search_times.append(time.time() - start_time)

            # Correct Recall Calculation: Based on TEXT content
            new_texts = {result.text for result in new_results}

            # Get golden texts directly from the searcher's passage manager
            golden_ids = golden_results_data["indices"][i][: args.top_k]
            golden_texts = get_golden_texts(searcher, golden_ids)

            overlap = len(new_texts & golden_texts)
            recall = overlap / len(golden_texts) if golden_texts else 0
            recall_scores.append(recall)

            print("\n--- EVALUATION RESULTS ---")
            print(f"Query: {queries[i]}")
            print(f"New Results: {new_texts}")
            print(f"Golden Results: {golden_texts}")
            print(f"Overlap: {overlap}")
            print(f"Recall: {recall}")
            print(f"Search Time: {search_times[-1]:.4f}s")
            print("--------------------------------")

        avg_recall = np.mean(recall_scores) if recall_scores else 0
        avg_time = np.mean(search_times) if search_times else 0

        print("\n🎉 --- Evaluation Complete ---")
        print(f"Avg. Recall@{args.top_k} (efSearch={args.ef_search}): {avg_recall:.4f}")
        print(f"Avg. Search Time: {avg_time:.4f}s")

    except Exception as e:
        print(f"\n❌ An error occurred during evaluation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
