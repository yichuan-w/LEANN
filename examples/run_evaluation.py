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
from typing import List, Dict, Any
import glob
import pickle

# Add project root to path to allow importing from leann
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from leann.api import LeannSearcher

# --- Configuration ---
NQ_QUERIES_FILE = Path("/opt/dlami/nvme/scaling_out/examples/nq_open.jsonl")

# Ground truth files for different datasets
GROUND_TRUTH_FILES = {
    "rpj_wiki": "/opt/dlami/nvme/scaling_out/indices/rpj_wiki/facebook/contriever-msmarco/flat_results_nq_k3.json",
    "dpr": "/opt/dlami/nvme/scaling_out/indices/dpr/facebook/contriever-msmarco/flat_results_nq_k3.json"
}

# Old passages for different datasets  
OLD_PASSAGES_GLOBS = {
    "rpj_wiki": "/opt/dlami/nvme/scaling_out/passages/rpj_wiki/8-shards/raw_passages-*-of-8.pkl.jsonl",
    "dpr": "/opt/dlami/nvme/scaling_out/passages/dpr/1-shards/raw_passages-*-of-1.pkl.jsonl"
}

# --- Helper Class to Load Original Passages ---
class OldPassageLoader:
    """A simplified version of the old LazyPassages class to fetch golden results by ID."""
    def __init__(self, passages_glob: str):
        self.jsonl_paths = sorted(glob.glob(passages_glob))
        self.offsets = {}
        self.fps = [open(p, "r", encoding="utf-8") for p in self.jsonl_paths]
        print("Building offset map for original passages...")
        for i, shard_path_str in enumerate(self.jsonl_paths):
            old_idx_path = Path(shard_path_str.replace(".jsonl", ".idx"))
            if not old_idx_path.exists(): continue
            with open(old_idx_path, 'rb') as f:
                shard_offsets = pickle.load(f)
                for pid, offset in shard_offsets.items():
                    self.offsets[str(pid)] = (i, offset)
        print("Offset map for original passages is ready.")

    def get_passage_by_id(self, pid: str) -> Dict[str, Any]:
        pid = str(pid)
        if pid not in self.offsets:
            raise ValueError(f"Passage ID {pid} not found in offsets")
        file_idx, offset = self.offsets[pid]
        fp = self.fps[file_idx]
        fp.seek(offset)
        return json.loads(fp.readline())

    def __del__(self):
        for fp in self.fps:
            fp.close()

def load_queries(file_path: Path) -> List[str]:
    queries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            queries.append(data['query'])
    return queries

def main():
    parser = argparse.ArgumentParser(description="Run recall evaluation on a LEANN index.")
    parser.add_argument("index_path", type=str, help="Path to the LEANN index to evaluate.")
    parser.add_argument("--num-queries", type=int, default=10, help="Number of queries to evaluate.")
    parser.add_argument("--top-k", type=int, default=3, help="The 'k' value for recall@k.")
    parser.add_argument("--ef-search", type=int, default=120, help="The 'efSearch' parameter for HNSW.")
    args = parser.parse_args()

    print(f"--- Final, Correct Recall Evaluation (efSearch={args.ef_search}) ---")
    
    # Detect dataset type from index path
    index_path_str = str(args.index_path)
    if "rpj_wiki" in index_path_str:
        dataset_type = "rpj_wiki"
    elif "dpr" in index_path_str:
        dataset_type = "dpr"
    else:
        print("WARNING: Unknown dataset type, defaulting to rpj_wiki")
        dataset_type = "rpj_wiki"
    
    print(f"INFO: Detected dataset type: {dataset_type}")

    try:
        searcher = LeannSearcher(args.index_path)
        queries = load_queries(NQ_QUERIES_FILE)
        
        golden_results_file = GROUND_TRUTH_FILES[dataset_type]
        old_passages_glob = OLD_PASSAGES_GLOBS[dataset_type]
        
        print(f"INFO: Using ground truth file: {golden_results_file}")
        print(f"INFO: Using old passages glob: {old_passages_glob}")
        
        with open(golden_results_file, 'r') as f:
            golden_results_data = json.load(f)
        
        old_passage_loader = OldPassageLoader(old_passages_glob)

        num_eval_queries = min(args.num_queries, len(queries))
        queries = queries[:num_eval_queries]
        
        print(f"\nRunning evaluation on {num_eval_queries} queries...")
        recall_scores = []
        search_times = []

        for i in range(num_eval_queries):
            start_time = time.time()
            new_results = searcher.search(queries[i], top_k=args.top_k, ef=args.ef_search)
            search_times.append(time.time() - start_time)

            # Correct Recall Calculation: Based on TEXT content
            new_texts = {result.text for result in new_results}
            golden_ids = golden_results_data["indices"][i][:args.top_k]
            golden_texts = {old_passage_loader.get_passage_by_id(str(gid))['text'] for gid in golden_ids}

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
            print(f"--------------------------------")

        avg_recall = np.mean(recall_scores) if recall_scores else 0
        avg_time = np.mean(search_times) if search_times else 0

        print(f"\n🎉 --- Evaluation Complete ---")
        print(f"Avg. Recall@{args.top_k} (efSearch={args.ef_search}): {avg_recall:.4f}")
        print(f"Avg. Search Time: {avg_time:.4f}s")

    except Exception as e:
        print(f"\n❌ An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()