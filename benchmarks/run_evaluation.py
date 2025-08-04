#!/usr/bin/env python3
"""
This script runs a recall evaluation on a given LEANN index.
It correctly compares results by fetching the text content for both the new search
results and the golden standard results, making the comparison robust to ID changes.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from leann.api import LeannBuilder, LeannSearcher


def download_data_if_needed(data_root: Path, download_embeddings: bool = False):
    """Checks if the data directory exists, and if not, downloads it from HF Hub."""
    if not data_root.exists():
        print(f"Data directory '{data_root}' not found.")
        print("Downloading evaluation data from Hugging Face Hub... (this may take a moment)")
        try:
            from huggingface_hub import snapshot_download

            if download_embeddings:
                # Download everything including embeddings (large files)
                snapshot_download(
                    repo_id="LEANN-RAG/leann-rag-evaluation-data",
                    repo_type="dataset",
                    local_dir=data_root,
                    local_dir_use_symlinks=False,
                )
                print("Data download complete (including embeddings)!")
            else:
                # Download only specific folders, excluding embeddings
                allow_patterns = [
                    "ground_truth/**",
                    "indices/**",
                    "queries/**",
                    "*.md",
                    "*.txt",
                ]
                snapshot_download(
                    repo_id="LEANN-RAG/leann-rag-evaluation-data",
                    repo_type="dataset",
                    local_dir=data_root,
                    local_dir_use_symlinks=False,
                    allow_patterns=allow_patterns,
                )
                print("Data download complete (excluding embeddings)!")
        except ImportError:
            print(
                "Error: huggingface_hub is not installed. Please install it to download the data:"
            )
            print("uv pip install -e '.[dev]'")
            sys.exit(1)
        except Exception as e:
            print(f"An error occurred during data download: {e}")
            sys.exit(1)


def download_embeddings_if_needed(data_root: Path, dataset_type: str | None = None):
    """Download embeddings files specifically."""
    embeddings_dir = data_root / "embeddings"

    if dataset_type:
        # Check if specific dataset embeddings exist
        target_file = embeddings_dir / dataset_type / "passages_00.pkl"
        if target_file.exists():
            print(f"Embeddings for {dataset_type} already exist")
            return str(target_file)

    print("Downloading embeddings from HuggingFace Hub...")
    try:
        from huggingface_hub import snapshot_download

        # Download only embeddings folder
        snapshot_download(
            repo_id="LEANN-RAG/leann-rag-evaluation-data",
            repo_type="dataset",
            local_dir=data_root,
            local_dir_use_symlinks=False,
            allow_patterns=["embeddings/**/*.pkl"],
        )
        print("Embeddings download complete!")

        if dataset_type:
            target_file = embeddings_dir / dataset_type / "passages_00.pkl"
            if target_file.exists():
                return str(target_file)

        return str(embeddings_dir)

    except Exception as e:
        print(f"Error downloading embeddings: {e}")
        sys.exit(1)


# --- Helper Function to get Golden Passages ---
def get_golden_texts(searcher: LeannSearcher, golden_ids: list[int]) -> set:
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
            print(f"Warning: Golden passage ID '{gid}' not found in the index's passage data.")
    return golden_texts


def load_queries(file_path: Path) -> list[str]:
    queries = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            queries.append(data["query"])
    return queries


def build_index_from_embeddings(embeddings_file: str, output_path: str, backend: str = "hnsw"):
    """
    Build a LEANN index from pre-computed embeddings.

    Args:
        embeddings_file: Path to pickle file with (ids, embeddings) tuple
        output_path: Path where to save the index
        backend: Backend to use ("hnsw" or "diskann")
    """
    print(f"Building {backend} index from embeddings: {embeddings_file}")

    # Create builder with appropriate parameters
    if backend == "hnsw":
        builder_kwargs = {
            "M": 32,  # Graph degree
            "efConstruction": 256,  # Construction complexity
            "is_compact": True,  # Use compact storage
            "is_recompute": True,  # Enable pruning for better recall
        }
    elif backend == "diskann":
        builder_kwargs = {
            "complexity": 64,
            "graph_degree": 32,
            "search_memory_maximum": 8.0,  # GB
            "build_memory_maximum": 16.0,  # GB
        }
    else:
        builder_kwargs = {}

    builder = LeannBuilder(
        backend_name=backend,
        embedding_model="facebook/contriever-msmarco",  # Model used to create embeddings
        dimensions=768,  # Will be auto-detected from embeddings
        **builder_kwargs,
    )

    # Build index from precomputed embeddings
    builder.build_index_from_embeddings(output_path, embeddings_file)
    print(f"Index saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run recall evaluation on a LEANN index.")
    parser.add_argument(
        "index_path",
        type=str,
        nargs="?",
        help="Path to the LEANN index to evaluate or build (optional).",
    )
    parser.add_argument(
        "--mode",
        choices=["evaluate", "build"],
        default="evaluate",
        help="Mode: 'evaluate' existing index or 'build' from embeddings",
    )
    parser.add_argument(
        "--embeddings-file",
        type=str,
        help="Path to embeddings pickle file (optional for build mode)",
    )
    parser.add_argument(
        "--backend",
        choices=["hnsw", "diskann"],
        default="hnsw",
        help="Backend to use for building index (default: hnsw)",
    )
    parser.add_argument(
        "--num-queries", type=int, default=10, help="Number of queries to evaluate."
    )
    parser.add_argument("--top-k", type=int, default=3, help="The 'k' value for recall@k.")
    parser.add_argument(
        "--ef-search", type=int, default=120, help="The 'efSearch' parameter for HNSW."
    )
    args = parser.parse_args()

    # --- Path Configuration ---
    # Assumes a project structure where the script is in 'benchmarks/'
    # and evaluation data is in 'benchmarks/data/'.
    script_dir = Path(__file__).resolve().parent
    data_root = script_dir / "data"

    # Download data based on mode
    if args.mode == "build":
        # For building mode, we need embeddings
        download_data_if_needed(data_root, download_embeddings=False)  # Basic data first

        # Auto-detect dataset type and download embeddings
        if args.embeddings_file:
            embeddings_file = args.embeddings_file
            # Try to detect dataset type from embeddings file path
            if "rpj_wiki" in str(embeddings_file):
                dataset_type = "rpj_wiki"
            elif "dpr" in str(embeddings_file):
                dataset_type = "dpr"
            else:
                dataset_type = "dpr"  # Default
        else:
            # Auto-detect from index path if provided, otherwise default to DPR
            if args.index_path:
                index_path_str = str(args.index_path)
                if "rpj_wiki" in index_path_str:
                    dataset_type = "rpj_wiki"
                elif "dpr" in index_path_str:
                    dataset_type = "dpr"
                else:
                    dataset_type = "dpr"  # Default to DPR
            else:
                dataset_type = "dpr"  # Default to DPR

            embeddings_file = download_embeddings_if_needed(data_root, dataset_type)

        # Auto-generate index path if not provided
        if not args.index_path:
            indices_dir = data_root / "indices" / dataset_type
            indices_dir.mkdir(parents=True, exist_ok=True)
            args.index_path = str(indices_dir / f"{dataset_type}_from_embeddings")
            print(f"Auto-generated index path: {args.index_path}")

        print(f"Building index from embeddings: {embeddings_file}")
        built_index_path = build_index_from_embeddings(
            embeddings_file, args.index_path, args.backend
        )
        print(f"Index built successfully: {built_index_path}")

        # Ask if user wants to run evaluation
        eval_response = input("Run evaluation on the built index? (y/n): ").strip().lower()
        if eval_response != "y":
            print("Index building complete. Exiting.")
            return
    else:
        # For evaluation mode, don't need embeddings
        download_data_if_needed(data_root, download_embeddings=False)

        # Auto-detect index path if not provided
        if not args.index_path:
            # Default to using downloaded indices
            indices_dir = data_root / "indices"

            # Try common datasets in order of preference
            for dataset in ["dpr", "rpj_wiki"]:
                dataset_dir = indices_dir / dataset
                if dataset_dir.exists():
                    # Look for index files
                    index_files = list(dataset_dir.glob("*.index")) + list(
                        dataset_dir.glob("*_disk.index")
                    )
                    if index_files:
                        args.index_path = str(
                            index_files[0].with_suffix("")
                        )  # Remove .index extension
                        print(f"Using index: {args.index_path}")
                        break

            if not args.index_path:
                print("No indices found. The data download should have included pre-built indices.")
                print(
                    "Please check the benchmarks/data/indices/ directory or provide --index-path manually."
                )
                sys.exit(1)

    # Detect dataset type from index path to select the correct ground truth
    index_path_str = str(args.index_path)
    if "rpj_wiki" in index_path_str:
        dataset_type = "rpj_wiki"
    elif "dpr" in index_path_str:
        dataset_type = "dpr"
    else:
        # Fallback: try to infer from the index directory name
        dataset_type = Path(args.index_path).name
        print(f"WARNING: Could not detect dataset type from path, inferred '{dataset_type}'.")

    queries_file = data_root / "queries" / "nq_open.jsonl"
    golden_results_file = data_root / "ground_truth" / dataset_type / "flat_results_nq_k3.json"

    print(f"INFO: Detected dataset type: {dataset_type}")
    print(f"INFO: Using queries file: {queries_file}")
    print(f"INFO: Using ground truth file: {golden_results_file}")

    try:
        searcher = LeannSearcher(args.index_path)
        queries = load_queries(queries_file)

        with open(golden_results_file) as f:
            golden_results_data = json.load(f)

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
