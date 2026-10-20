#!/usr/bin/env python3
"""
Modular script to reproduce NDCG results for ViDoRe v1 benchmark.

This script uses the interface from leann_multi_vector.py to:
1. Download ViDoRe v1 datasets
2. Build indexes (LEANN or Fast-Plaid)
3. Perform retrieval
4. Evaluate using NDCG metrics

Usage:
    # Evaluate all ViDoRe v1 tasks
    python vidore_v1_benchmark.py --model colqwen2 --tasks all

    # Evaluate specific task
    python vidore_v1_benchmark.py --model colqwen2 --task VidoreArxivQARetrieval

    # Use Fast-Plaid index
    python vidore_v1_benchmark.py --model colqwen2 --use-fast-plaid --fast-plaid-index-path ./indexes/vidore_fastplaid

    # Rebuild index
    python vidore_v1_benchmark.py --model colqwen2 --rebuild-index
"""

import argparse
import json
import os
from typing import Any, Optional, cast

from datasets import Dataset, load_dataset
from leann_multi_vector import (
    ViDoReBenchmarkEvaluator,
    _ensure_repo_paths_importable,
)

_ensure_repo_paths_importable(__file__)

# ViDoRe v1 task configurations
# Prompts match MTEB task metadata prompts
VIDORE_V1_TASKS = {
    "VidoreArxivQARetrieval": {
        "dataset_path": "vidore/arxivqa_test_subsampled_beir",
        "revision": "7d94d570960eac2408d3baa7a33f9de4822ae3e4",
        "prompt": {"query": "Find a screenshot that relevant to the user's question."},
    },
    "VidoreDocVQARetrieval": {
        "dataset_path": "vidore/docvqa_test_subsampled_beir",
        "revision": "162ba2fc1a8437eda8b6c37b240bc1c0f0deb092",
        "prompt": {"query": "Find a screenshot that relevant to the user's question."},
    },
    "VidoreInfoVQARetrieval": {
        "dataset_path": "vidore/infovqa_test_subsampled_beir",
        "revision": "b802cc5fd6c605df2d673a963667d74881d2c9a4",
        "prompt": {"query": "Find a screenshot that relevant to the user's question."},
    },
    "VidoreTabfquadRetrieval": {
        "dataset_path": "vidore/tabfquad_test_subsampled_beir",
        "revision": "61a2224bcd29b7b261a4892ff4c8bea353527a31",
        "prompt": {"query": "Find a screenshot that relevant to the user's question."},
    },
    "VidoreTatdqaRetrieval": {
        "dataset_path": "vidore/tatdqa_test_beir",
        "revision": "5feb5630fdff4d8d189ffedb2dba56862fdd45c0",
        "prompt": {"query": "Find a screenshot that relevant to the user's question."},
    },
    "VidoreShiftProjectRetrieval": {
        "dataset_path": "vidore/shiftproject_test_beir",
        "revision": "84a382e05c4473fed9cff2bbae95fe2379416117",
        "prompt": {"query": "Find a screenshot that relevant to the user's question."},
    },
    "VidoreSyntheticDocQAAIRetrieval": {
        "dataset_path": "vidore/syntheticDocQA_artificial_intelligence_test_beir",
        "revision": "2d9ebea5a1c6e9ef4a3b902a612f605dca11261c",
        "prompt": {"query": "Find a screenshot that relevant to the user's question."},
    },
    "VidoreSyntheticDocQAEnergyRetrieval": {
        "dataset_path": "vidore/syntheticDocQA_energy_test_beir",
        "revision": "9935aadbad5c8deec30910489db1b2c7133ae7a7",
        "prompt": {"query": "Find a screenshot that relevant to the user's question."},
    },
    "VidoreSyntheticDocQAGovernmentReportsRetrieval": {
        "dataset_path": "vidore/syntheticDocQA_government_reports_test_beir",
        "revision": "b4909afa930f81282fd20601e860668073ad02aa",
        "prompt": {"query": "Find a screenshot that relevant to the user's question."},
    },
    "VidoreSyntheticDocQAHealthcareIndustryRetrieval": {
        "dataset_path": "vidore/syntheticDocQA_healthcare_industry_test_beir",
        "revision": "f9e25d5b6e13e1ad9f5c3cce202565031b3ab164",
        "prompt": {"query": "Find a screenshot that relevant to the user's question."},
    },
}

# Task name aliases (short names -> full names)
TASK_ALIASES = {
    "arxivqa": "VidoreArxivQARetrieval",
    "docvqa": "VidoreDocVQARetrieval",
    "infovqa": "VidoreInfoVQARetrieval",
    "tabfquad": "VidoreTabfquadRetrieval",
    "tatdqa": "VidoreTatdqaRetrieval",
    "shiftproject": "VidoreShiftProjectRetrieval",
    "syntheticdocqa_ai": "VidoreSyntheticDocQAAIRetrieval",
    "syntheticdocqa_energy": "VidoreSyntheticDocQAEnergyRetrieval",
    "syntheticdocqa_government": "VidoreSyntheticDocQAGovernmentReportsRetrieval",
    "syntheticdocqa_healthcare": "VidoreSyntheticDocQAHealthcareIndustryRetrieval",
}


def normalize_task_name(task_name: str) -> str:
    """Normalize task name (handle aliases)."""
    task_name_lower = task_name.lower()
    if task_name in VIDORE_V1_TASKS:
        return task_name
    if task_name_lower in TASK_ALIASES:
        return TASK_ALIASES[task_name_lower]
    # Try partial match
    for alias, full_name in TASK_ALIASES.items():
        if alias in task_name_lower or task_name_lower in alias:
            return full_name
    return task_name


def get_safe_model_name(model_name: str) -> str:
    """Get a safe model name for use in file paths."""
    import hashlib
    import os

    # If it's a path, use basename or hash
    if os.path.exists(model_name) and os.path.isdir(model_name):
        # Use basename if it's reasonable, otherwise use hash
        basename = os.path.basename(model_name.rstrip("/"))
        if basename and len(basename) < 100 and not basename.startswith("."):
            return basename
        # Use hash for very long or problematic paths
        return hashlib.md5(model_name.encode()).hexdigest()[:16]
    # For HuggingFace model names, replace / with _
    return model_name.replace("/", "_").replace(":", "_")


def load_vidore_v1_data(
    dataset_path: str,
    revision: Optional[str] = None,
    split: str = "test",
):
    """
    Load ViDoRe v1 dataset.

    Returns:
        corpus: dict mapping corpus_id to PIL Image
        queries: dict mapping query_id to query text
        qrels: dict mapping query_id to dict of {corpus_id: relevance_score}
    """
    print(f"Loading dataset: {dataset_path} (split={split})")

    # Load queries - cast to Dataset since we know split returns Dataset not DatasetDict
    query_ds = cast(Dataset, load_dataset(dataset_path, "queries", split=split, revision=revision))

    queries: dict[str, str] = {}
    for row in query_ds:
        row_dict = cast(dict[str, Any], row)
        query_id = f"query-{split}-{row_dict['query-id']}"
        queries[query_id] = row_dict["query"]

    # Load corpus (images) - cast to Dataset
    corpus_ds = cast(Dataset, load_dataset(dataset_path, "corpus", split=split, revision=revision))

    corpus: dict[str, Any] = {}
    for row in corpus_ds:
        row_dict = cast(dict[str, Any], row)
        corpus_id = f"corpus-{split}-{row_dict['corpus-id']}"
        # Extract image from the dataset row
        if "image" in row_dict:
            corpus[corpus_id] = row_dict["image"]
        elif "page_image" in row_dict:
            corpus[corpus_id] = row_dict["page_image"]
        else:
            raise ValueError(
                f"No image field found in corpus. Available fields: {list(row_dict.keys())}"
            )

    # Load qrels (relevance judgments) - cast to Dataset
    qrels_ds = cast(Dataset, load_dataset(dataset_path, "qrels", split=split, revision=revision))

    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        row_dict = cast(dict[str, Any], row)
        query_id = f"query-{split}-{row_dict['query-id']}"
        corpus_id = f"corpus-{split}-{row_dict['corpus-id']}"
        if query_id not in qrels:
            qrels[query_id] = {}
        qrels[query_id][corpus_id] = int(row_dict["score"])

    print(
        f"Loaded {len(queries)} queries, {len(corpus)} corpus items, {len(qrels)} query-relevance mappings"
    )

    # Filter qrels to only include queries that exist
    qrels = {qid: rel_docs for qid, rel_docs in qrels.items() if qid in queries}

    # Filter out queries without any relevant documents (matching MTEB behavior)
    # This is important for correct NDCG calculation
    qrels_filtered = {qid: rel_docs for qid, rel_docs in qrels.items() if len(rel_docs) > 0}
    queries_filtered = {
        qid: query_text for qid, query_text in queries.items() if qid in qrels_filtered
    }

    print(
        f"After filtering queries without positives: {len(queries_filtered)} queries, {len(qrels_filtered)} query-relevance mappings"
    )

    return corpus, queries_filtered, qrels_filtered


def evaluate_task(
    task_name: str,
    model_name: str,
    index_path: str,
    use_fast_plaid: bool = False,
    fast_plaid_index_path: Optional[str] = None,
    rebuild_index: bool = False,
    top_k: int = 1000,
    first_stage_k: int = 500,
    k_values: Optional[list[int]] = None,
    output_dir: Optional[str] = None,
):
    """
    Evaluate a single ViDoRe v1 task.
    """
    print(f"\n{'=' * 80}")
    print(f"Evaluating task: {task_name}")
    print(f"{'=' * 80}")

    # Normalize task name (handle aliases)
    task_name = normalize_task_name(task_name)

    # Get task config
    if task_name not in VIDORE_V1_TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(VIDORE_V1_TASKS.keys())}")

    task_config = VIDORE_V1_TASKS[task_name]
    dataset_path = str(task_config["dataset_path"])
    revision = str(task_config["revision"])

    # Load data
    corpus, queries, qrels = load_vidore_v1_data(
        dataset_path=dataset_path,
        revision=revision,
        split="test",
    )

    # Initialize k_values if not provided
    if k_values is None:
        k_values = [1, 3, 5, 10, 20, 100, 1000]

    # Check if we have any queries
    if len(queries) == 0:
        print(f"\nWarning: No queries found for task {task_name}. Skipping evaluation.")
        # Return zero scores
        scores = {}
        for k in k_values:
            scores[f"ndcg_at_{k}"] = 0.0
            scores[f"map_at_{k}"] = 0.0
            scores[f"recall_at_{k}"] = 0.0
            scores[f"precision_at_{k}"] = 0.0
            scores[f"mrr_at_{k}"] = 0.0
        return scores

    # Initialize evaluator
    evaluator = ViDoReBenchmarkEvaluator(
        model_name=model_name,
        use_fast_plaid=use_fast_plaid,
        top_k=top_k,
        first_stage_k=first_stage_k,
        k_values=k_values,
    )

    # Build or load index
    # Use safe model name for index path (different models need different indexes)
    safe_model_name = get_safe_model_name(model_name)
    index_path_full = index_path if not use_fast_plaid else fast_plaid_index_path
    if index_path_full is None:
        index_path_full = f"./indexes/{task_name}_{safe_model_name}"
        if use_fast_plaid:
            index_path_full = f"./indexes/{task_name}_{safe_model_name}_fastplaid"

    index_or_retriever, corpus_ids_ordered = evaluator.build_index_from_corpus(
        corpus=corpus,
        index_path=index_path_full,
        rebuild=rebuild_index,
    )

    # Search queries
    task_prompt = cast(Optional[dict[str, str]], task_config.get("prompt"))
    results = evaluator.search_queries(
        queries=queries,
        corpus_ids=corpus_ids_ordered,
        index_or_retriever=index_or_retriever,
        fast_plaid_index_path=fast_plaid_index_path,
        task_prompt=task_prompt,
    )

    # Evaluate
    scores = evaluator.evaluate_results(results, qrels, k_values=k_values)

    # Print results
    print(f"\n{'=' * 80}")
    print(f"Results for {task_name}:")
    print(f"{'=' * 80}")
    for metric, value in scores.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.5f}")

    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, f"{task_name}_results.json")
        scores_file = os.path.join(output_dir, f"{task_name}_scores.json")

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to: {results_file}")

        with open(scores_file, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"Saved scores to: {scores_file}")

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ViDoRe v1 benchmark using LEANN/Fast-Plaid indexing"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="colqwen2",
        help="Model to use: 'colqwen2', 'colpali', or path to a model directory (supports LoRA adapters)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Specific task to evaluate (or 'all' for all tasks)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Tasks to evaluate: 'all' or comma-separated list",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default=None,
        help="Path to LEANN index (auto-generated if not provided)",
    )
    parser.add_argument(
        "--use-fast-plaid",
        action="store_true",
        help="Use Fast-Plaid instead of LEANN",
    )
    parser.add_argument(
        "--fast-plaid-index-path",
        type=str,
        default=None,
        help="Path to Fast-Plaid index (auto-generated if not provided)",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild index even if it exists",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="Top-k results to retrieve (MTEB default is max(k_values)=1000)",
    )
    parser.add_argument(
        "--first-stage-k",
        type=int,
        default=500,
        help="First stage k for LEANN search",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="1,3,5,10,20,100,1000",
        help="Comma-separated k values for evaluation (e.g., '1,3,5,10,100')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./vidore_v1_results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Parse k_values
    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    # Determine tasks to evaluate
    if args.task:
        tasks_to_eval = [normalize_task_name(args.task)]
    elif args.tasks.lower() == "all":
        tasks_to_eval = list(VIDORE_V1_TASKS.keys())
    else:
        tasks_to_eval = [normalize_task_name(t.strip()) for t in args.tasks.split(",")]

    print(f"Tasks to evaluate: {tasks_to_eval}")

    # Evaluate each task
    all_scores = {}
    for task_name in tasks_to_eval:
        try:
            scores = evaluate_task(
                task_name=task_name,
                model_name=args.model,
                index_path=args.index_path,
                use_fast_plaid=args.use_fast_plaid,
                fast_plaid_index_path=args.fast_plaid_index_path,
                rebuild_index=args.rebuild_index,
                top_k=args.top_k,
                first_stage_k=args.first_stage_k,
                k_values=k_values,
                output_dir=args.output_dir,
            )
            all_scores[task_name] = scores
        except Exception as e:
            print(f"\nError evaluating {task_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Print summary
    if all_scores:
        print(f"\n{'=' * 80}")
        print("SUMMARY")
        print(f"{'=' * 80}")
        for task_name, scores in all_scores.items():
            print(f"\n{task_name}:")
            # Print main metrics
            for metric in ["ndcg_at_5", "ndcg_at_10", "ndcg_at_100", "map_at_10", "recall_at_10"]:
                if metric in scores:
                    print(f"  {metric}: {scores[metric]:.5f}")


if __name__ == "__main__":
    main()
