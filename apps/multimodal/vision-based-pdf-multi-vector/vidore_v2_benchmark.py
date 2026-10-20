#!/usr/bin/env python3
"""
Modular script to reproduce NDCG results for ViDoRe v2 benchmark.

This script uses the interface from leann_multi_vector.py to:
1. Download ViDoRe v2 datasets
2. Build indexes (LEANN or Fast-Plaid)
3. Perform retrieval
4. Evaluate using NDCG metrics

Usage:
    # Evaluate all ViDoRe v2 tasks
    python vidore_v2_benchmark.py --model colqwen2 --tasks all

    # Evaluate specific task
    python vidore_v2_benchmark.py --model colqwen2 --task Vidore2ESGReportsRetrieval

    # Use Fast-Plaid index
    python vidore_v2_benchmark.py --model colqwen2 --use-fast-plaid --fast-plaid-index-path ./indexes/vidore_fastplaid

    # Rebuild index
    python vidore_v2_benchmark.py --model colqwen2 --rebuild-index
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

# Language name to dataset language field value mapping
# Dataset uses ISO 639-3 + ISO 15924 format (e.g., "eng-Latn")
LANGUAGE_MAPPING = {
    "english": "eng-Latn",
    "french": "fra-Latn",
    "spanish": "spa-Latn",
    "german": "deu-Latn",
}

# ViDoRe v2 task configurations
# Prompts match MTEB task metadata prompts
VIDORE_V2_TASKS = {
    "Vidore2ESGReportsRetrieval": {
        "dataset_path": "vidore/esg_reports_v2",
        "revision": "0542c0d03da0ec1c8cbc517c8d78e7e95c75d3d3",
        "languages": ["french", "spanish", "english", "german"],
        "prompt": {"query": "Find a screenshot that relevant to the user's question."},
    },
    "Vidore2EconomicsReportsRetrieval": {
        "dataset_path": "vidore/economics_reports_v2",
        "revision": "b3e3a04b07fbbaffe79be49dabf92f691fbca252",
        "languages": ["french", "spanish", "english", "german"],
        "prompt": {"query": "Find a screenshot that relevant to the user's question."},
    },
    "Vidore2BioMedicalLecturesRetrieval": {
        "dataset_path": "vidore/biomedical_lectures_v2",
        "revision": "a29202f0da409034d651614d87cd8938d254e2ea",
        "languages": ["french", "spanish", "english", "german"],
        "prompt": {"query": "Find a screenshot that relevant to the user's question."},
    },
    "Vidore2ESGReportsHLRetrieval": {
        "dataset_path": "vidore/esg_reports_human_labeled_v2",
        "revision": "6d467dedb09a75144ede1421747e47cf036857dd",
        # Note: This dataset doesn't have language filtering - all queries are English
        "languages": None,  # No language filtering needed
        "prompt": {"query": "Find a screenshot that relevant to the user's question."},
    },
}


def load_vidore_v2_data(
    dataset_path: str,
    revision: Optional[str] = None,
    split: str = "test",
    language: Optional[str] = None,
):
    """
    Load ViDoRe v2 dataset.

    Returns:
        corpus: dict mapping corpus_id to PIL Image
        queries: dict mapping query_id to query text
        qrels: dict mapping query_id to dict of {corpus_id: relevance_score}
    """
    print(f"Loading dataset: {dataset_path} (split={split}, language={language})")

    # Load queries - cast to Dataset since we know split returns Dataset not DatasetDict
    query_ds = cast(Dataset, load_dataset(dataset_path, "queries", split=split, revision=revision))

    # Check if dataset has language field before filtering
    has_language_field = len(query_ds) > 0 and "language" in query_ds.column_names

    if language and has_language_field:
        # Map language name to dataset language field value (e.g., "english" -> "eng-Latn")
        dataset_language = LANGUAGE_MAPPING.get(language, language)
        query_ds_filtered = query_ds.filter(lambda x: x.get("language") == dataset_language)
        # Check if filtering resulted in empty dataset
        if len(query_ds_filtered) == 0:
            print(
                f"Warning: No queries found after filtering by language '{language}' (mapped to '{dataset_language}')."
            )
            # Try with original language value (dataset might use simple names like 'english')
            print(f"Trying with original language value '{language}'...")
            query_ds_filtered = query_ds.filter(lambda x: x.get("language") == language)
            if len(query_ds_filtered) == 0:
                # Try to get a sample to see actual language values
                try:
                    sample_ds = cast(
                        Dataset,
                        load_dataset(dataset_path, "queries", split=split, revision=revision),
                    )
                    if len(sample_ds) > 0 and "language" in sample_ds.column_names:
                        sample_langs = set(sample_ds["language"])
                        print(f"Available language values in dataset: {sample_langs}")
                except Exception:
                    pass
            else:
                print(
                    f"Found {len(query_ds_filtered)} queries using original language value '{language}'"
                )
        query_ds = query_ds_filtered

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
    language: Optional[str] = None,
    rebuild_index: bool = False,
    top_k: int = 100,
    first_stage_k: int = 500,
    k_values: Optional[list[int]] = None,
    output_dir: Optional[str] = None,
):
    """
    Evaluate a single ViDoRe v2 task.
    """
    print(f"\n{'=' * 80}")
    print(f"Evaluating task: {task_name}")
    print(f"{'=' * 80}")

    # Get task config
    if task_name not in VIDORE_V2_TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(VIDORE_V2_TASKS.keys())}")

    task_config = VIDORE_V2_TASKS[task_name]
    dataset_path = str(task_config["dataset_path"])
    revision = str(task_config["revision"])

    # Determine language
    if language is None:
        # Use first language if multiple available
        languages = cast(Optional[list[str]], task_config.get("languages"))
        if languages is None:
            # Task doesn't support language filtering (e.g., Vidore2ESGReportsHLRetrieval)
            language = None
        elif len(languages) == 1:
            language = languages[0]
        else:
            language = None

    # Initialize k_values if not provided
    if k_values is None:
        k_values = [1, 3, 5, 10, 100]

    # Load data
    corpus, queries, qrels = load_vidore_v2_data(
        dataset_path=dataset_path,
        revision=revision,
        split="test",
        language=language,
    )

    # Check if we have any queries
    if len(queries) == 0:
        print(
            f"\nWarning: No queries found for task {task_name} with language {language}. Skipping evaluation."
        )
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
    index_path_full = index_path if not use_fast_plaid else fast_plaid_index_path
    if index_path_full is None:
        index_path_full = f"./indexes/{task_name}_{model_name}"
        if use_fast_plaid:
            index_path_full = f"./indexes/{task_name}_{model_name}_fastplaid"

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
        description="Evaluate ViDoRe v2 benchmark using LEANN/Fast-Plaid indexing"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="colqwen2",
        choices=["colqwen2", "colpali"],
        help="Model to use",
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
        "--language",
        type=str,
        default=None,
        help="Language to evaluate (default: first available)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Top-k results to retrieve",
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
        default="1,3,5,10,100",
        help="Comma-separated k values for evaluation (e.g., '1,3,5,10,100')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./vidore_v2_results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Parse k_values
    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    # Determine tasks to evaluate
    if args.task:
        tasks_to_eval = [args.task]
    elif args.tasks.lower() == "all":
        tasks_to_eval = list(VIDORE_V2_TASKS.keys())
    else:
        tasks_to_eval = [t.strip() for t in args.tasks.split(",")]

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
                language=args.language,
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
