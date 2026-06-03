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
from typing import Any, Protocol, cast

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leann.api import LeannBuilder, LeannChat, LeannSearcher

from benchmarks.artifact_formatting import command_markdown_lines
from benchmarks.metrics import mean, timing_stats
from benchmarks.provenance import benchmark_command, environment_metadata, file_sha256


class PassageManagerLike(Protocol):
    def get_passage(self, passage_id: str) -> dict[str, Any]: ...


class RetrievalSearcherLike(Protocol):
    backend_name: str
    embedding_model: str
    embedding_mode: str
    passage_id_scheme: str
    passage_manager: PassageManagerLike

    def search(
        self,
        query: str,
        *,
        top_k: int,
        complexity: int,
        batch_size: int,
    ) -> list[Any]: ...


def download_data_if_needed(data_root: Path, download_embeddings: bool = False):
    """Checks if the data directory exists, and if not, downloads it from HF Hub."""
    if not _evaluation_data_ready(data_root, download_embeddings=download_embeddings):
        if data_root.exists():
            print(f"Data directory '{data_root}' is incomplete.")
        else:
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
            print("uv sync --only-group dev")
            sys.exit(1)
        except Exception as e:
            print(f"An error occurred during data download: {e}")
            sys.exit(1)


def _evaluation_data_ready(data_root: Path, *, download_embeddings: bool = False) -> bool:
    required_paths = [
        data_root / "queries" / "nq_open.jsonl",
        data_root / "ground_truth",
    ]
    if download_embeddings:
        required_paths.append(data_root / "embeddings")
    return all(path.exists() for path in required_paths) and _indices_data_ready(
        data_root / "indices"
    )


def _indices_data_ready(indices_root: Path) -> bool:
    if not indices_root.is_dir():
        return False
    return any(path.is_file() for path in indices_root.rglob("*.index"))


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
def get_golden_texts(
    searcher: RetrievalSearcherLike,
    golden_ids: list[int | str],
) -> tuple[set[str], int]:
    """
    Retrieves the text for golden passage IDs directly from the LeannSearcher's
    passage manager.
    """
    golden_texts = set()
    missing_count = 0
    for gid in golden_ids:
        try:
            # PassageManager uses string IDs
            passage_data = searcher.passage_manager.get_passage(str(gid))
            golden_texts.add(passage_data["text"])
        except KeyError:
            missing_count += 1
            print(f"Warning: Golden passage ID '{gid}' not found in the index's passage data.")
    return golden_texts, missing_count


def load_queries(file_path: Path) -> list[str]:
    queries = []
    with open(file_path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            data = json.loads(line)
            query = data.get("query")
            if not isinstance(query, str):
                raise ValueError(f"missing string query at {file_path}:{line_number}")
            queries.append(query)
    return queries


def run_retrieval_evaluation(
    searcher: RetrievalSearcherLike,
    queries: list[str],
    golden_results_data: dict[str, Any],
    *,
    index_path: str,
    dataset_type: str,
    queries_file: str | Path,
    ground_truth_file: str | Path,
    num_queries: int,
    top_k: int,
    complexity: int,
    batch_size: int,
    run_llm: bool = False,
    llm_type: str = "ollama",
    llm_model: str = "qwen3:1.7b",
    data_source: str = "LEANN-RAG/leann-rag-evaluation-data",
    data_revision: str | None = None,
    queries_sha256: str | None = None,
    ground_truth_sha256: str | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    """Run retrieval-only recall evaluation and return a reviewable summary."""
    if num_queries <= 0:
        raise ValueError("num_queries must be greater than 0")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if complexity <= 0:
        raise ValueError("complexity must be greater than 0")
    if batch_size < 0:
        raise ValueError("batch_size must be greater than or equal to 0")

    eval_queries = queries[: min(num_queries, len(queries))]
    per_query: list[dict[str, Any]] = []
    recall_scores: list[float] = []
    hits = 0
    missing_ground_truth_queries = 0
    missing_golden_passages = 0
    queries_with_missing_result_ids = 0
    missing_result_id_count = 0
    queries_with_duplicate_result_texts = 0
    queries_with_duplicate_golden_texts = 0

    chat = None
    if run_llm:
        llm_config = {"type": llm_type, "model": llm_model}
        chat = LeannChat(
            index_path,
            llm_config=llm_config,
            searcher=cast(LeannSearcher, searcher),
        )

    for query_index, query in enumerate(eval_queries):
        started = time.perf_counter()
        new_results = searcher.search(
            query,
            top_k=top_k,
            complexity=complexity,
            batch_size=batch_size,
        )
        latency_ms = (time.perf_counter() - started) * 1000

        answer = None
        if chat is not None:
            answer = chat.ask(
                query,
                top_k=top_k,
                complexity=complexity,
                batch_size=batch_size,
            )

        result_ids = _result_ids(new_results)
        missing_result_ids = len(new_results) - len(result_ids)
        if missing_result_ids:
            queries_with_missing_result_ids += 1
            missing_result_id_count += missing_result_ids
        new_texts = {result.text for result in new_results}
        result_duplicate_text_count = len(new_results) - len(new_texts)
        if result_duplicate_text_count:
            queries_with_duplicate_result_texts += 1
        golden_ids = _golden_ids_for_query(golden_results_data, query_index, top_k)
        if not golden_ids:
            per_query.append(
                {
                    "query_index": query_index,
                    "result_ids": result_ids,
                    "result_count": len(new_results),
                    "missing_result_id_count": missing_result_ids,
                    "result_duplicate_text_count": result_duplicate_text_count,
                    "golden_ids": [],
                    "golden_count": 0,
                    "golden_duplicate_text_count": 0,
                    "overlap": 0,
                    "recall": None,
                    "latency_ms": latency_ms,
                    "missing_ground_truth": True,
                }
            )
            missing_ground_truth_queries += 1
            continue
        golden_texts, missing_count = get_golden_texts(searcher, golden_ids)
        missing_golden_passages += missing_count
        golden_duplicate_text_count = max(0, len(golden_ids) - missing_count - len(golden_texts))
        if golden_duplicate_text_count:
            queries_with_duplicate_golden_texts += 1

        overlap = len(new_texts & golden_texts)
        recall = overlap / len(golden_texts) if golden_texts else 0.0
        recall_scores.append(recall)
        if overlap:
            hits += 1

        row: dict[str, Any] = {
            "query_index": query_index,
            "result_ids": result_ids,
            "result_count": len(new_results),
            "missing_result_id_count": missing_result_ids,
            "result_duplicate_text_count": result_duplicate_text_count,
            "golden_ids": [str(item) for item in golden_ids],
            "golden_count": len(golden_texts),
            "golden_duplicate_text_count": golden_duplicate_text_count,
            "overlap": overlap,
            "recall": recall,
            "latency_ms": latency_ms,
            "missing_ground_truth": False,
        }
        if answer is not None:
            row["answer"] = answer
        per_query.append(row)

    latency_values = [row["latency_ms"] for row in per_query]
    summary = {
        "schema_version": 2,
        "benchmark": "retrieval_evaluation",
        "mode": "retrieval_with_llm" if run_llm else "retrieval_only",
        "dataset": dataset_type,
        "dataset_type": dataset_type,
        "data_source": data_source,
        "data_revision": data_revision,
        "command": command,
        "index_path": str(index_path),
        "queries_file": str(queries_file),
        "queries_sha256": queries_sha256,
        "ground_truth_file": str(ground_truth_file),
        "ground_truth_sha256": ground_truth_sha256,
        "query_count": len(eval_queries),
        "requested_query_count": num_queries,
        "top_k": top_k,
        "complexity": complexity,
        "batch_size": batch_size,
        "run_llm": run_llm,
        "llm_used": run_llm,
        "llm_type": llm_type if run_llm else None,
        "llm_model": llm_model if run_llm else None,
        "backend_name": getattr(searcher, "backend_name", "unknown"),
        "embedding_model": getattr(searcher, "embedding_model", "unknown"),
        "embedding_mode": getattr(searcher, "embedding_mode", "unknown"),
        "passage_id_scheme": getattr(searcher, "passage_id_scheme", "unknown"),
        "average_result_count": mean([row["result_count"] for row in per_query]),
        "evaluated_queries": len(recall_scores),
        "missing_ground_truth_queries": missing_ground_truth_queries,
        "recall": {
            "evaluated_queries": len(recall_scores),
            "missing_ground_truth_queries": missing_ground_truth_queries,
            "missing_golden_passages": missing_golden_passages,
            "queries_with_missing_result_ids": queries_with_missing_result_ids,
            "missing_result_id_count": missing_result_id_count,
            "queries_with_duplicate_result_texts": queries_with_duplicate_result_texts,
            "queries_with_duplicate_golden_texts": queries_with_duplicate_golden_texts,
            "recall_at_k": mean(recall_scores),
            "hit_rate_at_k": hits / len(recall_scores) if recall_scores else 0.0,
        },
        "latency_ms": timing_stats(latency_values),
        "storage": _index_storage(index_path),
        "environment": environment_metadata(),
        "per_query": per_query,
    }
    return summary


def format_summary(summary: dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if output_format == "markdown":
        return format_markdown(summary)
    raise ValueError(f"unsupported output format: {output_format}")


def format_markdown(summary: dict[str, Any]) -> str:
    recall = summary["recall"]
    latency = summary["latency_ms"]
    storage = summary["storage"]
    lines = [
        "# LEANN Retrieval Evaluation",
        "",
        f"- Dataset: {summary['dataset_type']}",
        f"- Data source: {summary.get('data_source') or 'unknown'}",
        f"- Data revision: {summary.get('data_revision') or 'unknown'}",
    ]
    lines.extend(command_markdown_lines(summary.get("command")))
    lines.extend(
        [
            f"- Index path: `{summary['index_path']}`",
            f"- Queries SHA256: `{summary.get('queries_sha256') or 'unavailable'}`",
            f"- Ground truth SHA256: `{summary.get('ground_truth_sha256') or 'unavailable'}`",
            f"- Backend: {summary['backend_name']}",
            f"- Embedding model: {summary['embedding_model']}",
            f"- Queries: {summary['query_count']} of requested {summary['requested_query_count']}",
            f"- top_k: {summary['top_k']}",
            f"- complexity: {summary['complexity']}",
            f"- batch_size: {summary['batch_size']}",
            f"- LLM generation: {summary['llm_used']}",
            f"- Evaluated recall queries: {recall['evaluated_queries']}",
            f"- Missing ground-truth queries: {recall['missing_ground_truth_queries']}",
            f"- Recall@{summary['top_k']}: {recall['recall_at_k']:.4f}",
            f"- Hit rate@{summary['top_k']}: {recall['hit_rate_at_k']:.4f}",
            f"- Missing golden passages: {recall['missing_golden_passages']}",
            f"- Queries with missing result IDs: {recall['queries_with_missing_result_ids']}",
            f"- Missing result ID count: {recall['missing_result_id_count']}",
            f"- Queries with duplicate result text: {recall['queries_with_duplicate_result_texts']}",
            f"- Queries with duplicate golden text: {recall['queries_with_duplicate_golden_texts']}",
            f"- Average result count: {summary['average_result_count']:.3f}",
            "- Latency ms: "
            f"mean={latency['mean']:.3f}, median={latency['median']:.3f}, "
            f"p95={latency['p95']:.3f}, min={latency['min']:.3f}, max={latency['max']:.3f}",
        ]
    )
    if storage["exists"]:
        lines.append(f"- Storage bytes: {storage['bytes']}")
    else:
        lines.append("- Storage bytes: unavailable")
    return "\n".join(lines) + "\n"


def write_summary_artifact(path: str | Path, content: str) -> None:
    output_path = Path(path)
    if output_path.exists() and output_path.is_dir():
        raise IsADirectoryError(f"summary output path is a directory: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


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


def _golden_ids_for_query(
    golden_results_data: dict[str, Any],
    query_index: int,
    top_k: int,
) -> list[int | str]:
    indices = golden_results_data.get("indices")
    if not isinstance(indices, list) or query_index >= len(indices):
        return []
    row = indices[query_index]
    if not isinstance(row, list):
        return []
    return row[:top_k]


def _result_ids(results: list[Any]) -> list[str]:
    ids: list[str] = []
    for result in results:
        result_id = getattr(result, "id", None)
        if result_id is not None:
            ids.append(str(result_id))
    return ids


def _index_storage(index_path: str | Path) -> dict[str, Any]:
    path = Path(index_path)
    parent = path.parent
    logical_name = path.name
    native_stem = path.stem
    files: list[Path] = []
    if parent.exists():
        for candidate in parent.iterdir():
            if not candidate.is_file():
                continue
            if _is_index_artifact(candidate.name, logical_name, native_stem):
                files.append(candidate)
    return {
        "index_path": str(path),
        "exists": bool(files),
        "bytes": sum(file.stat().st_size for file in files),
        "files": [str(file) for file in sorted(files)],
    }


def _is_index_artifact(filename: str, logical_name: str, native_stem: str) -> bool:
    return (
        filename == logical_name
        or filename.startswith(f"{logical_name}.")
        or filename in {f"{native_stem}.index", f"{native_stem}.ids.txt"}
    )


def _validate_output_paths(
    parser: argparse.ArgumentParser,
    *,
    input_paths: list[str | Path],
    output_paths: list[str | None],
) -> None:
    resolved_inputs = {Path(path).resolve() for path in input_paths}
    seen_outputs: set[Path] = set()
    for output_path in output_paths:
        if not output_path:
            continue
        resolved_output = Path(output_path).resolve()
        if resolved_output in resolved_inputs:
            parser.error("summary output path must not overwrite an input file")
        if resolved_output in seen_outputs:
            parser.error("JSON and Markdown output paths must be different")
        seen_outputs.add(resolved_output)


def main(argv: list[str] | None = None) -> None:
    command = benchmark_command(__file__, argv)
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
        "--complexity",
        "--ef-search",
        dest="complexity",
        type=int,
        default=120,
        help="Search complexity parameter forwarded to LeannSearcher.search (default: 120).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Batch size for HNSW batched search (0 disables batching)",
    )
    parser.add_argument(
        "--llm-type",
        type=str,
        choices=["ollama", "hf", "openai", "gemini", "simulated"],
        default="ollama",
        help="LLM backend type to optionally query during evaluation (default: ollama)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="qwen3:1.7b",
        help="LLM model identifier used with --run-llm (default: qwen3:1.7b)",
    )
    parser.add_argument(
        "--run-llm",
        action="store_true",
        help="Also run LeannChat generation after retrieval. Excluded from retrieval latency.",
    )
    parser.add_argument(
        "--evaluate-after-build",
        action="store_true",
        help="In build mode, run evaluation after building. Build mode exits after build by default.",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset label for artifact metadata. Defaults to inferred index-path label.",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        help="JSONL queries file. Defaults to benchmarks/data/queries/nq_open.jsonl.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Ground-truth JSON file. Defaults to benchmarks/data/ground_truth/<dataset>/flat_results_nq_k3.json.",
    )
    parser.add_argument(
        "--data-source",
        default="LEANN-RAG/leann-rag-evaluation-data",
        help="Dataset/source identifier recorded in benchmark artifacts.",
    )
    parser.add_argument(
        "--data-revision",
        help="Dataset revision, commit, snapshot, or download date recorded in benchmark artifacts.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format for stdout in evaluate mode (default: json).",
    )
    parser.add_argument(
        "--json-output",
        help="Write the evaluation summary JSON artifact to this file.",
    )
    parser.add_argument(
        "--markdown-output",
        help="Write the evaluation summary Markdown artifact to this file.",
    )
    args = parser.parse_args(argv)
    if args.num_queries <= 0:
        parser.error("--num-queries must be greater than 0")
    if args.top_k <= 0:
        parser.error("--top-k must be greater than 0")
    if args.complexity <= 0:
        parser.error("--complexity must be greater than 0")
    if args.batch_size < 0:
        parser.error("--batch-size must be greater than or equal to 0")

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

        if not args.evaluate_after_build:
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
    if args.dataset:
        dataset_type = args.dataset
    elif "rpj_wiki" in index_path_str:
        dataset_type = "rpj_wiki"
    elif "dpr" in index_path_str:
        dataset_type = "dpr"
    else:
        # Fallback: try to infer from the index directory name
        dataset_type = Path(args.index_path).name
        print(f"WARNING: Could not detect dataset type from path, inferred '{dataset_type}'.")

    queries_file = args.queries_file or data_root / "queries" / "nq_open.jsonl"
    golden_results_file = (
        args.ground_truth or data_root / "ground_truth" / dataset_type / "flat_results_nq_k3.json"
    )
    _validate_output_paths(
        parser,
        input_paths=[queries_file, golden_results_file],
        output_paths=[args.json_output, args.markdown_output],
    )

    print(f"INFO: Detected dataset type: {dataset_type}")
    print(f"INFO: Using queries file: {queries_file}")
    print(f"INFO: Using ground truth file: {golden_results_file}")

    try:
        searcher = LeannSearcher(args.index_path)
        queries_sha256 = file_sha256(queries_file)
        ground_truth_sha256 = file_sha256(golden_results_file)
        queries = load_queries(queries_file)
        if not queries:
            parser.error("queries file did not contain any queries")

        with open(golden_results_file) as f:
            golden_results_data = json.load(f)

        num_eval_queries = min(args.num_queries, len(queries))
        queries = queries[:num_eval_queries]

        summary = run_retrieval_evaluation(
            searcher,
            queries,
            golden_results_data,
            index_path=args.index_path,
            dataset_type=dataset_type,
            queries_file=queries_file,
            ground_truth_file=golden_results_file,
            num_queries=args.num_queries,
            top_k=args.top_k,
            complexity=args.complexity,
            batch_size=args.batch_size,
            run_llm=args.run_llm,
            llm_type=args.llm_type,
            llm_model=args.llm_model,
            data_source=args.data_source,
            data_revision=args.data_revision,
            queries_sha256=queries_sha256,
            ground_truth_sha256=ground_truth_sha256,
            command=command,
        )
        if args.json_output:
            write_summary_artifact(args.json_output, format_summary(summary, "json"))
        if args.markdown_output:
            write_summary_artifact(args.markdown_output, format_summary(summary, "markdown"))
        print(format_summary(summary, args.format), end="")

    except Exception as e:
        print(f"\n❌ An error occurred during evaluation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
