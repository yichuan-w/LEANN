# /// script
# dependencies = [
#   "leann-backend-diskann"
# ]
# ///

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.metrics import observed_percentile, timing_stats
from benchmarks.provenance import benchmark_command, environment_metadata, file_sha256
from benchmarks.storage import directory_storage

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_queries(path: Path, limit: int | None) -> list[str]:
    out: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out.append(obj["query"])
            if limit is not None and len(out) >= limit:
                break
    return out


def latency_report(times: list[float]) -> dict[str, float | int]:
    stats = timing_stats(times)
    return {
        "queries": len(times),
        "avg_s": stats["mean"],
        "p50_s": stats["median"],
        "p90_s": observed_percentile(times, 90),
        "p95_s": stats["p95"],
        "p99_s": observed_percentile(times, 99),
        "min_s": stats["min"],
        "max_s": stats["max"],
        "total_time_s": sum(times),
        "qps": 1.0 / stats["mean"] if stats["mean"] > 0 else 0.0,
    }


def benchmark_report(
    *,
    latency: dict[str, float | int],
    queries_file: str | Path,
    index_dir: str | Path,
    index_prefix: str,
    top_k: int,
    complexity: int,
    threads: int,
    beam_width: int,
    cache_mechanism: int,
    num_nodes_to_cache: int,
    requested_query_count: int,
    data_source: str | None,
    data_revision: str | None,
    command: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark": "diskann_baseline_latency",
        "data_source": data_source,
        "data_revision": data_revision,
        "command": command,
        "queries_file": str(queries_file),
        "queries_sha256": file_sha256(queries_file),
        "index_dir": str(Path(index_dir).resolve()),
        "index_prefix": index_prefix,
        "index_prefix_path": str(Path(index_dir).resolve() / index_prefix),
        "storage": directory_storage(index_dir),
        "query_count": latency["queries"],
        "requested_query_count": requested_query_count,
        "embedding_model": "facebook/contriever-msmarco",
        "embedding_mode": "sentence-transformers",
        "embedding_in_latency": False,
        "timing_scope": "search_only",
        "settings": {
            "top_k": top_k,
            "complexity": complexity,
            "threads": threads,
            "beam_width": beam_width,
            "cache_mechanism": cache_mechanism,
            "num_nodes_to_cache": num_nodes_to_cache,
            "prune_ratio": 0.0,
            "recompute_embeddings": False,
            "batch_recompute": False,
            "dedup_node_dis": False,
        },
        "latency_s": {
            "mean": latency["avg_s"],
            "median": latency["p50_s"],
            "p90": latency["p90_s"],
            "p95": latency["p95_s"],
            "p99": latency["p99_s"],
            "min": latency["min_s"],
            "max": latency["max_s"],
            "total": latency["total_time_s"],
            "qps": latency["qps"],
        },
        "environment": environment_metadata(),
    }


def write_json_report(path: str | Path, payload: dict[str, Any]) -> None:
    report_path = Path(path)
    if report_path.exists() and report_path.is_dir():
        raise IsADirectoryError(f"report path is a directory: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_report_path(
    parser: argparse.ArgumentParser,
    *,
    report_path: str | None,
    queries_path: str | Path,
    index_dir: str | Path,
) -> None:
    if not report_path:
        return
    resolved_report = Path(report_path).resolve()
    if resolved_report == Path(queries_path).resolve():
        parser.error("report path must not overwrite the queries file")
    resolved_index_dir = Path(index_dir).resolve()
    if resolved_report == resolved_index_dir or resolved_index_dir in resolved_report.parents:
        parser.error("report path must not be inside the index directory")


def main(argv: list[str] | None = None) -> None:
    command = benchmark_command(__file__, argv)
    ap = argparse.ArgumentParser(
        description="DiskANN baseline on real NQ queries (search-only timing)"
    )
    ap.add_argument(
        "--index-dir",
        default=str(REPO_ROOT / "benchmarks/data/indices/diskann_rpj_wiki"),
        help="Directory containing DiskANN files",
    )
    ap.add_argument("--index-prefix", default="ann")
    ap.add_argument(
        "--queries-file",
        default=str(REPO_ROOT / "benchmarks/data/queries/nq_open.jsonl"),
    )
    ap.add_argument("--num-queries", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--complexity", type=int, default=62)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--beam-width", type=int, default=1)
    ap.add_argument("--cache-mechanism", type=int, default=2)
    ap.add_argument("--num-nodes-to-cache", type=int, default=0)
    ap.add_argument(
        "--data-source",
        help="Dataset/source identifier recorded in benchmark report artifacts.",
    )
    ap.add_argument(
        "--data-revision",
        help="Dataset revision, snapshot, or download date recorded in benchmark reports.",
    )
    ap.add_argument("--report", help="Optional JSON report path.")
    args = ap.parse_args(argv)

    if args.num_queries <= 0:
        ap.error("--num-queries must be greater than 0")
    if args.top_k <= 0:
        ap.error("--top-k must be greater than 0")
    if args.complexity <= 0:
        ap.error("--complexity must be greater than 0")
    if args.threads <= 0:
        ap.error("--threads must be greater than 0")
    if args.beam_width <= 0:
        ap.error("--beam-width must be greater than 0")
    if args.num_nodes_to_cache < 0:
        ap.error("--num-nodes-to-cache must be greater than or equal to 0")
    _validate_report_path(
        ap,
        report_path=args.report,
        queries_path=args.queries_file,
        index_dir=args.index_dir,
    )

    index_dir = Path(args.index_dir).resolve()
    if not index_dir.is_dir():
        raise SystemExit(f"Index dir not found: {index_dir}")

    qpath = Path(args.queries_file).resolve()
    if not qpath.exists():
        raise SystemExit(f"Queries file not found: {qpath}")

    queries = load_queries(qpath, args.num_queries)
    print(f"Loaded {len(queries)} queries from {qpath}")

    # Compute embeddings once (exclude from timing)
    from leann.api import compute_embeddings as _compute

    embs = _compute(
        queries,
        model_name="facebook/contriever-msmarco",
        mode="sentence-transformers",
        use_server=False,
    ).astype(np.float32)
    if embs.ndim != 2:
        raise SystemExit("Embedding compute failed or returned wrong shape")

    # Build searcher
    from leann_backend_diskann.diskann_backend import DiskannSearcher as _DiskannSearcher

    index_prefix_path = str(index_dir / args.index_prefix)
    searcher = _DiskannSearcher(
        index_prefix_path,
        num_threads=int(args.threads),
        cache_mechanism=int(args.cache_mechanism),
        num_nodes_to_cache=int(args.num_nodes_to_cache),
    )

    # Warmup (not timed)
    _ = searcher.search(
        embs[0:1],
        top_k=args.top_k,
        complexity=args.complexity,
        beam_width=args.beam_width,
        prune_ratio=0.0,
        recompute_embeddings=False,
        batch_recompute=False,
        dedup_node_dis=False,
    )

    # Timed loop
    times: list[float] = []
    for i in range(embs.shape[0]):
        t0 = time.time()
        _ = searcher.search(
            embs[i : i + 1],
            top_k=args.top_k,
            complexity=args.complexity,
            beam_width=args.beam_width,
            prune_ratio=0.0,
            recompute_embeddings=False,
            batch_recompute=False,
            dedup_node_dis=False,
        )
        times.append(time.time() - t0)

    latency = latency_report(times)

    print("\nDiskANN (NQ, search-only) Report")
    print(f"  queries: {latency['queries']}")
    print(
        f"  k: {args.top_k}, complexity: {args.complexity}, beam_width: {args.beam_width}, threads: {args.threads}"
    )
    print(f"  avg per query: {latency['avg_s']:.6f} s")
    print(f"  p50/p95: {latency['p50_s']:.6f}/{latency['p95_s']:.6f} s")
    print(f"  QPS: {latency['qps']:.2f}")

    if args.report:
        payload = benchmark_report(
            latency=latency,
            queries_file=qpath,
            index_dir=index_dir,
            index_prefix=args.index_prefix,
            top_k=args.top_k,
            complexity=args.complexity,
            threads=args.threads,
            beam_width=args.beam_width,
            cache_mechanism=args.cache_mechanism,
            num_nodes_to_cache=args.num_nodes_to_cache,
            requested_query_count=args.num_queries,
            data_source=args.data_source,
            data_revision=args.data_revision,
            command=command,
        )
        write_json_report(args.report, payload)
        print(f"Saved report to {args.report}")


if __name__ == "__main__":
    main()
