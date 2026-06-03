# /// script
# dependencies = [
#   "pyserini"
# ]
# ///
# sudo pacman -S jdk21-openjdk
# export JAVA_HOME=/usr/lib/jvm/java-21-openjdk
# sudo archlinux-java status
# sudo archlinux-java set java-21-openjdk
# set -Ux JAVA_HOME /usr/lib/jvm/java-21-openjdk
# fish_add_path --global $JAVA_HOME/bin
# set -Ux LD_LIBRARY_PATH $JAVA_HOME/lib/server $LD_LIBRARY_PATH
# which javac # Should be /usr/lib/jvm/java-21-openjdk/bin/javac

import argparse
import json
import os
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.metrics import observed_percentile, timing_stats
from benchmarks.provenance import benchmark_command, environment_metadata, file_sha256
from benchmarks.storage import directory_storage

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_queries(path: str, limit: int | None) -> list[str]:
    queries: list[str] = []
    # Try JSONL with a 'query' or 'text' field; fallback to plain text (one query per line)
    _, ext = os.path.splitext(path)
    if ext.lower() in {".jsonl", ".json"}:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Not strict JSONL? treat the whole line as the query
                    queries.append(line)
                    continue
                q = obj.get("query") or obj.get("text") or obj.get("question")
                if q:
                    queries.append(str(q))
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    queries.append(s)

    if limit is not None and limit > 0:
        queries = queries[:limit]
    return queries


def latency_report(
    latencies: list[float],
    *,
    total_searches: int,
    total_time: float,
) -> dict[str, float | int]:
    stats = timing_stats(latencies)
    return {
        "queries": total_searches,
        "avg_s": stats["mean"],
        "p50_s": stats["median"],
        "p90_s": observed_percentile(latencies, 90),
        "p95_s": stats["p95"],
        "p99_s": observed_percentile(latencies, 99),
        "min_s": stats["min"],
        "max_s": stats["max"],
        "total_time_s": total_time,
        "qps": total_searches / total_time if total_time > 0 else 0.0,
    }


def benchmark_report(
    *,
    latency: dict[str, float | int],
    queries_file: str | Path,
    index_dir: str | Path,
    k: int,
    k1: float,
    b: float,
    warmup: int,
    fetch_docs: bool,
    requested_query_count: int,
    data_source: str | None,
    data_revision: str | None,
    command: str | None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "benchmark": "bm25_baseline_latency",
        "data_source": data_source,
        "data_revision": data_revision,
        "command": command,
        "queries_file": str(queries_file),
        "queries_sha256": file_sha256(queries_file),
        "index_dir": str(Path(index_dir).resolve()),
        "storage": directory_storage(index_dir),
        "query_count": latency["queries"],
        "requested_query_count": requested_query_count,
        "timing_scope": "search_with_doc_fetch" if fetch_docs else "search_only",
        "settings": {
            "k": k,
            "k1": k1,
            "b": b,
            "warmup": warmup,
            "fetch_docs": fetch_docs,
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


def write_json_report(path: str | Path, payload: dict[str, object]) -> None:
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


def main(argv: list[str] | None = None):
    command = benchmark_command(__file__, argv)
    ap = argparse.ArgumentParser(description="Standalone BM25 latency benchmark (Pyserini)")
    ap.add_argument(
        "--bm25-index",
        default=str(REPO_ROOT / "benchmarks/data/indices/bm25_index"),
        help="Path to Pyserini Lucene index directory",
    )
    ap.add_argument(
        "--queries",
        default=str(REPO_ROOT / "benchmarks/data/queries/nq_open.jsonl"),
        help="Path to queries file (JSONL with 'query'/'text' or plain txt one-per-line)",
    )
    ap.add_argument("--k", type=int, default=10, help="Top-k to retrieve (default: 10)")
    ap.add_argument("--k1", type=float, default=0.9, help="BM25 k1 (default: 0.9)")
    ap.add_argument("--b", type=float, default=0.4, help="BM25 b (default: 0.4)")
    ap.add_argument("--limit", type=int, default=100, help="Max queries to run (default: 100)")
    ap.add_argument(
        "--warmup", type=int, default=5, help="Warmup queries not counted in latency (default: 5)"
    )
    ap.add_argument(
        "--fetch-docs", action="store_true", help="Also fetch doc contents (slower; default: off)"
    )
    ap.add_argument(
        "--data-source",
        help="Dataset/source identifier recorded in benchmark report artifacts.",
    )
    ap.add_argument(
        "--data-revision",
        help="Dataset revision, snapshot, or download date recorded in benchmark reports.",
    )
    ap.add_argument("--report", type=str, default=None, help="Optional JSON report path")
    args = ap.parse_args(argv)

    if args.k <= 0:
        ap.error("--k must be greater than 0")
    if args.limit <= 0:
        ap.error("--limit must be greater than 0")
    if args.warmup < 0:
        ap.error("--warmup must be greater than or equal to 0")
    _validate_report_path(
        ap,
        report_path=args.report,
        queries_path=args.queries,
        index_dir=args.bm25_index,
    )

    try:
        from pyserini.search.lucene import LuceneSearcher
    except Exception:
        print("Pyserini not found. Install with: pip install pyserini", file=sys.stderr)
        raise

    if not os.path.isdir(args.bm25_index):
        print(f"Index directory not found: {args.bm25_index}", file=sys.stderr)
        sys.exit(1)

    queries = load_queries(args.queries, args.limit)
    if not queries:
        print("No queries loaded.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(queries)} queries from {args.queries}")
    print(f"Opening BM25 index: {args.bm25_index}")
    searcher = LuceneSearcher(args.bm25_index)
    # Some builds of pyserini require explicit set_bm25; others ignore
    try:
        searcher.set_bm25(k1=args.k1, b=args.b)
    except Exception:
        pass

    latencies: list[float] = []
    total_searches = 0

    # Warmup
    for i in range(min(args.warmup, len(queries))):
        _ = searcher.search(queries[i], k=args.k)

    t0 = time.time()
    for i, q in enumerate(queries):
        t1 = time.time()
        hits = searcher.search(q, k=args.k)

        if args.fetch_docs:
            # Optional doc fetch to include I/O time
            for h in hits:
                try:
                    _ = searcher.doc(h.docid)
                except Exception:
                    pass
        t2 = time.time()
        latencies.append(t2 - t1)
        total_searches += 1

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(queries)} queries")

    t1 = time.time()
    total_time = t1 - t0

    latency = latency_report(latencies, total_searches=total_searches, total_time=total_time)

    print("BM25 Latency Report")
    print(f"  queries: {total_searches}")
    print(f"  k: {args.k}, k1: {args.k1}, b: {args.b}")
    print(f"  avg per query: {latency['avg_s']:.6f} s")
    print(
        "  p50/p90/p95/p99: "
        f"{latency['p50_s']:.6f}/{latency['p90_s']:.6f}/"
        f"{latency['p95_s']:.6f}/{latency['p99_s']:.6f} s"
    )
    print(f"  total time: {total_time:.3f} s, qps: {latency['qps']:.2f}")

    if args.report:
        payload = benchmark_report(
            latency=latency,
            queries_file=args.queries,
            index_dir=args.bm25_index,
            k=args.k,
            k1=args.k1,
            b=args.b,
            warmup=args.warmup,
            fetch_docs=bool(args.fetch_docs),
            requested_query_count=args.limit,
            data_source=args.data_source,
            data_revision=args.data_revision,
            command=command,
        )
        write_json_report(args.report, payload)
        print(f"Saved report to {args.report}")


if __name__ == "__main__":
    main()
