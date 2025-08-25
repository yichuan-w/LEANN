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
from statistics import mean


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


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser(description="Standalone BM25 latency benchmark (Pyserini)")
    ap.add_argument(
        "--bm25-index",
        default="benchmarks/data/indices/bm25_index",
        help="Path to Pyserini Lucene index directory",
    )
    ap.add_argument(
        "--queries",
        default="benchmarks/data/queries/nq_open.jsonl",
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
    ap.add_argument("--report", type=str, default=None, help="Optional JSON report path")
    args = ap.parse_args()

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
        t2 = time.time()
        latencies.append(t2 - t1)
        total_searches += 1

        if args.fetch_docs:
            # Optional doc fetch to include I/O time
            for h in hits:
                try:
                    _ = searcher.doc(h.docid)
                except Exception:
                    pass

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(queries)} queries")

    t1 = time.time()
    total_time = t1 - t0

    if latencies:
        avg = mean(latencies)
        p50 = percentile(latencies, 50)
        p90 = percentile(latencies, 90)
        p95 = percentile(latencies, 95)
        p99 = percentile(latencies, 99)
        qps = total_searches / total_time if total_time > 0 else 0.0
    else:
        avg = p50 = p90 = p95 = p99 = qps = 0.0

    print("BM25 Latency Report")
    print(f"  queries: {total_searches}")
    print(f"  k: {args.k}, k1: {args.k1}, b: {args.b}")
    print(f"  avg per query: {avg:.6f} s")
    print(f"  p50/p90/p95/p99: {p50:.6f}/{p90:.6f}/{p95:.6f}/{p99:.6f} s")
    print(f"  total time: {total_time:.3f} s, qps: {qps:.2f}")

    if args.report:
        payload = {
            "queries": total_searches,
            "k": args.k,
            "k1": args.k1,
            "b": args.b,
            "avg_s": avg,
            "p50_s": p50,
            "p90_s": p90,
            "p95_s": p95,
            "p99_s": p99,
            "total_time_s": total_time,
            "qps": qps,
            "index_dir": os.path.abspath(args.bm25_index),
            "fetch_docs": bool(args.fetch_docs),
        }
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved report to {args.report}")


if __name__ == "__main__":
    main()
