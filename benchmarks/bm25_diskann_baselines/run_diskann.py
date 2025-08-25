# /// script
# dependencies = [
#   "leann-backend-diskann"
# ]
# ///

import argparse
import json
import time
from pathlib import Path

import numpy as np


def load_queries(path: Path, limit: int | None) -> list[str]:
    out: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out.append(obj["query"])
            if limit and len(out) >= limit:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="DiskANN baseline on real NQ queries (search-only timing)"
    )
    ap.add_argument(
        "--index-dir",
        default="benchmarks/data/indices/diskann_rpj_wiki",
        help="Directory containing DiskANN files",
    )
    ap.add_argument("--index-prefix", default="ann")
    ap.add_argument("--queries-file", default="benchmarks/data/queries/nq_open.jsonl")
    ap.add_argument("--num-queries", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--complexity", type=int, default=62)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--beam-width", type=int, default=1)
    ap.add_argument("--cache-mechanism", type=int, default=2)
    ap.add_argument("--num-nodes-to-cache", type=int, default=0)
    args = ap.parse_args()

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

    times_sorted = sorted(times)
    avg = float(sum(times) / len(times))
    p50 = times_sorted[len(times) // 2]
    p95 = times_sorted[max(0, int(len(times) * 0.95) - 1)]

    print("\nDiskANN (NQ, search-only) Report")
    print(f"  queries: {len(times)}")
    print(
        f"  k: {args.top_k}, complexity: {args.complexity}, beam_width: {args.beam_width}, threads: {args.threads}"
    )
    print(f"  avg per query: {avg:.6f} s")
    print(f"  p50/p95: {p50:.6f}/{p95:.6f} s")
    print(f"  QPS: {1.0 / avg:.2f}")


if __name__ == "__main__":
    main()
