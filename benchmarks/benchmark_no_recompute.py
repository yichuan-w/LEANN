import argparse
import os
import time
from pathlib import Path

from leann import LeannBuilder, LeannSearcher


def _meta_exists(index_path: str) -> bool:
    p = Path(index_path)
    return (p.parent / f"{p.stem}.meta.json").exists()


def ensure_index(index_path: str, backend_name: str, num_docs: int, is_recompute: bool) -> None:
    # if _meta_exists(index_path):
    #     return
    kwargs = {}
    if backend_name == "hnsw":
        kwargs["is_compact"] = is_recompute
    builder = LeannBuilder(
        backend_name=backend_name,
        embedding_model=os.getenv("LEANN_EMBED_MODEL", "facebook/contriever"),
        embedding_mode=os.getenv("LEANN_EMBED_MODE", "sentence-transformers"),
        graph_degree=32,
        complexity=64,
        is_recompute=is_recompute,
        num_threads=4,
        **kwargs,
    )
    for i in range(num_docs):
        builder.add_text(
            f"This is a test document number {i}. It contains some repeated text for benchmarking."
        )
    builder.build_index(index_path)


def _bench_group(
    index_path: str,
    recompute: bool,
    query: str,
    repeats: int,
    complexity: int = 32,
    top_k: int = 10,
) -> float:
    # Independent searcher per group; fixed port when recompute
    searcher = LeannSearcher(index_path=index_path)

    # Warm-up once
    _ = searcher.search(
        query,
        top_k=top_k,
        complexity=complexity,
        recompute_embeddings=recompute,
    )

    def _once() -> float:
        t0 = time.time()
        _ = searcher.search(
            query,
            top_k=top_k,
            complexity=complexity,
            recompute_embeddings=recompute,
        )
        return time.time() - t0

    if repeats <= 1:
        t = _once()
    else:
        vals = [_once() for _ in range(repeats)]
        vals.sort()
        t = vals[len(vals) // 2]

    searcher.cleanup()
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-docs", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--complexity", type=int, default=32)
    args = parser.parse_args()

    base = Path.cwd() / ".leann" / "indexes" / f"bench_n{args.num_docs}"
    base.parent.mkdir(parents=True, exist_ok=True)
    # ---------- Build HNSW variants ----------
    hnsw_r = str(base / f"hnsw_recompute_n{args.num_docs}.leann")
    hnsw_nr = str(base / f"hnsw_norecompute_n{args.num_docs}.leann")
    ensure_index(hnsw_r, "hnsw", args.num_docs, True)
    ensure_index(hnsw_nr, "hnsw", args.num_docs, False)

    # ---------- Build DiskANN variants ----------
    diskann_r = str(base / "diskann_r.leann")
    diskann_nr = str(base / "diskann_nr.leann")
    ensure_index(diskann_r, "diskann", args.num_docs, True)
    ensure_index(diskann_nr, "diskann", args.num_docs, False)

    # ---------- Helpers ----------
    def _size_for(prefix: str) -> int:
        p = Path(prefix)
        base_dir = p.parent
        stem = p.stem
        total = 0
        for f in base_dir.iterdir():
            if f.is_file() and f.name.startswith(stem):
                total += f.stat().st_size
        return total

    # ---------- HNSW benchmark ----------
    t_hnsw_r = _bench_group(
        hnsw_r, True, "test document number 42", repeats=args.repeats, complexity=args.complexity
    )
    t_hnsw_nr = _bench_group(
        hnsw_nr, False, "test document number 42", repeats=args.repeats, complexity=args.complexity
    )
    size_hnsw_r = _size_for(hnsw_r)
    size_hnsw_nr = _size_for(hnsw_nr)

    print("Benchmark results (HNSW):")
    print(f"  recompute=True:  search_time={t_hnsw_r:.3f}s, size={size_hnsw_r / 1024 / 1024:.1f}MB")
    print(
        f"  recompute=False: search_time={t_hnsw_nr:.3f}s, size={size_hnsw_nr / 1024 / 1024:.1f}MB"
    )
    print("  Expectation: no-recompute should be faster but larger on disk.")

    # ---------- DiskANN benchmark ----------
    t_diskann_r = _bench_group(
        diskann_r, True, "DiskANN R test doc 123", repeats=args.repeats, complexity=args.complexity
    )
    t_diskann_nr = _bench_group(
        diskann_nr,
        False,
        "DiskANN NR test doc 123",
        repeats=args.repeats,
        complexity=args.complexity,
    )
    size_diskann_r = _size_for(diskann_r)
    size_diskann_nr = _size_for(diskann_nr)

    print("\nBenchmark results (DiskANN):")
    print(f"  build(recompute=True, partition): size={size_diskann_r / 1024 / 1024:.1f}MB")
    print(f"  build(recompute=False):          size={size_diskann_nr / 1024 / 1024:.1f}MB")
    print(f"  search recompute=True (final rerank): {t_diskann_r:.3f}s")
    print(f"  search recompute=False (PQ only):     {t_diskann_nr:.3f}s")


if __name__ == "__main__":
    main()
