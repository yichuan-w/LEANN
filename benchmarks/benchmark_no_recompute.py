import os
import time
from pathlib import Path

from leann import LeannBuilder, LeannSearcher


def ensure_index(
    index_path: str, num_docs: int = 5000, is_recompute: bool = True, is_compact: bool = True
):
    path = Path(index_path)
    if (path.parent / f"{path.stem}.meta.json").exists():
        return

    builder = LeannBuilder(
        backend_name="hnsw",
        embedding_model=os.getenv("LEANN_EMBED_MODEL", "facebook/contriever"),
        embedding_mode=os.getenv("LEANN_EMBED_MODE", "sentence-transformers"),
        graph_degree=32,
        complexity=64,
        is_compact=is_compact,
        is_recompute=is_recompute,
        num_threads=4,
    )

    for i in range(num_docs):
        builder.add_text(
            f"This is a test document number {i}. It contains some repeated text for benchmarking."
        )

    builder.build_index(index_path)


def bench_once(index_path: str, recompute: bool, top_k: int = 10) -> float:
    searcher = LeannSearcher(index_path=index_path)
    t0 = time.time()
    _ = searcher.search(
        "test document number 42",
        top_k=top_k,
        complexity=64,
        prune_ratio=0.0,
        recompute_embeddings=recompute,
    )
    return time.time() - t0


def main():
    base = Path.cwd() / ".leann" / "indexes" / "bench"
    base.parent.mkdir(parents=True, exist_ok=True)
    index_path_recompute = str(base / "recompute.leann")
    index_path_norecompute = str(base / "norecompute.leann")

    # Build two variants: pruned (recompute) and non-compact (no-recompute)
    ensure_index(index_path_recompute, is_recompute=True, is_compact=True)
    ensure_index(index_path_norecompute, is_recompute=False, is_compact=False)

    # Warm up
    bench_once(index_path_recompute, recompute=True)
    bench_once(index_path_norecompute, recompute=False)

    t_recompute = bench_once(index_path_recompute, recompute=True)
    t_norecompute = bench_once(index_path_norecompute, recompute=False)

    # Compute sizes only for files belonging to each index prefix
    def _size_for(prefix: str) -> int:
        p = Path(prefix)
        base = p.parent
        stem = p.stem  # e.g., 'recompute.leann'
        total = 0
        for f in base.iterdir():
            if f.is_file() and f.name.startswith(stem):
                total += f.stat().st_size
        return total

    size_recompute = _size_for(index_path_recompute)
    size_norecompute = _size_for(index_path_norecompute)

    print("Benchmark results (HNSW):")
    print(
        f"  recompute=True:  search_time={t_recompute:.3f}s, size={size_recompute / 1024 / 1024:.1f}MB"
    )
    print(
        f"  recompute=False: search_time={t_norecompute:.3f}s, size={size_norecompute / 1024 / 1024:.1f}MB"
    )
    print("Expectation: no-recompute should be faster but larger on disk.")


if __name__ == "__main__":
    main()
