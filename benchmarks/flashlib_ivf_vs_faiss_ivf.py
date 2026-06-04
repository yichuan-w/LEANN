#!/usr/bin/env python3
"""
FlashLib IVF (GPU) vs FAISS IVF (CPU) head-to-head for LEANN.

This is the apples-to-apples *approximate* comparison: both backends are IVF-Flat
(inverted file) indexes that coarse-quantize the corpus into ``nlist`` cells and, at
search time, scan only the ``nprobe`` nearest cells. At a fixed ``(nlist, nprobe)``
the two probe (almost) the same candidate set, so recall is comparable - the only
difference is GPU vs CPU kernels. (Contrast with ``flashlib_vs_hnsw_speed_comparison.py``,
which compares *exact* GPU k-NN against exact CPU flat search.)

Backends compared, per corpus size, at a shared ``nlist`` and across an ``nprobe`` sweep:

- ``flashlib_ivf (GPU)`` -> the LEANN ``flashlib_ivf`` backend
  (``packages/leann-backend-flashlib-ivf``): FlashLib ``flash_ivf_flat`` on CUDA tensors.
- ``ivf (CPU)``          -> the LEANN ``ivf`` backend
  (``packages/leann-backend-ivf``): FAISS ``IndexIVFFlat`` on CPU.

Both are driven through the LEANN backend registry (the real builders/searchers), so
this measures what each backend actually does. Distance metric is cosine (vectors are
L2-normalized; FlashLib IVF ranks by squared-L2, FAISS by inner product - equivalent
on normalized vectors).

Metrics per (size, nprobe): single-query latency (median ms), batched throughput
(queries/s), recall@k vs exact ground truth (for BOTH backends), and the GPU/CPU
speedup. Build time and index size are reported once per (size, nlist).

Data is a mixture-of-Gaussians (clustered + L2-normalized) to mimic the local
structure of real embeddings, so IVF coarse quantization behaves realistically.

Requirements: a CUDA GPU, ``flashlib``, ``torch``+CUDA, ``faiss-cpu``, ``leann-core``,
``leann-backend-ivf`` and ``leann-backend-flashlib-ivf``.

Examples:

    # laptop-like CPU budget (8 threads) for the FAISS baseline
    python benchmarks/flashlib_ivf_vs_faiss_ivf.py --sizes 100000 1000000 --cpu-threads 8

    # single 1M run, custom nlist and nprobe sweep
    python benchmarks/flashlib_ivf_vs_faiss_ivf.py --sizes 1000000 \
        --nlist 4096 --nprobe-sweep 1 8 32 128

Note: importing ``leann`` pulls in ``leann_backend_hnsw`` (LEANN's API imports it at
module load). From a source checkout whose compiled HNSW backend is not installed
(e.g. glibc < 2.35), put the pure-Python package on the path first:

    PYTHONPATH=packages/leann-backend-hnsw python benchmarks/flashlib_ivf_vs_faiss_ivf.py
"""

# ruff: noqa: E402  (BLAS env vars must be set before importing numpy / faiss)
import os
import sys


def _argv_value(flag: str, default: str) -> str:
    """Read ``--flag value`` from argv before argparse, so we can pin BLAS thread
    counts BEFORE numpy/faiss import (their thread pools are fixed at import time)."""
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


# FAISS CPU search latency is governed by the BLAS thread pool, which is read at
# import time - so pin it here, before importing numpy/faiss, to the requested CPU
# budget. ``--cpu-threads 0`` means "all cores" (capped at 32: 192-thread OpenBLAS
# both crashes with "too many memory regions" and yields no benefit here).
_cpu_threads = int(_argv_value("--cpu-threads", "0"))
_blas = str(_cpu_threads) if _cpu_threads > 0 else str(min(os.cpu_count() or 1, 32))
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_v] = _blas

import argparse
import gc
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


def _fail(msg: str) -> None:
    print(f"\n[ERROR] {msg}")
    sys.exit(1)


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(x / norms)


def make_clustered_data(n_db: int, n_query: int, dim: int, seed: int, cluster_std: float):
    """Mixture-of-Gaussians, L2-normalized: a stand-in for real embeddings that have
    local cluster structure (so IVF coarse quantization is representative)."""
    rng = np.random.default_rng(seed)
    n_clusters = max(16, min(n_db // 100, 8192))
    centers = rng.standard_normal((n_clusters, dim), dtype=np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    assign = rng.integers(0, n_clusters, size=n_db)
    db = centers[assign] + cluster_std * rng.standard_normal((n_db, dim)).astype(np.float32)

    q_assign = rng.integers(0, n_clusters, size=n_query)
    queries = centers[q_assign] + cluster_std * rng.standard_normal((n_query, dim)).astype(
        np.float32
    )
    return _normalize(db.astype(np.float32)), _normalize(queries.astype(np.float32))


def exact_ground_truth(db: np.ndarray, queries: np.ndarray, top_k: int):
    """Exact top-k by cosine (== inner product on normalized vectors), on GPU,
    chunked over queries to bound memory."""
    import torch

    db_t = torch.from_numpy(db).cuda()
    q_t = torch.from_numpy(queries).cuda()
    out = np.empty((queries.shape[0], top_k), dtype=np.int64)
    step = 256
    for i in range(0, q_t.shape[0], step):
        scores = q_t[i : i + step] @ db_t.T
        out[i : i + step] = scores.topk(top_k, dim=1, largest=True).indices.cpu().numpy()
    del db_t, q_t
    torch.cuda.empty_cache()
    return out


def recall_at_k(found: np.ndarray, truth: np.ndarray) -> float:
    k = truth.shape[1]
    return float(np.mean([len(set(found[i]) & set(truth[i])) / k for i in range(truth.shape[0])]))


def best_time(fn, n_repeat: int) -> float:
    best = float("inf")
    for _ in range(n_repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def measure(search_fn, queries: np.ndarray, n_single: int, n_repeat: int):
    """Single-query latency (median ms over n_single individual queries) and batched
    throughput (q/s, best of n_repeat for all queries at once)."""
    for _ in range(3):  # warmup (FlashLib JIT-compiles per batch shape)
        search_fn(queries[:1])
        search_fn(queries)

    per_query = []
    for i in range(min(n_single, queries.shape[0])):
        t = time.perf_counter()
        search_fn(queries[i : i + 1])
        per_query.append((time.perf_counter() - t) * 1000.0)
    single_ms = float(np.median(per_query))

    batch_time = best_time(lambda: search_fn(queries), n_repeat)
    return single_ms, queries.shape[0] / batch_time


def auto_nlist(n_db: int) -> int:
    """A sane default nlist ~ 4*sqrt(N), clamped, rounded to a power of two."""
    target = 4 * math.sqrt(max(n_db, 1))
    p = 2 ** round(math.log2(max(target, 256)))
    return int(max(256, min(p, 16384)))


def _write_meta(index_path: str, backend_name: str, dim: int) -> None:
    Path(f"{index_path}.meta.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "backend_name": backend_name,
                "embedding_model": "synthetic",
                "dimensions": dim,
                "backend_kwargs": {"distance_metric": "cosine"},
                "embedding_mode": "sentence-transformers",
                "passage_sources": [],
            }
        )
    )


def _index_size_mb(index_path: str) -> float:
    stem = Path(index_path).stem
    parent = Path(index_path).parent
    total = 0
    for p in parent.glob(f"{stem}.*"):
        if p.name.endswith(".meta.json"):
            continue
        total += p.stat().st_size
    return total / (1024 * 1024)


def build_backend(name: str, db, ids, index_path: str, nlist: int, nprobe: int) -> dict[str, Any]:
    from leann.registry import BACKEND_REGISTRY

    kwargs = {
        "dimensions": db.shape[1],
        "distance_metric": "cosine",
        "nlist": nlist,
        "nprobe": nprobe,
    }
    t = time.perf_counter()
    BACKEND_REGISTRY[name].builder(**kwargs).build(db, ids, index_path, **kwargs)
    build_time = time.perf_counter() - t
    _write_meta(index_path, name, db.shape[1])
    return {"build_time": build_time, "index_size_mb": _index_size_mb(index_path)}


def make_searcher(name: str, index_path: str):
    from leann.registry import BACKEND_REGISTRY

    return BACKEND_REGISTRY[name].searcher(index_path, enable_warmup=False, use_daemon=False)


def run_search(searcher, queries, top_k, truth, nprobe, n_single, n_repeat) -> dict[str, Any]:
    def search(x):
        return searcher.search(x, top_k=top_k, nprobe=nprobe, recompute_embeddings=False)

    single_ms, qps = measure(search, queries, n_single, n_repeat)
    out = search(queries)
    found = np.array(
        [[int(x) if x.lstrip("-").isdigit() else -1 for x in row] for row in out["labels"]],
        dtype=np.int64,
    )
    recall = recall_at_k(found, truth)
    return {"single_ms": single_ms, "throughput_qps": qps, "recall": recall}


def run_size(n_db: int, args) -> dict[str, Any]:
    import faiss
    import torch

    nlist = args.nlist if args.nlist > 0 else auto_nlist(n_db)
    print(f"\n{'=' * 80}")
    print(
        f"Corpus: {n_db:,} vectors x {args.dim} dims | cosine | nlist={nlist} | "
        f"{args.queries} queries | top_k={args.top_k}"
    )
    print(f"{'=' * 80}")

    db, queries = make_clustered_data(n_db, args.queries, args.dim, args.seed, args.cluster_std)
    ids = [str(i) for i in range(n_db)]
    print("Computing exact ground truth (GPU)...")
    truth = exact_ground_truth(db, queries, args.top_k)

    nprobe_sweep = [p for p in args.nprobe_sweep if p <= nlist]
    result: dict[str, Any] = {
        "n_db": n_db,
        "nlist": nlist,
        "nprobe_sweep": nprobe_sweep,
        "rows": [],
    }

    with tempfile.TemporaryDirectory() as tmp:
        # ---- Build both backends once (build cost is a one-time offline step). ----
        faiss.omp_set_num_threads(min(os.cpu_count() or 1, 64))  # build uses many cores
        gpu_path = str(Path(tmp) / "flashlib_ivf.leann")
        cpu_path = str(Path(tmp) / "ivf.leann")

        print("Building flashlib_ivf (GPU)...")
        gb = build_backend("flashlib_ivf", db, ids, gpu_path, nlist, max(nprobe_sweep))
        print(f"  build {gb['build_time']:.2f}s | index {gb['index_size_mb']:.1f} MB")

        print("Building ivf (FAISS, CPU)...")
        cb = build_backend("ivf", db, ids, cpu_path, nlist, max(nprobe_sweep))
        print(f"  build {cb['build_time']:.2f}s | index {cb['index_size_mb']:.1f} MB")
        result["flashlib_ivf_build"] = gb
        result["ivf_build"] = cb

        # ---- Sweep nprobe; reuse a single searcher per backend across the sweep. ----
        gpu_searcher = make_searcher("flashlib_ivf", gpu_path)
        cpu_searcher = make_searcher("ivf", cpu_path)
        faiss.omp_set_num_threads(args.n_threads)  # constrain SEARCH to CPU budget

        for nprobe in nprobe_sweep:
            g = run_search(
                gpu_searcher, queries, args.top_k, truth, nprobe, args.single_queries, args.repeat
            )
            c = run_search(
                cpu_searcher, queries, args.top_k, truth, nprobe, args.single_queries, args.repeat
            )
            row = {"nprobe": nprobe, "gpu": g, "cpu": c}
            result["rows"].append(row)
            lat = c["single_ms"] / g["single_ms"] if g["single_ms"] else float("nan")
            tpt = g["throughput_qps"] / c["throughput_qps"] if c["throughput_qps"] else float("nan")
            print(
                f"  nprobe={nprobe:<4} | "
                f"GPU {g['single_ms']:7.3f}ms {g['throughput_qps']:>10,.0f}q/s r{g['recall']:.3f} | "
                f"CPU {c['single_ms']:7.3f}ms {c['throughput_qps']:>10,.0f}q/s r{c['recall']:.3f} | "
                f"speedup {lat:5.1f}x lat {tpt:6.1f}x tpt"
            )

        del gpu_searcher, cpu_searcher
        gc.collect()
        torch.cuda.empty_cache()

    return result


def print_summary(rows: list[dict[str, Any]], args) -> None:
    print(f"\n\n{'#' * 84}")
    print("# SUMMARY: flashlib_ivf (GPU) vs ivf (FAISS, CPU) - matched nlist, nprobe sweep")
    print(f"# CPU baseline used {args.n_threads} thread(s); metric=cosine; top_k={args.top_k}")
    print(f"{'#' * 84}")

    rcol = f"R@{args.top_k}"
    hdr = (
        f"{'nprobe':>6} | {'GPU ms':>8} {'GPU q/s':>11} {rcol:>6} | "
        f"{'CPU ms':>8} {'CPU q/s':>11} {rcol:>6} | {'lat x':>6} {'tpt x':>6}"
    )
    for r in rows:
        gb, cb = r["flashlib_ivf_build"], r["ivf_build"]
        print(f"\n{'-' * len(hdr)}")
        print(
            f"Corpus {r['n_db']:,} x {args.dim}d | nlist={r['nlist']} | "
            f"build: GPU {gb['build_time']:.1f}s ({gb['index_size_mb']:.0f}MB) vs "
            f"CPU {cb['build_time']:.1f}s ({cb['index_size_mb']:.0f}MB)"
        )
        print("-" * len(hdr))
        print(hdr)
        print("-" * len(hdr))
        for row in r["rows"]:
            g, c = row["gpu"], row["cpu"]
            lat = c["single_ms"] / g["single_ms"] if g["single_ms"] else float("nan")
            tpt = g["throughput_qps"] / c["throughput_qps"] if c["throughput_qps"] else float("nan")
            print(
                f"{row['nprobe']:>6} | {g['single_ms']:>8.3f} {g['throughput_qps']:>11,.0f} "
                f"{g['recall']:>6.3f} | {c['single_ms']:>8.3f} {c['throughput_qps']:>11,.0f} "
                f"{c['recall']:>6.3f} | {lat:>6.1f} {tpt:>6.1f}"
            )


def main() -> None:
    p = argparse.ArgumentParser(
        description="FlashLib IVF (GPU) vs FAISS IVF (CPU) comparison for LEANN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sizes", type=int, nargs="+", default=[100_000, 1_000_000])
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--nlist", type=int, default=0, help="IVF partitions (0 = auto ~4*sqrt(N))")
    p.add_argument("--nprobe-sweep", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    p.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="FAISS CPU search threads (0 = all cores, capped at 32).",
    )
    p.add_argument("--cluster-std", type=float, default=0.1, help="Cluster spread (lower=tighter)")
    p.add_argument("--single-queries", type=int, default=200)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    try:
        import torch
    except ImportError:
        _fail("PyTorch is required (with CUDA).")
    if not torch.cuda.is_available():
        _fail("FlashLib IVF is GPU-only, but no CUDA GPU is available.")
    try:
        import faiss
        import flashlib
    except ImportError as e:
        _fail(f"Missing dependency: {e}. Need 'flashlib' and 'faiss-cpu'.")

    from leann.registry import BACKEND_REGISTRY, autodiscover_backends

    autodiscover_backends()
    for need in ("ivf", "flashlib_ivf"):
        if need not in BACKEND_REGISTRY:
            _fail(
                f"Backend '{need}' not registered. Install it: "
                f"pip install -e packages/leann-backend-{need.replace('_', '-')}"
            )

    all_cores = os.cpu_count() or 1
    args.n_threads = args.cpu_threads if args.cpu_threads > 0 else min(all_cores, 32)

    print("FlashLib IVF (GPU) vs FAISS IVF (CPU) comparison for LEANN")
    print(f"GPU: {torch.cuda.get_device_name(0)} | CPU cores available: {all_cores}")
    print(
        f"flashlib {flashlib.__version__} | faiss {faiss.__version__} | torch {torch.__version__}"
    )
    print(
        f"Config: dim={args.dim}, queries={args.queries}, top_k={args.top_k}, "
        f"FAISS search threads={args.n_threads} (build uses up to 64), "
        f"nprobe_sweep={args.nprobe_sweep}"
    )

    rows = [run_size(n, args) for n in args.sizes]
    print_summary(rows, args)
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
