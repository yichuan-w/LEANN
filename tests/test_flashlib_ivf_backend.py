"""
Correctness tests for the FlashLib IVF backend (``flashlib_ivf``).

Registry-level (no embedding model needed): builds a small clustered corpus through
the real LEANN backend builders/searchers and checks that ``flashlib_ivf`` (GPU)
registers, persists/reloads its index, and returns recall comparable to the FAISS
``ivf`` backend and to exact ground truth at a matched ``(nlist, nprobe)``.

Requires a CUDA GPU + ``flashlib`` (skipped otherwise, e.g. in CI).
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


def _cuda_or_skip():
    torch = pytest.importorskip("torch")
    pytest.importorskip("flashlib")
    if not torch.cuda.is_available():
        pytest.skip("FlashLib IVF backend requires a CUDA GPU")


def _registry():
    from leann.registry import BACKEND_REGISTRY, autodiscover_backends

    autodiscover_backends()
    return BACKEND_REGISTRY


def _clustered(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_clusters = 64
    centers = rng.standard_normal((n_clusters, dim), dtype=np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    assign = rng.integers(0, n_clusters, size=n)
    db = centers[assign] + 0.05 * rng.standard_normal((n, dim)).astype(np.float32)
    db /= np.linalg.norm(db, axis=1, keepdims=True)
    return np.ascontiguousarray(db.astype(np.float32))


def _exact_gt(db: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    scores = q @ db.T
    return np.argsort(-scores, axis=1)[:, :k]


def _recall(found: np.ndarray, truth: np.ndarray) -> float:
    k = truth.shape[1]
    return float(np.mean([len(set(found[i]) & set(truth[i])) / k for i in range(len(truth))]))


def _write_meta(index_path: str, backend: str, dim: int) -> None:
    Path(f"{index_path}.meta.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "backend_name": backend,
                "embedding_model": "synthetic",
                "dimensions": dim,
                "backend_kwargs": {"distance_metric": "cosine"},
                "embedding_mode": "sentence-transformers",
                "passage_sources": [],
            }
        )
    )


def _labels_to_int(out: dict) -> np.ndarray:
    return np.array(
        [[int(x) if x.lstrip("-").isdigit() else -1 for x in row] for row in out["labels"]],
        dtype=np.int64,
    )


def test_flashlib_ivf_recall_parity_with_faiss_ivf():
    _cuda_or_skip()
    reg = _registry()
    assert "flashlib_ivf" in reg, "flashlib_ivf backend not registered"
    if "ivf" not in reg:
        pytest.skip("faiss ivf backend not installed")

    dim, n, k, nlist, nprobe = 64, 20_000, 10, 128, 16
    db = _clustered(n, dim, seed=0)
    queries = db[:100].copy()
    ids = [str(i) for i in range(n)]
    truth = _exact_gt(db, queries, k)

    kw = {"dimensions": dim, "distance_metric": "cosine", "nlist": nlist, "nprobe": nprobe}
    with tempfile.TemporaryDirectory() as tmp:
        gpu_path = str(Path(tmp) / "g.leann")
        cpu_path = str(Path(tmp) / "c.leann")

        reg["flashlib_ivf"].builder(**kw).build(db, ids, gpu_path, **kw)
        reg["ivf"].builder(**kw).build(db, ids, cpu_path, **kw)

        # Persistence: the GPU index + id map are written to disk.
        assert (Path(tmp) / "g.flashlib_ivf.pt").exists()
        assert (Path(tmp) / "g.flashlib_ivf_id_map.json").exists()

        _write_meta(gpu_path, "flashlib_ivf", dim)
        _write_meta(cpu_path, "ivf", dim)

        gpu = reg["flashlib_ivf"].searcher(gpu_path, enable_warmup=False, use_daemon=False)
        cpu = reg["ivf"].searcher(cpu_path, enable_warmup=False, use_daemon=False)

        gpu_found = _labels_to_int(
            gpu.search(queries, top_k=k, nprobe=nprobe, recompute_embeddings=False)
        )
        cpu_found = _labels_to_int(
            cpu.search(queries, top_k=k, nprobe=nprobe, recompute_embeddings=False)
        )

    gpu_recall = _recall(gpu_found, truth)
    cpu_recall = _recall(cpu_found, truth)

    # At matched (nlist, nprobe) both IVF backends probe ~the same candidate set, so
    # recall should be high and comparable (GPU is often a touch higher: independent
    # coarse-quantizer training, not "more exact").
    assert gpu_recall > 0.7, f"flashlib_ivf recall too low: {gpu_recall:.3f}"
    assert abs(gpu_recall - cpu_recall) < 0.15, (
        f"recall parity off: gpu {gpu_recall:.3f} vs cpu {cpu_recall:.3f}"
    )
