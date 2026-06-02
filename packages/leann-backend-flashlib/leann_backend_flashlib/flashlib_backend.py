"""
FlashLib backend: GPU-accelerated nearest-neighbor search built on Triton / CuteDSL.

FlashLib (https://github.com/FlashML-org/flashlib) is a GPU library of classical
ML primitives. This backend uses its ``NearestNeighbors`` application (a fused
flash-knn search) which runs entirely on CUDA tensors:

    from flashlib import NearestNeighbors
    index = NearestNeighbors().fit(db_cuda)            # db_cuda: (N, D) CUDA tensor
    distances, indices = index.kneighbors(queries, n_neighbors=10)

FlashLib's search has no on-disk format, so this backend persists the raw float32
vectors (``.flashlib.npy``) plus an id map, and reconstructs the GPU index at
searcher start-up via ``NearestNeighbors().fit(db)``.

FlashLib ranks by squared L2. For ``mips`` / ``cosine`` we L2-normalize both the
database and query vectors, on which squared-L2 ranking is equivalent to
inner-product / cosine ranking.

Requires a CUDA GPU at search time. Index building only needs numpy (no GPU).
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from leann.interface import (
    LeannBackendBuilderInterface,
    LeannBackendFactoryInterface,
    LeannBackendSearcherInterface,
)
from leann.registry import register_backend
from leann.searcher_base import BaseSearcher

logger = logging.getLogger(__name__)

VECTORS_SUFFIX = "flashlib.npy"
ID_MAP_SUFFIX = "flashlib_id_map.json"


def _import_flashlib():
    try:
        import torch  # noqa: F401
        from flashlib import NearestNeighbors
    except ImportError as e:
        raise ImportError(
            "The FlashLib backend requires 'flashlib' and 'torch' with CUDA. "
            "Install with: pip install flashlib (a CUDA GPU is required at search time)."
        ) from e
    return NearestNeighbors


def _normalize_l2(data: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return data / norms


def _needs_normalize(distance_metric: str) -> bool:
    return distance_metric.lower() in ("mips", "cosine")


def _vectors_path(index_dir: Path, index_prefix: str) -> Path:
    return index_dir / f"{index_prefix}.{VECTORS_SUFFIX}"


def _id_map_path(index_dir: Path, index_prefix: str) -> Path:
    return index_dir / f"{index_prefix}.{ID_MAP_SUFFIX}"


def _save_id_map(index_dir: Path, index_prefix: str, ids: list[str]) -> None:
    with open(_id_map_path(index_dir, index_prefix), "w", encoding="utf-8") as f:
        json.dump({"ids": ids}, f)


def _load_id_map(index_dir: Path, index_prefix: str) -> list[str]:
    path = _id_map_path(index_dir, index_prefix)
    if not path.exists():
        raise FileNotFoundError(f"FlashLib id map not found at {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["ids"]


@register_backend("flashlib")
class FlashlibBackend(LeannBackendFactoryInterface):
    @staticmethod
    def builder(**kwargs) -> LeannBackendBuilderInterface:
        return FlashlibBuilder(**kwargs)

    @staticmethod
    def searcher(index_path: str, **kwargs) -> LeannBackendSearcherInterface:
        return FlashlibSearcher(index_path, **kwargs)


class FlashlibBuilder(LeannBackendBuilderInterface):
    def __init__(self, **kwargs):
        self.build_params = kwargs.copy()
        self.distance_metric = self.build_params.setdefault("distance_metric", "mips")
        self.dimensions = self.build_params.get("dimensions")

    def build(self, data: np.ndarray, ids: list[str], index_path: str, **kwargs) -> None:
        path = Path(index_path)
        index_dir = path.parent
        index_prefix = path.stem
        index_dir.mkdir(parents=True, exist_ok=True)

        if data.dtype != np.float32:
            data = data.astype(np.float32)
        data = np.ascontiguousarray(data)

        if _needs_normalize(self.distance_metric):
            data = _normalize_l2(data)

        # FlashLib search has no disk format, so we persist the raw vectors and
        # rebuild the GPU index when the searcher starts.
        np.save(_vectors_path(index_dir, index_prefix), data)
        _save_id_map(index_dir, index_prefix, list(ids))
        logger.info(
            "FlashLib build: stored %d vectors (dim=%d, metric=%s) at %s",
            data.shape[0],
            data.shape[1],
            self.distance_metric,
            _vectors_path(index_dir, index_prefix),
        )


class FlashlibSearcher(BaseSearcher):
    def __init__(self, index_path: str, **kwargs):
        # Reuse the HNSW embedding server (if present) to embed queries, exactly like
        # the other non-recompute backends; falls back to direct model loading.
        super().__init__(
            index_path,
            backend_module_name="leann_backend_hnsw.hnsw_embedding_server",
            **kwargs,
        )
        backend_kwargs = self.meta.get("backend_kwargs", {})
        self.distance_metric = backend_kwargs.get("distance_metric", "mips").lower()

        index_prefix = self.index_path.stem
        vectors_file = _vectors_path(self.index_dir, index_prefix)
        if not vectors_file.exists():
            raise FileNotFoundError(f"FlashLib vectors file not found at {vectors_file}")

        self._ids = _load_id_map(self.index_dir, index_prefix)

        NearestNeighbors = _import_flashlib()
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "FlashLib backend requires a CUDA GPU at search time, but none is available."
            )

        vectors = np.load(vectors_file)
        self._ntotal = vectors.shape[0]
        db = torch.from_numpy(np.ascontiguousarray(vectors)).cuda()
        self._index = NearestNeighbors().fit(db)
        logger.info(
            "FlashLib searcher ready: %d vectors, metric=%s",
            self._ntotal,
            self.distance_metric,
        )

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        complexity: int = 64,
        **kwargs,
    ) -> dict[str, Any]:
        import torch

        if query.dtype != np.float32:
            query = query.astype(np.float32)
        if _needs_normalize(self.distance_metric):
            query = _normalize_l2(query)

        k = min(top_k, self._ntotal)
        q = torch.from_numpy(np.ascontiguousarray(query)).cuda()
        distances, indices = self._index.kneighbors(q, n_neighbors=k)
        distances_np = distances.detach().cpu().numpy().astype(np.float32)
        indices_np = indices.detach().cpu().numpy()

        string_labels = [[self._ids[int(i)] for i in row] for row in indices_np]
        return {"labels": string_labels, "distances": distances_np}

    def compute_query_embedding(
        self,
        query: str,
        use_server_if_available: bool = True,
        zmq_port: Optional[int] = None,
        query_template: Optional[str] = None,
    ) -> np.ndarray:
        return super().compute_query_embedding(
            query,
            use_server_if_available=use_server_if_available,
            zmq_port=zmq_port,
            query_template=query_template,
        )
