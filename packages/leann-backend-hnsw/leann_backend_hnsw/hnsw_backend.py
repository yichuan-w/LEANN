import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, List, Literal
import pickle
import shutil
import time

from leann.searcher_base import BaseSearcher
from .convert_to_csr import convert_hnsw_graph_to_csr

from leann.registry import register_backend
from leann.interface import (
    LeannBackendFactoryInterface,
    LeannBackendBuilderInterface,
    LeannBackendSearcherInterface,
)


def get_metric_map():
    from . import faiss  # type: ignore

    return {
        "mips": faiss.METRIC_INNER_PRODUCT,
        "l2": faiss.METRIC_L2,
        "cosine": faiss.METRIC_INNER_PRODUCT,
    }


@register_backend("hnsw")
class HNSWBackend(LeannBackendFactoryInterface):
    @staticmethod
    def builder(**kwargs) -> LeannBackendBuilderInterface:
        return HNSWBuilder(**kwargs)

    @staticmethod
    def searcher(index_path: str, **kwargs) -> LeannBackendSearcherInterface:
        return HNSWSearcher(index_path, **kwargs)


class HNSWBuilder(LeannBackendBuilderInterface):
    def __init__(self, **kwargs):
        self.build_params = kwargs.copy()
        self.is_compact = self.build_params.setdefault("is_compact", True)
        self.is_recompute = self.build_params.setdefault("is_recompute", True)
        self.M = self.build_params.setdefault("M", 32)
        self.efConstruction = self.build_params.setdefault("efConstruction", 200)
        self.distance_metric = self.build_params.setdefault("distance_metric", "mips")
        self.dimensions = self.build_params.get("dimensions")

    def build(self, data: np.ndarray, ids: List[str], index_path: str, **kwargs):
        from . import faiss  # type: ignore

        path = Path(index_path)
        index_dir = path.parent
        index_prefix = path.stem
        index_dir.mkdir(parents=True, exist_ok=True)

        if data.dtype != np.float32:
            data = data.astype(np.float32)


        metric_enum = get_metric_map().get(self.distance_metric.lower())
        if metric_enum is None:
            raise ValueError(f"Unsupported distance_metric '{self.distance_metric}'.")

        dim = self.dimensions or data.shape[1]
        index = faiss.IndexHNSWFlat(dim, self.M, metric_enum)
        index.hnsw.efConstruction = self.efConstruction

        if self.distance_metric.lower() == "cosine":
            faiss.normalize_L2(data)

        index.add(data.shape[0], faiss.swig_ptr(data))
        index_file = index_dir / f"{index_prefix}.index"
        faiss.write_index(index, str(index_file))

        if self.is_compact:
            self._convert_to_csr(index_file)

    def _convert_to_csr(self, index_file: Path):
        """Convert built index to CSR format"""
        mode_str = "CSR-pruned" if self.is_recompute else "CSR-standard"
        print(f"INFO: Converting HNSW index to {mode_str} format...")

        csr_temp_file = index_file.with_suffix(".csr.tmp")

        success = convert_hnsw_graph_to_csr(
            str(index_file), str(csr_temp_file), prune_embeddings=self.is_recompute
        )

        if success:
            print("✅ CSR conversion successful.")
            index_file_old = index_file.with_suffix(".old")
            shutil.move(str(index_file), str(index_file_old))
            shutil.move(str(csr_temp_file), str(index_file))
            print(
                f"INFO: Replaced original index with {mode_str} version at '{index_file}'"
            )
        else:
            # Clean up and fail fast
            if csr_temp_file.exists():
                os.remove(csr_temp_file)
            raise RuntimeError(
                "CSR conversion failed - cannot proceed with compact format"
            )


class HNSWSearcher(BaseSearcher):
    def __init__(self, index_path: str, **kwargs):
        super().__init__(
            index_path,
            backend_module_name="leann_backend_hnsw.hnsw_embedding_server",
            **kwargs,
        )
        from . import faiss  # type: ignore

        self.distance_metric = self.meta.get("distance_metric", "mips").lower()
        metric_enum = get_metric_map().get(self.distance_metric)
        if metric_enum is None:
            raise ValueError(f"Unsupported distance_metric '{self.distance_metric}'.")

        self.is_compact, self.is_pruned = (
            self.meta.get("is_compact", True),
            self.meta.get("is_pruned", True),
        )

        index_file = self.index_dir / f"{self.index_path.stem}.index"
        if not index_file.exists():
            raise FileNotFoundError(f"HNSW index file not found at {index_file}")

        hnsw_config = faiss.HNSWIndexConfig()
        hnsw_config.is_compact = self.is_compact
        hnsw_config.is_recompute = self.is_pruned or kwargs.get("is_recompute", False)

        if self.is_pruned and not hnsw_config.is_recompute:
            raise RuntimeError("Index is pruned but recompute is disabled.")

        self._index = faiss.read_index(str(index_file), faiss.IO_FLAG_MMAP, hnsw_config)
        

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        complexity: int = 64,
        beam_width: int = 1,
        prune_ratio: float = 0.0,
        recompute_embeddings: bool = False,
        pruning_strategy: Literal["global", "local", "proportional"] = "global",
        zmq_port: int = 5557,
        batch_size: int = 0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Search for nearest neighbors using HNSW index.

        Args:
            query: Query vectors (B, D) where B is batch size, D is dimension
            top_k: Number of nearest neighbors to return
            complexity: Search complexity/efSearch, higher = more accurate but slower
            beam_width: Number of parallel search paths/beam_size
            prune_ratio: Ratio of neighbors to prune via PQ (0.0-1.0)
            recompute_embeddings: Whether to fetch fresh embeddings from server
            pruning_strategy: PQ candidate selection strategy:
                - "global": Use global PQ queue size for selection (default)
                - "local": Local pruning, sort and select best candidates
                - "proportional": Base selection on new neighbor count ratio
            zmq_port: ZMQ port for embedding server
            batch_size: Neighbor processing batch size, 0=disabled (HNSW-specific)
            **kwargs: Additional HNSW-specific parameters (for legacy compatibility)

        Returns:
            Dict with 'labels' (list of lists) and 'distances' (ndarray)
        """
        from . import faiss  # type: ignore

        # Use recompute_embeddings parameter
        use_recompute = recompute_embeddings or self.is_pruned
        if use_recompute:
            meta_file_path = self.index_dir / f"{self.index_path.name}.meta.json"
            if not meta_file_path.exists():
                raise RuntimeError(
                    f"FATAL: Recompute enabled but metadata file not found: {meta_file_path}"
                )
            self._ensure_server_running(str(meta_file_path), port=zmq_port, **kwargs)

        if query.dtype != np.float32:
            query = query.astype(np.float32)
        if self.distance_metric == "cosine":
            faiss.normalize_L2(query)

        params = faiss.SearchParametersHNSW()
        params.zmq_port = zmq_port
        params.efSearch = complexity
        params.beam_size = beam_width

        # PQ pruning: direct mapping to HNSW's pq_pruning_ratio
        params.pq_pruning_ratio = prune_ratio

        # Map pruning_strategy to HNSW parameters
        if pruning_strategy == "local":
            params.local_prune = True
            params.send_neigh_times_ratio = 0.0
        elif pruning_strategy == "proportional":
            params.local_prune = False
            params.send_neigh_times_ratio = (
                1.0  # Any value > 1e-6 triggers proportional mode
            )
        else:  # "global"
            params.local_prune = False
            params.send_neigh_times_ratio = 0.0

        # HNSW-specific batch processing parameter
        params.batch_size = batch_size

        batch_size_query = query.shape[0]
        distances = np.empty((batch_size_query, top_k), dtype=np.float32)
        labels = np.empty((batch_size_query, top_k), dtype=np.int64)

        self._index.search(
            query.shape[0],
            faiss.swig_ptr(query),
            top_k,
            faiss.swig_ptr(distances),
            faiss.swig_ptr(labels),
            params,
        )

        string_labels = [
            [str(int_label) for int_label in batch_labels]
            for batch_labels in labels
        ]

        return {"labels": string_labels, "distances": distances}
