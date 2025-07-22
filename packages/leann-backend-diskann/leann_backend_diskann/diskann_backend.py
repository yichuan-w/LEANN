import numpy as np
import os
import struct
from pathlib import Path
from typing import Dict, Any, List, Literal
import contextlib

from leann.searcher_base import BaseSearcher
from leann.registry import register_backend
from leann.interface import (
    LeannBackendFactoryInterface,
    LeannBackendBuilderInterface,
    LeannBackendSearcherInterface,
)


def _get_diskann_metrics():
    from . import _diskannpy as diskannpy  # type: ignore

    return {
        "mips": diskannpy.Metric.INNER_PRODUCT,
        "l2": diskannpy.Metric.L2,
        "cosine": diskannpy.Metric.COSINE,
    }


@contextlib.contextmanager
def chdir(path):
    original_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_dir)


def _write_vectors_to_bin(data: np.ndarray, file_path: Path):
    num_vectors, dim = data.shape
    with open(file_path, "wb") as f:
        f.write(struct.pack("I", num_vectors))
        f.write(struct.pack("I", dim))
        f.write(data.tobytes())


@register_backend("diskann")
class DiskannBackend(LeannBackendFactoryInterface):
    @staticmethod
    def builder(**kwargs) -> LeannBackendBuilderInterface:
        return DiskannBuilder(**kwargs)

    @staticmethod
    def searcher(index_path: str, **kwargs) -> LeannBackendSearcherInterface:
        return DiskannSearcher(index_path, **kwargs)


class DiskannBuilder(LeannBackendBuilderInterface):
    def __init__(self, **kwargs):
        self.build_params = kwargs

    def build(self, data: np.ndarray, ids: List[str], index_path: str, **kwargs):
        path = Path(index_path)
        index_dir = path.parent
        index_prefix = path.stem
        index_dir.mkdir(parents=True, exist_ok=True)

        if data.dtype != np.float32:
            data = data.astype(np.float32)

        data_filename = f"{index_prefix}_data.bin"
        _write_vectors_to_bin(data, index_dir / data_filename)

        build_kwargs = {**self.build_params, **kwargs}
        metric_enum = _get_diskann_metrics().get(
            build_kwargs.get("distance_metric", "mips").lower()
        )
        if metric_enum is None:
            raise ValueError("Unsupported distance_metric.")

        try:
            from . import _diskannpy as diskannpy  # type: ignore

            with chdir(index_dir):
                diskannpy.build_disk_float_index(
                    metric_enum,
                    data_filename,
                    index_prefix,
                    build_kwargs.get("complexity", 64),
                    build_kwargs.get("graph_degree", 32),
                    build_kwargs.get("search_memory_maximum", 4.0),
                    build_kwargs.get("build_memory_maximum", 8.0),
                    build_kwargs.get("num_threads", 8),
                    build_kwargs.get("pq_disk_bytes", 0),
                    "",
                )
        finally:
            temp_data_file = index_dir / data_filename
            if temp_data_file.exists():
                os.remove(temp_data_file)


class DiskannSearcher(BaseSearcher):
    def __init__(self, index_path: str, **kwargs):
        super().__init__(
            index_path,
            backend_module_name="leann_backend_diskann.embedding_server",
            **kwargs,
        )
        from . import _diskannpy as diskannpy  # type: ignore

        distance_metric = kwargs.get("distance_metric", "mips").lower()
        metric_enum = _get_diskann_metrics().get(distance_metric)
        if metric_enum is None:
            raise ValueError(f"Unsupported distance_metric '{distance_metric}'.")

        self.num_threads = kwargs.get("num_threads", 8)
        self.zmq_port = kwargs.get("zmq_port", 6666)

        full_index_prefix = str(self.index_dir / self.index_path.stem)
        self._index = diskannpy.StaticDiskFloatIndex(
            metric_enum,
            full_index_prefix,
            self.num_threads,
            kwargs.get("num_nodes_to_cache", 0),
            1,
            self.zmq_port,
            "",
            "",
        )

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
        batch_recompute: bool = False,
        dedup_node_dis: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Search for nearest neighbors using DiskANN index.

        Args:
            query: Query vectors (B, D) where B is batch size, D is dimension
            top_k: Number of nearest neighbors to return
            complexity: Search complexity/candidate list size, higher = more accurate but slower
            beam_width: Number of parallel IO requests per iteration
            prune_ratio: Ratio of neighbors to prune via approximate distance (0.0-1.0)
            recompute_embeddings: Whether to fetch fresh embeddings from server
            pruning_strategy: PQ candidate selection strategy:
                - "global": Use global pruning strategy (default)
                - "local": Use local pruning strategy
                - "proportional": Not supported in DiskANN, falls back to global
            zmq_port: ZMQ port for embedding server
            batch_recompute: Whether to batch neighbor recomputation (DiskANN-specific)
            dedup_node_dis: Whether to cache and reuse distance computations (DiskANN-specific)
            **kwargs: Additional DiskANN-specific parameters (for legacy compatibility)

        Returns:
            Dict with 'labels' (list of lists) and 'distances' (ndarray)
        """
        # DiskANN doesn't support "proportional" strategy
        if pruning_strategy == "proportional":
            raise NotImplementedError(
                "DiskANN backend does not support 'proportional' pruning strategy. Use 'global' or 'local' instead."
            )

        # Use recompute_embeddings parameter
        use_recompute = recompute_embeddings
        if use_recompute:
            meta_file_path = self.index_dir / f"{self.index_path.name}.meta.json"
            if not meta_file_path.exists():
                raise RuntimeError(
                    f"FATAL: Recompute enabled but metadata file not found: {meta_file_path}"
                )
            self._ensure_server_running(str(meta_file_path), port=zmq_port, **kwargs)

        if query.dtype != np.float32:
            query = query.astype(np.float32)

        # Map pruning_strategy to DiskANN's global_pruning parameter
        if pruning_strategy == "local":
            use_global_pruning = False
        else:  # "global"
            use_global_pruning = True

        labels, distances = self._index.batch_search(
            query,
            query.shape[0],
            top_k,
            complexity,
            beam_width,
            self.num_threads,
            kwargs.get("USE_DEFERRED_FETCH", False),
            kwargs.get("skip_search_reorder", False),
            use_recompute,
            dedup_node_dis,
            prune_ratio,
            batch_recompute,
            use_global_pruning,
        )

        string_labels = [
            [str(int_label) for int_label in batch_labels] for batch_labels in labels
        ]

        return {"labels": string_labels, "distances": distances}
