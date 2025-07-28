import contextlib
import logging
import os
import struct
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
from leann.interface import (
    LeannBackendBuilderInterface,
    LeannBackendFactoryInterface,
    LeannBackendSearcherInterface,
)
from leann.registry import register_backend
from leann.searcher_base import BaseSearcher

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_cpp_output_if_needed():
    """Suppress C++ stdout/stderr based on LEANN_LOG_LEVEL"""
    log_level = os.getenv("LEANN_LOG_LEVEL", "WARNING").upper()

    # Only suppress if log level is WARNING or higher (ERROR, CRITICAL)
    should_suppress = log_level in ["WARNING", "ERROR", "CRITICAL"]

    if not should_suppress:
        # Don't suppress, just yield
        yield
        return

    # Save original file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()

    # Save original stdout/stderr
    stdout_dup = os.dup(stdout_fd)
    stderr_dup = os.dup(stderr_fd)

    try:
        # Redirect to /dev/null
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)

        yield

    finally:
        # Restore original file descriptors
        os.dup2(stdout_dup, stdout_fd)
        os.dup2(stderr_dup, stderr_fd)
        os.close(stdout_dup)
        os.close(stderr_dup)


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

    def build(self, data: np.ndarray, ids: list[str], index_path: str, **kwargs):
        path = Path(index_path)
        index_dir = path.parent
        index_prefix = path.stem
        index_dir.mkdir(parents=True, exist_ok=True)

        if data.dtype != np.float32:
            logger.warning(f"Converting data to float32, shape: {data.shape}")
            data = data.astype(np.float32)

        data_filename = f"{index_prefix}_data.bin"
        _write_vectors_to_bin(data, index_dir / data_filename)

        build_kwargs = {**self.build_params, **kwargs}
        metric_enum = _get_diskann_metrics().get(
            build_kwargs.get("distance_metric", "mips").lower()
        )
        if metric_enum is None:
            raise ValueError(
                f"Unsupported distance_metric '{build_kwargs.get('distance_metric', 'unknown')}'."
            )

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
                logger.debug(f"Cleaned up temporary data file: {temp_data_file}")


class DiskannSearcher(BaseSearcher):
    def __init__(self, index_path: str, **kwargs):
        super().__init__(
            index_path,
            backend_module_name="leann_backend_diskann.diskann_embedding_server",
            **kwargs,
        )

        # Initialize DiskANN index with suppressed C++ output based on log level
        with suppress_cpp_output_if_needed():
            from . import _diskannpy as diskannpy  # type: ignore

            distance_metric = kwargs.get("distance_metric", "mips").lower()
            metric_enum = _get_diskann_metrics().get(distance_metric)
            if metric_enum is None:
                raise ValueError(f"Unsupported distance_metric '{distance_metric}'.")

            self.num_threads = kwargs.get("num_threads", 8)

            # For DiskANN, we need to reinitialize the index when zmq_port changes
            # Store the initialization parameters for later use
            full_index_prefix = str(self.index_dir / self.index_path.stem)
            self._init_params = {
                "metric_enum": metric_enum,
                "full_index_prefix": full_index_prefix,
                "num_threads": self.num_threads,
                "num_nodes_to_cache": kwargs.get("num_nodes_to_cache", 0),
                "cache_mechanism": 1,
                "pq_prefix": "",
                "partition_prefix": "",
            }
            self._diskannpy = diskannpy
            self._current_zmq_port = None
            self._index = None
            logger.debug("DiskANN searcher initialized (index will be loaded on first search)")

    def _ensure_index_loaded(self, zmq_port: int):
        """Ensure the index is loaded with the correct zmq_port."""
        if self._index is None or self._current_zmq_port != zmq_port:
            # Need to (re)load the index with the correct zmq_port
            with suppress_cpp_output_if_needed():
                if self._index is not None:
                    logger.debug(f"Reloading DiskANN index with new zmq_port: {zmq_port}")
                else:
                    logger.debug(f"Loading DiskANN index with zmq_port: {zmq_port}")

                self._index = self._diskannpy.StaticDiskFloatIndex(
                    self._init_params["metric_enum"],
                    self._init_params["full_index_prefix"],
                    self._init_params["num_threads"],
                    self._init_params["num_nodes_to_cache"],
                    self._init_params["cache_mechanism"],
                    zmq_port,
                    self._init_params["pq_prefix"],
                    self._init_params["partition_prefix"],
                )
                self._current_zmq_port = zmq_port

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        complexity: int = 64,
        beam_width: int = 1,
        prune_ratio: float = 0.0,
        recompute_embeddings: bool = False,
        pruning_strategy: Literal["global", "local", "proportional"] = "global",
        zmq_port: int | None = None,
        batch_recompute: bool = False,
        dedup_node_dis: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
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
            zmq_port: ZMQ port for embedding server communication. Must be provided if recompute_embeddings is True.
            batch_recompute: Whether to batch neighbor recomputation (DiskANN-specific)
            dedup_node_dis: Whether to cache and reuse distance computations (DiskANN-specific)
            **kwargs: Additional DiskANN-specific parameters (for legacy compatibility)

        Returns:
            Dict with 'labels' (list of lists) and 'distances' (ndarray)
        """
        # Handle zmq_port compatibility: Ensure index is loaded with correct port
        if recompute_embeddings:
            if zmq_port is None:
                raise ValueError("zmq_port must be provided if recompute_embeddings is True")
            self._ensure_index_loaded(zmq_port)
        else:
            # If not recomputing, we still need an index, use a default port
            if self._index is None:
                self._ensure_index_loaded(6666)  # Default port when not recomputing

        # DiskANN doesn't support "proportional" strategy
        if pruning_strategy == "proportional":
            raise NotImplementedError(
                "DiskANN backend does not support 'proportional' pruning strategy. Use 'global' or 'local' instead."
            )

        if query.dtype != np.float32:
            query = query.astype(np.float32)

        # Map pruning_strategy to DiskANN's global_pruning parameter
        if pruning_strategy == "local":
            use_global_pruning = False
        else:  # "global"
            use_global_pruning = True

        # Perform search with suppressed C++ output based on log level
        with suppress_cpp_output_if_needed():
            labels, distances = self._index.batch_search(
                query,
                query.shape[0],
                top_k,
                complexity,
                beam_width,
                self.num_threads,
                kwargs.get("USE_DEFERRED_FETCH", False),
                kwargs.get("skip_search_reorder", False),
                recompute_embeddings,
                dedup_node_dis,
                prune_ratio,
                batch_recompute,
                use_global_pruning,
            )

        string_labels = [[str(int_label) for int_label in batch_labels] for batch_labels in labels]

        return {"labels": string_labels, "distances": distances}
