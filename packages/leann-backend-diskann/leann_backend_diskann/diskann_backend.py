import numpy as np
import os
import json
import struct
from pathlib import Path
from typing import Dict, Any, List
import contextlib
import pickle

from leann.searcher_base import BaseSearcher
from leann.registry import register_backend
from leann.interface import (
    LeannBackendFactoryInterface,
    LeannBackendBuilderInterface,
    LeannBackendSearcherInterface
)

def _get_diskann_metrics():
    from . import _diskannpy as diskannpy
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
    with open(file_path, 'wb') as f:
        f.write(struct.pack('I', num_vectors))
        f.write(struct.pack('I', dim))
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

        label_map = {i: str_id for i, str_id in enumerate(ids)}
        label_map_file = index_dir / "leann.labels.map"
        with open(label_map_file, 'wb') as f:
            pickle.dump(label_map, f)

        build_kwargs = {**self.build_params, **kwargs}
        metric_enum = _get_diskann_metrics().get(build_kwargs.get("distance_metric", "mips").lower())
        if metric_enum is None:
            raise ValueError(f"Unsupported distance_metric.")

        try:
            from . import _diskannpy as diskannpy
            with chdir(index_dir):
                diskannpy.build_disk_float_index(
                    metric_enum, data_filename, index_prefix,
                    build_kwargs.get("complexity", 64), build_kwargs.get("graph_degree", 32),
                    build_kwargs.get("search_memory_maximum", 4.0), build_kwargs.get("build_memory_maximum", 8.0),
                    build_kwargs.get("num_threads", 8), build_kwargs.get("pq_disk_bytes", 0), ""
                )
        finally:
            temp_data_file = index_dir / data_filename
            if temp_data_file.exists():
                os.remove(temp_data_file)

class DiskannSearcher(BaseSearcher):
    def __init__(self, index_path: str, **kwargs):
        super().__init__(index_path, backend_module_name="leann_backend_diskann.embedding_server", **kwargs)
        from . import _diskannpy as diskannpy

        distance_metric = kwargs.get("distance_metric", "mips").lower()
        metric_enum = _get_diskann_metrics().get(distance_metric)
        if metric_enum is None:
            raise ValueError(f"Unsupported distance_metric '{distance_metric}'.")

        self.num_threads = kwargs.get("num_threads", 8)
        self.zmq_port = kwargs.get("zmq_port", 6666)

        full_index_prefix = str(self.index_dir / self.index_path.stem)
        self._index = diskannpy.StaticDiskFloatIndex(
            metric_enum, full_index_prefix, self.num_threads, 
            kwargs.get("num_nodes_to_cache", 0), 1, self.zmq_port, "", ""
        )

    def search(self, query: np.ndarray, top_k: int, **kwargs) -> Dict[str, Any]:
        recompute = kwargs.get("recompute_beighbor_embeddings", False)
        if recompute:
            meta_file_path = self.index_dir / f"{self.index_path.name}.meta.json"
            if not meta_file_path.exists():
                raise RuntimeError(f"FATAL: Recompute mode enabled but metadata file not found: {meta_file_path}")
            zmq_port = kwargs.get("zmq_port", self.zmq_port)
            self._ensure_server_running(str(meta_file_path), port=zmq_port, **kwargs)

        if query.dtype != np.float32:
            query = query.astype(np.float32)

        labels, distances = self._index.batch_search(
            query, query.shape[0], top_k,
            kwargs.get("complexity", 256), kwargs.get("beam_width", 4), self.num_threads,
            kwargs.get("USE_DEFERRED_FETCH", False), kwargs.get("skip_search_reorder", False),
            recompute, kwargs.get("dedup_node_dis", False), kwargs.get("prune_ratio", 0.0),
            kwargs.get("batch_recompute", False), kwargs.get("global_pruning", False)
        )

        string_labels = [[self.label_map.get(int_label, f"unknown_{int_label}") for int_label in batch_labels] for batch_labels in labels]

        return {"labels": string_labels, "distances": distances}