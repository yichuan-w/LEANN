import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, Any, List
import pickle

from leann.searcher_base import BaseSearcher
from .convert_to_csr import convert_hnsw_graph_to_csr

from leann.registry import register_backend
from leann.interface import (
    LeannBackendFactoryInterface,
    LeannBackendBuilderInterface,
    LeannBackendSearcherInterface
)

def get_metric_map():
    from . import faiss
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
        from . import faiss
        path = Path(index_path)
        index_dir = path.parent
        index_prefix = path.stem
        index_dir.mkdir(parents=True, exist_ok=True)

        if data.dtype != np.float32:
            data = data.astype(np.float32)

        label_map = {i: str_id for i, str_id in enumerate(ids)}
        label_map_file = index_dir / "leann.labels.map"
        with open(label_map_file, 'wb') as f:
            pickle.dump(label_map, f)

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
        csr_temp_file = index_file.with_suffix(".csr.tmp")
        success = convert_hnsw_graph_to_csr(
            str(index_file), str(csr_temp_file), prune_embeddings=self.is_recompute
        )
        if success:
            import shutil
            shutil.move(str(csr_temp_file), str(index_file))
        else:
            if csr_temp_file.exists():
                os.remove(csr_temp_file)
            raise RuntimeError("CSR conversion failed")

class HNSWSearcher(BaseSearcher):
    def __init__(self, index_path: str, **kwargs):
        super().__init__(index_path, backend_module_name="leann_backend_hnsw.hnsw_embedding_server", **kwargs)
        from . import faiss

        self.distance_metric = self.meta.get("distance_metric", "mips").lower()
        metric_enum = get_metric_map().get(self.distance_metric)
        if metric_enum is None:
            raise ValueError(f"Unsupported distance_metric '{self.distance_metric}'.")

        self.is_compact, self.is_pruned = self._get_index_storage_status_from_meta()

        index_file = self.index_dir / f"{self.index_path.stem}.index"
        if not index_file.exists():
            raise FileNotFoundError(f"HNSW index file not found at {index_file}")

        hnsw_config = faiss.HNSWIndexConfig()
        hnsw_config.is_compact = self.is_compact
        hnsw_config.is_recompute = self.is_pruned or kwargs.get("is_recompute", False)

        if self.is_pruned and not hnsw_config.is_recompute:
            raise RuntimeError("Index is pruned but recompute is disabled.")

        self._index = faiss.read_index(str(index_file), faiss.IO_FLAG_MMAP, hnsw_config)

    def _get_index_storage_status_from_meta(self) -> tuple[bool, bool]:
        is_compact = self.meta.get('is_compact', True)
        is_pruned = self.meta.get('is_pruned', True)
        return is_compact, is_pruned

    def search(self, query: np.ndarray, top_k: int, **kwargs) -> Dict[str, Any]:
        from . import faiss

        if self.is_pruned:
            meta_file_path = self.index_dir / f"{self.index_path.name}.meta.json"
            if not meta_file_path.exists():
                raise RuntimeError(f"FATAL: Index is pruned but metadata file not found: {meta_file_path}")
            zmq_port = kwargs.get("zmq_port", 5557)
            self._ensure_server_running(str(meta_file_path), port=zmq_port, **kwargs)

        if query.dtype != np.float32:
            query = query.astype(np.float32)
        if self.distance_metric == "cosine":
            faiss.normalize_L2(query)

        params = faiss.SearchParametersHNSW()
        params.zmq_port = kwargs.get("zmq_port", 5557)
        params.efSearch = kwargs.get("ef", 128)
        params.beam_size = 2

        batch_size = query.shape[0]
        distances = np.empty((batch_size, top_k), dtype=np.float32)
        labels = np.empty((batch_size, top_k), dtype=np.int64)

        self._index.search(query.shape[0], faiss.swig_ptr(query), top_k, faiss.swig_ptr(distances), faiss.swig_ptr(labels), params)

        string_labels = [[self.label_map.get(int_label, f"unknown_{int_label}") for int_label in batch_labels] for batch_labels in labels]

        return {"labels": string_labels, "distances": distances}