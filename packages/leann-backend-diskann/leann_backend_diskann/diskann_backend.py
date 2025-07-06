import numpy as np
import os
import json
import struct
from pathlib import Path
from typing import Dict, Any
import contextlib
import threading
import time
import atexit
import socket
import subprocess
import sys

from leann.embedding_server_manager import EmbeddingServerManager
from leann.registry import register_backend
from leann.interface import (
    LeannBackendFactoryInterface,
    LeannBackendBuilderInterface,
    LeannBackendSearcherInterface
)
from . import _diskannpy as diskannpy

METRIC_MAP = {
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
        path = Path(index_path)
        meta_path = path.parent / f"{path.name}.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Leann metadata file not found at {meta_path}.")

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        # Pass essential metadata to the searcher
        kwargs['meta'] = meta
        return DiskannSearcher(index_path, **kwargs)

class DiskannBuilder(LeannBackendBuilderInterface):
    def __init__(self, **kwargs):
        self.build_params = kwargs

    def _generate_passages_file(self, index_dir: Path, index_prefix: str, **kwargs):
        """Generate passages file for recompute mode, mirroring HNSW backend."""
        try:
            chunks = kwargs.get('chunks', [])
            if not chunks:
                print("INFO: No chunks data provided, skipping passages file generation for DiskANN.")
                return
            
            passages_data = {str(node_id): chunk["text"] for node_id, chunk in enumerate(chunks)}
            
            passages_file = index_dir / f"{index_prefix}.passages.json"
            with open(passages_file, 'w', encoding='utf-8') as f:
                json.dump(passages_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Generated passages file for recompute mode at '{passages_file}' ({len(passages_data)} passages)")
            
        except Exception as e:
            print(f"💥 ERROR: Failed to generate passages file for DiskANN. Exception: {e}")
            pass

    def build(self, data: np.ndarray, index_path: str, **kwargs):
        path = Path(index_path)
        index_dir = path.parent
        index_prefix = path.stem

        index_dir.mkdir(parents=True, exist_ok=True)

        if data.dtype != np.float32:
            data = data.astype(np.float32)
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
            
        data_filename = f"{index_prefix}_data.bin"
        _write_vectors_to_bin(data, index_dir / data_filename)

        build_kwargs = {**self.build_params, **kwargs}
        metric_str = build_kwargs.get("distance_metric", "mips").lower()
        metric_enum = METRIC_MAP.get(metric_str)
        if metric_enum is None:
            raise ValueError(f"Unsupported distance_metric '{metric_str}'.")

        complexity = build_kwargs.get("complexity", 64)
        graph_degree = build_kwargs.get("graph_degree", 32)
        final_index_ram_limit = build_kwargs.get("search_memory_maximum", 4.0)
        indexing_ram_budget = build_kwargs.get("build_memory_maximum", 8.0)
        num_threads = build_kwargs.get("num_threads", 8)
        pq_disk_bytes = build_kwargs.get("pq_disk_bytes", 0)
        codebook_prefix = ""
        is_recompute = build_kwargs.get("is_recompute", False)

        print(f"INFO: Building DiskANN index for {data.shape[0]} vectors with metric {metric_enum}...")
        
        try:
            with chdir(index_dir):
                diskannpy.build_disk_float_index(
                    metric_enum,
                    data_filename,
                    index_prefix,
                    complexity,
                    graph_degree,
                    final_index_ram_limit,
                    indexing_ram_budget,
                    num_threads,
                    pq_disk_bytes,
                    codebook_prefix
                )
            print(f"✅ DiskANN index built successfully at '{index_dir / index_prefix}'")
            if is_recompute:
                self._generate_passages_file(index_dir, index_prefix, **build_kwargs)
        except Exception as e:
            print(f"💥 ERROR: DiskANN index build failed. Exception: {e}")
            raise
        finally:
            temp_data_file = index_dir / data_filename
            if temp_data_file.exists():
                os.remove(temp_data_file)

class DiskannSearcher(LeannBackendSearcherInterface):
    def __init__(self, index_path: str, **kwargs):
        self.meta = kwargs.get("meta", {})
        if not self.meta:
            raise ValueError("DiskannSearcher requires metadata from .meta.json.")

        dimensions = self.meta.get("dimensions")
        if not dimensions:
            raise ValueError("Dimensions not found in Leann metadata.")
        
        self.distance_metric = self.meta.get("distance_metric", "mips").lower()
        metric_enum = METRIC_MAP.get(self.distance_metric)
        if metric_enum is None:
            raise ValueError(f"Unsupported distance_metric '{self.distance_metric}'.")

        self.embedding_model = self.meta.get("embedding_model")
        if not self.embedding_model:
            print("WARNING: embedding_model not found in meta.json. Recompute will fail if attempted.")

        path = Path(index_path)
        self.index_dir = path.parent
        self.index_prefix = path.stem
        
        num_threads = kwargs.get("num_threads", 8)
        num_nodes_to_cache = kwargs.get("num_nodes_to_cache", 0)
        self.zmq_port = kwargs.get("zmq_port", 6666)
        
        try:
            full_index_prefix = str(self.index_dir / self.index_prefix)
            self._index = diskannpy.StaticDiskFloatIndex(
                metric_enum, full_index_prefix, num_threads, num_nodes_to_cache, 1, self.zmq_port, "", ""
            )
            self.num_threads = num_threads
            self.embedding_server_manager = EmbeddingServerManager(
                backend_module_name="leann_backend_diskann.embedding_server"
            )
            print("✅ DiskANN index loaded successfully.")
        except Exception as e:
            print(f"💥 ERROR: Failed to load DiskANN index. Exception: {e}")
            raise

    def search(self, query: np.ndarray, top_k: int, **kwargs) -> Dict[str, Any]:
        complexity = kwargs.get("complexity", 256)
        beam_width = kwargs.get("beam_width", 4)
        
        USE_DEFERRED_FETCH = kwargs.get("USE_DEFERRED_FETCH", False)
        skip_search_reorder = kwargs.get("skip_search_reorder", False)
        recompute_beighbor_embeddings = kwargs.get("recompute_beighbor_embeddings", False)
        dedup_node_dis = kwargs.get("dedup_node_dis", False)
        prune_ratio = kwargs.get("prune_ratio", 0.0)
        batch_recompute = kwargs.get("batch_recompute", False)
        global_pruning = kwargs.get("global_pruning", False)
        port = kwargs.get("zmq_port", self.zmq_port)
        
        if recompute_beighbor_embeddings:
            print(f"INFO: DiskANN ZMQ mode enabled - ensuring embedding server is running")
            if not self.embedding_model:
                raise ValueError("Cannot use recompute_beighbor_embeddings without 'embedding_model' in meta.json.")

            passages_file = kwargs.get("passages_file")
            if not passages_file:
                potential_passages_file = self.index_dir / f"{self.index_prefix}.passages.json"
                if potential_passages_file.exists():
                    passages_file = str(potential_passages_file)
                    print(f"INFO: Automatically found passages file: {passages_file}")

            if not passages_file:
                raise RuntimeError(
                    f"Recompute mode is enabled, but no passages file was found. "
                    f"A '{self.index_prefix}.passages.json' file should exist in the index directory "
                    f"'{self.index_dir}'. Ensure you build the index with 'recompute=True'."
                )

            server_started = self.embedding_server_manager.start_server(
                port=self.zmq_port,
                model_name=self.embedding_model,
                distance_metric=self.distance_metric,
                passages_file=passages_file
            )
            
            if not server_started:
                raise RuntimeError(f"Failed to start DiskANN embedding server on port {self.zmq_port}")
        
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        if query.ndim == 1:
            query = np.expand_dims(query, axis=0)
            
        try:
            labels, distances = self._index.batch_search(
                query,
                query.shape[0],
                top_k,
                complexity,
                beam_width,
                self.num_threads,
                USE_DEFERRED_FETCH,
                skip_search_reorder,
                recompute_beighbor_embeddings,
                dedup_node_dis,
                prune_ratio,
                batch_recompute,
                global_pruning
            )
            return {"labels": labels, "distances": distances}
        except Exception as e:
            print(f"💥 ERROR: DiskANN search failed. Exception: {e}")
            batch_size = query.shape[0]
            return {"labels": np.full((batch_size, top_k), -1, dtype=np.int64), 
                   "distances": np.full((batch_size, top_k), float('inf'), dtype=np.float32)}
    
    def __del__(self):
        if hasattr(self, 'embedding_server_manager'):
            self.embedding_server_manager.stop_server()
