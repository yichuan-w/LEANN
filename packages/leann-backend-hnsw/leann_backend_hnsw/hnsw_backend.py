import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, Any, List
import contextlib
import threading
import time
import atexit
import socket
import subprocess
import sys
import pickle

from leann.embedding_server_manager import EmbeddingServerManager
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
        path = Path(index_path)
        meta_path = path.parent / f"{path.name}.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Leann metadata file not found at {meta_path}.")
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        kwargs['meta'] = meta
        return HNSWSearcher(index_path, **kwargs)

class HNSWBuilder(LeannBackendBuilderInterface):
    def __init__(self, **kwargs):
        self.build_params = kwargs.copy()
        
        # --- Configuration defaults with standardized names ---
        self.is_compact = self.build_params.setdefault("is_compact", True)
        self.is_recompute = self.build_params.setdefault("is_recompute", True)
        
        # --- Additional Options ---
        self.is_skip_neighbors = self.build_params.setdefault("is_skip_neighbors", False) 
        self.disk_cache_ratio = self.build_params.setdefault("disk_cache_ratio", 0.0)
        self.external_storage_path = self.build_params.get("external_storage_path", None)
        
        # --- Standard HNSW parameters ---
        self.M = self.build_params.setdefault("M", 32)
        self.efConstruction = self.build_params.setdefault("efConstruction", 200)
        self.distance_metric = self.build_params.setdefault("distance_metric", "mips")
        self.dimensions = self.build_params.get("dimensions")

        if self.is_skip_neighbors and not self.is_compact:
            raise ValueError("is_skip_neighbors can only be used with is_compact=True")
        
        if self.is_recompute and not self.is_compact:
            raise ValueError("is_recompute requires is_compact=True for efficiency")

    def build(self, data: np.ndarray, ids: List[str], index_path: str, **kwargs):
        """Build HNSW index using FAISS"""
        from . import faiss
        
        path = Path(index_path)
        index_dir = path.parent
        index_prefix = path.stem

        index_dir.mkdir(parents=True, exist_ok=True)

        if data.dtype != np.float32:
            data = data.astype(np.float32)
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
            
        # Create label map: integer -> string_id
        label_map = {i: str_id for i, str_id in enumerate(ids)}
        label_map_file = index_dir / "leann.labels.map"
        with open(label_map_file, 'wb') as f:
            pickle.dump(label_map, f)
            
        metric_str = self.distance_metric.lower()
        metric_enum = get_metric_map().get(metric_str)
        if metric_enum is None:
            raise ValueError(f"Unsupported distance_metric '{metric_str}'.")

        M = self.M
        efConstruction = self.efConstruction
        dim = self.dimensions
        if not dim:
            dim = data.shape[1]

        print(f"INFO: Building HNSW index for {data.shape[0]} vectors with metric {metric_enum}...")
        
        try:
            index = faiss.IndexHNSWFlat(dim, M, metric_enum)
            index.hnsw.efConstruction = efConstruction
            
            if metric_str == "cosine":
                faiss.normalize_L2(data)
            
            index.add(data.shape[0], faiss.swig_ptr(data))
            
            index_file = index_dir / f"{index_prefix}.index"
            faiss.write_index(index, str(index_file))
            
            print(f"✅ HNSW index built successfully at '{index_file}'")

            if self.is_compact:
                self._convert_to_csr(index_file)
            
        except Exception as e:
            print(f"💥 ERROR: HNSW index build failed. Exception: {e}")
            raise

    def _convert_to_csr(self, index_file: Path):
        """Convert built index to CSR format"""
        try:
            mode_str = "CSR-pruned" if self.is_recompute else "CSR-standard"
            print(f"INFO: Converting HNSW index to {mode_str} format...")
            
            csr_temp_file = index_file.with_suffix(".csr.tmp")
            
            success = convert_hnsw_graph_to_csr(
                str(index_file), 
                str(csr_temp_file),
                prune_embeddings=self.is_recompute
            )
            
            if success:
                print("✅ CSR conversion successful.")
                import shutil
                shutil.move(str(csr_temp_file), str(index_file))
                print(f"INFO: Replaced original index with {mode_str} version at '{index_file}'")
            else:
                # Clean up and fail fast
                if csr_temp_file.exists():
                    os.remove(csr_temp_file)
                raise RuntimeError("CSR conversion failed - cannot proceed with compact format")
                
        except Exception as e:
            print(f"💥 ERROR: CSR conversion failed. Exception: {e}")
            raise


class HNSWSearcher(LeannBackendSearcherInterface):
    def _get_index_storage_status_from_meta(self) -> tuple[bool, bool]:
        """
        Get storage status from metadata with sensible defaults.
        
        Returns:
            A tuple (is_compact, is_pruned).
        """
        # Check if metadata has these flags
        is_compact = self.meta.get('is_compact', True)  # Default to compact (CSR format)
        is_pruned = self.meta.get('is_pruned', True)    # Default to pruned (embeddings removed)
        
        print(f"INFO: Storage status from metadata: is_compact={is_compact}, is_pruned={is_pruned}")
        return is_compact, is_pruned

    def __init__(self, index_path: str, **kwargs):
        from . import faiss
        self.meta = kwargs.get("meta", {})
        if not self.meta:
            raise ValueError("HNSWSearcher requires metadata from .meta.json.")

        self.dimensions = self.meta.get("dimensions")
        if not self.dimensions:
            raise ValueError("Dimensions not found in Leann metadata.")
            
        self.distance_metric = self.meta.get("distance_metric", "mips").lower()
        metric_enum = get_metric_map().get(self.distance_metric)
        if metric_enum is None:
            raise ValueError(f"Unsupported distance_metric '{self.distance_metric}'.")

        self.embedding_model = self.meta.get("embedding_model")
        if not self.embedding_model:
            print("WARNING: embedding_model not found in meta.json. Recompute will fail if attempted.")

        # Check for embedding model override (not allowed)
        if 'embedding_model' in kwargs and kwargs['embedding_model'] != self.embedding_model:
            raise ValueError(f"Embedding model override not allowed. Index uses '{self.embedding_model}', but got '{kwargs['embedding_model']}'")

        path = Path(index_path)
        self.index_dir = path.parent
        self.index_prefix = path.stem
        
        # Load the label map
        label_map_file = self.index_dir / "leann.labels.map"
        if not label_map_file.exists():
            raise FileNotFoundError(f"Label map file not found: {label_map_file}")
        
        with open(label_map_file, 'rb') as f:
            self.label_map = pickle.load(f)
        
        index_file = self.index_dir / f"{self.index_prefix}.index"
        if not index_file.exists():
            raise FileNotFoundError(f"HNSW index file not found at {index_file}")

        # Get storage status from metadata with user overrides
        self.is_compact, self.is_pruned = self._get_index_storage_status_from_meta()
        
        # Allow override of storage parameters via kwargs
        if 'is_compact' in kwargs:
            self.is_compact = kwargs['is_compact']
        if 'is_pruned' in kwargs:
            self.is_pruned = kwargs['is_pruned']
        
        # Validate configuration constraints
        if not self.is_compact and kwargs.get("is_skip_neighbors", False):
            raise ValueError("is_skip_neighbors can only be used with is_compact=True")
        
        if kwargs.get("is_recompute", False) and kwargs.get("external_storage_path"):
            raise ValueError("Cannot use both is_recompute and external_storage_path simultaneously")
            
        hnsw_config = faiss.HNSWIndexConfig()
        hnsw_config.is_compact = self.is_compact
        
        # Apply additional configuration options with strict validation
        hnsw_config.is_skip_neighbors = kwargs.get("is_skip_neighbors", False)
        hnsw_config.is_recompute = self.is_pruned or kwargs.get("is_recompute", False)
        hnsw_config.disk_cache_ratio = kwargs.get("disk_cache_ratio", 0.0)
        hnsw_config.external_storage_path = kwargs.get("external_storage_path")
        
        self.zmq_port = kwargs.get("zmq_port", 5557)
        
        if self.is_pruned and not hnsw_config.is_recompute:
            raise RuntimeError("Index is pruned (embeddings removed) but recompute is disabled. This is impossible - recompute must be enabled for pruned indices.")
        
        print(f"INFO: Loading index with is_compact={self.is_compact}, is_pruned={self.is_pruned}")
        print(f"INFO: Config - skip_neighbors={hnsw_config.is_skip_neighbors}, recompute={hnsw_config.is_recompute}")
        
        self._index = faiss.read_index(str(index_file), faiss.IO_FLAG_MMAP, hnsw_config)
        
        if self.is_compact:
            print("✅ Compact CSR format HNSW index loaded successfully.")
        else:
            print("✅ Standard HNSW index loaded successfully.")

        self.embedding_server_manager = EmbeddingServerManager(
            backend_module_name="leann_backend_hnsw.hnsw_embedding_server"
        )

    def search(self, query: np.ndarray, top_k: int, **kwargs) -> Dict[str, Any]:
        """Search using HNSW index with optional recompute functionality"""
        from . import faiss
        
        ef = kwargs.get("ef", 128)
        
        if self.is_pruned:
            print(f"INFO: Index is pruned - ensuring embedding server is running for recompute.")
            if not self.embedding_model:
                raise ValueError("Cannot use recompute mode without 'embedding_model' in meta.json.")

            passages_file = kwargs.get("passages_file")
            if not passages_file:
                # Pass the metadata file instead of a single passage file
                meta_file_path = self.index_dir / f"{self.index_prefix}.index.meta.json"
                if meta_file_path.exists():
                    passages_file = str(meta_file_path)
                    print(f"INFO: Using metadata file for lazy loading: {passages_file}")
                else:
                    raise RuntimeError(f"FATAL: Index is pruned but metadata file not found: {meta_file_path}")

            zmq_port = kwargs.get("zmq_port", 5557)
            server_started = self.embedding_server_manager.start_server(
                port=zmq_port,
                model_name=self.embedding_model,
                passages_file=passages_file,
                distance_metric=self.distance_metric
            )
            if not server_started:
                raise RuntimeError(f"Failed to start HNSW embedding server on port {zmq_port}")
        
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        if query.ndim == 1:
            query = np.expand_dims(query, axis=0)
            
        if self.distance_metric == "cosine":
            faiss.normalize_L2(query)
        
        try:
            self._index.hnsw.efSearch = ef
            params = faiss.SearchParametersHNSW()
            params.zmq_port = kwargs.get("zmq_port", self.zmq_port)
            params.efSearch = ef
            params.beam_size = 2  # Match research system beam_size
            
            batch_size = query.shape[0]
            distances = np.empty((batch_size, top_k), dtype=np.float32)
            labels = np.empty((batch_size, top_k), dtype=np.int64)
            
            self._index.search(query.shape[0], faiss.swig_ptr(query), top_k, faiss.swig_ptr(distances), faiss.swig_ptr(labels), params)
            
            # 🐛 DEBUG: Print raw faiss results before conversion
            print(f"🔍 DEBUG HNSW Search Results:")
            print(f"  Query shape: {query.shape}")
            print(f"  Top_k: {top_k}")
            print(f"  Raw faiss indices: {labels[0] if len(labels) > 0 else 'No results'}")
            print(f"  Raw faiss distances: {distances[0] if len(distances) > 0 else 'No results'}")
            
            # Convert integer labels to string IDs
            string_labels = []
            for batch_idx, batch_labels in enumerate(labels):
                batch_string_labels = []
                print(f"  Batch {batch_idx} conversion:")
                for i, int_label in enumerate(batch_labels):
                    if int_label in self.label_map:
                        string_id = self.label_map[int_label]
                        batch_string_labels.append(string_id)
                        print(f"    faiss[{int_label}] -> passage_id '{string_id}' (distance: {distances[batch_idx][i]:.4f})")
                    else:
                        unknown_id = f"unknown_{int_label}"
                        batch_string_labels.append(unknown_id)
                        print(f"    faiss[{int_label}] -> {unknown_id} (NOT FOUND in label_map!)")
                string_labels.append(batch_string_labels)
            
            return {"labels": string_labels, "distances": distances}
            
        except Exception as e:
            print(f"💥 ERROR: HNSW search failed. Exception: {e}")
            raise
    
    def __del__(self):
        if hasattr(self, 'embedding_server_manager'):
            self.embedding_server_manager.stop_server()
