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

    def build(self, data: np.ndarray, index_path: str, **kwargs):
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
            
            if self.is_recompute:
                self._generate_passages_file(index_dir, index_prefix, **kwargs)
            
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
                # rename index_file to index_file.old
                index_file_old = index_file.with_suffix(".old")
                shutil.move(str(index_file), str(index_file_old))
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

    def _generate_passages_file(self, index_dir: Path, index_prefix: str, **kwargs):
        """Generate passages file for recompute mode"""
        try:
            chunks = kwargs.get('chunks', [])
            if not chunks:
                print("INFO: No chunks data provided, skipping passages file generation")
                return
            
            # Generate node_id to text mapping
            passages_data = {}
            for node_id, chunk in enumerate(chunks):
                passages_data[str(node_id)] = chunk["text"]
            
            # Save passages file
            passages_file = index_dir / f"{index_prefix}.passages.json"
            with open(passages_file, 'w', encoding='utf-8') as f:
                json.dump(passages_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Generated passages file for recompute mode at '{passages_file}' ({len(passages_data)} passages)")
            
        except Exception as e:
            print(f"💥 ERROR: Failed to generate passages file. Exception: {e}")
            # Don't raise - this is not critical for index building
            pass

class HNSWSearcher(LeannBackendSearcherInterface):
    def _get_index_storage_status(self, index_file: Path) -> tuple[bool, bool]:
        """
        Robustly determines the index's storage status by parsing the file.
        
        Returns:
            A tuple (is_compact, is_pruned).
        """
        if not index_file.exists():
            return False, False
        
        with open(index_file, 'rb') as f:
            try:
                def read_struct(fmt):
                    size = struct.calcsize(fmt)
                    data = f.read(size)
                    if len(data) != size:
                        raise EOFError(f"File ended unexpectedly reading struct fmt '{fmt}'.")
                    return struct.unpack(fmt, data)[0]

                def skip_vector(element_size):
                    count = read_struct('<Q')
                    f.seek(count * element_size, 1)

                # 1. Read up to the compact flag
                read_struct('<I'); read_struct('<i'); read_struct('<q'); 
                read_struct('<q'); read_struct('<q'); read_struct('<?')
                metric_type = read_struct('<i')
                if metric_type > 1: read_struct('<f')
                skip_vector(8); skip_vector(4); skip_vector(4)
                
                # 2. Check if there's a compact flag byte
                # Try to read the compact flag, but handle both old and new formats
                pos_before_compact = f.tell()
                try:
                    is_compact = read_struct('<?')
                    print(f"INFO: Detected is_compact flag as: {is_compact}")
                except (EOFError, struct.error):
                    # Old format without compact flag - assume non-compact
                    f.seek(pos_before_compact)
                    is_compact = False
                    print(f"INFO: No compact flag found, assuming is_compact=False")

                # 3. Read storage FourCC to determine if pruned
                is_pruned = False
                try:
                    if is_compact:
                        # For compact, we need to skip pointers and scalars to get to the storage FourCC
                        skip_vector(8) # level_ptr
                        skip_vector(8) # node_offsets
                        read_struct('<i'); read_struct('<i'); read_struct('<i');
                        read_struct('<i'); read_struct('<i')
                        storage_fourcc = read_struct('<I')
                    else:
                        # For non-compact, we need to read the flag probe, then skip offsets and neighbors
                        pos_before_probe = f.tell()
                        flag_byte = f.read(1)
                        if not (flag_byte and flag_byte == b'\x00'):
                            f.seek(pos_before_probe)
                        skip_vector(8); skip_vector(4) # offsets, neighbors
                        read_struct('<i'); read_struct('<i'); read_struct('<i');
                        read_struct('<i'); read_struct('<i')
                        # Now we are at the storage. The entire rest is storage blob.
                        storage_fourcc = struct.unpack('<I', f.read(4))[0]
                        
                    NULL_INDEX_FOURCC = int.from_bytes(b'null', 'little')
                    if storage_fourcc == NULL_INDEX_FOURCC:
                        is_pruned = True
                except (EOFError, struct.error):
                    # Cannot determine pruning status, assume not pruned
                    pass
                
                print(f"INFO: Detected is_pruned as: {is_pruned}")
                return is_compact, is_pruned

            except (EOFError, struct.error) as e:
                print(f"WARNING: Could not parse index file to detect format: {e}. Assuming standard, not pruned.")
                return False, False

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

        path = Path(index_path)
        self.index_dir = path.parent
        self.index_prefix = path.stem
        
        index_file = self.index_dir / f"{self.index_prefix}.index"
        if not index_file.exists():
            raise FileNotFoundError(f"HNSW index file not found at {index_file}")

        self.is_compact, self.is_pruned = self._get_index_storage_status(index_file)
        
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
        
        ef = kwargs.get("ef", 200)
        
        if self.is_pruned:
            print(f"INFO: Index is pruned - ensuring embedding server is running for recompute.")
            if not self.embedding_model:
                raise ValueError("Cannot use recompute mode without 'embedding_model' in meta.json.")

            passages_file = kwargs.get("passages_file")
            if not passages_file:
                potential_passages_file = self.index_dir / f"{self.index_prefix}.passages.json"
                if potential_passages_file.exists():
                    passages_file = str(potential_passages_file)
                    print(f"INFO: Automatically found passages file: {passages_file}")
                else:
                    raise RuntimeError(f"FATAL: Index is pruned but no passages file found.")

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
            params = faiss.SearchParametersHNSW()
            params.efSearch = ef
            params.zmq_port = kwargs.get("zmq_port", self.zmq_port)
            
            batch_size = query.shape[0]
            distances = np.empty((batch_size, top_k), dtype=np.float32)
            labels = np.empty((batch_size, top_k), dtype=np.int64)
            
            self._index.search(query.shape[0], faiss.swig_ptr(query), top_k, faiss.swig_ptr(distances), faiss.swig_ptr(labels), params)
            
            return {"labels": labels, "distances": distances}
            
        except Exception as e:
            print(f"💥 ERROR: HNSW search failed. Exception: {e}")
            raise
    
    def __del__(self):
        if hasattr(self, 'embedding_server_manager'):
            self.embedding_server_manager.stop_server()
