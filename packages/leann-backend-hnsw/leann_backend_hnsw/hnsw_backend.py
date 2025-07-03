import numpy as np
import os
import json
import struct
from pathlib import Path
from typing import Dict
import contextlib
import threading
import time
import atexit
import socket
import subprocess
import sys

# 文件: packages/leann-backend-hnsw/leann_backend_hnsw/hnsw_backend.py

# ... (其他 import 保持不变) ...

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
    "cosine": faiss.METRIC_INNER_PRODUCT,  # Will need normalization
    }

def _check_port(port: int) -> bool:
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

class HNSWEmbeddingServerManager:
    """
    HNSW-specific embedding server manager that handles the lifecycle of the embedding server process.
    Mirrors the DiskANN EmbeddingServerManager architecture.
    """
    def __init__(self):
        self.server_process = None
        self.server_port = None
        atexit.register(self.stop_server)

    def start_server(self, port=5556, model_name="sentence-transformers/all-mpnet-base-v2", passages_file=None):
        """
        Start the HNSW embedding server process.
        
        Args:
            port: ZMQ port for the server
            model_name: Name of the embedding model to use
            passages_file: Optional path to passages JSON file
        """
        if self.server_process and self.server_process.poll() is None:
            print(f"INFO: Reusing existing HNSW server process for this session (PID {self.server_process.pid})")
            return True
            
        # Check if port is already in use
        if _check_port(port):
            print(f"WARNING: Port {port} is already in use. Assuming an external HNSW server is running and connecting to it.")
            return True
        
        print(f"INFO: Starting session-level HNSW embedding server as a background process...")
        
        try:
            command = [
                sys.executable,
                "-m", "packages.leann-backend-hnsw.src.leann_backend_hnsw.hnsw_embedding_server",
                "--zmq-port", str(port), 
                "--model-name", model_name
            ]
            
            # Add passages file if provided
            if passages_file:
                command.extend(["--passages-file", str(passages_file)])
            
            project_root = Path(__file__).parent.parent.parent.parent
            print(f"INFO: Running HNSW command from project root: {project_root}")
            
            self.server_process = subprocess.Popen(
                command,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            self.server_port = port
            print(f"INFO: HNSW server process started with PID: {self.server_process.pid}")

            max_wait, wait_interval = 30, 0.5
            for _ in range(int(max_wait / wait_interval)):
                if _check_port(port):
                    print(f"✅ HNSW embedding server is up and ready for this session.")
                    log_thread = threading.Thread(target=self._log_monitor, daemon=True)
                    log_thread.start()
                    return True
                if self.server_process.poll() is not None:
                    print("❌ ERROR: HNSW server process terminated unexpectedly during startup.")
                    self._log_monitor()
                    return False
                time.sleep(wait_interval)
            
            print(f"❌ ERROR: HNSW server process failed to start listening within {max_wait} seconds.")
            self.stop_server()
            return False
                
        except Exception as e:
            print(f"❌ ERROR: Failed to start HNSW embedding server process: {e}")
            return False

    def _log_monitor(self):
        """Monitor server logs"""
        if not self.server_process:
            return
        try:
            if self.server_process.stdout:
                for line in iter(self.server_process.stdout.readline, ''):
                    print(f"[HNSWEmbeddingServer LOG]: {line.strip()}")
                self.server_process.stdout.close()
            if self.server_process.stderr:
                for line in iter(self.server_process.stderr.readline, ''):
                    print(f"[HNSWEmbeddingServer ERROR]: {line.strip()}")
                self.server_process.stderr.close()
        except Exception as e:
            print(f"HNSW Log monitor error: {e}")

    def stop_server(self):
        """Stop the HNSW embedding server process"""
        if self.server_process and self.server_process.poll() is None:
            print(f"INFO: Terminating HNSW session server process (PID: {self.server_process.pid})...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
                print("INFO: HNSW server process terminated.")
            except subprocess.TimeoutExpired:
                print("WARNING: HNSW server process did not terminate gracefully, killing it.")
                self.server_process.kill()
        self.server_process = None

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
             raise FileNotFoundError(f"Leann metadata file not found at {meta_path}. Cannot infer vector dimension for searcher.")
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(meta.get("embedding_model"))
            dimensions = model.get_sentence_embedding_dimension()
            kwargs['dimensions'] = dimensions
        except ImportError:
            raise ImportError("sentence-transformers is required to infer embedding dimensions. Please install it.")
        except Exception as e:
            raise RuntimeError(f"Could not load SentenceTransformer model to get dimension: {e}")

        return HNSWSearcher(index_path, **kwargs)

class HNSWBuilder(LeannBackendBuilderInterface):
    def __init__(self, **kwargs):
        self.build_params = kwargs

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
            
        build_kwargs = {**self.build_params, **kwargs}
        metric_str = build_kwargs.get("distance_metric", "mips").lower()
        metric_enum = get_metric_map().get(metric_str)
        print('metric_enum', metric_enum,' metric_str', metric_str)
        if metric_enum is None:
            raise ValueError(f"Unsupported distance_metric '{metric_str}'.")

        # HNSW parameters
        M = build_kwargs.get("M", 32)  # Max connections per layer
        efConstruction = build_kwargs.get("efConstruction", 200)  # Size of the dynamic candidate list for construction
        dim = data.shape[1]

        print(f"INFO: Building HNSW index for {data.shape[0]} vectors with metric {metric_enum}...")
        
        try:
            # Create HNSW index
            if metric_enum == faiss.METRIC_INNER_PRODUCT:
                index = faiss.IndexHNSWFlat(dim, M, metric_enum)
            else:  # L2
                index = faiss.IndexHNSWFlat(dim, M, metric_enum)
            
            # Set construction parameters
            index.hnsw.efConstruction = efConstruction
            
            # Normalize vectors if using cosine similarity
            if metric_str == "cosine":
                faiss.normalize_L2(data)
            
            # Add vectors to index
            print('starting to add vectors to index')
            index.add(data.shape[0], faiss.swig_ptr(data))
            print('vectors added to index')
            
            # Save index
            index_file = index_dir / f"{index_prefix}.index"
            faiss.write_index(index, str(index_file))
            
            print(f"✅ HNSW index built successfully at '{index_file}'")
            
        except Exception as e:
            print(f"💥 ERROR: HNSW index build failed. Exception: {e}")
            raise

class HNSWSearcher(LeannBackendSearcherInterface):
    def __init__(self, index_path: str, **kwargs):
        from . import faiss
        path = Path(index_path)
        index_dir = path.parent
        index_prefix = path.stem
        
        metric_str = kwargs.get("distance_metric", "mips").lower()
        metric_enum = get_metric_map().get(metric_str)
        if metric_enum is None:
            raise ValueError(f"Unsupported distance_metric '{metric_str}'.")
        
        dimensions = kwargs.get("dimensions")
        if not dimensions:
            raise ValueError("Vector dimension not provided to HNSWSearcher.")
        
        try:
            # Load FAISS HNSW index
            index_file = index_dir / f"{index_prefix}.index"
            if not index_file.exists():
                raise FileNotFoundError(f"HNSW index file not found at {index_file}")
            
            self._index = faiss.read_index(str(index_file))
            self.metric_str = metric_str
            self.embedding_server_manager = HNSWEmbeddingServerManager()
            print("✅ HNSW index loaded successfully.")
            
        except Exception as e:
            print(f"💥 ERROR: Failed to load HNSW index. Exception: {e}")
            raise

    def search(self, query: np.ndarray, top_k: int, **kwargs) -> Dict[str, any]:
        """Search using HNSW index with optional recompute functionality"""
        ef = kwargs.get("ef", 200)  # Size of the dynamic candidate list for search
        
        # Recompute parameters
        recompute_neighbor_embeddings = kwargs.get("recompute_neighbor_embeddings", False)
        zmq_port = kwargs.get("zmq_port", 5556)
        embedding_model = kwargs.get("embedding_model", "sentence-transformers/all-mpnet-base-v2")
        passages_file = kwargs.get("passages_file", None)
        
        if recompute_neighbor_embeddings:
            print(f"INFO: HNSW ZMQ mode enabled - ensuring embedding server is running")
            
            if not self.embedding_server_manager.start_server(zmq_port, embedding_model, passages_file):
                print(f"WARNING: Failed to start HNSW embedding server, falling back to standard search")
                kwargs['recompute_neighbor_embeddings'] = False
        
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        if query.ndim == 1:
            query = np.expand_dims(query, axis=0)
            
        # Normalize query if using cosine similarity
        if self.metric_str == "cosine":
            faiss.normalize_L2(query)
        
        try:
            # Set search parameter
            self._index.hnsw.efSearch = ef
            
            if recompute_neighbor_embeddings:
                # Use custom search with recompute
                # This would require implementing custom HNSW search logic
                # For now, we'll fall back to standard search
                print("WARNING: Recompute functionality for HNSW not yet implemented, using standard search")
                distances, labels = self._index.search(query, top_k)
            else:
                # Standard FAISS search
                distances, labels = self._index.search(query, top_k)
            
            return {"labels": labels, "distances": distances}
            
        except Exception as e:
            print(f"💥 ERROR: HNSW search failed. Exception: {e}")
            batch_size = query.shape[0]
            return {"labels": np.full((batch_size, top_k), -1, dtype=np.int64), 
                   "distances": np.full((batch_size, top_k), float('inf'), dtype=np.float32)}
    
    def __del__(self):
        if hasattr(self, 'embedding_server_manager'):
            self.embedding_server_manager.stop_server()