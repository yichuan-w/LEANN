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

def _write_vectors_to_bin(data: np.ndarray, file_path: str):
    num_vectors, dim = data.shape
    with open(file_path, 'wb') as f:
        f.write(struct.pack('I', num_vectors))
        f.write(struct.pack('I', dim))
        f.write(data.tobytes())

def _check_port(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

class EmbeddingServerManager:
    def __init__(self):
        self.server_process = None
        self.server_port = None
        atexit.register(self.stop_server)

    def start_server(self, port=5555, model_name="sentence-transformers/all-mpnet-base-v2"):
        if self.server_process and self.server_process.poll() is None:
            print(f"INFO: Reusing existing server process for this session (PID {self.server_process.pid})")
            return True
            
        # 检查端口是否已被其他无关进程占用
        if _check_port(port):
            print(f"WARNING: Port {port} is already in use. Assuming an external server is running and connecting to it.")
            return True
        
        print(f"INFO: Starting session-level embedding server as a background process...")
        
        try:
            command = [
                sys.executable,
                "-m", "packages.leann-backend-diskann.leann_backend_diskann.embedding_server",
                "--zmq-port", str(port), 
                "--model-name", model_name
            ]
            project_root = Path(__file__).parent.parent.parent.parent
            print(f"INFO: Running command from project root: {project_root}")
            self.server_process = subprocess.Popen(
                command,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            self.server_port = port
            print(f"INFO: Server process started with PID: {self.server_process.pid}")

            max_wait, wait_interval = 30, 0.5
            for _ in range(int(max_wait / wait_interval)):
                if _check_port(port):
                    print(f"✅ Embedding server is up and ready for this session.")
                    log_thread = threading.Thread(target=self._log_monitor, daemon=True)
                    log_thread.start()
                    return True
                if self.server_process.poll() is not None:
                    print("❌ ERROR: Server process terminated unexpectedly during startup.")
                    self._log_monitor()
                    return False
                time.sleep(wait_interval)
            
            print(f"❌ ERROR: Server process failed to start listening within {max_wait} seconds.")
            self.stop_server()
            return False
                
        except Exception as e:
            print(f"❌ ERROR: Failed to start embedding server process: {e}")
            return False

    def _log_monitor(self):
        if not self.server_process:
            return
        try:
            if self.server_process.stdout:
                for line in iter(self.server_process.stdout.readline, ''):
                    print(f"[EmbeddingServer LOG]: {line.strip()}")
                self.server_process.stdout.close()
            if self.server_process.stderr:
                for line in iter(self.server_process.stderr.readline, ''):
                    print(f"[EmbeddingServer ERROR]: {line.strip()}")
                self.server_process.stderr.close()
        except Exception as e:
            print(f"Log monitor error: {e}")

    def stop_server(self):
        if self.server_process and self.server_process.poll() is None:
            print(f"INFO: Terminating session server process (PID: {self.server_process.pid})...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
                print("INFO: Server process terminated.")
            except subprocess.TimeoutExpired:
                print("WARNING: Server process did not terminate gracefully, killing it.")
                self.server_process.kill()
        self.server_process = None

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
            raise FileNotFoundError(f"Leann metadata file not found at {meta_path}. Cannot infer vector dimension for searcher.")
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        dimensions = meta.get("dimensions")
        if not dimensions:
            raise ValueError("Dimensions not found in Leann metadata. Please rebuild the index with a newer version of Leann.")
        
        kwargs['dimensions'] = dimensions
        return DiskannSearcher(index_path, **kwargs)

class DiskannBuilder(LeannBackendBuilderInterface):
    def __init__(self, **kwargs):
        self.build_params = kwargs

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
        except Exception as e:
            print(f"💥 ERROR: DiskANN index build failed. Exception: {e}")
            raise
        finally:
            temp_data_file = index_dir / data_filename
            if temp_data_file.exists():
                os.remove(temp_data_file)

class DiskannSearcher(LeannBackendSearcherInterface):
    def __init__(self, index_path: str, **kwargs):
        path = Path(index_path)
        index_dir = path.parent
        index_prefix = path.stem
        metric_str = kwargs.get("distance_metric", "mips").lower()
        metric_enum = METRIC_MAP.get(metric_str)
        if metric_enum is None:
            raise ValueError(f"Unsupported distance_metric '{metric_str}'.")
        
        num_threads = kwargs.get("num_threads", 8)
        num_nodes_to_cache = kwargs.get("num_nodes_to_cache", 0)
        dimensions = kwargs.get("dimensions")
        if not dimensions:
            raise ValueError("Vector dimension not provided to DiskannSearcher.")
        
        try:
            full_index_prefix = str(index_dir / index_prefix)
            self._index = diskannpy.StaticDiskFloatIndex(
                metric_enum, full_index_prefix, num_threads, num_nodes_to_cache, 1, "", ""
            )
            self.num_threads = num_threads
            self.embedding_server_manager = EmbeddingServerManager()
            print("✅ DiskANN index loaded successfully.")
        except Exception as e:
            print(f"💥 ERROR: Failed to load DiskANN index. Exception: {e}")
            raise

    def search(self, query: np.ndarray, top_k: int, **kwargs) -> Dict[str, any]:
        complexity = kwargs.get("complexity", 32)
        beam_width = kwargs.get("beam_width", 4)
        
        USE_DEFERRED_FETCH = kwargs.get("USE_DEFERRED_FETCH", False)
        skip_search_reorder = kwargs.get("skip_search_reorder", False)
        recompute_beighbor_embeddings = kwargs.get("recompute_beighbor_embeddings", False)
        dedup_node_dis = kwargs.get("dedup_node_dis", False)
        prune_ratio = kwargs.get("prune_ratio", 0.0)
        batch_recompute = kwargs.get("batch_recompute", False)
        global_pruning = kwargs.get("global_pruning", False)
        
        if recompute_beighbor_embeddings:
            print(f"INFO: DiskANN ZMQ mode enabled - ensuring embedding server is running")
            zmq_port = kwargs.get("zmq_port", 5555)
            embedding_model = kwargs.get("embedding_model", "sentence-transformers/all-mpnet-base-v2")
            
            if not self.embedding_server_manager.start_server(zmq_port, embedding_model):
                print(f"WARNING: Failed to start embedding server, falling back to PQ computation")
                kwargs['recompute_beighbor_embeddings'] = False
        
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