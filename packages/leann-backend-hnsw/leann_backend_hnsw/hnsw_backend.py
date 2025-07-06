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

    def start_server(self, port=5556, model_name="sentence-transformers/all-mpnet-base-v2", passages_file=None, distance_metric="mips"):
        """
        Start the HNSW embedding server process.
        
        Args:
            port: ZMQ port for the server
            model_name: Name of the embedding model to use
            passages_file: Optional path to passages JSON file
            distance_metric: The distance metric to use
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
                "-m", "leann_backend_hnsw.hnsw_embedding_server",
                "--zmq-port", str(port), 
                "--model-name", model_name,
                "--distance-metric", distance_metric
            ]
            
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

        dimensions = meta.get("dimensions")
        if not dimensions:
            raise ValueError("Dimensions not found in Leann metadata. Please rebuild the index with a newer version of Leann.")
        
        kwargs['dimensions'] = dimensions
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
        path = Path(index_path)
        index_dir = path.parent
        index_prefix = path.stem
        
        # Store configuration and paths for later use
        self.config = kwargs.copy()
        self.config["index_path"] = index_path
        self.index_dir = index_dir
        self.index_prefix = index_prefix
        
        metric_str = self.config.get("distance_metric", "mips").lower()
        metric_enum = get_metric_map().get(metric_str)
        if metric_enum is None:
            raise ValueError(f"Unsupported distance_metric '{metric_str}'.")
        
        dimensions = self.config.get("dimensions")
        if not dimensions:
            raise ValueError("Vector dimension not provided to HNSWSearcher.")
        
        index_file = index_dir / f"{index_prefix}.index"
        if not index_file.exists():
            raise FileNotFoundError(f"HNSW index file not found at {index_file}")

        self.is_compact, self.is_pruned = self._get_index_storage_status(index_file)
        
        # Validate configuration constraints
        if not self.is_compact and self.config.get("is_skip_neighbors", False):
            raise ValueError("is_skip_neighbors can only be used with is_compact=True")
        
        if self.config.get("is_recompute", False) and self.config.get("external_storage_path"):
            raise ValueError("Cannot use both is_recompute and external_storage_path simultaneously")
            
        hnsw_config = faiss.HNSWIndexConfig()
        hnsw_config.is_compact = self.is_compact
        
        # Apply additional configuration options with strict validation
        hnsw_config.is_skip_neighbors = self.config.get("is_skip_neighbors", False)
        hnsw_config.is_recompute = self.is_pruned or self.config.get("is_recompute", False)
        hnsw_config.disk_cache_ratio = self.config.get("disk_cache_ratio", 0.0)
        hnsw_config.external_storage_path = self.config.get("external_storage_path")
        hnsw_config.zmq_port = self.config.get("zmq_port", 5557)
        
        if self.is_pruned and not hnsw_config.is_recompute:
            raise RuntimeError("Index is pruned (embeddings removed) but recompute is disabled. This is impossible - recompute must be enabled for pruned indices.")
        
        print(f"INFO: Loading index with is_compact={self.is_compact}, is_pruned={self.is_pruned}")
        print(f"INFO: Config - skip_neighbors={hnsw_config.is_skip_neighbors}, recompute={hnsw_config.is_recompute}")
        
        self._index = faiss.read_index(str(index_file), faiss.IO_FLAG_MMAP, hnsw_config)
        
        if self.is_compact:
            print("✅ Compact CSR format HNSW index loaded successfully.")
        else:
            print("✅ Standard HNSW index loaded successfully.")

        self.metric_str = metric_str
        self.embedding_server_manager = HNSWEmbeddingServerManager()

    def _get_index_file(self, index_dir: Path, index_prefix: str) -> Path:
        """Get the appropriate index file path based on format"""
        # We always use the same filename now, format is detected internally
        return index_dir / f"{index_prefix}.index"

    def search(self, query: np.ndarray, top_k: int, **kwargs) -> Dict[str, Any]:
        """Search using HNSW index with optional recompute functionality"""
        from . import faiss
        # Merge config with search-time kwargs
        search_config = self.config.copy()
        search_config.update(kwargs)
        
        ef = search_config.get("ef", 200)  # Size of the dynamic candidate list for search
        
        # Recompute parameters
        zmq_port = search_config.get("zmq_port", 5557)
        embedding_model = search_config.get("embedding_model", "sentence-transformers/all-mpnet-base-v2")
        passages_file = search_config.get("passages_file", None)
        
        # For recompute mode, try to find the passages file automatically
        if self.is_pruned and not passages_file:
            potential_passages_file = self.index_dir / f"{self.index_prefix}.passages.json"
            print(f"DEBUG: Checking for passages file at: {potential_passages_file}")
            if potential_passages_file.exists():
                passages_file = str(potential_passages_file)
                print(f"INFO: Found passages file for recompute mode: {passages_file}")
            else:
                print(f"WARNING: No passages file found for recompute mode at {potential_passages_file}")
        
        # If index is pruned (embeddings removed), we MUST start embedding server for recompute
        if self.is_pruned:
            print(f"INFO: Index is pruned - starting embedding server for recompute")
            
            # CRITICAL: Check passages file exists - fail fast if not
            if not passages_file:
                raise RuntimeError(f"FATAL: Index is pruned but no passages file found. Cannot proceed with recompute mode.")
            
            # Check if server is already running first
            if _check_port(zmq_port):
                print(f"INFO: Embedding server already running on port {zmq_port}")
            else:
                if not self.embedding_server_manager.start_server(zmq_port, embedding_model, passages_file, self.metric_str):
                    raise RuntimeError(f"Failed to start HNSW embedding server on port {zmq_port}")
                
                # Give server extra time to fully initialize
                print(f"INFO: Waiting for embedding server to fully initialize...")
                time.sleep(3)
                
                # Final verification
                if not _check_port(zmq_port):
                    raise RuntimeError(f"Embedding server failed to start listening on port {zmq_port}")
        else:
            print(f"INFO: Index has embeddings stored - no recompute needed")
        
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
            
            # Prepare output arrays for the older FAISS SWIG API
            batch_size = query.shape[0]
            distances = np.empty((batch_size, top_k), dtype=np.float32)
            labels = np.empty((batch_size, top_k), dtype=np.int64)
            
            # Use standard FAISS search - recompute is handled internally by FAISS
            self._index.search(query.shape[0], faiss.swig_ptr(query), top_k, faiss.swig_ptr(distances), faiss.swig_ptr(labels))
            
            return {"labels": labels, "distances": distances}
            
        except Exception as e:
            print(f"💥 ERROR: HNSW search failed. Exception: {e}")
            raise
    
    def __del__(self):
        if hasattr(self, 'embedding_server_manager'):
            self.embedding_server_manager.stop_server()