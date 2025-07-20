#!/usr/bin/env python3
"""
Embedding server for leann-backend-diskann - Fixed ZMQ REQ-REP pattern
"""

import pickle
import argparse
import time
import json
from typing import Dict, Any, Optional, Union

from transformers import AutoTokenizer, AutoModel
import os
from contextlib import contextmanager
import zmq
import numpy as np
import msgpack
from pathlib import Path
import logging

RED = "\033[91m"

# Set up logging based on environment variable
LOG_LEVEL = os.getenv('LEANN_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
RESET = "\033[0m"

# --- New Passage Loader from HNSW backend ---
class SimplePassageLoader:
    """
    Simple passage loader that replaces config.py dependencies
    """
    def __init__(self, passages_data: Optional[Dict[str, Any]] = None):
        self.passages_data = passages_data or {}
        self._meta_path = ''
    
    def __getitem__(self, passage_id: Union[str, int]) -> Dict[str, str]:
        """Get passage by ID"""
        str_id = str(passage_id)
        if str_id in self.passages_data:
            return {"text": self.passages_data[str_id]}
        else:
            # Return empty text for missing passages
            return {"text": ""}
    
    def __len__(self) -> int:
        return len(self.passages_data)
    
    def keys(self):
        return self.passages_data.keys()

def load_passages_from_metadata(meta_file: str) -> SimplePassageLoader:
    """
    Load passages using metadata file with PassageManager for lazy loading
    """
    # Load metadata to get passage sources
    with open(meta_file, 'r') as f:
        meta = json.load(f)
    
    # Import PassageManager dynamically to avoid circular imports
    import sys
    from pathlib import Path
    
    # Find the leann package directory relative to this file
    current_dir = Path(__file__).parent
    leann_core_path = current_dir.parent.parent / "leann-core" / "src"
    sys.path.insert(0, str(leann_core_path))
    
    try:
        from leann.api import PassageManager
        passage_manager = PassageManager(meta['passage_sources'])
    finally:
        sys.path.pop(0)
    
    # Load label map 
    passages_dir = Path(meta_file).parent
    label_map_file = passages_dir / "leann.labels.map"
    
    if label_map_file.exists():
        import pickle
        with open(label_map_file, 'rb') as f:
            label_map = pickle.load(f)
        print(f"Loaded label map with {len(label_map)} entries")
    else:
        raise FileNotFoundError(f"Label map file not found: {label_map_file}")
    
    print(f"Initialized lazy passage loading for {len(label_map)} passages")
    
    class LazyPassageLoader(SimplePassageLoader):
        def __init__(self, passage_manager, label_map):
            self.passage_manager = passage_manager
            self.label_map = label_map
            # Initialize parent with empty data
            super().__init__({})
        
        def __getitem__(self, passage_id: Union[str, int]) -> Dict[str, str]:
            """Get passage by ID with lazy loading"""
            try:
                int_id = int(passage_id)
                if int_id in self.label_map:
                    string_id = self.label_map[int_id]
                    passage_data = self.passage_manager.get_passage(string_id)
                    if passage_data and passage_data.get("text"):
                        return {"text": passage_data["text"]}
                    else:
                        raise RuntimeError(f"FATAL: Empty text for ID {int_id} -> {string_id}")
                else:
                    raise RuntimeError(f"FATAL: ID {int_id} not found in label_map")
            except Exception as e:
                raise RuntimeError(f"FATAL: Exception getting passage {passage_id}: {e}")
        
        def __len__(self) -> int:
            return len(self.label_map)
        
        def keys(self):
            return self.label_map.keys()
    
    loader = LazyPassageLoader(passage_manager, label_map)
    loader._meta_path = meta_file
    return loader

def load_passages_from_file(passages_file: str) -> SimplePassageLoader:
    """
    Load passages from a JSONL file with label map support
    Expected format: {"id": "passage_id", "text": "passage_text", "metadata": {...}} (one per line)
    """
    
    if not os.path.exists(passages_file):
        raise FileNotFoundError(f"Passages file {passages_file} not found.")
    
    if not passages_file.endswith('.jsonl'):
        raise ValueError(f"Expected .jsonl file format, got: {passages_file}")
    
    # Load label map (int -> string_id)
    passages_dir = Path(passages_file).parent
    label_map_file = passages_dir / "leann.labels.map"
    
    label_map = {}
    if label_map_file.exists():
        with open(label_map_file, 'rb') as f:
            label_map = pickle.load(f)
        print(f"Loaded label map with {len(label_map)} entries")
    else:
        raise FileNotFoundError(f"Label map file not found: {label_map_file}")
    
    # Load passages by string ID
    string_id_passages = {}
    with open(passages_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                passage = json.loads(line)
                string_id_passages[passage['id']] = passage['text']
    
    # Create int ID -> text mapping using label map
    passages_data = {}
    for int_id, string_id in label_map.items():
        if string_id in string_id_passages:
            passages_data[str(int_id)] = string_id_passages[string_id]
        else:
            print(f"WARNING: String ID {string_id} from label map not found in passages")
    
    print(f"Loaded {len(passages_data)} passages from JSONL file {passages_file} using label map")
    return SimplePassageLoader(passages_data)

def create_embedding_server_thread(
    zmq_port=5555,
    model_name="sentence-transformers/all-mpnet-base-v2",
    max_batch_size=128,
    passages_file: Optional[str] = None,
    embedding_mode: str = "sentence-transformers",
    enable_warmup: bool = False,
):
    """
    Create and run embedding server in the current thread
    This function is designed to be called in a separate thread
    """
    logger.info(f"Initializing embedding server thread on port {zmq_port}")
    
    try:
        # Check if port is already occupied
        import socket
        def check_port(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0

        if check_port(zmq_port):
            print(f"{RED}Port {zmq_port} is already in use{RESET}")
            return

        # Auto-detect mode based on model name if not explicitly set
        if embedding_mode == "sentence-transformers" and model_name.startswith("text-embedding-"):
            embedding_mode = "openai"
        
        if embedding_mode == "mlx":
            from leann.api import compute_embeddings_mlx
            import torch
            logger.info("Using MLX for embeddings")
            # Set device to CPU for compatibility with DeviceTimer class
            device = torch.device("cpu")
            cuda_available = False
            mps_available = False
        elif embedding_mode == "openai":
            from leann.api import compute_embeddings_openai
            import torch
            logger.info("Using OpenAI API for embeddings")
            # Set device to CPU for compatibility with DeviceTimer class
            device = torch.device("cpu")
            cuda_available = False
            mps_available = False
        elif embedding_mode == "sentence-transformers":
            # Initialize model
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            import torch

            # Select device
            mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                device = torch.device("cuda")
                logger.info("Using CUDA device")
            elif mps_available:
                device = torch.device("mps")
                logger.info("Using MPS device (Apple Silicon)")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU device")
            
            # Load model
            logger.info(f"Loading model {model_name}")
            model = AutoModel.from_pretrained(model_name).to(device).eval()

            # Optimize model
            if cuda_available or mps_available:
                try:
                    model = model.half()
                    model = torch.compile(model)
                    logger.info(f"Using FP16 precision with model: {model_name}")
                except Exception as e:
                    print(f"WARNING: Model optimization failed: {e}")
        else:
            raise ValueError(f"Unsupported embedding mode: {embedding_mode}. Supported modes: sentence-transformers, mlx, openai")

        # Load passages from file if provided
        if passages_file and os.path.exists(passages_file):
            # Check if it's a metadata file or a single passages file
            if passages_file.endswith('.meta.json'):
                passages = load_passages_from_metadata(passages_file)
            else:
                # Try to find metadata file in same directory
                passages_dir = Path(passages_file).parent
                meta_files = list(passages_dir.glob("*.meta.json"))
                if meta_files:
                    print(f"Found metadata file: {meta_files[0]}, using lazy loading")
                    passages = load_passages_from_metadata(str(meta_files[0]))
                else:
                    # Fallback to original single file loading (will cause warnings)
                    print("WARNING: No metadata file found, using single file loading (may cause missing passage warnings)")
                    passages = load_passages_from_file(passages_file)
        else:
            print("WARNING: No passages file provided or file not found. Using an empty passage loader.")
            passages = SimplePassageLoader()

        logger.info(f"Loaded {len(passages)} passages.")

        def client_warmup(zmq_port):
            """Perform client-side warmup for DiskANN server"""
            time.sleep(2)
            print(f"Performing client-side warmup with model {model_name}...")
            
            # Get actual passage IDs from the loaded passages
            sample_ids = []
            if hasattr(passages, 'keys') and len(passages) > 0:
                available_ids = list(passages.keys())
                # Take up to 5 actual IDs, but at least 1
                sample_ids = available_ids[:min(5, len(available_ids))]
                print(f"Using actual passage IDs for warmup: {sample_ids}")
            else:
                print("No passages available for warmup, skipping warmup...")
                return

            try:
                context = zmq.Context()
                socket = context.socket(zmq.REQ)
                socket.connect(f"tcp://localhost:{zmq_port}")
                socket.setsockopt(zmq.RCVTIMEO, 30000)
                socket.setsockopt(zmq.SNDTIMEO, 30000)

                try:
                    ids_to_send = [int(x) for x in sample_ids]
                except ValueError:
                    print("Warning: Could not convert sample IDs to integers, skipping warmup")
                    return

                if not ids_to_send:
                    print("Skipping warmup send.")
                    return

                # Use protobuf format for warmup
                from . import embedding_pb2
                req_proto = embedding_pb2.NodeEmbeddingRequest()
                req_proto.node_ids.extend(ids_to_send)
                request_bytes = req_proto.SerializeToString()

                for i in range(3):
                    print(f"Sending warmup request {i + 1}/3 via ZMQ (Protobuf)...")
                    socket.send(request_bytes)
                    response_bytes = socket.recv()
                    
                    resp_proto = embedding_pb2.NodeEmbeddingResponse()
                    resp_proto.ParseFromString(response_bytes)
                    embeddings_count = resp_proto.dimensions[0] if resp_proto.dimensions else 0
                    print(f"Warmup request {i + 1}/3 successful, received {embeddings_count} embeddings")
                    time.sleep(0.1)

                print("Client-side Protobuf ZMQ warmup complete")
                socket.close()
                context.term()
            except Exception as e:
                print(f"Error during Protobuf ZMQ warmup: {e}")

        class DeviceTimer:
            """Device timer"""
            def __init__(self, name="", device=device):
                self.name = name
                self.device = device
                self.start_time = 0
                self.end_time = 0
                
                if embedding_mode == "sentence-transformers" and torch.cuda.is_available():
                    self.start_event = torch.cuda.Event(enable_timing=True)
                    self.end_event = torch.cuda.Event(enable_timing=True)
                else:
                    self.start_event = None
                    self.end_event = None

            @contextmanager
            def timing(self):
                self.start()
                yield
                self.end()

            def start(self):
                if embedding_mode == "sentence-transformers" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    self.start_event.record()
                else:
                    if embedding_mode == "sentence-transformers" and self.device.type == "mps":
                        torch.mps.synchronize()
                    self.start_time = time.time()

            def end(self):
                if embedding_mode == "sentence-transformers" and torch.cuda.is_available():
                    self.end_event.record()
                    torch.cuda.synchronize()
                else:
                    if embedding_mode == "sentence-transformers" and self.device.type == "mps":
                        torch.mps.synchronize()
                    self.end_time = time.time()

            def elapsed_time(self):
                if embedding_mode == "sentence-transformers" and torch.cuda.is_available():
                    return self.start_event.elapsed_time(self.end_event) / 1000.0
                else:
                    return self.end_time - self.start_time

            def print_elapsed(self):
                elapsed = self.elapsed_time()
                print(f"[{self.name}] Elapsed time: {elapsed:.3f}s")

        def process_batch_pytorch(texts_batch, ids_batch, missing_ids):
            """Process text batch"""
            if not texts_batch:
                return np.array([])

            # Filter out empty texts and their corresponding IDs
            valid_texts = []
            valid_ids = []
            for i, text in enumerate(texts_batch):
                if text.strip():  # Only include non-empty texts
                    valid_texts.append(text)
                    valid_ids.append(ids_batch[i])

            if not valid_texts:
                print("WARNING: No valid texts in batch")
                return np.array([])

            # Tokenize
            token_timer = DeviceTimer("tokenization")
            with token_timer.timing():
                inputs = tokenizer(
                    valid_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(device)

            # Compute embeddings
            embed_timer = DeviceTimer("embedding computation")
            with embed_timer.timing():
                with torch.no_grad():
                    outputs = model(**inputs)
                    hidden_states = outputs.last_hidden_state
                    
                    # Mean pooling
                    attention_mask = inputs['attention_mask']
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = sum_embeddings / sum_mask
                embed_timer.print_elapsed()

            return batch_embeddings.cpu().numpy()

        # ZMQ server main loop - modified to use REP socket
        context = zmq.Context()
        socket = context.socket(zmq.ROUTER)  # Changed to REP socket
        socket.bind(f"tcp://127.0.0.1:{zmq_port}")
        print(f"INFO: ZMQ ROUTER server listening on port {zmq_port}")

        # Set timeouts
        socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second receive timeout
        socket.setsockopt(zmq.SNDTIMEO, 300000)  # 300 second send timeout

        from . import embedding_pb2

        print(f"INFO: Embedding server ready to serve requests")

        # Start warmup thread if enabled
        if enable_warmup and len(passages) > 0:
            import threading
            print(f"Warmup enabled: starting warmup thread")
            warmup_thread = threading.Thread(target=client_warmup, args=(zmq_port,))
            warmup_thread.daemon = True
            warmup_thread.start()
        else:
            print(f"Warmup disabled or no passages available (enable_warmup={enable_warmup}, passages={len(passages)})")

        while True:
            try:
                parts = socket.recv_multipart()

                # --- Restore robust message format detection ---
                # Must check parts length to avoid IndexError
                if len(parts) >= 3:
                    identity = parts[0]
                    # empty = parts[1]  # We usually don't care about the middle empty frame
                    message = parts[2]
                elif len(parts) == 2:
                    # Can also handle cases without empty frame
                    identity = parts[0]
                    message = parts[1]
                else:
                    # If received message format is wrong, print warning and ignore it instead of crashing
                    print(f"WARNING: Received unexpected message format with {len(parts)} parts. Ignoring.")
                    continue
                print(f"INFO: Received ZMQ request from client {identity.hex()[:8]}, size {len(message)} bytes")

                # Handle control messages (MessagePack format)
                try:
                    request_payload = msgpack.unpackb(message)
                    if isinstance(request_payload, list) and len(request_payload) >= 1:
                        if request_payload[0] == "__QUERY_META_PATH__":
                            # Return the current meta path being used by the server
                            current_meta_path = getattr(passages, '_meta_path', '') if hasattr(passages, '_meta_path') else ''
                            response = [current_meta_path]
                            socket.send_multipart([identity, b'', msgpack.packb(response)])
                            continue
                            
                        elif request_payload[0] == "__UPDATE_META_PATH__" and len(request_payload) >= 2:
                            # Update the server's meta path and reload passages
                            new_meta_path = request_payload[1]
                            try:
                                print(f"INFO: Updating server meta path to: {new_meta_path}")
                                # Reload passages from the new meta file
                                passages = load_passages_from_metadata(new_meta_path)
                                # Store the meta path for future queries
                                passages._meta_path = new_meta_path
                                response = ["SUCCESS"]
                                print(f"INFO: Successfully updated meta path and reloaded {len(passages)} passages")
                            except Exception as e:
                                print(f"ERROR: Failed to update meta path: {e}")
                                response = ["FAILED", str(e)]
                            socket.send_multipart([identity, b'', msgpack.packb(response)])
                            continue
                            
                        elif request_payload[0] == "__QUERY_MODEL__":
                            # Return the current model being used by the server
                            response = [model_name]
                            socket.send_multipart([identity, b'', msgpack.packb(response)])
                            continue
                            
                        elif request_payload[0] == "__UPDATE_MODEL__" and len(request_payload) >= 2:
                            # Update the server's embedding model
                            new_model_name = request_payload[1]
                            try:
                                print(f"INFO: Updating server model from {model_name} to: {new_model_name}")
                                
                                # Clean up old model to free memory
                                if not use_mlx:
                                    print("INFO: Releasing old model from memory...")
                                    old_model = model
                                    old_tokenizer = tokenizer
                                    
                                    # Load new tokenizer first
                                    print(f"Loading new tokenizer for {new_model_name}...")
                                    tokenizer = AutoTokenizer.from_pretrained(new_model_name, use_fast=True)
                                    
                                    # Load new model
                                    print(f"Loading new model {new_model_name}...")
                                    model = AutoModel.from_pretrained(new_model_name).to(device).eval()
                                    
                                    # Optimize new model
                                    if cuda_available or mps_available:
                                        try:
                                            model = model.half()
                                            model = torch.compile(model)
                                            print(f"INFO: Using FP16 precision with model: {new_model_name}")
                                        except Exception as e:
                                            print(f"WARNING: Model optimization failed: {e}")
                                    
                                    # Now safely delete old model after new one is loaded
                                    del old_model
                                    del old_tokenizer
                                    
                                    # Clear GPU cache if available
                                    if device.type == "cuda":
                                        torch.cuda.empty_cache()
                                        print("INFO: Cleared CUDA cache")
                                    elif device.type == "mps":
                                        torch.mps.empty_cache()
                                        print("INFO: Cleared MPS cache")
                                    
                                    # Force garbage collection
                                    import gc
                                    gc.collect()
                                    print("INFO: Memory cleanup completed")
                                
                                # Update model name
                                model_name = new_model_name
                                
                                response = ["SUCCESS"]
                                print(f"INFO: Successfully updated model to: {new_model_name}")
                            except Exception as e:
                                print(f"ERROR: Failed to update model: {e}")
                                response = ["FAILED", str(e)]
                            socket.send_multipart([identity, b'', msgpack.packb(response)])
                            continue
                except:
                    # Not a control message, continue with normal protobuf processing
                    pass

                e2e_start = time.time()
                lookup_timer = DeviceTimer("text lookup")

                # Parse request
                req_proto = embedding_pb2.NodeEmbeddingRequest()
                req_proto.ParseFromString(message)
                node_ids = req_proto.node_ids
                print(f"INFO: Request for {len(node_ids)} node embeddings: {list(node_ids)}")

                # Add debug information
                if len(node_ids) > 0:
                    print(f"DEBUG: Node ID range: {min(node_ids)} to {max(node_ids)}")
                
                # Look up texts
                texts = []
                missing_ids = []
                with lookup_timer.timing():
                    for nid in node_ids:
                        txtinfo = passages[nid]
                        txt = txtinfo["text"]
                        if txt:
                            texts.append(txt)
                        else:
                            # If text is empty, we still need a placeholder for batch processing,
                            # but record its ID as missing
                            texts.append("") 
                            missing_ids.append(nid)
                lookup_timer.print_elapsed()

                if missing_ids:
                    print(f"WARNING: Missing passages for IDs: {missing_ids}")

                # Process batch
                total_size = len(texts)
                print(f"INFO: Total batch size: {total_size}, max_batch_size: {max_batch_size}")
                
                all_embeddings = []
                
                if total_size > max_batch_size:
                    print(f"INFO: Splitting batch of size {total_size} into chunks of {max_batch_size}")
                    for i in range(0, total_size, max_batch_size):
                        end_idx = min(i + max_batch_size, total_size)
                        print(f"INFO: Processing chunk {i//max_batch_size + 1}/{(total_size + max_batch_size - 1)//max_batch_size}: items {i} to {end_idx-1}")
                        
                        chunk_texts = texts[i:end_idx]
                        chunk_ids = node_ids[i:end_idx]
                        
                        if embedding_mode == "mlx":
                            embeddings_chunk = compute_embeddings_mlx(chunk_texts, model_name, batch_size=16)
                        elif embedding_mode == "openai":
                            embeddings_chunk = compute_embeddings_openai(chunk_texts, model_name)
                        else:  # sentence-transformers
                            embeddings_chunk = process_batch_pytorch(chunk_texts, chunk_ids, missing_ids)
                        all_embeddings.append(embeddings_chunk)
                        
                        if embedding_mode == "sentence-transformers":
                            if cuda_available:
                                torch.cuda.empty_cache()
                            elif device.type == "mps":
                                torch.mps.empty_cache()
                            
                    hidden = np.vstack(all_embeddings)
                    print(f"INFO: Combined embeddings shape: {hidden.shape}")
                else:
                    if embedding_mode == "mlx":
                        hidden = compute_embeddings_mlx(texts, model_name, batch_size=16)
                    elif embedding_mode == "openai":
                        hidden = compute_embeddings_openai(texts, model_name)
                    else:  # sentence-transformers
                        hidden = process_batch_pytorch(texts, node_ids, missing_ids)

                # Serialize response
                ser_start = time.time()

                resp_proto = embedding_pb2.NodeEmbeddingResponse()
                hidden_contiguous = np.ascontiguousarray(hidden, dtype=np.float32)
                resp_proto.embeddings_data = hidden_contiguous.tobytes()
                resp_proto.dimensions.append(hidden_contiguous.shape[0])
                resp_proto.dimensions.append(hidden_contiguous.shape[1])
                resp_proto.missing_ids.extend(missing_ids)

                response_data = resp_proto.SerializeToString()
                
                # REP socket sends a single response
                socket.send_multipart([identity, b'', response_data])

                ser_end = time.time()

                print(f"INFO: Serialize time: {ser_end - ser_start:.6f} seconds")

                if embedding_mode == "sentence-transformers":
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elif device.type == "mps":
                        torch.mps.synchronize()
                e2e_end = time.time()
                print(f"INFO: ZMQ E2E time: {e2e_end - e2e_start:.6f} seconds")

            except zmq.Again:
                print("INFO: ZMQ socket timeout, continuing to listen")
                continue
            except Exception as e:
                print(f"ERROR: Error in ZMQ server: {e}")
                try:
                    # Send empty response to maintain REQ-REP state
                    empty_resp = embedding_pb2.NodeEmbeddingResponse()
                    socket.send(empty_resp.SerializeToString())
                except:
                    # If sending fails, recreate socket
                    socket.close()
                    socket = context.socket(zmq.REP)
                    socket.bind(f"tcp://127.0.0.1:{zmq_port}")
                    socket.setsockopt(zmq.RCVTIMEO, 5000)
                    socket.setsockopt(zmq.SNDTIMEO, 300000)
                    print("INFO: ZMQ socket recreated after error")

    except Exception as e:
        print(f"ERROR: Failed to start embedding server: {e}")
        raise


def create_embedding_server(
    domain="demo",
    load_passages=True,
    load_embeddings=False,
    use_fp16=True,
    use_int8=False,
    use_cuda_graphs=False,
    zmq_port=5555,
    max_batch_size=128,
    lazy_load_passages=False,
    model_name="sentence-transformers/all-mpnet-base-v2",
    passages_file: Optional[str] = None,
    embedding_mode: str = "sentence-transformers",
    enable_warmup: bool = False,
):
    """
    原有的 create_embedding_server 函数保持不变
    这个是阻塞版本，用于直接运行
    """
    create_embedding_server_thread(zmq_port, model_name, max_batch_size, passages_file, embedding_mode, enable_warmup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding service")
    parser.add_argument("--zmq-port", type=int, default=5555, help="ZMQ port to run on")
    parser.add_argument("--domain", type=str, default="demo", help="Domain name")
    parser.add_argument("--passages-file", type=str, help="JSON file containing passage ID to text mapping")
    parser.add_argument("--load-passages", action="store_true", default=True)
    parser.add_argument("--load-embeddings", action="store_true", default=False)
    parser.add_argument("--use-fp16", action="store_true", default=False)
    parser.add_argument("--use-int8", action="store_true", default=False)
    parser.add_argument("--use-cuda-graphs", action="store_true", default=False)
    parser.add_argument("--max-batch-size", type=int, default=128, help="Maximum batch size before splitting")
    parser.add_argument("--lazy-load-passages", action="store_true", default=True)
    parser.add_argument("--model-name", type=str, default="sentence-transformers/all-mpnet-base-v2", 
                        help="Embedding model name")
    parser.add_argument("--embedding-mode", type=str, default="sentence-transformers", 
                        choices=["sentence-transformers", "mlx", "openai"],
                        help="Embedding backend mode")
    parser.add_argument("--use-mlx", action="store_true", default=False, help="Use MLX backend for embeddings (deprecated: use --embedding-mode mlx)")
    parser.add_argument("--disable-warmup", action="store_true", default=False, help="Disable warmup requests on server start")
    args = parser.parse_args()
    
    # Handle backward compatibility with use_mlx
    embedding_mode = args.embedding_mode
    if args.use_mlx:
        embedding_mode = "mlx"

    create_embedding_server(
        domain=args.domain,
        load_passages=args.load_passages,
        load_embeddings=args.load_embeddings,
        use_fp16=args.use_fp16,
        use_int8=args.use_int8,
        use_cuda_graphs=args.use_cuda_graphs,
        zmq_port=args.zmq_port,
        max_batch_size=args.max_batch_size,
        lazy_load_passages=args.lazy_load_passages,
        model_name=args.model_name,
        passages_file=args.passages_file,
        embedding_mode=embedding_mode,
        enable_warmup=not args.disable_warmup,
    )
