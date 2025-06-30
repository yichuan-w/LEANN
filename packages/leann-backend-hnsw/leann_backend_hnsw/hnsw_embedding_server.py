#!/usr/bin/env python3
"""
HNSW-specific embedding server with removed config.py dependencies
Based on DiskANN embedding server architecture
"""

import pickle
import argparse
import threading
import time
from transformers import AutoTokenizer, AutoModel
import os
from contextlib import contextmanager
import zmq
import numpy as np
import msgpack
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union

RED = "\033[91m"
RESET = "\033[0m"

def is_similarity_metric():
    """
    Check if the metric type is similarity-based (like inner product).
    0 = L2 (distance metric), 1 = Inner Product (similarity metric)
    """
    return True  # 1 is METRIC_INNER_PRODUCT in FAISS

# Function for E5-style average pooling
import torch
from torch import Tensor
import torch.nn.functional as F

def e5_average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

class SimplePassageLoader:
    """
    Simple passage loader that replaces config.py dependencies
    """
    def __init__(self, passages_data: Optional[Dict[str, Any]] = None):
        self.passages_data = passages_data or {}
    
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

def load_passages_from_file(passages_file: str) -> SimplePassageLoader:
    """
    Load passages from a JSON file
    Expected format: {"passage_id": "passage_text", ...}
    """
    if not os.path.exists(passages_file):
        print(f"Warning: Passages file {passages_file} not found. Using empty loader.")
        return SimplePassageLoader()
    
    try:
        with open(passages_file, 'r', encoding='utf-8') as f:
            passages_data = json.load(f)
        print(f"Loaded {len(passages_data)} passages from {passages_file}")
        return SimplePassageLoader(passages_data)
    except Exception as e:
        print(f"Error loading passages from {passages_file}: {e}")
        return SimplePassageLoader()

def create_hnsw_embedding_server(
    passages_file: Optional[str] = None,
    passages_data: Optional[Dict[str, str]] = None,
    embeddings_file: Optional[str] = None,
    use_fp16: bool = True,
    use_int8: bool = False,
    use_cuda_graphs: bool = False,
    zmq_port: int = 5555,
    max_batch_size: int = 128,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    custom_max_length_param: Optional[int] = None,
):
    """
    Create and start a ZMQ-based embedding server for HNSW backend.
    
    Args:
        passages_file: Path to JSON file containing passage ID -> text mapping
        passages_data: Direct passage data dict (alternative to passages_file)
        embeddings_file: Path to pre-computed embeddings file (optional)
        use_fp16: Whether to use FP16 precision
        use_int8: Whether to use INT8 quantization
        use_cuda_graphs: Whether to use CUDA graphs
        zmq_port: ZMQ port to bind to
        max_batch_size: Maximum batch size for processing
        model_name: Transformer model name
        custom_max_length_param: Custom max sequence length
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # Device setup
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()
    
    print(f"MPS available: {mps_available}")
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        device = torch.device("cuda")
        print("Using CUDA device")
    elif mps_available:
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU device (no GPU acceleration available)")
    
    # Load model to the appropriate device
    print(f"Starting HNSW server on port {zmq_port} with model {model_name}")
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    # Check port availability
    import socket
    def check_port(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    if check_port(zmq_port):
        print(f"{RED}Port {zmq_port} is already in use{RESET}")
        return

    # Apply model optimizations (similar to DiskANN version)
    if use_fp16 and (cuda_available or mps_available):
        model = model.half()
        model = torch.compile(model)
        print(f"Using FP16 precision with model: {model_name}")
    elif use_int8:
        print("- Using TorchAO for Int8 dynamic activation and Int8 weight quantization")
        from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
        quantize_(model, Int8DynamicActivationInt8WeightConfig())
        model = torch.compile(model)
        model.eval()
        print("- Model successfully quantized and compiled")

    # Load passages
    if passages_data:
        passages = SimplePassageLoader(passages_data)
        print(f"Using provided passages data: {len(passages)} passages")
    elif passages_file:
        passages = load_passages_from_file(passages_file)
    else:
        passages = SimplePassageLoader()
        print("No passages provided, using empty loader")

    # Load embeddings if provided
    _embeddings = None
    if embeddings_file and os.path.exists(embeddings_file):
        try:
            with open(embeddings_file, "rb") as f:
                _embeddings = pickle.load(f)
            print(f"Loaded embeddings from {embeddings_file}")
        except Exception as e:
            print(f"Error loading embeddings: {e}")

    class DeviceTimer:
        """Device event-based timer for accurate timing."""
        def __init__(self, name="", device=device):
            self.name = name
            self.device = device
            self.start_time = 0
            self.end_time = 0
            
            if cuda_available:
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
            if cuda_available:
                torch.cuda.synchronize()
                self.start_event.record()
            else:
                if self.device.type == "mps":
                    torch.mps.synchronize()
                self.start_time = time.time()

        def end(self):
            if cuda_available:
                self.end_event.record()
                torch.cuda.synchronize()
            else:
                if self.device.type == "mps":
                    torch.mps.synchronize()
                self.end_time = time.time()

        def elapsed_time(self):
            if cuda_available:
                return self.start_event.elapsed_time(self.end_event) / 1000.0
            else:
                return self.end_time - self.start_time

        def print_elapsed(self):
            return  # Disabled for now

    def process_batch(texts_batch, ids_batch, missing_ids):
        """Process a batch of texts and return embeddings"""
        _is_e5_model = "e5" in model_name.lower()
        batch_size = len(texts_batch)
        
        # E5 model preprocessing
        if _is_e5_model:
            processed_texts_batch = [f"passage: {text}" for text in texts_batch]
        else:
            processed_texts_batch = texts_batch
        
        # Set max length
        if _is_e5_model:
            current_max_length = custom_max_length_param if custom_max_length_param is not None else 512
        else:
            current_max_length = custom_max_length_param if custom_max_length_param is not None else 256
        
        tokenize_timer = DeviceTimer("tokenization (batch)", device)
        to_device_timer = DeviceTimer("transfer to device (batch)", device)
        embed_timer = DeviceTimer("embedding (batch)", device)
        pool_timer = DeviceTimer("pooling (batch)", device)
        norm_timer = DeviceTimer("normalization (batch)", device)
        
        with tokenize_timer.timing():
            encoded_batch = tokenizer(
                processed_texts_batch,
                padding="max_length",
                truncation=True,
                max_length=current_max_length,
                return_tensors="pt",
                return_token_type_ids=False,
            )
        
        seq_length = encoded_batch["input_ids"].size(1)
        
        with to_device_timer.timing():
            enc = {k: v.to(device) for k, v in encoded_batch.items()}
        
        with torch.no_grad():
            with embed_timer.timing():
                out = model(enc["input_ids"], enc["attention_mask"])
            
            with pool_timer.timing():
                if not hasattr(out, 'last_hidden_state'):
                    if isinstance(out, torch.Tensor) and len(out.shape) == 2:
                        pooled_embeddings = out
                    else:
                        print(f"{RED}ERROR: Cannot determine how to pool. Output shape: {out.shape if isinstance(out, torch.Tensor) else 'N/A'}{RESET}")
                        hidden_dim = getattr(model.config, 'hidden_size', 384 if _is_e5_model else 768)
                        pooled_embeddings = torch.zeros((batch_size, hidden_dim), device=device, dtype=enc["input_ids"].dtype if hasattr(enc["input_ids"], "dtype") else torch.float32)
                elif _is_e5_model:
                    pooled_embeddings = e5_average_pool(out.last_hidden_state, enc['attention_mask'])
                else:
                    hidden_states = out.last_hidden_state
                    mask_expanded = enc["attention_mask"].unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    pooled_embeddings = sum_embeddings / sum_mask
            
            final_embeddings = pooled_embeddings
            if _is_e5_model:
                with norm_timer.timing():
                    final_embeddings = F.normalize(pooled_embeddings, p=2, dim=1)
        
        if torch.isnan(final_embeddings).any() or torch.isinf(final_embeddings).any():
            print(f"{RED}!!! In process_batch: NaN or Inf detected in final_embeddings! "
                  f"Model: {model_name}, E5: {_is_e5_model}. IDs (sample): {ids_batch[:5]}...{RESET}")
            dim_size = final_embeddings.shape[-1]
            error_output = torch.zeros((batch_size, dim_size), device='cpu', dtype=torch.float32).numpy()
            print(f"{RED}Returning zero embeddings of shape ({batch_size}, {dim_size}) due to NaN/Inf.{RESET}")
            return error_output
        
        return final_embeddings.cpu().numpy()

    def client_warmup(zmq_port):
        """Perform client-side warmup"""
        time.sleep(2)
        print(f"Performing client-side warmup with model {model_name}...")
        sample_ids = ["1", "2", "3", "4", "5"]
        
        try:
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.connect(f"tcp://localhost:{zmq_port}")
            socket.setsockopt(zmq.RCVTIMEO, 30000)
            socket.setsockopt(zmq.SNDTIMEO, 30000)

            try: 
                ids_to_send = [int(x) for x in sample_ids]
            except ValueError: 
                ids_to_send = []

            if not ids_to_send: 
                print("Skipping warmup send.")
                return

            request_payload = [ids_to_send]
            request_bytes = msgpack.packb(request_payload)

            for i in range(3):
                print(f"Sending warmup request {i+1}/3 via ZMQ (MessagePack)...")
                socket.send(request_bytes)
                response_bytes = socket.recv()

                response_payload = msgpack.unpackb(response_bytes)
                dimensions = response_payload[0]
                embeddings_count = dimensions[0] if dimensions and len(dimensions) > 0 else 0
                print(f"Warmup request {i+1}/3 successful, received {embeddings_count} embeddings")
                time.sleep(0.1)

            print("Client-side MessagePack ZMQ warmup complete")
            socket.close()
            context.term()
        except Exception as e:
            print(f"Error during MessagePack ZMQ warmup: {e}")

    def zmq_server_thread():
        """ZMQ server thread"""
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{zmq_port}")
        print(f"HNSW ZMQ server listening on port {zmq_port}")

        socket.setsockopt(zmq.RCVTIMEO, 300000)
        socket.setsockopt(zmq.SNDTIMEO, 300000)

        while True:
            try:
                message_bytes = socket.recv()
                print(f"Received ZMQ request of size {len(message_bytes)} bytes")

                e2e_start = time.time()
                lookup_timer = DeviceTimer("text lookup", device)

                try:
                    request_payload = msgpack.unpackb(message_bytes)
                    
                    # Handle distance calculation requests
                    if isinstance(request_payload, list) and len(request_payload) == 2 and isinstance(request_payload[0], list) and isinstance(request_payload[1], list):
                        node_ids = request_payload[0]
                        query_vector = np.array(request_payload[1], dtype=np.float32)
                        
                        print(f"Request for distance calculation: {len(node_ids)} nodes, query vector dim: {len(query_vector)}")
                        
                        # Get embeddings for node IDs
                        texts = []
                        missing_ids = []
                        with lookup_timer.timing():
                            for nid in node_ids:
                                txtinfo = passages[nid]
                                if txtinfo is None or txtinfo["text"] == "":
                                    print(f"Warning: Passage with ID {nid} not found")
                                    missing_ids.append(nid)
                                    txt = ""
                                else:
                                    txt = txtinfo["text"]
                                texts.append(txt)
                        lookup_timer.print_elapsed()
                        
                        # Process embeddings in chunks if needed
                        all_node_embeddings = []
                        total_size = len(texts)
                        
                        if total_size > max_batch_size:
                            for i in range(0, total_size, max_batch_size):
                                end_idx = min(i + max_batch_size, total_size)
                                chunk_texts = texts[i:end_idx]
                                chunk_ids = node_ids[i:end_idx]
                                
                                embeddings_chunk = process_batch(chunk_texts, chunk_ids, missing_ids)
                                all_node_embeddings.append(embeddings_chunk)
                                
                                if cuda_available:
                                    torch.cuda.empty_cache()
                                elif device.type == "mps":
                                    torch.mps.empty_cache()
                            
                            node_embeddings = np.vstack(all_node_embeddings)
                        else:
                            node_embeddings = process_batch(texts, node_ids, missing_ids)
                        
                        # Calculate distances
                        query_tensor = torch.tensor(query_vector, device=device).float()
                        node_embeddings_tensor = torch.tensor(node_embeddings, device=device).float()
                        
                        calc_timer = DeviceTimer("distance calculation", device)
                        with calc_timer.timing():
                            with torch.no_grad():
                                if is_similarity_metric():
                                    node_embeddings_np = node_embeddings_tensor.cpu().numpy()
                                    query_np = query_tensor.cpu().numpy()
                                    distances = -np.dot(node_embeddings_np, query_np)
                                else:
                                    node_embeddings_np = node_embeddings_tensor.cpu().numpy().astype(np.float32)
                                    query_np = query_tensor.cpu().numpy().astype(np.float32)
                                    distances = np.sum(np.square(node_embeddings_np - query_np.reshape(1, -1)), axis=1)
                        calc_timer.print_elapsed()
                        
                        try:
                            response_payload = distances.flatten().tolist()
                            response_bytes = msgpack.packb([response_payload], use_single_float=True)
                            print(f"Sending distance response with {len(distances)} distances")
                        except Exception as pack_error:
                            print(f"Error packing MessagePack distance response: {pack_error}")
                            response_bytes = msgpack.packb([[]])
                        
                        socket.send(response_bytes)
                        
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        elif device.type == "mps":
                            torch.mps.synchronize()
                        e2e_end = time.time()
                        print(f"Distance calculation E2E time: {e2e_end - e2e_start:.6f} seconds")
                        continue
                    
                    # Standard embedding request
                    if not isinstance(request_payload, list) or len(request_payload) != 1 or not isinstance(request_payload[0], list):
                        print(f"Error: Invalid MessagePack request format. Expected [[ids...]], got: {type(request_payload)}")
                        socket.send(msgpack.packb([[], []]))
                        continue
                    
                    node_ids = request_payload[0]
                    print(f"Request for {len(node_ids)} node embeddings")
                    
                except Exception as unpack_error:
                    print(f"Error unpacking MessagePack request: {unpack_error}")
                    socket.send(msgpack.packb([[], []]))
                    continue

                # Look up texts by node IDs
                texts = []
                missing_ids = []
                with lookup_timer.timing():
                    for nid in node_ids:
                        txtinfo = passages[nid]
                        if txtinfo is None or txtinfo["text"] == "":
                            print(f"Warning: Passage with ID {nid} not found")
                            missing_ids.append(nid)
                            txt = ""
                        else:
                            txt = txtinfo["text"]
                        texts.append(txt)
                lookup_timer.print_elapsed()

                if missing_ids:
                    print(f"Missing passages for IDs: {missing_ids}")

                # Process in chunks
                total_size = len(texts)
                print(f"Total batch size: {total_size}, max_batch_size: {max_batch_size}")
                
                all_embeddings = []
                
                if total_size > max_batch_size:
                    print(f"Splitting batch of size {total_size} into chunks of {max_batch_size}")
                    for i in range(0, total_size, max_batch_size):
                        end_idx = min(i + max_batch_size, total_size)
                        print(f"Processing chunk {i//max_batch_size + 1}/{(total_size + max_batch_size - 1)//max_batch_size}: items {i} to {end_idx-1}")
                        
                        chunk_texts = texts[i:end_idx]
                        chunk_ids = node_ids[i:end_idx]
                        
                        embeddings_chunk = process_batch(chunk_texts, chunk_ids, missing_ids)
                        all_embeddings.append(embeddings_chunk)
                        
                        if cuda_available:
                            torch.cuda.empty_cache()
                        elif device.type == "mps":
                            torch.mps.empty_cache()
                            
                    hidden = np.vstack(all_embeddings)
                    print(f"Combined embeddings shape: {hidden.shape}")
                else:
                    hidden = process_batch(texts, node_ids, missing_ids)

                # Serialization and response
                ser_start = time.time()

                print(f"DEBUG zmq_server_thread: Final 'hidden' array | Shape: {hidden.shape} | Dtype: {hidden.dtype} | Has NaN/Inf: {np.isnan(hidden).any() or np.isinf(hidden).any()}")
                if np.isnan(hidden).any() or np.isinf(hidden).any():
                    print(f"{RED}!!! ERROR: NaN or Inf detected in final 'hidden' numpy array BEFORE sending! "
                          f"Requested IDs (sample): {node_ids[:5]}...{RESET}")
                    assert False

                try:
                    hidden_contiguous_f32 = np.ascontiguousarray(hidden, dtype=np.float32)
                    response_payload = [
                        list(hidden_contiguous_f32.shape),
                        hidden_contiguous_f32.flatten().tolist()
                    ]
                    response_bytes = msgpack.packb(response_payload, use_single_float=True)
                except Exception as pack_error:
                     print(f"Error packing MessagePack response: {pack_error}")
                     response_bytes = msgpack.packb([[], []])

                socket.send(response_bytes)
                ser_end = time.time()

                print(f"Serialize time: {ser_end - ser_start:.6f} seconds")

                if device.type == "cuda":
                    torch.cuda.synchronize()
                elif device.type == "mps":
                    torch.mps.synchronize()
                e2e_end = time.time()
                print(f"ZMQ E2E time: {e2e_end - e2e_start:.6f} seconds")

            except zmq.Again:
                print("ZMQ socket timeout, continuing to listen")
                continue
            except Exception as e:
                print(f"Error in ZMQ server loop: {e}")
                import traceback
                traceback.print_exc()
                try: 
                    socket.send(msgpack.packb([[], []]))
                except:
                    pass

    # Start warmup and server threads
    if len(passages) > 0:
        warmup_thread = threading.Thread(target=client_warmup, args=(zmq_port,))
        warmup_thread.daemon = True
        warmup_thread.start()

    zmq_thread = threading.Thread(target=zmq_server_thread, daemon=True)
    zmq_thread.start()
    print(f"Started HNSW ZMQ server thread on port {zmq_port}")
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("HNSW Server shutting down...")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HNSW Embedding service")
    parser.add_argument("--zmq-port", type=int, default=5555, help="ZMQ port to run on")
    parser.add_argument("--passages-file", type=str, help="JSON file containing passage ID to text mapping")
    parser.add_argument("--embeddings-file", type=str, help="Pickle file containing pre-computed embeddings")
    parser.add_argument("--use-fp16", action="store_true", default=False)
    parser.add_argument("--use-int8", action="store_true", default=False)
    parser.add_argument("--use-cuda-graphs", action="store_true", default=False)
    parser.add_argument("--max-batch-size", type=int, default=128, help="Maximum batch size before splitting")
    parser.add_argument("--model-name", type=str, default="sentence-transformers/all-mpnet-base-v2", 
                        help="Embedding model name")
    parser.add_argument("--custom-max-length", type=int, default=None, help="Override model's default max sequence length")
    
    args = parser.parse_args()

    # Create and start the HNSW embedding server
    create_hnsw_embedding_server(
        passages_file=args.passages_file,
        embeddings_file=args.embeddings_file,
        use_fp16=args.use_fp16,
        use_int8=args.use_int8,
        use_cuda_graphs=args.use_cuda_graphs,
        zmq_port=args.zmq_port,
        max_batch_size=args.max_batch_size,
        model_name=args.model_name,
        custom_max_length_param=args.custom_max_length,
    )