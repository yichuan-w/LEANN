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
import sys
import logging

RED = "\033[91m"
RESET = "\033[0m"

# Set up logging based on environment variable
LOG_LEVEL = os.getenv('LEANN_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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

# Timing utilities
@contextmanager
def timer(name: str, sync_cuda: bool = True):
    """Context manager for timing operations with optional CUDA sync"""
    start_time = time.time()
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    try:
        yield
    finally:
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif sync_cuda and torch.backends.mps.is_available():
            torch.mps.synchronize()
        elapsed = time.time() - start_time
        logger.info(f"⏱️  {name}: {elapsed:.4f}s")


def e5_average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class SimplePassageLoader:
    """
    Simple passage loader that replaces config.py dependencies
    """

    def __init__(self, passages_data: Optional[Dict[str, Any]] = None):
        self.passages_data = passages_data or {}
        self._meta_path = ""

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
    with open(meta_file, "r") as f:
        meta = json.load(f)

    # Import PassageManager dynamically to avoid circular imports
    # Find the leann package directory relative to this file
    current_dir = Path(__file__).parent
    leann_core_path = current_dir.parent.parent / "leann-core" / "src"
    sys.path.insert(0, str(leann_core_path))

    try:
        from leann.api import PassageManager

        passage_manager = PassageManager(meta["passage_sources"])
    finally:
        sys.path.pop(0)

    print(f"Initialized lazy passage loading for {len(passage_manager.global_offset_map)} passages")

    class LazyPassageLoader(SimplePassageLoader):
        def __init__(self, passage_manager):
            self.passage_manager = passage_manager
            # Initialize parent with empty data
            super().__init__({})

        def __getitem__(self, passage_id: Union[str, int]) -> Dict[str, str]:
            """Get passage by ID with lazy loading"""
            try:
                int_id = int(passage_id)
                string_id = str(int_id)
                passage_data = self.passage_manager.get_passage(string_id)
                if passage_data and passage_data.get("text"):
                    return {"text": passage_data["text"]}
                else:
                    logger.debug(f"Empty text for ID {int_id} -> {string_id}")
                    return {"text": ""}
            except Exception as e:
                logger.debug(f"Exception getting passage {passage_id}: {e}")
                return {"text": ""}

        def __len__(self) -> int:
            return len(self.passage_manager.global_offset_map)

        def keys(self):
            return self.passage_manager.global_offset_map.keys()

    return LazyPassageLoader(passage_manager)


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
    distance_metric: str = "mips",
    embedding_mode: str = "sentence-transformers",
    enable_warmup: bool = False,
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
        distance_metric: The distance metric to use
        enable_warmup: Whether to perform warmup requests on server start
    """
    # Handle different embedding modes directly in HNSW server
    
    # Auto-detect mode based on model name if not explicitly set
    if embedding_mode == "sentence-transformers" and model_name.startswith("text-embedding-"):
        embedding_mode = "openai"
    
    if embedding_mode == "openai":
        print(f"Using OpenAI API mode for {model_name}")
        tokenizer = None  # No local tokenizer needed for OpenAI API
    elif embedding_mode == "mlx":
        print(f"Using MLX mode for {model_name}")
        tokenizer = None  # MLX handles tokenization separately
    else:  # sentence-transformers
        print(f"Loading tokenizer for {model_name}...")
        # Optimized tokenizer loading: try local first, then fallback
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                use_fast=True,       # Use fast tokenizer (better runtime perf)
                local_files_only=True  # Avoid network delays
            )
            print(f"Tokenizer loaded successfully! (local + fast)")
        except Exception as e:
            print(f"Local tokenizer failed ({e}), trying network download...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                use_fast=True  # Use fast tokenizer
            )
            print(f"Tokenizer loaded successfully! (network)")

    # Device setup
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
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
    print(f"Loading model {model_name}... (this may take a while if downloading)")

    if embedding_mode == "mlx":
        # For MLX models, we need to use the MLX embedding computation
        print("MLX model detected - using MLX backend for embeddings")
        model = None  # We'll handle MLX separately
    elif embedding_mode == "openai":
        # For OpenAI API, no local model needed
        print("OpenAI API mode - no local model loading required")
        model = None
    else:
        # Use optimized transformers loading for sentence-transformers models
        print(f"Loading model with optimizations...")
        try:
            # Ultra-fast loading: preload config + fast_init
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name, local_files_only=True)
            model = AutoModel.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.float16,  # Half precision for speed
                low_cpu_mem_usage=True,     # Reduce memory peaks  
                local_files_only=True,     # Avoid network delays
                _fast_init=True             # Skip weight init checks
            ).to(device).eval()
            print(f"Model {model_name} loaded successfully! (ultra-fast)")
        except Exception as e:
            print(f"Ultra-fast loading failed ({e}), trying optimized...")
            try:
                # Fallback: regular optimized loading
                model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    local_files_only=True
                ).to(device).eval()
                print(f"Model {model_name} loaded successfully! (optimized)")
            except Exception as e2:
                print(f"Optimized loading failed ({e2}), trying network...")
                try:
                    # Fallback: optimized network loading
                    model = AutoModel.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True
                    ).to(device).eval()
                    print(f"Model {model_name} loaded successfully! (network + optimized)")
                except Exception as e3:
                    print(f"All optimized methods failed ({e3}), using standard...")
                    # Final fallback: standard loading
                    model = AutoModel.from_pretrained(model_name).to(device).eval()
                    print(f"Model {model_name} loaded successfully! (standard)")

    # Check port availability
    import socket

    def check_port(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    if check_port(zmq_port):
        print(f"{RED}Port {zmq_port} is already in use{RESET}")
        return

    # Apply model optimizations (similar to DiskANN version)
    if use_fp16 and (cuda_available or mps_available):
        model = model.half()
        model = torch.compile(model)
        print(f"Using FP16 precision with model: {model_name}")
    elif use_int8:
        print(
            "- Using TorchAO for Int8 dynamic activation and Int8 weight quantization"
        )
        from torchao.quantization import (
            quantize_,
            Int8DynamicActivationInt8WeightConfig,
        )

        quantize_(model, Int8DynamicActivationInt8WeightConfig())
        model = torch.compile(model)
        model.eval()
        print("- Model successfully quantized and compiled")

    # Load passages
    if passages_data:
        passages = SimplePassageLoader(passages_data)
        print(f"Using provided passages data: {len(passages)} passages")
    elif passages_file:
        # Check if it's a metadata file or a single passages file
        if passages_file.endswith(".meta.json"):
            passages = load_passages_from_metadata(passages_file)
            # Store the meta path for future reference
            passages._meta_path = passages_file
        else:
            # Try to find metadata file in same directory
            passages_dir = Path(passages_file).parent
            meta_files = list(passages_dir.glob("*.meta.json"))
            if meta_files:
                print(f"Found metadata file: {meta_files[0]}, using lazy loading")
                passages = load_passages_from_metadata(str(meta_files[0]))
            else:
                # Fallback to original single file loading (will cause warnings)
                print(
                    "WARNING: No metadata file found, using single file loading (may cause missing passage warnings)"
                )
                passages = (
                    SimplePassageLoader()
                )  # Use empty loader to avoid massive warnings
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

    def _process_batch_mlx(texts_batch, ids_batch, missing_ids):
        """Process a batch of texts using MLX backend"""
        try:
            # Import MLX embedding computation from main API
            from leann.api import compute_embeddings

            # Compute embeddings using MLX
            embeddings = compute_embeddings(texts_batch, model_name, mode="mlx", use_server=False)

            print(
                f"[leann_backend_hnsw.hnsw_embedding_server LOG]: MLX embeddings computed for {len(texts_batch)} texts"
            )
            print(
                f"[leann_backend_hnsw.hnsw_embedding_server LOG]: Embedding shape: {embeddings.shape}"
            )

            return embeddings

        except Exception as e:
            print(
                f"[leann_backend_hnsw.hnsw_embedding_server LOG]: ERROR in MLX processing: {e}"
            )
            raise

    def process_batch(texts_batch, ids_batch, missing_ids):
        """Process a batch of texts and return embeddings"""

        # Handle different embedding modes
        if embedding_mode == "mlx":
            return _process_batch_mlx(texts_batch, ids_batch, missing_ids)
        elif embedding_mode == "openai":
            with timer("OpenAI API call", sync_cuda=False):
                from leann.api import compute_embeddings_openai
                return compute_embeddings_openai(texts_batch, model_name)

        _is_e5_model = "e5" in model_name.lower()
        _is_bge_model = "bge" in model_name.lower()
        batch_size = len(texts_batch)

        # Allow empty texts to pass through (remove validation)

        # E5 model preprocessing
        if _is_e5_model:
            processed_texts_batch = [f"passage: {text}" for text in texts_batch]
        else:
            processed_texts_batch = texts_batch

        # Set max length
        if _is_e5_model:
            current_max_length = (
                custom_max_length_param if custom_max_length_param is not None else 512
            )
        else:
            current_max_length = (
                custom_max_length_param if custom_max_length_param is not None else 256
            )

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
            with timer("Model forward pass"):
                with embed_timer.timing():
                    out = model(enc["input_ids"], enc["attention_mask"])

            with timer("Pooling"):
                with pool_timer.timing():
                    if _is_bge_model:
                        pooled_embeddings = out.last_hidden_state[:, 0]
                    elif not hasattr(out, "last_hidden_state"):
                        if isinstance(out, torch.Tensor) and len(out.shape) == 2:
                            pooled_embeddings = out
                        else:
                            print(
                                f"{RED}ERROR: Cannot determine how to pool. Output shape: {out.shape if isinstance(out, torch.Tensor) else 'N/A'}{RESET}"
                            )
                            hidden_dim = getattr(
                                model.config, "hidden_size", 384 if _is_e5_model else 768
                            )
                            pooled_embeddings = torch.zeros(
                                (batch_size, hidden_dim),
                                device=device,
                                dtype=enc["input_ids"].dtype
                                if hasattr(enc["input_ids"], "dtype")
                                else torch.float32,
                            )
                    elif _is_e5_model:
                        pooled_embeddings = e5_average_pool(
                            out.last_hidden_state, enc["attention_mask"]
                        )
                    else:
                        hidden_states = out.last_hidden_state
                        mask_expanded = (
                            enc["attention_mask"]
                            .unsqueeze(-1)
                            .expand(hidden_states.size())
                            .float()
                        )
                        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                        pooled_embeddings = sum_embeddings / sum_mask

            final_embeddings = pooled_embeddings
            if _is_e5_model or _is_bge_model:
                with norm_timer.timing():
                    final_embeddings = F.normalize(pooled_embeddings, p=2, dim=1)

        if torch.isnan(final_embeddings).any() or torch.isinf(final_embeddings).any():
            print(
                f"{RED}!!! In process_batch: NaN or Inf detected in final_embeddings! "
                f"Model: {model_name}, E5: {_is_e5_model}. IDs (sample): {ids_batch[:5]}...{RESET}"
            )
            dim_size = final_embeddings.shape[-1]
            error_output = torch.zeros(
                (batch_size, dim_size), device="cpu", dtype=torch.float32
            ).numpy()
            print(
                f"{RED}Returning zero embeddings of shape ({batch_size}, {dim_size}) due to NaN/Inf.{RESET}"
            )
            return error_output

        return final_embeddings.cpu().numpy()

    def client_warmup(zmq_port):
        """Perform client-side warmup"""
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

            request_payload = [ids_to_send]
            request_bytes = msgpack.packb(request_payload)

            for i in range(3):
                print(f"Sending warmup request {i + 1}/3 via ZMQ (MessagePack)...")
                socket.send(request_bytes)
                response_bytes = socket.recv()

                response_payload = msgpack.unpackb(response_bytes)
                dimensions = response_payload[0]
                embeddings_count = (
                    dimensions[0] if dimensions and len(dimensions) > 0 else 0
                )
                print(
                    f"Warmup request {i + 1}/3 successful, received {embeddings_count} embeddings"
                )
                time.sleep(0.1)

            print("Client-side MessagePack ZMQ warmup complete")
            socket.close()
            context.term()
        except Exception as e:
            print(f"Error during MessagePack ZMQ warmup: {e}")

    def zmq_server_thread():
        """ZMQ server thread"""
        nonlocal passages, model, tokenizer, model_name, embedding_mode
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
                    if isinstance(request_payload, list):
                        logger.debug(f"request_payload length: {len(request_payload)}")
                        for i, item in enumerate(request_payload):
                            print(
                                f"DEBUG: request_payload[{i}]: {type(item)} - {item if len(str(item)) < 100 else str(item)[:100] + '...'}"
                            )

                    # Handle control messages for meta path and model management FIRST
                    if isinstance(request_payload, list) and len(request_payload) >= 1:
                        if request_payload[0] == "__QUERY_META_PATH__":
                            # Return the current meta path being used by the server
                            current_meta_path = (
                                getattr(passages, "_meta_path", "")
                                if hasattr(passages, "_meta_path")
                                else ""
                            )
                            response = [current_meta_path]
                            socket.send(msgpack.packb(response))
                            continue

                        elif (
                            request_payload[0] == "__UPDATE_META_PATH__"
                            and len(request_payload) >= 2
                        ):
                            # Update the server's meta path and reload passages
                            new_meta_path = request_payload[1]
                            try:
                                print(
                                    f"INFO: Updating server meta path to: {new_meta_path}"
                                )
                                # Reload passages from the new meta file
                                passages = load_passages_from_metadata(new_meta_path)
                                # Store the meta path for future queries
                                passages._meta_path = new_meta_path
                                response = ["SUCCESS"]
                                print(
                                    f"INFO: Successfully updated meta path and reloaded {len(passages)} passages"
                                )
                            except Exception as e:
                                print(f"ERROR: Failed to update meta path: {e}")
                                response = ["FAILED", str(e)]
                            socket.send(msgpack.packb(response))
                            continue

                        elif request_payload[0] == "__QUERY_MODEL__":
                            # Return the current model being used by the server
                            response = [model_name]
                            socket.send(msgpack.packb(response))
                            continue

                        elif (
                            request_payload[0] == "__UPDATE_MODEL__"
                            and len(request_payload) >= 2
                        ):
                            # Update the server's embedding model
                            new_model_name = request_payload[1]
                            try:
                                print(
                                    f"INFO: Updating server model from {model_name} to: {new_model_name}"
                                )

                                # Clean up old model to free memory
                                logger.info("Releasing old model from memory...")
                                old_model = model
                                old_tokenizer = tokenizer

                                # Load new tokenizer first (optimized)
                                print(f"Loading new tokenizer for {new_model_name}...")
                                try:
                                    tokenizer = AutoTokenizer.from_pretrained(
                                        new_model_name, 
                                        use_fast=True,
                                        local_files_only=True
                                    )
                                    print(f"New tokenizer loaded! (local + fast)")
                                except:
                                    tokenizer = AutoTokenizer.from_pretrained(
                                        new_model_name, 
                                        use_fast=True
                                    )
                                    print(f"New tokenizer loaded! (network + fast)")

                                # Load new model (optimized)
                                print(f"Loading new model {new_model_name}...")
                                try:
                                    # Ultra-fast model switching
                                    from transformers import AutoConfig
                                    config = AutoConfig.from_pretrained(new_model_name, local_files_only=True)
                                    model = AutoModel.from_pretrained(
                                        new_model_name,
                                        config=config,
                                        torch_dtype=torch.float16,
                                        low_cpu_mem_usage=True,
                                        local_files_only=True,
                                        _fast_init=True
                                    )
                                    print(f"New model loaded! (ultra-fast)")
                                except:
                                    try:
                                        model = AutoModel.from_pretrained(
                                            new_model_name,
                                            torch_dtype=torch.float16,
                                            low_cpu_mem_usage=True,
                                            local_files_only=True
                                        )
                                        print(f"New model loaded! (optimized)")
                                    except:
                                        try:
                                            model = AutoModel.from_pretrained(
                                                new_model_name,
                                                torch_dtype=torch.float16,
                                                low_cpu_mem_usage=True
                                            )
                                            print(f"New model loaded! (network + optimized)")
                                        except:
                                            model = AutoModel.from_pretrained(new_model_name)
                                            print(f"New model loaded! (standard)")
                                model.to(device)
                                model.eval()

                                # Now safely delete old model after new one is loaded
                                del old_model
                                del old_tokenizer

                                # Clear GPU cache if available
                                if device.type == "cuda":
                                    torch.cuda.empty_cache()
                                    logger.info("Cleared CUDA cache")
                                elif device.type == "mps":
                                    torch.mps.empty_cache()
                                    logger.info("Cleared MPS cache")

                                # Update model name
                                model_name = new_model_name
                                
                                # Re-detect embedding mode based on new model name
                                if model_name.startswith("text-embedding-"):
                                    embedding_mode = "openai"
                                    logger.info(f"Auto-detected embedding mode: openai for {model_name}")
                                else:
                                    embedding_mode = "sentence-transformers"
                                    logger.info(f"Auto-detected embedding mode: sentence-transformers for {model_name}")

                                # Force garbage collection
                                import gc

                                gc.collect()
                                logger.info("Memory cleanup completed")

                                response = ["SUCCESS"]
                                print(
                                    f"INFO: Successfully updated model to: {new_model_name}"
                                )
                            except Exception as e:
                                print(f"ERROR: Failed to update model: {e}")
                                response = ["FAILED", str(e)]
                            socket.send(msgpack.packb(response))
                            continue

                    # Handle direct text embedding request (for OpenAI and sentence-transformers)
                    if isinstance(request_payload, list) and len(request_payload) > 0:
                        # Check if this is a direct text request (list of strings) and NOT a control message
                        if (all(isinstance(item, str) for item in request_payload) and 
                            not request_payload[0].startswith("__")):
                            logger.info(f"Processing direct text embedding request for {len(request_payload)} texts in {embedding_mode} mode")
                            
                            try:
                                if embedding_mode == "openai":
                                    from leann.api import compute_embeddings_openai
                                    embeddings = compute_embeddings_openai(request_payload, model_name)
                                else:
                                    # sentence-transformers mode - compute directly
                                    with timer(f"Direct text embedding ({len(request_payload)} texts)"):
                                        embeddings = process_batch(request_payload, [], [])
                                
                                response = embeddings.tolist()
                                socket.send(msgpack.packb(response))
                                e2e_end = time.time()
                                logger.info(f"⏱️  Text embedding E2E time: {e2e_end - e2e_start:.6f}s")
                                continue
                            except Exception as e:
                                logger.error(f"ERROR: Failed to compute {embedding_mode} embeddings: {e}")
                                socket.send(msgpack.packb([]))
                                continue

                    # Handle distance calculation requests
                    if (
                        isinstance(request_payload, list)
                        and len(request_payload) == 2
                        and isinstance(request_payload[0], list)
                        and isinstance(request_payload[1], list)
                    ):
                        node_ids = request_payload[0]
                        query_vector = np.array(request_payload[1], dtype=np.float32)

                        logger.debug("Distance calculation request received")
                        print(f"    Node IDs: {node_ids}")
                        print(f"    Query vector dim: {len(query_vector)}")
                        print(f"    Passages loaded: {len(passages)}")

                        # Get embeddings for node IDs
                        texts = []
                        missing_ids = []
                        with lookup_timer.timing():
                            for nid in node_ids:
                                logger.debug(f"Looking up passage ID {nid}")
                                try:
                                    txtinfo = passages[nid]
                                    if txtinfo is None:
                                        print(
                                            f"ERROR: Passage with ID {nid} returned None"
                                        )
                                        print(f"ERROR: txtinfo: {txtinfo}")
                                        raise RuntimeError(
                                            f"FATAL: Passage with ID {nid} returned None"
                                        )
                                    txt = txtinfo[
                                        "text"
                                    ]  # Allow empty text to pass through
                                    print(
                                        f"DEBUG: Found text for ID {nid}, length: {len(txt)}"
                                    )
                                    texts.append(txt)
                                except KeyError:
                                    print(
                                        f"ERROR: Passage ID {nid} not found in passages dict"
                                    )
                                    print(
                                        f"ERROR: Available passage IDs: {list(passages.keys())}..."
                                    )
                                    raise RuntimeError(
                                        f"FATAL: Passage with ID {nid} not found"
                                    )
                                except Exception as e:
                                    print(
                                        f"ERROR: Exception looking up passage ID {nid}: {e}"
                                    )
                                    raise
                        lookup_timer.print_elapsed()

                        # Process embeddings in chunks if needed
                        all_node_embeddings = []
                        total_size = len(texts)

                        if total_size > max_batch_size:
                            for i in range(0, total_size, max_batch_size):
                                end_idx = min(i + max_batch_size, total_size)
                                chunk_texts = texts[i:end_idx]
                                chunk_ids = node_ids[i:end_idx]

                                embeddings_chunk = process_batch(
                                    chunk_texts, chunk_ids, missing_ids
                                )
                                all_node_embeddings.append(embeddings_chunk)

                                if cuda_available:
                                    torch.cuda.empty_cache()
                                elif device.type == "mps":
                                    torch.mps.empty_cache()

                            node_embeddings = np.vstack(all_node_embeddings)
                        else:
                            node_embeddings = process_batch(
                                texts, node_ids, missing_ids
                            )

                        # Calculate distances
                        query_tensor = torch.tensor(query_vector, device=device).float()
                        node_embeddings_tensor = torch.tensor(
                            node_embeddings, device=device
                        ).float()

                        calc_timer = DeviceTimer("distance calculation", device)
                        with calc_timer.timing():
                            with torch.no_grad():
                                if distance_metric == "l2":
                                    node_embeddings_np = (
                                        node_embeddings_tensor.cpu()
                                        .numpy()
                                        .astype(np.float32)
                                    )
                                    query_np = (
                                        query_tensor.cpu().numpy().astype(np.float32)
                                    )
                                    distances = np.sum(
                                        np.square(
                                            node_embeddings_np - query_np.reshape(1, -1)
                                        ),
                                        axis=1,
                                    )
                                else:  # mips or cosine
                                    node_embeddings_np = (
                                        node_embeddings_tensor.cpu().numpy()
                                    )
                                    query_np = query_tensor.cpu().numpy()
                                    distances = -np.dot(node_embeddings_np, query_np)
                        calc_timer.print_elapsed()

                        try:
                            response_payload = distances.flatten().tolist()
                            response_bytes = msgpack.packb(
                                [response_payload], use_single_float=True
                            )
                            print(
                                f"Sending distance response with {len(distances)} distances"
                            )
                        except Exception as pack_error:
                            print(
                                f"ERROR: Error packing MessagePack distance response: {pack_error}"
                            )
                            print(f"ERROR: distances shape: {distances.shape}")
                            print(f"ERROR: distances dtype: {distances.dtype}")
                            print(f"ERROR: distances content: {distances}")
                            print(f"ERROR: node_ids: {node_ids}")
                            print(f"ERROR: query_vector shape: {query_vector.shape}")
                            # Still return empty for now but with full error info
                            response_bytes = msgpack.packb([[]])

                        socket.send(response_bytes)

                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        elif device.type == "mps":
                            torch.mps.synchronize()
                        e2e_end = time.time()
                        logger.info(
                            f"⏱️  Distance calculation E2E time: {e2e_end - e2e_start:.6f}s"
                        )
                        continue


                    # Standard embedding request (passage ID lookup)
                    if (
                        not isinstance(request_payload, list)
                        or len(request_payload) != 1
                        or not isinstance(request_payload[0], list)
                    ):
                        print(
                            f"Error: Invalid MessagePack request format. Expected [[ids...]] or [texts...], got: {type(request_payload)}"
                        )
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
                        try:
                            txtinfo = passages[nid]
                            if txtinfo is None or txtinfo["text"] == "":
                                raise RuntimeError(
                                    f"FATAL: Passage with ID {nid} not found - failing fast"
                                )
                            else:
                                txt = txtinfo["text"]
                        except (KeyError, IndexError):
                            raise RuntimeError(
                                f"FATAL: Passage with ID {nid} not found - failing fast"
                            )
                        texts.append(txt)
                lookup_timer.print_elapsed()

                if missing_ids:
                    print(f"Missing passages for IDs: {missing_ids}")

                # Process in chunks
                total_size = len(texts)
                print(
                    f"Total batch size: {total_size}, max_batch_size: {max_batch_size}"
                )

                all_embeddings = []

                if total_size > max_batch_size:
                    print(
                        f"Splitting batch of size {total_size} into chunks of {max_batch_size}"
                    )
                    for i in range(0, total_size, max_batch_size):
                        end_idx = min(i + max_batch_size, total_size)
                        print(
                            f"Processing chunk {i // max_batch_size + 1}/{(total_size + max_batch_size - 1) // max_batch_size}: items {i} to {end_idx - 1}"
                        )

                        chunk_texts = texts[i:end_idx]
                        chunk_ids = node_ids[i:end_idx]

                        embeddings_chunk = process_batch(
                            chunk_texts, chunk_ids, missing_ids
                        )
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

                print(
                    f"DEBUG zmq_server_thread: Final 'hidden' array | Shape: {hidden.shape} | Dtype: {hidden.dtype} | Has NaN/Inf: {np.isnan(hidden).any() or np.isinf(hidden).any()}"
                )
                if np.isnan(hidden).any() or np.isinf(hidden).any():
                    print(
                        f"{RED}!!! ERROR: NaN or Inf detected in final 'hidden' numpy array BEFORE sending! "
                        f"Requested IDs (sample): {node_ids[:5]}...{RESET}"
                    )
                    assert False

                try:
                    hidden_contiguous_f32 = np.ascontiguousarray(
                        hidden, dtype=np.float32
                    )
                    response_payload = [
                        list(hidden_contiguous_f32.shape),
                        hidden_contiguous_f32.flatten().tolist(),
                    ]
                    response_bytes = msgpack.packb(
                        response_payload, use_single_float=True
                    )
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
                logger.info(f"⏱️  ZMQ E2E time: {e2e_end - e2e_start:.6f}s")

            except zmq.Again:
                logger.debug("ZMQ socket timeout, continuing to listen")
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
    if enable_warmup and len(passages) > 0:
        print(f"Warmup enabled: starting warmup thread")
        warmup_thread = threading.Thread(target=client_warmup, args=(zmq_port,))
        warmup_thread.daemon = True
        warmup_thread.start()
    else:
        print(f"Warmup disabled or no passages available (enable_warmup={enable_warmup}, passages={len(passages)})")

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
    parser.add_argument(
        "--passages-file",
        type=str,
        help="JSON file containing passage ID to text mapping",
    )
    parser.add_argument(
        "--embeddings-file",
        type=str,
        help="Pickle file containing pre-computed embeddings",
    )
    parser.add_argument("--use-fp16", action="store_true", default=False)
    parser.add_argument("--use-int8", action="store_true", default=False)
    parser.add_argument("--use-cuda-graphs", action="store_true", default=False)
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=128,
        help="Maximum batch size before splitting",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Embedding model name",
    )
    parser.add_argument(
        "--custom-max-length",
        type=int,
        default=None,
        help="Override model's default max sequence length",
    )
    parser.add_argument(
        "--distance-metric", type=str, default="mips", help="Distance metric to use"
    )
    parser.add_argument(
        "--embedding-mode", 
        type=str, 
        default="sentence-transformers", 
        choices=["sentence-transformers", "mlx", "openai"],
        help="Embedding backend mode"
    )
    parser.add_argument(
        "--use-mlx",
        action="store_true",
        default=False,
        help="Use MLX for model inference (deprecated: use --embedding-mode mlx)",
    )
    parser.add_argument(
        "--disable-warmup",
        action="store_true",
        default=False,
        help="Disable warmup requests on server start",
    )

    args = parser.parse_args()
    
    # Handle backward compatibility with use_mlx
    embedding_mode = args.embedding_mode
    if args.use_mlx:
        embedding_mode = "mlx"

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
        distance_metric=args.distance_metric,
        embedding_mode=embedding_mode,
        enable_warmup=not args.disable_warmup,
    )
