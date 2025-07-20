"""
This file contains the core API for the LEANN project, now definitively updated
with the correct, original embedding logic from the user's reference code.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
import uuid
import torch

from .registry import BACKEND_REGISTRY
from .interface import LeannBackendFactoryInterface
from .chat import get_llm


def compute_embeddings(
    chunks: List[str],
    model_name: str,
    mode: str = "sentence-transformers",
    use_server: bool = True,
    use_mlx: bool = False  # Backward compatibility: if True, override mode to 'mlx',
) -> np.ndarray:
    """
    Computes embeddings using different backends.

    Args:
        chunks: List of text chunks to embed
        model_name: Name of the embedding model
        mode: Embedding backend mode. Options:
            - "sentence-transformers": Use sentence-transformers library (default)
            - "mlx": Use MLX backend for Apple Silicon
            - "openai": Use OpenAI embedding API
        use_server: Whether to use embedding server (True for search, False for build)

    Returns:
        numpy array of embeddings
    """
    # Override mode for backward compatibility
    if use_mlx:
        mode = "mlx"

    # Auto-detect mode based on model name if not explicitly set
    if mode == "sentence-transformers" and model_name.startswith("text-embedding-"):
        mode = "openai"

    if mode == "mlx":
        return compute_embeddings_mlx(chunks, model_name, batch_size=16)
    elif mode == "openai":
        return compute_embeddings_openai(chunks, model_name)
    elif mode == "sentence-transformers":
        return compute_embeddings_sentence_transformers(
            chunks, model_name, use_server=use_server
        )
    else:
        raise ValueError(
            f"Unsupported embedding mode: {mode}. Supported modes: sentence-transformers, mlx, openai"
        )


def compute_embeddings_sentence_transformers(
    chunks: List[str], model_name: str, use_server: bool = True
) -> np.ndarray:
    """Computes embeddings using sentence-transformers.

    Args:
        chunks: List of text chunks to embed
        model_name: Name of the sentence transformer model
        use_server: If True, use embedding server (good for search). If False, use direct computation (good for build).
    """
    if not use_server:
        print(
            f"INFO: Computing embeddings for {len(chunks)} chunks using SentenceTransformer model '{model_name}' (direct)..."
        )
        return _compute_embeddings_sentence_transformers_direct(chunks, model_name)

    print(
        f"INFO: Computing embeddings for {len(chunks)} chunks using SentenceTransformer model '{model_name}' (via embedding server)..."
    )

    # Use embedding server for sentence-transformers too
    # This avoids loading the model twice (once in API, once in server)
    try:
        # Import ZMQ client functionality and server manager
        import zmq
        import msgpack
        import numpy as np
        from .embedding_server_manager import EmbeddingServerManager

        # Ensure embedding server is running
        port = 5557
        server_manager = EmbeddingServerManager(
            backend_module_name="leann_backend_hnsw.hnsw_embedding_server"
        )

        server_started = server_manager.start_server(
            port=port,
            model_name=model_name,
            embedding_mode="sentence-transformers",
            enable_warmup=False,
        )

        if not server_started:
            raise RuntimeError(f"Failed to start embedding server on port {port}")

        # Connect to embedding server
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://localhost:{port}")

        # Send chunks to server for embedding computation
        request = chunks
        socket.send(msgpack.packb(request))

        # Receive embeddings from server
        response = socket.recv()
        embeddings_list = msgpack.unpackb(response)

        # Convert back to numpy array
        embeddings = np.array(embeddings_list, dtype=np.float32)

        socket.close()
        context.term()

        return embeddings

    except Exception as e:
        # Fallback to direct sentence-transformers if server connection fails
        print(
            f"Warning: Failed to connect to embedding server, falling back to direct computation: {e}"
        )
        return _compute_embeddings_sentence_transformers_direct(chunks, model_name)


def _compute_embeddings_sentence_transformers_direct(
    chunks: List[str], model_name: str
) -> np.ndarray:
    """Direct sentence-transformers computation (fallback)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers not available. Install with: uv pip install sentence-transformers"
        ) from e

    # Load model using sentence-transformers
    model = SentenceTransformer(model_name)

    model = model.half()
    print(
        f"INFO: Computing embeddings for {len(chunks)} chunks using SentenceTransformer model '{model_name}' (direct)..."
    )
    # use acclerater GPU or MAC GPU

    if torch.cuda.is_available():
        model = model.to("cuda")
    elif torch.backends.mps.is_available():
        model = model.to("mps")

    # Generate embeddings
    # give use an warning if OOM here means we need to turn down the batch size
    embeddings = model.encode(
        chunks, convert_to_numpy=True, show_progress_bar=True, batch_size=16
    )

    return embeddings


def compute_embeddings_openai(chunks: List[str], model_name: str) -> np.ndarray:
    """Computes embeddings using OpenAI API."""
    try:
        import openai
        import os
    except ImportError as e:
        raise RuntimeError(
            "openai not available. Install with: uv pip install openai"
        ) from e

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    client = openai.OpenAI(api_key=api_key)

    print(
        f"INFO: Computing embeddings for {len(chunks)} chunks using OpenAI model '{model_name}'..."
    )

    # OpenAI has a limit on batch size and input length
    max_batch_size = 100  # Conservative batch size
    all_embeddings = []
    
    try:
        from tqdm import tqdm
        total_batches = (len(chunks) + max_batch_size - 1) // max_batch_size
        batch_range = range(0, len(chunks), max_batch_size)
        batch_iterator = tqdm(batch_range, desc="Computing embeddings", unit="batch", total=total_batches)
    except ImportError:
        # Fallback without progress bar
        batch_iterator = range(0, len(chunks), max_batch_size)
    
    for i in batch_iterator:
        batch_chunks = chunks[i:i + max_batch_size]
        
        try:
            response = client.embeddings.create(model=model_name, input=batch_chunks)
            batch_embeddings = [embedding.embedding for embedding in response.data]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"ERROR: Failed to get embeddings for batch starting at {i}: {e}")
            raise

    embeddings = np.array(all_embeddings, dtype=np.float32)
    print(
        f"INFO: Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}"
    )
    return embeddings


def compute_embeddings_mlx(chunks: List[str], model_name: str, batch_size: int = 16) -> np.ndarray:
    """Computes embeddings using an MLX model."""
    try:
        import mlx.core as mx
        from mlx_lm.utils import load
        from tqdm import tqdm
    except ImportError as e:
        raise RuntimeError(
            "MLX or related libraries not available. Install with: uv pip install mlx mlx-lm"
        ) from e

    print(
        f"INFO: Computing embeddings for {len(chunks)} chunks using MLX model '{model_name}' with batch_size={batch_size}..."
    )

    # Load model and tokenizer
    model, tokenizer = load(model_name)

    # Process chunks in batches with progress bar
    all_embeddings = []
    
    try:
        from tqdm import tqdm
        batch_iterator = tqdm(range(0, len(chunks), batch_size), desc="Computing embeddings", unit="batch")
    except ImportError:
        batch_iterator = range(0, len(chunks), batch_size)
    
    for i in batch_iterator:
        batch_chunks = chunks[i:i + batch_size]
        
        # Tokenize all chunks in the batch
        batch_token_ids = []
        for chunk in batch_chunks:
            token_ids = tokenizer.encode(chunk)  # type: ignore
            batch_token_ids.append(token_ids)
        
        # Pad sequences to the same length for batch processing
        max_length = max(len(ids) for ids in batch_token_ids)
        padded_token_ids = []
        for token_ids in batch_token_ids:
            # Pad with tokenizer.pad_token_id or 0
            padded = token_ids + [0] * (max_length - len(token_ids))
            padded_token_ids.append(padded)
        
        # Convert to MLX array with batch dimension
        input_ids = mx.array(padded_token_ids)

        # Get embeddings for the batch
        embeddings = model(input_ids)

        # Mean pooling for each sequence in the batch
        pooled = embeddings.mean(axis=1)  # Shape: (batch_size, hidden_size)

        # Convert batch embeddings to numpy
        for j in range(len(batch_chunks)):
            pooled_list = pooled[j].tolist()  # Convert to list
            pooled_numpy = np.array(pooled_list, dtype=np.float32)
            all_embeddings.append(pooled_numpy)

    # Stack numpy arrays
    return np.stack(all_embeddings)


@dataclass
class SearchResult:
    id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PassageManager:
    def __init__(self, passage_sources: List[Dict[str, Any]]):
        self.offset_maps = {}
        self.passage_files = {}
        self.global_offset_map = {}  # Combined map for fast lookup

        for source in passage_sources:
            if source["type"] == "jsonl":
                passage_file = source["path"]
                index_file = source["index_path"]
                if not Path(index_file).exists():
                    raise FileNotFoundError(
                        f"Passage index file not found: {index_file}"
                    )
                with open(index_file, "rb") as f:
                    offset_map = pickle.load(f)
                    self.offset_maps[passage_file] = offset_map
                    self.passage_files[passage_file] = passage_file

                    # Build global map for O(1) lookup
                    for passage_id, offset in offset_map.items():
                        self.global_offset_map[passage_id] = (passage_file, offset)

    def get_passage(self, passage_id: str) -> Dict[str, Any]:
        if passage_id in self.global_offset_map:
            passage_file, offset = self.global_offset_map[passage_id]
            with open(passage_file, "r", encoding="utf-8") as f:
                f.seek(offset)
                return json.loads(f.readline())
        raise KeyError(f"Passage ID not found: {passage_id}")


class LeannBuilder:
    def __init__(
        self,
        backend_name: str,
        embedding_model: str = "facebook/contriever-msmarco",
        dimensions: Optional[int] = None,
        embedding_mode: str = "sentence-transformers",
        **backend_kwargs,
    ):
        self.backend_name = backend_name
        backend_factory: LeannBackendFactoryInterface | None = BACKEND_REGISTRY.get(
            backend_name
        )
        if backend_factory is None:
            raise ValueError(f"Backend '{backend_name}' not found or not registered.")
        self.backend_factory = backend_factory
        self.embedding_model = embedding_model
        self.dimensions = dimensions
        self.embedding_mode = embedding_mode
        self.backend_kwargs = backend_kwargs
        if 'mlx' in self.embedding_model:
            self.embedding_mode = "mlx"
        self.chunks: List[Dict[str, Any]] = []

    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        if metadata is None:
            metadata = {}
        passage_id = metadata.get("id", str(uuid.uuid4()))
        chunk_data = {"id": passage_id, "text": text, "metadata": metadata}
        self.chunks.append(chunk_data)

    def build_index(self, index_path: str):
        if not self.chunks:
            raise ValueError("No chunks added.")
        if self.dimensions is None:
            self.dimensions = len(
                compute_embeddings(
                    ["dummy"],
                    self.embedding_model,
                    self.embedding_mode,
                    use_server=False,
                )[0]
            )
        path = Path(index_path)
        index_dir = path.parent
        index_name = path.name
        index_dir.mkdir(parents=True, exist_ok=True)
        passages_file = index_dir / f"{index_name}.passages.jsonl"
        offset_file = index_dir / f"{index_name}.passages.idx"
        offset_map = {}
        with open(passages_file, "w", encoding="utf-8") as f:
            try:
                from tqdm import tqdm
                chunk_iterator = tqdm(self.chunks, desc="Writing passages", unit="chunk")
            except ImportError:
                chunk_iterator = self.chunks
            
            for chunk in chunk_iterator:
                offset = f.tell()
                json.dump(
                    {
                        "id": chunk["id"],
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")
                offset_map[chunk["id"]] = offset
        with open(offset_file, "wb") as f:
            pickle.dump(offset_map, f)
        texts_to_embed = [c["text"] for c in self.chunks]
        embeddings = compute_embeddings(
            texts_to_embed, self.embedding_model, self.embedding_mode, use_server=False
        )
        string_ids = [chunk["id"] for chunk in self.chunks]
        current_backend_kwargs = {**self.backend_kwargs, "dimensions": self.dimensions}
        builder_instance = self.backend_factory.builder(**current_backend_kwargs)
        builder_instance.build(
            embeddings, string_ids, index_path, **current_backend_kwargs
        )
        leann_meta_path = index_dir / f"{index_name}.meta.json"
        meta_data = {
            "version": "1.0",
            "backend_name": self.backend_name,
            "embedding_model": self.embedding_model,
            "dimensions": self.dimensions,
            "backend_kwargs": self.backend_kwargs,
            "embedding_mode": self.embedding_mode,
            "passage_sources": [
                {
                    "type": "jsonl",
                    "path": str(passages_file),
                    "index_path": str(offset_file),
                }
            ],
        }

        # Add storage status flags for HNSW backend
        if self.backend_name == "hnsw":
            is_compact = self.backend_kwargs.get("is_compact", True)
            is_recompute = self.backend_kwargs.get("is_recompute", True)
            meta_data["is_compact"] = is_compact
            meta_data["is_pruned"] = (
                is_compact and is_recompute
            )  # Pruned only if compact and recompute
        with open(leann_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)

    def build_index_from_embeddings(self, index_path: str, embeddings_file: str):
        """
        Build an index from pre-computed embeddings stored in a pickle file.

        Args:
            index_path: Path where the index will be saved
            embeddings_file: Path to pickle file containing (ids, embeddings) tuple
        """
        # Load pre-computed embeddings
        with open(embeddings_file, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, tuple) or len(data) != 2:
            raise ValueError(
                f"Invalid embeddings file format. Expected tuple with 2 elements, got {type(data)}"
            )

        ids, embeddings = data

        if not isinstance(embeddings, np.ndarray):
            raise ValueError(
                f"Expected embeddings to be numpy array, got {type(embeddings)}"
            )

        if len(ids) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch between number of IDs ({len(ids)}) and embeddings ({embeddings.shape[0]})"
            )

        # Validate/set dimensions
        embedding_dim = embeddings.shape[1]
        if self.dimensions is None:
            self.dimensions = embedding_dim
        elif self.dimensions != embedding_dim:
            raise ValueError(
                f"Dimension mismatch: expected {self.dimensions}, got {embedding_dim}"
            )

        print(
            f"Building index from precomputed embeddings: {len(ids)} items, {embedding_dim} dimensions"
        )

        # Ensure we have text data for each embedding
        if len(self.chunks) != len(ids):
            # If no text chunks provided, create placeholder text entries
            if not self.chunks:
                print("No text chunks provided, creating placeholder entries...")
                for id_val in ids:
                    self.add_text(
                        f"Document {id_val}",
                        metadata={"id": str(id_val), "from_embeddings": True},
                    )
            else:
                raise ValueError(
                    f"Number of text chunks ({len(self.chunks)}) doesn't match number of embeddings ({len(ids)})"
                )

        # Build file structure
        path = Path(index_path)
        index_dir = path.parent
        index_name = path.name
        index_dir.mkdir(parents=True, exist_ok=True)
        passages_file = index_dir / f"{index_name}.passages.jsonl"
        offset_file = index_dir / f"{index_name}.passages.idx"

        # Write passages and create offset map
        offset_map = {}
        with open(passages_file, "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                offset = f.tell()
                json.dump(
                    {
                        "id": chunk["id"],
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")
                offset_map[chunk["id"]] = offset

        with open(offset_file, "wb") as f:
            pickle.dump(offset_map, f)

        # Build the vector index using precomputed embeddings
        string_ids = [str(id_val) for id_val in ids]
        current_backend_kwargs = {**self.backend_kwargs, "dimensions": self.dimensions}
        builder_instance = self.backend_factory.builder(**current_backend_kwargs)
        builder_instance.build(embeddings, string_ids, index_path)

        # Create metadata file
        leann_meta_path = index_dir / f"{index_name}.meta.json"
        meta_data = {
            "version": "1.0",
            "backend_name": self.backend_name,
            "embedding_model": self.embedding_model,
            "dimensions": self.dimensions,
            "backend_kwargs": self.backend_kwargs,
            "embedding_mode": self.embedding_mode,
            "passage_sources": [
                {
                    "type": "jsonl",
                    "path": str(passages_file),
                    "index_path": str(offset_file),
                }
            ],
            "built_from_precomputed_embeddings": True,
            "embeddings_source": str(embeddings_file),
        }

        # Add storage status flags for HNSW backend
        if self.backend_name == "hnsw":
            is_compact = self.backend_kwargs.get("is_compact", True)
            is_recompute = self.backend_kwargs.get("is_recompute", True)
            meta_data["is_compact"] = is_compact
            meta_data["is_pruned"] = is_compact and is_recompute

        with open(leann_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)

        print(f"Index built successfully from precomputed embeddings: {index_path}")


class LeannSearcher:
    def __init__(self, index_path: str, enable_warmup: bool = False, **backend_kwargs):
        meta_path_str = f"{index_path}.meta.json"
        if not Path(meta_path_str).exists():
            raise FileNotFoundError(f"Leann metadata file not found at {meta_path_str}")
        with open(meta_path_str, "r", encoding="utf-8") as f:
            self.meta_data = json.load(f)
        backend_name = self.meta_data["backend_name"]
        self.embedding_model = self.meta_data["embedding_model"]
        # Support both old and new format
        self.embedding_mode = self.meta_data.get(
            "embedding_mode", "sentence-transformers"
        )
        # Backward compatibility with use_mlx
        if self.meta_data.get("use_mlx", False):
            self.embedding_mode = "mlx"
        self.passage_manager = PassageManager(self.meta_data.get("passage_sources", []))
        backend_factory = BACKEND_REGISTRY.get(backend_name)
        if backend_factory is None:
            raise ValueError(f"Backend '{backend_name}' not found.")
        final_kwargs = {**self.meta_data.get("backend_kwargs", {}), **backend_kwargs}
        final_kwargs["enable_warmup"] = enable_warmup
        self.backend_impl = backend_factory.searcher(index_path, **final_kwargs)

    def search(
        self,
        query: str,
        top_k: int = 5,
        complexity: int = 64,
        beam_width: int = 1,
        prune_ratio: float = 0.0,
        recompute_embeddings: bool = False,
        pruning_strategy: Literal["global", "local", "proportional"] = "global",
        zmq_port: int = 5557,
        **kwargs,
    ) -> List[SearchResult]:
        print("🔍 DEBUG LeannSearcher.search() called:")
        print(f"  Query: '{query}'")
        print(f"  Top_k: {top_k}")
        print(f"  Additional kwargs: {kwargs}")

        # Use backend's compute_query_embedding method
        # This will automatically use embedding server if available and needed
        import time

        start_time = time.time()

        query_embedding = self.backend_impl.compute_query_embedding(query, zmq_port)
        print(f"  Generated embedding shape: {query_embedding.shape}")
        embedding_time = time.time() - start_time
        print(f"  Embedding time: {embedding_time} seconds")

        start_time = time.time()
        results = self.backend_impl.search(
            query_embedding,
            top_k,
            complexity=complexity,
            beam_width=beam_width,
            prune_ratio=prune_ratio,
            recompute_embeddings=recompute_embeddings,
            pruning_strategy=pruning_strategy,
            zmq_port=zmq_port,
            **kwargs,
        )
        search_time = time.time() - start_time
        print(f"  Search time: {search_time} seconds")
        print(
            f"  Backend returned: labels={len(results.get('labels', [[]])[0])} results"
        )

        enriched_results = []
        if "labels" in results and "distances" in results:
            print(f"  Processing {len(results['labels'][0])} passage IDs:")
            for i, (string_id, dist) in enumerate(
                zip(results["labels"][0], results["distances"][0])
            ):
                try:
                    passage_data = self.passage_manager.get_passage(string_id)
                    enriched_results.append(
                        SearchResult(
                            id=string_id,
                            score=dist,
                            text=passage_data["text"],
                            metadata=passage_data.get("metadata", {}),
                        )
                    )
                    print(
                        f"    {i + 1}. passage_id='{string_id}' -> SUCCESS: {passage_data['text']}..."
                    )
                except KeyError:
                    print(
                        f"    {i + 1}. passage_id='{string_id}' -> ERROR: Passage not found in PassageManager!"
                    )

        print(f"  Final enriched results: {len(enriched_results)} passages")
        return enriched_results


class LeannChat:
    def __init__(
        self,
        index_path: str,
        llm_config: Optional[Dict[str, Any]] = None,
        enable_warmup: bool = False,
        **kwargs,
    ):
        self.searcher = LeannSearcher(index_path, enable_warmup=enable_warmup, **kwargs)
        self.llm = get_llm(llm_config)

    def ask(
        self,
        question: str,
        top_k: int = 5,
        complexity: int = 64,
        beam_width: int = 1,
        prune_ratio: float = 0.0,
        recompute_embeddings: bool = False,
        pruning_strategy: Literal["global", "local", "proportional"] = "global",
        zmq_port: int = 5557,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **search_kwargs,
    ):
        if llm_kwargs is None:
            llm_kwargs = {}

        results = self.searcher.search(
            question,
            top_k=top_k,
            complexity=complexity,
            beam_width=beam_width,
            prune_ratio=prune_ratio,
            recompute_embeddings=recompute_embeddings,
            pruning_strategy=pruning_strategy,
            zmq_port=zmq_port,
            **search_kwargs,
        )
        context = "\n\n".join([r.text for r in results])
        prompt = (
            "Here is some retrieved context that might help answer your question:\n\n"
            f"{context}\n\n"
            f"Question: {question}\n\n"
            "Please provide the best answer you can based on this context and your knowledge."
        )

        ans = self.llm.ask(prompt, **llm_kwargs)
        return ans

    def start_interactive(self):
        print("\nLeann Chat started (type 'quit' to exit)")
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ["quit", "exit"]:
                    break
                if not user_input:
                    continue
                response = self.ask(user_input)
                print(f"Leann: {response}")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
