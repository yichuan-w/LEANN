"""
Optimized version of searcher_base.py with:
1. Query embedding caching
2. ZMQ connection reuse
3. Model persistence checks
"""

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

from .embedding_server_manager import EmbeddingServerManager
from .interface import LeannBackendSearcherInterface


class QueryEmbeddingCache:
    """Hash-based cache for query embeddings."""

    def __init__(self, max_size: int = 1000):
        self.cache: dict[str, np.ndarray] = {}
        self.max_size = max_size

    def _hash_query(self, query: str, query_template: Optional[str] = None) -> str:
        """Create hash key for query."""
        key_data = {
            "query": query,
            "template": query_template or "",
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, query: str, query_template: Optional[str] = None) -> Optional[np.ndarray]:
        """Get cached embedding if exists."""
        key = self._hash_query(query, query_template)
        return self.cache.get(key)

    def put(self, query: str, embedding: np.ndarray, query_template: Optional[str] = None):
        """Cache embedding."""
        key = self._hash_query(query, query_template)

        # Simple LRU: remove oldest if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Remove first item (oldest)
            first_key = next(iter(self.cache))
            del self.cache[first_key]

        self.cache[key] = embedding.copy()

    def clear(self):
        """Clear cache."""
        self.cache.clear()


class ReusableZMQConnection:
    """Reusable ZMQ connection to avoid creating new context/socket per request."""

    def __init__(self):
        self.context = None
        self.socket = None
        self.port = None

    def connect(self, port: int):
        """Connect to ZMQ server on given port."""
        import zmq

        if self.port == port and self.socket is not None:
            # Already connected to this port
            return

        # Close existing connection
        self.close()

        # Create new connection
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30 second timeout
        self.socket.setsockopt(zmq.LINGER, 0)  # Don't wait on close
        self.socket.connect(f"tcp://localhost:{port}")
        self.port = port

    def send_recv(self, data: list) -> list:
        """Send data and receive response."""
        import msgpack

        if self.socket is None:
            raise RuntimeError("ZMQ connection not established")

        # Send request
        request_bytes = msgpack.packb(data)
        self.socket.send(request_bytes)

        # Receive response
        response_bytes = self.socket.recv()
        response = msgpack.unpackb(response_bytes)

        return response

    def close(self):
        """Close ZMQ connection."""
        if self.socket is not None:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None

        if self.context is not None:
            try:
                self.context.term()
            except Exception:
                pass
            self.context = None

        self.port = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class BaseSearcherOptimized(LeannBackendSearcherInterface, ABC):
    """
    Optimized base searcher with query embedding caching and ZMQ connection reuse.
    """

    def __init__(self, index_path: str, backend_module_name: str, **kwargs):
        """
        Initializes the Optimized BaseSearcher.

        Args:
            index_path: Path to the Leann index file (e.g., '.../my_index.leann').
            backend_module_name: The specific embedding server module to use
                                 (e.g., 'leann_backend_hnsw.hnsw_embedding_server').
            **kwargs: Additional keyword arguments.
        """
        self.index_path = Path(index_path)
        self.index_dir = self.index_path.parent
        self.meta = kwargs.get("meta", self._load_meta())

        if not self.meta:
            raise ValueError("Searcher requires metadata from .meta.json.")

        self.dimensions = self.meta.get("dimensions")
        if not self.dimensions:
            raise ValueError("Dimensions not found in Leann metadata.")

        self.embedding_model = self.meta.get("embedding_model")
        if not self.embedding_model:
            print("WARNING: embedding_model not found in meta.json. Recompute will fail.")

        self.embedding_mode = self.meta.get("embedding_mode", "sentence-transformers")
        self.embedding_options = self.meta.get("embedding_options", {})

        self.embedding_server_manager = EmbeddingServerManager(
            backend_module_name=backend_module_name,
        )

        # Optimization: Query embedding cache
        cache_size = kwargs.get("query_cache_size", 1000)
        self.query_cache = QueryEmbeddingCache(max_size=cache_size)

        # Optimization: Reusable ZMQ connection
        self.zmq_connection = ReusableZMQConnection()
        self._zmq_port: Optional[int] = None

    def _load_meta(self) -> dict[str, Any]:
        """Loads the metadata file associated with the index."""
        meta_path = self.index_dir / f"{self.index_path.name}.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Leann metadata file not found at {meta_path}")
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)

    def _ensure_server_running(
        self, passages_source_file: str, port: Optional[int], **kwargs
    ) -> int:
        """
        Ensures the embedding server is running if recompute is needed.
        This is a helper for subclasses.
        """
        if not self.embedding_model:
            raise ValueError("Cannot use recompute mode without 'embedding_model' in meta.json.")

        # Get distance_metric from meta if not provided in kwargs
        distance_metric = (
            kwargs.get("distance_metric")
            or self.meta.get("backend_kwargs", {}).get("distance_metric")
            or "mips"
        )

        # Filter out ALL prompt templates from provider_options during search
        search_provider_options = {
            k: v
            for k, v in self.embedding_options.items()
            if k not in ("build_prompt_template", "query_prompt_template", "prompt_template")
        }

        server_started, actual_port = self.embedding_server_manager.start_server(
            port=port if port is not None else 5557,
            model_name=self.embedding_model,
            embedding_mode=self.embedding_mode,
            passages_file=passages_source_file,
            distance_metric=distance_metric,
            enable_warmup=kwargs.get("enable_warmup", False),
            use_daemon=kwargs.get("use_daemon", True),
            daemon_ttl_seconds=kwargs.get("daemon_ttl_seconds", 900),
            provider_options=search_provider_options,
        )
        if not server_started:
            raise RuntimeError(f"Failed to start embedding server on port {actual_port}")

        # Update ZMQ connection if port changed
        if self._zmq_port != actual_port:
            self.zmq_connection.connect(actual_port)
            self._zmq_port = actual_port

        return actual_port

    def compute_query_embedding(
        self,
        query: str,
        use_server_if_available: bool = True,
        zmq_port: Optional[int] = None,
        query_template: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute embedding for a query string with caching and connection reuse.

        Args:
            query: The query string to embed
            zmq_port: ZMQ port for embedding server
            use_server_if_available: Whether to try using embedding server first
            query_template: Optional prompt template to prepend to query

        Returns:
            Query embedding as numpy array
        """
        # Check cache first
        cached = self.query_cache.get(query, query_template)
        if cached is not None:
            return cached

        # Apply query template BEFORE any computation path
        if query_template:
            query = f"{query_template}{query}"

        # Try to use embedding server if available and requested
        if use_server_if_available:
            try:
                # Ensure we have a server with passages_file for compatibility
                passages_source_file = self.index_dir / f"{self.index_path.name}.meta.json"
                # Convert to absolute path to ensure server can find it
                actual_port = self._ensure_server_running(
                    str(passages_source_file.resolve()), zmq_port
                )

                # Use reusable connection
                embedding = self._compute_embedding_via_server_optimized([query], actual_port)[
                    0:1
                ]  # Return (1, D) shape

                # Cache the result (use original query before template for cache key)
                if query_template:
                    original_query = query[len(query_template) :]
                else:
                    original_query = query
                self.query_cache.put(original_query, embedding[0], query_template)

                return embedding
            except Exception as e:
                print(f"⚠️ Embedding server failed: {e}")
                print("⏭️ Falling back to direct model loading...")

        # Fallback to direct computation
        from .embedding_compute import compute_embeddings

        embedding_mode = self.meta.get("embedding_mode", "sentence-transformers")
        embedding = compute_embeddings(
            [query],
            self.embedding_model,
            embedding_mode,
            provider_options=self.embedding_options,
        )

        # Cache the result
        if query_template:
            original_query = query[len(query_template) :]
        else:
            original_query = query
        self.query_cache.put(original_query, embedding[0], query_template)

        return embedding

    def _compute_embedding_via_server_optimized(self, chunks: list, zmq_port: int) -> np.ndarray:
        """Compute embeddings using the ZMQ embedding server with connection reuse."""
        # Ensure connection is established
        self.zmq_connection.connect(zmq_port)

        # Send request and get response
        response = self.zmq_connection.send_recv(chunks)

        # Convert response to numpy array
        if isinstance(response, list) and len(response) > 0:
            return np.array(response, dtype=np.float32)
        else:
            raise RuntimeError("Invalid response from embedding server")

    @abstractmethod
    def search(
        self,
        query: np.ndarray,
        top_k: int,
        complexity: int = 64,
        beam_width: int = 1,
        prune_ratio: float = 0.0,
        recompute_embeddings: bool = False,
        pruning_strategy: Literal["global", "local", "proportional"] = "global",
        zmq_port: Optional[int] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Search for the top_k nearest neighbors of the query vector.

        Args:
            query: Query vectors (B, D) where B is batch size, D is dimension
            top_k: Number of nearest neighbors to return
            complexity: Search complexity/candidate list size, higher = more accurate but slower
            beam_width: Number of parallel search paths/IO requests per iteration
            prune_ratio: Ratio of neighbors to prune via approximate distance (0.0-1.0)
            recompute_embeddings: Whether to fetch fresh embeddings from server vs use stored PQ codes
            pruning_strategy: PQ candidate selection strategy - "global" (default), "local", or "proportional"
            zmq_port: ZMQ port for embedding server communication. Must be provided if recompute_embeddings is True.
            **kwargs: Backend-specific parameters (e.g., batch_size, dedup_node_dis, etc.)

        Returns:
            Dict with 'labels' (list of lists) and 'distances' (ndarray)
        """
        pass

    def __del__(self):
        """Ensures cleanup when the searcher is destroyed."""
        if hasattr(self, "zmq_connection"):
            self.zmq_connection.close()
        if hasattr(self, "embedding_server_manager"):
            self.embedding_server_manager.stop_server()
