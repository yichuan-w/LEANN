import json
from abc import ABC, abstractmethod
from pathlib import Path
import threading
from typing import Any, Literal, Optional

import numpy as np

from .embedding_server_manager import EmbeddingServerManager
from .interface import LeannBackendSearcherInterface


class BaseSearcher(LeannBackendSearcherInterface, ABC):
    """
    Abstract base class for Leann searchers, containing common logic for
    loading metadata, managing embedding servers, and handling file paths.
    """

    def __init__(self, index_path: str, backend_module_name: str, **kwargs):
        """
        Initializes the BaseSearcher.

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

        # Persistent ZMQ connection state
        self._zmq_lock = threading.Lock()
        self._zmq_context = None
        self._zmq_socket = None
        self._zmq_current_host = None
        self._zmq_current_port = None

    def _load_meta(self) -> dict[str, Any]:
        """Loads the metadata file associated with the index."""
        # This is the corrected logic for finding the meta file.
        meta_path = self.index_dir / f"{self.index_path.name}.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Leann metadata file not found at {meta_path}")
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)

    def _ensure_server_running(
        self, passages_source_file: str, port: Optional[int], **kwargs
    ) -> tuple[str, int]:
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
        # Templates are applied in compute_query_embedding (line 109-110) BEFORE server call
        # The server should never apply templates during search to avoid double-templating
        search_provider_options = {
            k: v
            for k, v in self.embedding_options.items()
            if k not in ("build_prompt_template", "query_prompt_template", "prompt_template")
        }

        server_started, host, actual_port = self.embedding_server_manager.start_server(
            port=port if port is not None else 5557,
            model_name=self.embedding_model,
            embedding_mode=self.embedding_mode,
            passages_file=passages_source_file,
            distance_metric=distance_metric,
            enable_warmup=kwargs.get("enable_warmup", False),
            provider_options=search_provider_options,
        )
        if not server_started:
            raise RuntimeError(f"Failed to start embedding server on port {actual_port}")

        return host, actual_port

    def compute_query_embedding(
        self,
        query: str,
        use_server_if_available: bool = True,
        zmq_port: Optional[int] = None,
        zmq_host: str = "localhost",
        query_template: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute embedding for a query string.

        Args:
            query: The query string to embed
            zmq_port: ZMQ port for embedding server
            use_server_if_available: Whether to try using embedding server first
            query_template: Optional prompt template to prepend to query

        Returns:
            Query embedding as numpy array
        """
        # Apply query template BEFORE any computation path
        # This ensures template is applied consistently for both server and fallback paths
        if query_template:
            query = f"{query_template}{query}"

        # Try to use embedding server if available and requested
        if use_server_if_available:
            try:
                # TODO: Maybe we can directly use this port here?
                # For this internal method, it's ok to assume that the server is running
                # on that port?

                # Ensure we have a server with passages_file for compatibility
                passages_source_file = self.index_dir / f"{self.index_path.name}.meta.json"
                # Convert to absolute path to ensure server can find it
                zmq_host, zmq_port = self._ensure_server_running(
                    str(passages_source_file.resolve()), zmq_port
                )

                return self._compute_embedding_via_server([query], zmq_host, zmq_port)[
                    0:1
                ]  # Return (1, D) shape
            except Exception as e:
                print(f"⚠️ Embedding server failed: {e}")
                print("⏭️ Falling back to direct model loading...")

        # Fallback to direct computation
        from .embedding_compute import compute_embeddings

        embedding_mode = self.meta.get("embedding_mode", "sentence-transformers")
        return compute_embeddings(
            [query],
            self.embedding_model,
            embedding_mode,
            provider_options=self.embedding_options,
        )

    def _close_zmq(self):
        """Closes the ZMQ socket and context safely."""
        try:
            if self._zmq_socket:
                self._zmq_socket.close()
                self._zmq_socket = None
            if self._zmq_context:
                self._zmq_context.term()
                self._zmq_context = None
            self._zmq_current_host = None
            self._zmq_current_port = None
        except Exception as e:
            print(f"Error closing ZMQ socket: {e}")

    def _compute_embedding_via_server(self, chunks: list, zmq_host: str, zmq_port: int) -> np.ndarray:
        """Compute embeddings using the ZMQ embedding server with persistent connection."""
        import msgpack
        import zmq

        with self._zmq_lock:
            # Reconnect if setting changed or socket missing
            if (
                self._zmq_socket is None
                or zmq_host != self._zmq_current_host
                or zmq_port != self._zmq_current_port
            ):
                if self._zmq_socket:
                    self._zmq_socket.close()
                
                if self._zmq_context is None:
                    self._zmq_context = zmq.Context()
                
                self._zmq_socket = self._zmq_context.socket(zmq.REQ)
                self._zmq_socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30 second timeout
                self._zmq_socket.setsockopt(zmq.LINGER, 0)
                try:
                    self._zmq_socket.connect(f"tcp://{zmq_host}:{zmq_port}")
                except Exception as e:
                    self._zmq_socket.close()
                    self._zmq_socket = None
                    raise RuntimeError(f"Failed to connect to ZMQ server: {e}")

                self._zmq_current_host = zmq_host
                self._zmq_current_port = zmq_port

            try:
                # Send embedding request
                request = chunks
                request_bytes = msgpack.packb(request)
                self._zmq_socket.send(request_bytes)

                # Wait for response
                response_bytes = self._zmq_socket.recv()
                response = msgpack.unpackb(response_bytes)

                # Convert response to numpy array
                if isinstance(response, list) and len(response) > 0:
                    return np.array(response, dtype=np.float32)
                else:
                    raise RuntimeError("Invalid response from embedding server")

            except (zmq.ZMQError, Exception) as e:
                # On error, force reconnect next time
                self._close_zmq()
                raise RuntimeError(f"Failed to compute embeddings via server: {e}")

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
        zmq_host: str = "localhost",
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
        """Ensures the embedding server is stopped when the searcher is destroyed."""
        self._close_zmq()
        if hasattr(self, "embedding_server_manager"):
            self.embedding_server_manager.stop_server()
