import json
from abc import ABC, abstractmethod
from pathlib import Path
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

        self.embedding_server_manager = EmbeddingServerManager(
            backend_module_name=backend_module_name,
        )

    def _load_meta(self) -> dict[str, Any]:
        """Loads the metadata file associated with the index."""
        # This is the corrected logic for finding the meta file.
        meta_path = self.index_dir / f"{self.index_path.name}.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Leann metadata file not found at {meta_path}")
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)

    def _ensure_server_running(self, passages_source_file: str, port: int, **kwargs) -> int:
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

        server_started, actual_port = self.embedding_server_manager.start_server(
            port=port,
            model_name=self.embedding_model,
            embedding_mode=self.embedding_mode,
            passages_file=passages_source_file,
            distance_metric=distance_metric,
            enable_warmup=kwargs.get("enable_warmup", False),
        )
        if not server_started:
            raise RuntimeError(f"Failed to start embedding server on port {actual_port}")

        return actual_port

    def compute_query_embedding(
        self,
        query: str,
        use_server_if_available: bool = True,
        zmq_port: int = 5557,
    ) -> np.ndarray:
        """
        Compute embedding for a query string.

        Args:
            query: The query string to embed
            zmq_port: ZMQ port for embedding server
            use_server_if_available: Whether to try using embedding server first

        Returns:
            Query embedding as numpy array
        """
        # Try to use embedding server if available and requested
        if use_server_if_available:
            try:
                # TODO: Maybe we can directly use this port here?
                # For this internal method, it's ok to assume that the server is running
                # on that port?

                # Ensure we have a server with passages_file for compatibility
                passages_source_file = self.index_dir / f"{self.index_path.name}.meta.json"
                # Convert to absolute path to ensure server can find it
                zmq_port = self._ensure_server_running(
                    str(passages_source_file.resolve()), zmq_port
                )

                return self._compute_embedding_via_server([query], zmq_port)[
                    0:1
                ]  # Return (1, D) shape
            except Exception as e:
                print(f"⚠️ Embedding server failed: {e}")
                print("⏭️ Falling back to direct model loading...")

        # Fallback to direct computation
        from .embedding_compute import compute_embeddings

        embedding_mode = self.meta.get("embedding_mode", "sentence-transformers")
        return compute_embeddings([query], self.embedding_model, embedding_mode)

    def _compute_embedding_via_server(self, chunks: list, zmq_port: int) -> np.ndarray:
        """Compute embeddings using the ZMQ embedding server."""
        import msgpack
        import zmq

        try:
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30 second timeout
            socket.connect(f"tcp://localhost:{zmq_port}")

            # Send embedding request
            request = chunks
            request_bytes = msgpack.packb(request)
            socket.send(request_bytes)

            # Wait for response
            response_bytes = socket.recv()
            response = msgpack.unpackb(response_bytes)

            socket.close()
            context.term()

            # Convert response to numpy array
            if isinstance(response, list) and len(response) > 0:
                return np.array(response, dtype=np.float32)
            else:
                raise RuntimeError("Invalid response from embedding server")

        except Exception as e:
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
        if hasattr(self, "embedding_server_manager"):
            self.embedding_server_manager.stop_server()
