from abc import ABC, abstractmethod
from typing import Any, Literal, Union

import numpy as np


class LeannBackendBuilderInterface(ABC):
    """Backend interface for building indexes"""

    @abstractmethod
    def build(self, data: np.ndarray, ids: list[str], index_path: str, **kwargs) -> None:
        """Build index

        Args:
            data: Vector data (N, D)
            ids: List of string IDs for each vector
            index_path: Path to save index
            **kwargs: Backend-specific build parameters
        """
        pass


class LeannBackendSearcherInterface(ABC):
    """Backend interface for searching"""

    @abstractmethod
    def __init__(self, index_path: str, **kwargs):
        """Initialize searcher

        Args:
            index_path: Path to index file
            **kwargs: Backend-specific loading parameters
        """
        pass

    @abstractmethod
    def _ensure_server_running(
        self, passages_source_file: str, port: Union[int, None], **kwargs
    ) -> int:
        """Ensure server is running"""
        pass

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
        zmq_port: Union[int, None] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Search for nearest neighbors

        Args:
            query: Query vectors (B, D) where B is batch size, D is dimension
            top_k: Number of nearest neighbors to return
            complexity: Search complexity/candidate list size, higher = more accurate but slower
            beam_width: Number of parallel search paths/IO requests per iteration
            prune_ratio: Ratio of neighbors to prune via approximate distance (0.0-1.0)
            recompute_embeddings: Whether to fetch fresh embeddings from server vs use stored PQ codes
            pruning_strategy: PQ candidate selection strategy - "global" (default), "local", or "proportional"
            zmq_port: ZMQ port for embedding server communication. Must be provided if recompute_embeddings is True.
            **kwargs: Backend-specific parameters

        Returns:
            {"labels": [...], "distances": [...]}
        """
        pass

    @abstractmethod
    def compute_query_embedding(
        self,
        query: str,
        use_server_if_available: bool = True,
        zmq_port: Union[int, None] = None,
    ) -> np.ndarray:
        """Compute embedding for a query string

        Args:
            query: The query string to embed
            zmq_port: ZMQ port for embedding server
            use_server_if_available: Whether to try using embedding server first

        Returns:
            Query embedding as numpy array with shape (1, D)
        """
        pass


class LeannBackendFactoryInterface(ABC):
    """Backend factory interface"""

    @staticmethod
    @abstractmethod
    def builder(**kwargs) -> LeannBackendBuilderInterface:
        """Create Builder instance"""
        pass

    @staticmethod
    @abstractmethod
    def searcher(index_path: str, **kwargs) -> LeannBackendSearcherInterface:
        """Create Searcher instance"""
        pass
