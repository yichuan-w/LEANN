
import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List

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

        self.label_map = self._load_label_map()

        self.embedding_server_manager = EmbeddingServerManager(
            backend_module_name=backend_module_name
        )

    def _load_meta(self) -> Dict[str, Any]:
        """Loads the metadata file associated with the index."""
        # This is the corrected logic for finding the meta file.
        meta_path = self.index_dir / f"{self.index_path.name}.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Leann metadata file not found at {meta_path}")
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_label_map(self) -> Dict[int, str]:
        """Loads the mapping from integer IDs to string IDs."""
        label_map_file = self.index_dir / "leann.labels.map"
        if not label_map_file.exists():
            raise FileNotFoundError(f"Label map file not found: {label_map_file}")
        with open(label_map_file, 'rb') as f:
            return pickle.load(f)

    def _ensure_server_running(self, passages_source_file: str, port: int, **kwargs) -> None:
        """
        Ensures the embedding server is running if recompute is needed.
        This is a helper for subclasses.
        """
        if not self.embedding_model:
            raise ValueError("Cannot use recompute mode without 'embedding_model' in meta.json.")

        server_started = self.embedding_server_manager.start_server(
            port=port,
            model_name=self.embedding_model,
            passages_file=passages_source_file,
            distance_metric=kwargs.get("distance_metric"),
        )
        if not server_started:
            raise RuntimeError(f"Failed to start embedding server on port {kwargs.get('zmq_port')}")

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int, **kwargs) -> Dict[str, Any]:
        """
        Search for the top_k nearest neighbors of the query vector.
        Must be implemented by subclasses.
        """
        pass

    def __del__(self):
        """Ensures the embedding server is stopped when the searcher is destroyed."""
        if hasattr(self, 'embedding_server_manager'):
            self.embedding_server_manager.stop_server()

