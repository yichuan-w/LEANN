"""
FAISS-based vector search backend for LEANN.

Provides GPU-accelerated similarity search with automatic CPU fallback.
Uses adaptive indexing strategy based on dataset size.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Literal, Optional

import faiss
import numpy as np
from leann.interface import (
    LeannBackendBuilderInterface,
    LeannBackendFactoryInterface,
    LeannBackendSearcherInterface,
)
from leann.registry import register_backend
from leann.searcher_base import BaseSearcher

from . import faiss_embedding_server

logger = logging.getLogger(__name__)

__all__ = [
    "FaissBackendBuilder",
    "FaissBackendFactory",
    "FaissBackendSearcher",
    "faiss_embedding_server",
]


class FaissBackendBuilder(LeannBackendBuilderInterface):
    """FAISS-based index builder with GPU acceleration.

    Uses adaptive indexing strategy:
    - Small datasets (<100k): GpuIndexFlatIP (brute-force, exact, fast on GPU)
    - Large datasets (>=100k): IVF{nlist},Flat (approximate, partitioned search)

    CPU fallback uses IndexFlatIP which benefits from AVX2 SIMD optimizations
    when available.
    """

    # Batch size for adding vectors to prevent OOM on large datasets
    ADD_BATCH_SIZE = 65536

    def build(self, data: np.ndarray, ids: list[str], index_path: str, **kwargs) -> None:
        """Build FAISS index with optional GPU acceleration."""
        logger.info(f"Building FAISS index with shape {data.shape}")

        # Extract config from kwargs to save in metadata
        embedding_model = kwargs.get("embedding_model", "nomic-ai/nomic-embed-text-v1.5")
        embedding_mode = kwargs.get("embedding_mode", "sentence-transformers")

        d = data.shape[1]

        # Use GPU resources
        try:
            res = faiss.StandardGpuResources()
            logger.info("FAISS: GPU resources initialized")
            use_gpu = True
        except Exception as e:
            logger.warning(f"FAISS: Could not initialize GPU resources: {e}. Falling back to CPU.")
            use_gpu = False

        # Create index with adaptive strategy based on dataset size
        # Metric: Inner Product with L2 normalization = Cosine Similarity
        metric = faiss.METRIC_INNER_PRODUCT

        if use_gpu:
            try:
                if data.shape[0] < 100000:
                    # Brute-force exact search - fast on GPU for small-medium datasets
                    config = faiss.GpuIndexFlatConfig()
                    config.useFloat16 = True  # Halve VRAM usage
                    index = faiss.GpuIndexFlatIP(res, d, config)
                    logger.info("FAISS: Created GpuIndexFlatIP (exact search, fp16)")
                else:
                    # IVF for larger datasets - trades small recall for massive speed gains
                    nlist = int(np.sqrt(data.shape[0]))
                    index = faiss.index_factory(d, f"IVF{nlist},Flat", metric)
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                    logger.info(f"FAISS: Created GPU IVF{nlist},Flat index")
            except Exception as e:
                logger.warning(f"FAISS: Failed to create GPU index: {e}. Falling back to CPU.")
                use_gpu = False

        if not use_gpu:
            # CPU fallback - IndexFlatIP benefits from AVX2 SIMD optimizations
            index = faiss.IndexFlatIP(d)
            logger.info("FAISS: Created CPU IndexFlatIP (AVX2 optimized when available)")

        # Normalize for cosine similarity (IP + L2 norm = cosine)
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        faiss.normalize_L2(data)

        # Train if needed (IVF indices require training)
        if not index.is_trained:
            logger.info("FAISS: Training index...")
            index.train(data)

        # Add vectors in batches to prevent OOM on large datasets
        n_vectors = len(data)
        for i in range(0, n_vectors, self.ADD_BATCH_SIZE):
            end_idx = min(i + self.ADD_BATCH_SIZE, n_vectors)
            index.add(data[i:end_idx])
            if n_vectors > self.ADD_BATCH_SIZE:
                logger.debug(
                    f"FAISS: Added batch {i // self.ADD_BATCH_SIZE + 1} ({end_idx}/{n_vectors})"
                )

        logger.info(f"FAISS: Added {index.ntotal} vectors to index")

        # Convert GPU index to CPU for serialization
        if use_gpu:
            index_cpu = faiss.index_gpu_to_cpu(index)
        else:
            index_cpu = index

        # Save FAISS index
        index_file = Path(index_path)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index_cpu, str(index_file))

        # Save IDs separately (FAISS only handles integer indices)
        ids_file = index_file.with_suffix(".ids.pkl")
        with open(ids_file, "wb") as f:
            pickle.dump(ids, f)

        # Save metadata for Searcher to load embedding config
        meta_file = f"{index_path}.meta.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "embedding_model": embedding_model,
                    "embedding_mode": embedding_mode,
                    "count": len(ids),
                    "dims": d,
                },
                f,
                indent=2,
            )

        logger.info(f"FAISS: Saved index, IDs, and metadata to {index_file.parent}")


class FaissBackendSearcher(BaseSearcher):
    """FAISS-based searcher with GPU acceleration.

    Extends BaseSearcher to inherit proper embedding server lifecycle management
    via EmbeddingServerManager.
    """

    def __init__(self, index_path: str, **kwargs):
        # Initialize BaseSearcher with FAISS embedding server module
        super().__init__(
            index_path,
            backend_module_name="leann_backend_faiss.faiss_embedding_server",
            **kwargs,
        )

        logger.info(f"FAISS: Loading index from {self.index_path}")

        # Load FAISS index
        self.index_cpu = faiss.read_index(str(self.index_path))

        # Load IDs
        ids_file = self.index_path.with_suffix(".ids.pkl")
        with open(ids_file, "rb") as f:
            self.ids = pickle.load(f)

        # Move to GPU if available
        try:
            self.res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index_cpu)
            logger.info("FAISS: Moved index to GPU")
        except Exception as e:
            logger.warning(f"FAISS: Could not move index to GPU: {e}. Using CPU.")
            self.index = self.index_cpu

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
        """Search for nearest neighbors.

        Args:
            query: Query vectors (B, D) where B is batch size, D is dimension
            top_k: Number of nearest neighbors to return
            complexity: Search complexity (unused for FAISS Flat, kept for interface compat)
            beam_width: Beam width (unused for FAISS Flat, kept for interface compat)
            prune_ratio: Pruning ratio (unused, kept for interface compat)
            recompute_embeddings: Whether to use embedding server (unused for FAISS)
            pruning_strategy: Pruning strategy (unused, kept for interface compat)
            zmq_port: ZMQ port (unused for FAISS direct search)
            **kwargs: Additional parameters

        Returns:
            Dict with 'labels' (list of lists) and 'distances' (list of lists)
        """
        # Normalize query for cosine similarity
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        faiss.normalize_L2(query)

        # Search
        distances, indices = self.index.search(query, top_k)

        # Map indices to IDs
        # indices is (B, K)
        results_labels = []
        results_distances = []

        for i in range(query.shape[0]):
            row_labels = []
            row_dists = []
            for j in range(top_k):
                idx = indices[i][j]
                if idx != -1:
                    row_labels.append(self.ids[idx])
                    row_dists.append(float(distances[i][j]))
            results_labels.append(row_labels)
            results_distances.append(row_dists)

        return {"labels": results_labels, "distances": results_distances}


@register_backend("faiss")
class FaissBackendFactory(LeannBackendFactoryInterface):
    """Factory for FAISS backend."""

    @staticmethod
    def builder(**kwargs) -> LeannBackendBuilderInterface:
        return FaissBackendBuilder()

    @staticmethod
    def searcher(index_path: str, **kwargs) -> LeannBackendSearcherInterface:
        return FaissBackendSearcher(index_path, **kwargs)
