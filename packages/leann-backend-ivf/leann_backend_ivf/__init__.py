"""LEANN IVF backend: FAISS IndexIVFFlat with add/delete APIs for incremental updates."""

from .ivf_backend import (
    IVFBackend,
    IVFBuilder,
    IVFSearcher,
    add_vectors,
    remove_ids,
)

__all__ = [
    "IVFBackend",
    "IVFBuilder",
    "IVFSearcher",
    "add_vectors",
    "remove_ids",
]
