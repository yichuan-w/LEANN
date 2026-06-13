"""Embedding providers — one module per embedding backend.

Each module exports a ``compute_embeddings_<name>`` function.  The registry
in ``embedding_compute.py`` auto-discovers them and wires up the public
``compute_embeddings()`` entry point.

To add a new provider:

1. Create ``providers/my_provider.py`` with a
   ``compute_embeddings_my_provider(texts, model_name, **kwargs) -> np.ndarray``
   function.
2. Import and register it in ``embedding_compute._init_providers()``.
3. (Optional) Call ``embedding_compute.register_provider("my-mode", fn)``
   from third-party code to avoid touching core.
"""
