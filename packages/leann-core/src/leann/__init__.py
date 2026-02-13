# packages/leann-core/src/leann/__init__.py
import os
import platform

# ruff: noqa: E402  (env vars must be set before importing the rest of the package)

# Fix OpenMP/FAISS threading defaults for common platforms
system = platform.system()

if system == "Darwin":
    # macOS ARM64: prevent runaway threading and duplicate lib issues
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["KMP_BLOCKTIME"] = "0"
    # Additional fixes for PyTorch/sentence-transformers on macOS ARM64 only in CI
    if os.environ.get("CI") == "true":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
elif system == "Linux":
    # Linux CPU-only: default to single-thread to avoid FAISS/ZMQ hangs (issue #208)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("FAISS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

from .api import LeannBuilder, LeannChat, LeannSearcher
from .react_agent import ReActAgent, create_react_agent
from .registry import BACKEND_REGISTRY, autodiscover_backends

autodiscover_backends()

__all__ = [
    "BACKEND_REGISTRY",
    "LeannBuilder",
    "LeannChat",
    "LeannSearcher",
    "ReActAgent",
    "create_react_agent",
]
