"""
LEANN - Low-storage Embedding Approximation for Neural Networks

A revolutionary vector database that democratizes personal AI.
"""

__version__ = "0.1.0"

# Re-export main API from leann-core
from leann_core import LeannBuilder, LeannSearcher, LeannChat

__all__ = ["LeannBuilder", "LeannSearcher", "LeannChat"]
