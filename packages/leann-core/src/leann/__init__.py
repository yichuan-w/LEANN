# packages/leann-core/src/leann/__init__.py
from .api import LeannBuilder, LeannChat, LeannSearcher
from .registry import BACKEND_REGISTRY, autodiscover_backends

autodiscover_backends()

__all__ = ["LeannBuilder", "LeannSearcher", "LeannChat", "BACKEND_REGISTRY"]