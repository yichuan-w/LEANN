"""Optional framework integrations for LEANN.

Integration modules are imported lazily so importing `leann.integrations` does
not load framework packages unless the caller asks for that integration.
"""

from importlib import import_module
from typing import Any

__all__ = ["LeannHybridRetriever", "LeannRetriever"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module("leann.integrations.llamaindex")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
