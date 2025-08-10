# packages/leann-core/src/leann/registry.py

import importlib
import importlib.metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from leann.interface import LeannBackendFactoryInterface

BACKEND_REGISTRY: dict[str, "LeannBackendFactoryInterface"] = {}


def register_backend(name: str):
    """A decorator to register a new backend class."""

    def decorator(cls):
        print(f"INFO: Registering backend '{name}'")
        BACKEND_REGISTRY[name] = cls
        return cls

    return decorator


def autodiscover_backends():
    """Automatically discovers and imports all 'leann-backend-*' packages."""
    # print("INFO: Starting backend auto-discovery...")
    discovered_backends = []
    for dist in importlib.metadata.distributions():
        dist_name = dist.metadata["name"]
        if dist_name.startswith("leann-backend-"):
            backend_module_name = dist_name.replace("-", "_")
            discovered_backends.append(backend_module_name)

    for backend_module_name in sorted(discovered_backends):  # sort for deterministic loading
        try:
            importlib.import_module(backend_module_name)
            # Registration message is printed by the decorator
        except ImportError:
            # print(f"WARN: Could not import backend module '{backend_module_name}': {e}")
            pass
    # print("INFO: Backend auto-discovery finished.")
