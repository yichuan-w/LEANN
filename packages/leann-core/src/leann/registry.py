# packages/leann-core/src/leann/registry.py

from typing import Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from leann.interface import LeannBackendFactoryInterface

BACKEND_REGISTRY: Dict[str, 'LeannBackendFactoryInterface'] = {}

def register_backend(name: str):
    """A decorator to register a new backend class."""
    def decorator(cls):
        print(f"INFO: Registering backend '{name}'")
        BACKEND_REGISTRY[name] = cls
        return cls
    return decorator