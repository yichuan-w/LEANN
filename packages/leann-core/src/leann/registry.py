# packages/leann-core/src/leann/registry.py

import importlib
import importlib.metadata
import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from leann.interface import LeannBackendFactoryInterface

# Set up logger for this module
logger = logging.getLogger(__name__)

BACKEND_REGISTRY: dict[str, "LeannBackendFactoryInterface"] = {}
DEFAULT_INDEX_SCAN_DEPTH = 5
INDEX_SCAN_SKIP_DIRS = frozenset(
    {".git", ".venv", "venv", "node_modules", "__pycache__", "Library"}
)


def iter_index_meta_files(
    root: Union[str, Path], max_depth: Optional[int] = DEFAULT_INDEX_SCAN_DEPTH
) -> Iterator[Path]:
    """Yield LEANN metadata files within an optionally bounded directory tree.

    The root directory is depth zero. A max_depth of None scans all depths.
    Known dependency, cache, and system directories are always pruned before traversal.
    """
    if max_depth is not None and max_depth < 0:
        raise ValueError("max_depth must be non-negative")

    root_path = Path(root)
    for current_dir, dirnames, filenames in os.walk(root_path):
        current_path = Path(current_dir)
        depth = len(current_path.relative_to(root_path).parts)
        if max_depth is not None and depth >= max_depth:
            dirnames.clear()
        else:
            dirnames[:] = [name for name in dirnames if name not in INDEX_SCAN_SKIP_DIRS]

        for filename in filenames:
            if filename.endswith(".leann.meta.json"):
                yield current_path / filename


def register_backend(name: str):
    """A decorator to register a new backend class."""

    def decorator(cls):
        logger.debug(f"Registering backend '{name}'")
        BACKEND_REGISTRY[name] = cls
        return cls

    return decorator


def autodiscover_backends():
    """Automatically discovers and imports all 'leann-backend-*' packages."""
    # print("INFO: Starting backend auto-discovery...")
    discovered_backends = []
    for dist in importlib.metadata.distributions():
        dist_name = dist.metadata["name"]
        if dist_name is None:
            continue
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


def register_project_directory(
    project_dir: Optional[Union[str, Path]] = None,
    max_depth: Optional[int] = None,
):
    """
    Register a project directory in the global LEANN registry.

    This allows `leann list` to discover indexes created by apps or other tools.

    Args:
        project_dir: Directory to register. If None, uses current working directory.
        max_depth: Maximum directory depth used when looking for App-format indexes.
            None preserves full-depth discovery for existing API callers.
    """
    if os.environ.get("LEANN_NO_REGISTER") == "1":
        return
    if project_dir is None:
        project_dir = Path.cwd()
    else:
        project_dir = Path(project_dir)

    # Only register directories that have some kind of LEANN content.
    # Check CLI-format first to avoid even a bounded scan when it is unnecessary.
    has_cli_indexes = (project_dir / ".leann" / "indexes").exists()
    if not has_cli_indexes and not any(iter_index_meta_files(project_dir, max_depth=max_depth)):
        # Don't register if there are no LEANN indexes
        return

    global_registry = Path.home() / ".leann" / "projects.json"
    global_registry.parent.mkdir(exist_ok=True)

    project_str = str(project_dir.resolve())

    # Load existing registry
    projects = []
    if global_registry.exists():
        try:
            with open(global_registry) as f:
                projects = json.load(f)
        except Exception:
            logger.debug("Could not load existing project registry")
            projects = []

    # Add project if not already present
    if project_str not in projects:
        projects.append(project_str)

        # Save updated registry
        try:
            with open(global_registry, "w") as f:
                json.dump(projects, f, indent=2)
            logger.debug(f"Registered project directory: {project_str}")
        except Exception as e:
            logger.warning(f"Could not save project registry: {e}")
