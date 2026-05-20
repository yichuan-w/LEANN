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

# Directories we never descend into during index discovery. These are caches,
# dependencies, and build outputs that won't contain LEANN indexes and that
# dominate walk latency under $HOME (especially macOS Library/). Keep this in
# sync across cli.py callsites that scan for *.leann.meta.json files.
INDEX_SCAN_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".cache",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".tox",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "Library",
        "target",
        "dist",
        "build",
        ".next",
        ".nuxt",
        ".gradle",
    }
)


def walk_index_meta_files(root: Path) -> Iterator[Path]:
    """Yield *.leann.meta.json files under root, pruning huge irrelevant dirs.

    Faster than Path.rglob('*.leann.meta.json') on large home directories where
    Library/, node_modules/, .venv/, .git/, etc. would otherwise dominate the walk.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in INDEX_SCAN_SKIP_DIRS]
        for fname in filenames:
            if fname.endswith(".leann.meta.json"):
                yield Path(dirpath) / fname


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


def register_project_directory(project_dir: Optional[Union[str, Path]] = None):
    """
    Register a project directory in the global LEANN registry.

    This allows `leann list` to discover indexes created by apps or other tools.

    Args:
        project_dir: Directory to register. If None, uses current working directory.
    """
    if project_dir is None:
        project_dir = Path.cwd()
    else:
        project_dir = Path(project_dir)

    # Only register directories that have some kind of LEANN content.
    # Check CLI-format first to avoid an expensive walk on large directories.
    has_cli_indexes = (project_dir / ".leann" / "indexes").exists()
    if not has_cli_indexes and not any(walk_index_meta_files(project_dir)):
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
