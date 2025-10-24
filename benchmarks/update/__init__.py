"""Benchmarks for LEANN update workflows."""

# Expose helper to locate repository root for other modules that need it.
from pathlib import Path


def find_repo_root() -> Path:
    """Return the project root containing pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[1]


__all__ = ["find_repo_root"]
