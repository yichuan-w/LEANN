"""Storage accounting helpers for benchmark artifacts."""

from __future__ import annotations

from pathlib import Path


def directory_storage(path: str | Path) -> dict[str, object]:
    """Return recursive storage statistics for a benchmark data directory."""
    root = Path(path)
    files = (
        [candidate for candidate in root.rglob("*") if candidate.is_file()] if root.exists() else []
    )
    return {
        "path": str(root.resolve()),
        "exists": root.exists(),
        "bytes": sum(file.stat().st_size for file in files),
        "file_count": len(files),
    }
