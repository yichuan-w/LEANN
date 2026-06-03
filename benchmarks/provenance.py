"""Shared provenance helpers for benchmark artifacts."""

from __future__ import annotations

import hashlib
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def environment_metadata() -> dict[str, Any]:
    """Return local environment metadata for reviewable benchmark artifacts."""
    return {
        "leann_commit": _git_commit_sha(),
        "leann_branch": _git_branch_name(),
        "leann_dirty": _git_worktree_dirty(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def file_sha256(path: str | Path) -> str:
    """Return the SHA256 hex digest for a benchmark input file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_command(script_path: str | Path, argv: list[str] | None) -> str:
    """Return a repo-relative, shell-quoted benchmark command string."""
    script = _repo_relative_path(script_path)
    args = list(argv) if argv is not None else sys.argv[1:]
    return shlex.join([script, *args])


def _repo_relative_path(path: str | Path) -> str:
    raw_path = Path(path)
    try:
        return str(raw_path.resolve().relative_to(Path(__file__).resolve().parents[1]))
    except (OSError, ValueError):
        return str(raw_path)


def _git_commit_sha() -> str | None:
    return _git_output(["git", "rev-parse", "HEAD"])


def _git_branch_name() -> str | None:
    return _git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])


def _git_worktree_dirty() -> bool | None:
    output = _git_output(["git", "status", "--short"])
    return None if output is None else bool(output)


def _git_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None
