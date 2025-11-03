"""Unified chunking utilities facade.

This module re-exports the packaged utilities from `leann.chunking_utils` so
that both repo apps (importing `chunking`) and installed wheels share one
single implementation. When running from the repo without installation, it
adds the `packages/leann-core/src` directory to `sys.path` as a fallback.
"""

import sys
from pathlib import Path

try:
    from leann.chunking_utils import (
        CODE_EXTENSIONS,
        _traditional_chunks_as_dicts,
        create_ast_chunks,
        create_text_chunks,
        create_traditional_chunks,
        detect_code_files,
        get_language_from_extension,
    )
except Exception:  # pragma: no cover - best-effort fallback for dev environment
    repo_root = Path(__file__).resolve().parents[2]
    leann_src = repo_root / "packages" / "leann-core" / "src"
    if leann_src.exists():
        sys.path.insert(0, str(leann_src))
        from leann.chunking_utils import (
            CODE_EXTENSIONS,
            _traditional_chunks_as_dicts,
            create_ast_chunks,
            create_text_chunks,
            create_traditional_chunks,
            detect_code_files,
            get_language_from_extension,
        )
    else:
        raise

__all__ = [
    "CODE_EXTENSIONS",
    "_traditional_chunks_as_dicts",
    "create_ast_chunks",
    "create_text_chunks",
    "create_traditional_chunks",
    "detect_code_files",
    "get_language_from_extension",
]
