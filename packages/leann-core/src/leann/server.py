"""
Minimal HTTP API server for LEANN.

This exposes LEANN indexes over HTTP so clients can:
- List available indexes
- Run semantic search against an index

The design intentionally keeps dependencies optional:
- FastAPI + pydantic are imported lazily inside `create_app()`
- uvicorn is imported lazily inside `main()`

This way, core LEANN usage is unaffected unless you actually run the server.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .api import LeannSearcher
from .cli import LeannCLI


def _ensure_fastapi():
    """Lazy import FastAPI and Pydantic, with a clear error if missing."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as e:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "FastAPI and pydantic are required for the LEANN HTTP server.\n"
            "Install them with:\n\n"
            "  uv pip install 'fastapi>=0.115' 'pydantic>=2' 'uvicorn[standard]'\n"
        ) from e

    return FastAPI, HTTPException, BaseModel


def _resolve_index_path(index_name: str) -> str:
    """
    Resolve an index path for the HTTP server.

    For now we use the same convention as the CLI:
    - Look in the current project's `.leann/indexes/<name>/documents.leann`.

    This keeps behavior predictable when running `leann serve` from a project root.
    """
    cli = LeannCLI()
    index_path = cli.get_index_path(index_name)
    if not cli.index_exists(index_name):
        raise FileNotFoundError(
            f"Index '{index_name}' not found in current project. "
            f"Build it with: leann build {index_name} --docs ./your_docs"
        )
    return index_path


def _list_current_project_indexes() -> list[dict[str, Any]]:
    """
    Return machine-readable index metadata for the current project.

    This mirrors `LeannCLI.list_indexes()` but only for the current project
    and without printing to stdout.
    """
    cli = LeannCLI()
    current_path = Path.cwd()
    indexes: list[dict[str, Any]] = []

    for idx in cli._discover_indexes_in_project(current_path):
        # `idx` includes keys like: name, type (cli/app), status, size_mb
        indexes.append(
            {
                "name": idx.get("name", ""),
                "type": idx.get("type", "cli"),
                "status": idx.get("status", ""),
                "size_mb": idx.get("size_mb", 0.0),
                "project_path": str(current_path),
            }
        )

    return indexes


def create_app():
    """
    Create and return a FastAPI application exposing LEANN as a simple vector DB.

    Endpoints:
    - GET  /health                     -> basic health check
    - GET  /indexes                    -> list indexes in current project
    - POST /indexes/{name}/search      -> semantic search
    """

    FastAPI, HTTPException, BaseModel = _ensure_fastapi()

    class SearchRequest(BaseModel):
        query: str
        top_k: int = 5
        complexity: int = 64
        beam_width: int = 1
        prune_ratio: float = 0.0
        recompute_embeddings: bool = True
        pruning_strategy: str = "global"
        use_grep: bool = False

    class SearchResultModel(BaseModel):
        id: str
        score: float
        text: str
        metadata: dict[str, Any]

    app = FastAPI(
        title="LEANN Vector DB Server",
        description=(
            "HTTP API for querying LEANN indexes.\n\n"
            "This is a minimal first version focused on search. "
            "Run it from a project root where `.leann/indexes` exists."
        ),
        version="0.1.0",
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/indexes")
    async def list_indexes() -> list[dict[str, Any]]:
        """
        List indexes for the current project (the working directory where the server runs).
        """
        return _list_current_project_indexes()

    @app.post("/indexes/{index_name}/search", response_model=list[SearchResultModel])
    async def search_index(index_name: str, body: SearchRequest):
        """
        Run semantic search against an existing LEANN index.
        """
        try:
            index_path = _resolve_index_path(index_name)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

        searcher = LeannSearcher(index_path=index_path)
        results = searcher.search(
            query=body.query,
            top_k=body.top_k,
            complexity=body.complexity,
            beam_width=body.beam_width,
            prune_ratio=body.prune_ratio,
            recompute_embeddings=body.recompute_embeddings,
            pruning_strategy=body.pruning_strategy,  # type: ignore[arg-type]
            use_grep=body.use_grep,
        )

        # Normalize into JSON-serializable structures
        return [
            SearchResultModel(
                id=r.id,
                score=float(r.score),
                text=r.text,
                metadata=dict(r.metadata or {}),
            )
            for r in results
        ]

    return app


def main() -> None:
    """
    Entrypoint to run the HTTP server with uvicorn.

    Example:
        uv run python -m leann.server
        # or, after wiring a CLI command:
        leann serve
    """
    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "uvicorn is required to run the LEANN HTTP server.\n"
            "Install it with:\n\n"
            "  uv pip install 'uvicorn[standard]'\n"
        ) from e

    app = create_app()
    host = os.getenv("LEANN_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("LEANN_SERVER_PORT", "8000"))

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    main()
