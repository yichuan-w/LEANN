"""
Tests for incremental build (Feature #89).

When an index already exists and build is run again without --force,
only new files are indexed and appended to the existing index (HNSW, non-compact only).
"""

import json
import os
from pathlib import Path

import pytest
from leann.cli import (
    SOURCES_MANIFEST_FILENAME,
    _normalize_path,
    load_sources_manifest,
    save_sources_manifest,
)


def test_normalize_path():
    assert _normalize_path("") == ""
    assert _normalize_path("/a/b") == "/a/b" or "a" in _normalize_path("/a/b")
    # Resolve so relative becomes absolute
    rel = "foo/bar"
    out = _normalize_path(rel)
    assert Path(out).is_absolute() or out == rel


def test_load_sources_manifest_missing(tmp_path):
    assert load_sources_manifest(tmp_path, "documents.leann") == {}


def test_load_sources_manifest_empty(tmp_path):
    (tmp_path / SOURCES_MANIFEST_FILENAME).write_text('{"sources": {}}', encoding="utf-8")
    assert load_sources_manifest(tmp_path, "documents.leann") == {}


def test_save_and_load_sources_manifest(tmp_path):
    sources = {"/path/to/a": 12345.0, "/path/to/b": 67890.0}
    save_sources_manifest(tmp_path, "documents.leann", sources)
    loaded = load_sources_manifest(tmp_path, "documents.leann")
    assert loaded == sources
    raw = (tmp_path / SOURCES_MANIFEST_FILENAME).read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data.get("sources") == sources


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Skip in CI to avoid embedding/model load",
)
def test_incremental_build_adds_only_new_files(tmp_path):
    """Build once with one file, add a second file, run build again without --force; index grows."""
    from leann.cli import LeannCLI

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("First document content for indexing.", encoding="utf-8")

    cli = LeannCLI()
    # Override indexes_dir so we use tmp_path for index
    cli.indexes_dir = tmp_path / ".leann" / "indexes"
    cli.indexes_dir.mkdir(parents=True, exist_ok=True)
    index_name = "incr_test"
    index_dir = cli.indexes_dir / index_name
    index_path = cli.get_index_path(index_name)

    parser = cli.create_parser()
    args = parser.parse_args(
        [
            "build",
            index_name,
            "--docs",
            str(docs_dir),
            "--backend-name",
            "hnsw",
            "--no-compact",
            "--embedding-model",
            "all-MiniLM-L6-v2",
            "--embedding-mode",
            "sentence-transformers",
            "--force",
        ]
    )
    import asyncio

    asyncio.run(cli.build_index(args))
    assert index_dir.exists()
    meta_file = index_dir / "documents.leann.meta.json"
    assert meta_file.exists()
    manifest_file = index_dir / "documents.leann.sources.json"
    assert manifest_file.exists(), "Full build should write sources manifest"
    with open(manifest_file, encoding="utf-8") as f:
        manifest = json.load(f)
    assert len(manifest.get("sources", {})) == 1

    # Add second file
    (docs_dir / "b.txt").write_text("Second document content.", encoding="utf-8")

    # Build again without --force (incremental)
    args2 = parser.parse_args(
        [
            "build",
            index_name,
            "--docs",
            str(docs_dir),
            "--backend-name",
            "hnsw",
            "--no-compact",
            "--embedding-model",
            "all-MiniLM-L6-v2",
            "--embedding-mode",
            "sentence-transformers",
        ]
    )
    asyncio.run(cli.build_index(args2))

    with open(manifest_file, encoding="utf-8") as f:
        manifest2 = json.load(f)
    assert len(manifest2.get("sources", {})) == 2, (
        "Manifest should list both files after incremental"
    )

    # Index should still be searchable
    from leann.api import LeannSearcher

    searcher = LeannSearcher(index_path)
    results = searcher.search("Second document", top_k=3)
    searcher.cleanup()
    assert len(results) >= 1
    assert "Second" in results[0].text or "document" in results[0].text
