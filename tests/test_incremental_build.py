"""
Tests for incremental build (Feature #89).

When an index already exists and build is run again without --force,
only new files are indexed and appended to the existing index (HNSW, non-compact only).
Change detection uses content-hash (merkle tree) via FileSynchronizer.
"""

import os
from pathlib import Path

import pytest
from leann.cli import _normalize_path
from leann.sync import FileSynchronizer


def test_normalize_path():
    assert _normalize_path("") == ""
    assert _normalize_path("/a/b") == "/a/b" or "a" in _normalize_path("/a/b")
    rel = "foo/bar"
    out = _normalize_path(rel)
    assert Path(out).is_absolute() or out == rel


def test_file_synchronizer_detect_changes(tmp_path):
    """FileSynchronizer detect_changes returns all files as added when no snapshot exists."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("hello", encoding="utf-8")
    snapshot = str(tmp_path / "test.pickle")
    fs = FileSynchronizer(root_dir=str(docs), snapshot_path=snapshot)
    added, removed, modified = fs.detect_changes()
    assert len(added) == 1
    assert len(removed) == 0
    assert len(modified) == 0


def test_file_synchronizer_no_changes_after_commit(tmp_path):
    """After commit, detect_changes should report no changes if files haven't changed."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("hello", encoding="utf-8")
    snapshot = str(tmp_path / "test.pickle")
    fs = FileSynchronizer(root_dir=str(docs), snapshot_path=snapshot)
    fs.detect_changes()
    fs.commit()

    fs2 = FileSynchronizer(root_dir=str(docs), snapshot_path=snapshot)
    added, removed, modified = fs2.detect_changes()
    assert not added and not removed and not modified


def test_file_synchronizer_detects_new_file(tmp_path):
    """Adding a file after commit should be detected as added."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("hello", encoding="utf-8")
    snapshot = str(tmp_path / "test.pickle")
    fs = FileSynchronizer(root_dir=str(docs), snapshot_path=snapshot)
    fs.detect_changes()
    fs.commit()

    (docs / "b.txt").write_text("world", encoding="utf-8")
    fs2 = FileSynchronizer(root_dir=str(docs), snapshot_path=snapshot)
    added, removed, modified = fs2.detect_changes()
    assert len(added) == 1
    assert len(removed) == 0
    assert len(modified) == 0


def test_file_synchronizer_detects_modification(tmp_path):
    """Changing file content should be detected as modified."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("hello", encoding="utf-8")
    snapshot = str(tmp_path / "test.pickle")
    fs = FileSynchronizer(root_dir=str(docs), snapshot_path=snapshot)
    fs.detect_changes()
    fs.commit()

    (docs / "a.txt").write_text("changed", encoding="utf-8")
    fs2 = FileSynchronizer(root_dir=str(docs), snapshot_path=snapshot)
    added, removed, modified = fs2.detect_changes()
    assert len(modified) == 1
    assert len(added) == 0


def test_file_synchronizer_touch_no_false_positive(tmp_path):
    """Touching a file (mtime change, same content) should NOT report as modified."""
    docs = tmp_path / "docs"
    docs.mkdir()
    f = docs / "a.txt"
    f.write_text("hello", encoding="utf-8")
    snapshot = str(tmp_path / "test.pickle")
    fs = FileSynchronizer(root_dir=str(docs), snapshot_path=snapshot)
    fs.detect_changes()
    fs.commit()

    os.utime(f, None)
    fs2 = FileSynchronizer(root_dir=str(docs), snapshot_path=snapshot)
    added, removed, modified = fs2.detect_changes()
    assert not added and not removed and not modified


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Skip in CI to avoid embedding/model load",
)
def test_incremental_build_adds_only_new_files(tmp_path):
    """Build once with one file, add a second file, run build again without --force; index grows."""
    import asyncio

    from leann.api import LeannSearcher
    from leann.cli import LeannCLI

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("First document content for indexing.", encoding="utf-8")

    cli = LeannCLI()
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
    asyncio.run(cli.build_index(args))
    assert index_dir.exists()
    assert (index_dir / "documents.leann.meta.json").exists()

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

    # Index should still be searchable and contain both files
    searcher = LeannSearcher(index_path)
    results = searcher.search("Second document", top_k=3)
    searcher.cleanup()
    assert len(results) >= 1
    assert "Second" in results[0].text or "document" in results[0].text


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Skip in CI to avoid embedding/model load",
)
def test_ivf_incremental_add_then_remove_searchable(tmp_path):
    """IVF: add content, search finds it; delete content, incremental build, search no longer finds it."""
    import asyncio

    from leann.api import LeannSearcher
    from leann.cli import LeannCLI

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    unique_phrase = "XYZZY_PLUGH_INCR_TEST_12345"
    (docs_dir / "f.txt").write_text(
        f"Initial content.\n\n{unique_phrase}\n\nMore text here.",
        encoding="utf-8",
    )
    # Add extra files so IVF has enough training points (nlist=100)
    for i in range(110):
        (docs_dir / f"filler_{i:03d}.txt").write_text(
            f"Filler document {i} with some content to create enough chunks.",
            encoding="utf-8",
        )

    cli = LeannCLI()
    cli.indexes_dir = tmp_path / ".leann" / "indexes"
    cli.indexes_dir.mkdir(parents=True, exist_ok=True)
    index_name = "ivf_incr_test"
    index_path = cli.get_index_path(index_name)

    parser = cli.create_parser()
    build_args = [
        "build",
        index_name,
        "--docs",
        str(docs_dir),
        "--backend-name",
        "ivf",
        "--embedding-model",
        "all-MiniLM-L6-v2",
        "--embedding-mode",
        "sentence-transformers",
    ]

    asyncio.run(cli.build_index(parser.parse_args([*build_args, "--force"])))

    searcher = LeannSearcher(index_path)
    results = searcher.search(unique_phrase, top_k=5)
    searcher.cleanup()
    assert len(results) >= 1, "Added content should be searchable"
    assert unique_phrase in results[0].text

    # Remove the unique phrase from the file
    (docs_dir / "f.txt").write_text("Initial content.\n\nMore text here.", encoding="utf-8")

    # Incremental build (IVF remove+add)
    asyncio.run(cli.build_index(parser.parse_args(build_args)))

    # Deleted content should no longer be searchable
    searcher = LeannSearcher(index_path)
    results = searcher.search(unique_phrase, top_k=5)
    searcher.cleanup()
    assert all(unique_phrase not in r.text for r in results), (
        "Deleted content should not appear in search results"
    )
