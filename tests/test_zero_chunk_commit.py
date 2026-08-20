"""Failing tests for committing sync snapshots on safely-empty (zero-chunk) build deltas."""

import asyncio
import hashlib
import json
import os
from pathlib import Path

import pytest
from leann.cli import LeannCLI


def _make_fake_builder():
    class FakeBuilder:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def add_text(self, _text, metadata=None):
            pass

        def build_index(self, index_path):
            target_dir = Path(index_path).parent
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "documents.leann.meta.json").write_text(
                json.dumps(
                    {
                        "backend_name": "hnsw",
                        "embedding_model": self.kwargs.get("embedding_model", "m"),
                        "embedding_mode": self.kwargs.get(
                            "embedding_mode", "sentence-transformers"
                        ),
                        "passage_id_scheme": "sequential",
                        "backend_kwargs": {
                            "graph_degree": 32,
                            "complexity": 64,
                            "num_threads": 1,
                            "is_compact": False,
                            "is_recompute": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (target_dir / "documents.leann.index").write_bytes(b"fake")
            (target_dir / "documents.leann.passages.jsonl").write_text("", encoding="utf-8")

        def update_index(self, index_path):
            pass

    return FakeBuilder


def _loading_fake(docs_paths, custom_file_types=None, include_hidden=False, args=None):
    if isinstance(docs_paths, str):
        docs_paths = [docs_paths]
    files: list[Path] = []
    for p in docs_paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.txt")))
        elif path.is_file():
            files.append(path)
    return [{"text": f.name, "metadata": {"file_path": str(f.resolve())}} for f in files]


def _wire_cli(monkeypatch) -> LeannCLI:
    cli = LeannCLI()
    monkeypatch.setattr(cli, "load_documents", _loading_fake)
    monkeypatch.setattr(cli, "register_project_dir", lambda: None)
    monkeypatch.setattr("leann.cli.LeannBuilder", _make_fake_builder())
    return cli


def _build_args(index_name: str, docs: list[str], extra: list[str] | None = None) -> list[str]:
    return [
        "build",
        index_name,
        "--docs",
        *docs,
        "--backend-name",
        "hnsw",
        "--no-compact",
        "--embedding-model",
        "m",
        "--embedding-mode",
        "sentence-transformers",
        *(extra or []),
    ]


def _run_build(cli: LeannCLI, argv: list[str]) -> None:
    asyncio.run(cli.build_index(cli.create_parser().parse_args(argv)))


def _run_changes(cli: LeannCLI, argv: list[str]) -> int:
    args = cli.create_parser().parse_args(argv)
    result = cli.changes_command(args)
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
    return int(result or 0)


def _index_artifact_hashes(index_dir: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(index_dir.glob("documents.leann*"))
        if p.is_file()
    }


def test_add_only_zero_chunk_delta_commits_snapshot_and_leaves_index_untouched(
    tmp_path, monkeypatch, capsys
):
    # Arrange
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha", encoding="utf-8")
    cli = _wire_cli(monkeypatch)
    _run_build(cli, _build_args("idx", [str(docs)]))
    (docs / "b.txt").write_text("beta", encoding="utf-8")
    index_dir = tmp_path / ".leann" / "indexes" / "idx"
    hashes_before = _index_artifact_hashes(index_dir)
    zero_chunk_calls: list[list[str]] = []

    def zero_chunk_load(docs_paths, custom_file_types=None, include_hidden=False, args=None):
        zero_chunk_calls.append(list(docs_paths))
        return []

    monkeypatch.setattr(cli, "load_documents", zero_chunk_load)

    # Act
    _run_build(cli, _build_args("idx", [str(docs)]))
    monkeypatch.setattr(cli, "load_documents", _loading_fake)
    capsys.readouterr()
    rc = _run_changes(cli, ["changes", "idx"])
    report = json.loads(capsys.readouterr().out)

    # Assert
    assert zero_chunk_calls, "second build should have loaded the added file"
    assert rc == 0
    assert report["added"] == []
    assert report["modified"] == []
    assert report["removed"] == []
    assert (index_dir / "sync_roots.json").exists()
    assert _index_artifact_hashes(index_dir) == hashes_before


def test_loader_failure_does_not_advance_snapshot(tmp_path, monkeypatch, capsys):
    # Arrange
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha", encoding="utf-8")
    cli = _wire_cli(monkeypatch)
    _run_build(cli, _build_args("idx", [str(docs)]))
    new_file = docs / "b.txt"
    new_file.write_text("beta", encoding="utf-8")

    def failing_load(docs_paths, custom_file_types=None, include_hidden=False, args=None):
        raise RuntimeError("parser exploded")

    monkeypatch.setattr(cli, "load_documents", failing_load)

    # Act
    with pytest.raises(RuntimeError, match="parser exploded"):
        _run_build(cli, _build_args("idx", [str(docs)]))
    monkeypatch.setattr(cli, "load_documents", _loading_fake)
    capsys.readouterr()
    rc = _run_changes(cli, ["changes", "idx"])
    report = json.loads(capsys.readouterr().out)

    # Assert
    assert rc == 0
    assert report["added"] == [str(new_file.resolve())]


def test_snapshot_and_sync_config_writes_use_atomic_os_replace(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha", encoding="utf-8")
    cli = _wire_cli(monkeypatch)
    replaced_targets: list[str] = []
    real_replace = os.replace

    def spy_replace(src, dst, *args, **kwargs):
        replaced_targets.append(str(dst))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", spy_replace)

    # Act
    _run_build(cli, _build_args("idx", [str(docs)]))

    # Assert
    index_dir = tmp_path / ".leann" / "indexes" / "idx"
    assert any(t.endswith(".pickle") for t in replaced_targets), (
        "save_snapshot should publish the snapshot via os.replace"
    )
    assert any(t.endswith("sync_roots.json") for t in replaced_targets), (
        "_write_sync_config should publish sync_roots.json via os.replace"
    )
    assert not list(index_dir.glob("*.tmp"))


def test_swallowed_loader_failure_blocks_snapshot_commit(tmp_path, monkeypatch, capsys):
    # Arrange
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha", encoding="utf-8")
    cli = _wire_cli(monkeypatch)
    _run_build(cli, _build_args("idx", [str(docs)]))
    new_file = docs / "b.txt"
    new_file.write_text("beta", encoding="utf-8")

    def swallowed_failure_load(docs_paths, custom_file_types=None, include_hidden=False, args=None):
        cli._load_errors = 1  # simulate load_documents warn-and-continue on a broken file
        return []

    monkeypatch.setattr(cli, "load_documents", swallowed_failure_load)

    # Act
    with pytest.raises(RuntimeError, match="failed to load"):
        _run_build(cli, _build_args("idx", [str(docs)]))
    monkeypatch.setattr(cli, "load_documents", _loading_fake)
    capsys.readouterr()
    rc = _run_changes(cli, ["changes", "idx"])
    report = json.loads(capsys.readouterr().out)

    # Assert: snapshot not committed, so the failed file is still pending
    assert rc == 0
    assert report["added"] == [str(new_file.resolve())]


def test_partial_loader_failure_aborts_before_mutating_index(tmp_path, monkeypatch, capsys):
    # Arrange: two new files, one loads and one fails — nothing may be committed
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha", encoding="utf-8")
    cli = _wire_cli(monkeypatch)
    _run_build(cli, _build_args("idx", [str(docs)]))
    good = docs / "b.txt"
    bad = docs / "c.txt"
    good.write_text("beta", encoding="utf-8")
    bad.write_text("gamma", encoding="utf-8")
    index_dir = tmp_path / ".leann" / "indexes" / "idx"
    hashes_before = _index_artifact_hashes(index_dir)

    def partial_failure_load(docs_paths, custom_file_types=None, include_hidden=False, args=None):
        cli._load_errors = 1  # one file failed with a swallowed warning
        return [{"text": "beta", "metadata": {"file_path": str(good.resolve())}}]

    monkeypatch.setattr(cli, "load_documents", partial_failure_load)

    # Act
    with pytest.raises(RuntimeError, match="failed to load"):
        _run_build(cli, _build_args("idx", [str(docs)]))
    monkeypatch.setattr(cli, "load_documents", _loading_fake)
    capsys.readouterr()
    rc = _run_changes(cli, ["changes", "idx"])
    report = json.loads(capsys.readouterr().out)

    # Assert: index untouched, both files still pending
    assert rc == 0
    assert sorted(report["added"]) == [str(good.resolve()), str(bad.resolve())]
    assert _index_artifact_hashes(index_dir) == hashes_before
