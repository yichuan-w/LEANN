"""Failing tests for `leann changes` (non-mutating merkle diff vs stored snapshot)."""

import asyncio
import hashlib
import json
from pathlib import Path

from leann.cli import LeannCLI


def _keyed_snapshot_name(key: str) -> str:
    return f"sync_key_{hashlib.sha256(key.encode()).hexdigest()[:12]}.pickle"


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


def _fake_load_documents(docs_paths, custom_file_types=None, include_hidden=False, args=None):
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
    monkeypatch.setattr(cli, "load_documents", _fake_load_documents)
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


def _run_changes(cli: LeannCLI, argv: list[str]) -> int:
    args = cli.create_parser().parse_args(argv)
    handler = cli.changes_command
    try:
        result = handler(args)
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
        return int(result or 0)
    except SystemExit as e:
        return int(e.code or 0)


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_changes_reports_modified_and_added_as_single_sorted_json_doc(
    tmp_path, monkeypatch, capsys
):
    # Arrange
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    file_a = docs / "a.txt"
    file_b = docs / "b.txt"
    file_a.write_text("alpha", encoding="utf-8")
    file_b.write_text("beta", encoding="utf-8")
    cli = _wire_cli(monkeypatch)
    asyncio.run(
        cli.build_index(
            cli.create_parser().parse_args(
                _build_args("idx", [str(docs)], ["--sync-key", "corpus-v1"])
            )
        )
    )
    file_a.write_text("alpha modified", encoding="utf-8")
    file_c = docs / "c.txt"
    file_c.write_text("gamma", encoding="utf-8")
    capsys.readouterr()

    # Act
    rc = _run_changes(cli, ["changes", "idx", "--docs", str(docs), "--sync-key", "corpus-v1"])
    out = capsys.readouterr().out

    # Assert
    assert rc == 0
    report = json.loads(out)
    assert report["modified"] == [str(file_a.resolve())]
    assert report["added"] == [str(file_c.resolve())]
    assert report["removed"] == []
    assert report["added"] == sorted(report["added"])
    assert report["modified"] == sorted(report["modified"])


def test_changes_is_non_mutating_and_never_commits_snapshot(tmp_path, monkeypatch, capsys):
    # Arrange
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha", encoding="utf-8")
    cli = _wire_cli(monkeypatch)
    asyncio.run(
        cli.build_index(
            cli.create_parser().parse_args(
                _build_args("idx", [str(docs)], ["--sync-key", "corpus-v1"])
            )
        )
    )
    (docs / "b.txt").write_text("beta", encoding="utf-8")
    leann_dir = tmp_path / ".leann"
    hashes_before = _hash_tree(leann_dir)
    capsys.readouterr()

    # Act
    argv = ["changes", "idx", "--docs", str(docs), "--sync-key", "corpus-v1"]
    rc_first = _run_changes(cli, argv)
    out_first = capsys.readouterr().out
    rc_second = _run_changes(cli, argv)
    out_second = capsys.readouterr().out

    # Assert
    assert rc_first == 0
    assert rc_second == 0
    assert json.loads(out_first) == json.loads(out_second)
    assert json.loads(out_first)["added"] == [str((docs / "b.txt").resolve())]
    assert _hash_tree(leann_dir) == hashes_before


def test_changes_without_docs_uses_stored_sync_roots_scope(tmp_path, monkeypatch, capsys):
    # Arrange
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha", encoding="utf-8")
    cli = _wire_cli(monkeypatch)
    asyncio.run(cli.build_index(cli.create_parser().parse_args(_build_args("idx", [str(docs)]))))
    new_file = docs / "new.txt"
    new_file.write_text("new", encoding="utf-8")
    capsys.readouterr()

    # Act
    rc = _run_changes(cli, ["changes", "idx"])
    report = json.loads(capsys.readouterr().out)

    # Assert
    assert rc == 0
    assert report["added"] == [str(new_file.resolve())]
    assert report["modified"] == []
    assert report["removed"] == []


def test_changes_reports_empty_delta_immediately_after_build(tmp_path, monkeypatch, capsys):
    # Arrange
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha", encoding="utf-8")
    cli = _wire_cli(monkeypatch)
    asyncio.run(
        cli.build_index(
            cli.create_parser().parse_args(
                _build_args("idx", [str(docs)], ["--sync-key", "corpus-v1"])
            )
        )
    )
    capsys.readouterr()

    # Act
    rc = _run_changes(cli, ["changes", "idx", "--docs", str(docs), "--sync-key", "corpus-v1"])

    # Assert
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == {"added": [], "modified": [], "removed": []}


def test_changes_with_corrupt_snapshot_exits_nonzero_without_false_clean_report(
    tmp_path, monkeypatch, capsys
):
    # Arrange
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha", encoding="utf-8")
    cli = _wire_cli(monkeypatch)
    asyncio.run(
        cli.build_index(
            cli.create_parser().parse_args(
                _build_args("idx", [str(docs)], ["--sync-key", "corpus-v1"])
            )
        )
    )
    snapshot = tmp_path / ".leann" / "indexes" / "idx" / _keyed_snapshot_name("corpus-v1")
    assert snapshot.exists()
    snapshot.write_bytes(b"not a pickle at all")
    capsys.readouterr()

    # Act
    rc = _run_changes(cli, ["changes", "idx", "--docs", str(docs), "--sync-key", "corpus-v1"])
    out = capsys.readouterr().out

    # Assert
    assert rc != 0
    if out.strip():
        assert json.loads(out) != {"added": [], "modified": [], "removed": []}


def test_cli_construction_does_not_create_indexes_dir_in_cwd(tmp_path, monkeypatch):
    # Arrange
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    monkeypatch.chdir(fresh)

    # Act
    LeannCLI()

    # Assert
    assert not (fresh / ".leann" / "indexes").exists()


def test_changes_on_missing_index_exits_nonzero(tmp_path, monkeypatch, capsys):
    # Arrange
    monkeypatch.chdir(tmp_path)
    cli = _wire_cli(monkeypatch)

    # Act
    rc = _run_changes(cli, ["changes", "no-such-index"])
    captured = capsys.readouterr()

    # Assert
    assert rc != 0
    assert "not found" in captured.err
    assert captured.out.strip() == ""


def test_changes_with_wrong_sync_key_exits_nonzero(tmp_path, monkeypatch, capsys):
    # Arrange
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha", encoding="utf-8")
    cli = _wire_cli(monkeypatch)
    asyncio.run(
        cli.build_index(
            cli.create_parser().parse_args(
                _build_args("idx", [str(docs)], ["--sync-key", "corpus-v1"])
            )
        )
    )
    capsys.readouterr()

    # Act
    rc = _run_changes(cli, ["changes", "idx", "--docs", str(docs), "--sync-key", "typo-key"])
    captured = capsys.readouterr()

    # Assert
    assert rc != 0
    assert "corpus-v1" in captured.err
    assert captured.out.strip() == ""


def test_changes_with_empty_scope_and_no_docs_exits_nonzero(tmp_path, monkeypatch, capsys):
    # Arrange: index dir exists but has no sync_roots.json
    monkeypatch.chdir(tmp_path)
    index_dir = tmp_path / ".leann" / "indexes" / "idx"
    index_dir.mkdir(parents=True)
    cli = _wire_cli(monkeypatch)

    # Act
    rc = _run_changes(cli, ["changes", "idx"])
    captured = capsys.readouterr()

    # Assert
    assert rc != 0
    assert "sync scope" in captured.err
    assert captured.out.strip() == ""


def test_changes_on_missing_index_with_docs_exits_nonzero(tmp_path, monkeypatch, capsys):
    # Arrange: --docs given but the index itself does not exist (likely a typo)
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha", encoding="utf-8")
    cli = _wire_cli(monkeypatch)

    # Act
    rc = _run_changes(cli, ["changes", "no-such-index", "--docs", str(docs)])
    captured = capsys.readouterr()

    # Assert
    assert rc != 0
    assert "not found" in captured.err


def test_changes_with_key_on_unkeyed_index_exits_nonzero(tmp_path, monkeypatch, capsys):
    # Arrange: index built WITHOUT a sync key
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha", encoding="utf-8")
    cli = _wire_cli(monkeypatch)
    asyncio.run(cli.build_index(cli.create_parser().parse_args(_build_args("idx", [str(docs)]))))
    capsys.readouterr()

    # Act: a key against the unkeyed index would diff a never-written snapshot
    rc = _run_changes(cli, ["changes", "idx", "--docs", str(docs), "--sync-key", "typo"])
    captured = capsys.readouterr()

    # Assert
    assert rc != 0
    assert captured.out.strip() == ""
