"""Failing tests for `leann build --sync-key` (stable global snapshot identity)."""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from leann.cli import LeannCLI


def _keyed_snapshot_name(key: str) -> str:
    return f"sync_key_{hashlib.sha256(key.encode()).hexdigest()[:12]}.pickle"


def _make_fake_builder(recorded_builds: list[list[str]]):
    class FakeBuilder:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.paths: list[str] = []
            recorded_builds.append(self.paths)

        def add_text(self, _text, metadata=None):
            self.paths.append((metadata or {}).get("file_path", ""))

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


def _fake_load_documents(loaded_calls: list[set[str]]):
    def fake(docs_paths, custom_file_types=None, include_hidden=False, args=None):
        if isinstance(docs_paths, str):
            docs_paths = [docs_paths]
        files: list[Path] = []
        for p in docs_paths:
            path = Path(p)
            if path.is_dir():
                files.extend(sorted(path.rglob("*.txt")))
            elif path.is_file():
                files.append(path)
        resolved = {str(f.resolve()) for f in files}
        loaded_calls.append(resolved)
        return [{"text": f.name, "metadata": {"file_path": str(f.resolve())}} for f in files]

    return fake


def _wire_cli(monkeypatch, recorded_builds, loaded_calls):
    cli = LeannCLI()
    monkeypatch.setattr(cli, "load_documents", _fake_load_documents(loaded_calls))
    monkeypatch.setattr(cli, "register_project_dir", lambda: None)
    monkeypatch.setattr("leann.cli.LeannBuilder", _make_fake_builder(recorded_builds))
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


def test_same_key_different_docs_lists_share_snapshot_and_load_only_new_files(
    tmp_path, monkeypatch
):
    # Arrange
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    file_a = docs / "a.txt"
    file_b = docs / "b.txt"
    file_c = docs / "c.txt"
    file_a.write_text("alpha", encoding="utf-8")
    file_b.write_text("beta", encoding="utf-8")
    recorded_builds: list[list[str]] = []
    loaded_calls: list[set[str]] = []
    cli = _wire_cli(monkeypatch, recorded_builds, loaded_calls)
    parser = cli.create_parser()

    # Act
    asyncio.run(
        cli.build_index(
            parser.parse_args(
                _build_args("keyed", [str(file_a), str(file_b)], ["--sync-key", "corpus-v1"])
            )
        )
    )
    index_dir = tmp_path / ".leann" / "indexes" / "keyed"
    file_c.write_text("gamma", encoding="utf-8")
    calls_before_second = len(loaded_calls)
    asyncio.run(
        cli.build_index(
            parser.parse_args(
                _build_args(
                    "keyed",
                    [str(file_a), str(file_b), str(file_c)],
                    ["--sync-key", "corpus-v1"],
                )
            )
        )
    )

    # Assert
    assert (index_dir / _keyed_snapshot_name("corpus-v1")).exists()
    second_build_loaded: set[str] = set()
    for call in loaded_calls[calls_before_second:]:
        second_build_loaded |= call
    assert second_build_loaded == {str(file_c.resolve())}


def test_sync_key_persisted_in_sync_roots_json(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha", encoding="utf-8")
    cli = _wire_cli(monkeypatch, [], [])
    parser = cli.create_parser()

    # Act
    asyncio.run(
        cli.build_index(
            parser.parse_args(_build_args("keyed", [str(docs)], ["--sync-key", "corpus-v1"]))
        )
    )

    # Assert
    sync_config = json.loads(
        (tmp_path / ".leann" / "indexes" / "keyed" / "sync_roots.json").read_text(encoding="utf-8")
    )
    assert sync_config["sync_key"] == "corpus-v1"


def test_different_key_on_keyed_index_errors_without_force_and_succeeds_with_force(
    tmp_path, monkeypatch
):
    # Arrange
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha", encoding="utf-8")
    cli = _wire_cli(monkeypatch, [], [])
    parser = cli.create_parser()
    asyncio.run(
        cli.build_index(
            parser.parse_args(_build_args("keyed", [str(docs)], ["--sync-key", "key-one"]))
        )
    )
    index_dir = tmp_path / ".leann" / "indexes" / "keyed"

    # Act / Assert
    with pytest.raises((SystemExit, RuntimeError, ValueError)):
        asyncio.run(
            cli.build_index(
                parser.parse_args(_build_args("keyed", [str(docs)], ["--sync-key", "key-two"]))
            )
        )

    asyncio.run(
        cli.build_index(
            parser.parse_args(
                _build_args("keyed", [str(docs)], ["--sync-key", "key-two", "--force"])
            )
        )
    )
    sync_config = json.loads((index_dir / "sync_roots.json").read_text(encoding="utf-8"))
    assert sync_config["sync_key"] == "key-two"


def test_directory_dropped_from_docs_triggers_rebuild_with_remaining_files_only(
    tmp_path, monkeypatch
):
    # Arrange
    monkeypatch.chdir(tmp_path)
    dir_one = tmp_path / "d1"
    dir_two = tmp_path / "d2"
    dir_one.mkdir()
    dir_two.mkdir()
    kept = dir_one / "kept.txt"
    kept.write_text("kept", encoding="utf-8")
    (dir_two / "dropped.txt").write_text("dropped", encoding="utf-8")
    recorded_builds: list[list[str]] = []
    cli = _wire_cli(monkeypatch, recorded_builds, [])
    parser = cli.create_parser()
    asyncio.run(
        cli.build_index(
            parser.parse_args(
                _build_args("keyed", [str(dir_one), str(dir_two)], ["--sync-key", "corpus-v1"])
            )
        )
    )
    builds_before = len(recorded_builds)

    # Act
    asyncio.run(
        cli.build_index(
            parser.parse_args(_build_args("keyed", [str(dir_one)], ["--sync-key", "corpus-v1"]))
        )
    )

    # Assert
    rebuilt_paths: set[str] = set()
    for build in recorded_builds[builds_before:]:
        rebuilt_paths |= set(build)
    assert rebuilt_paths == {str(kept.resolve())}


def test_load_snapshot_raises_snapshot_corrupt_error_on_corrupt_pickle(tmp_path):
    from leann.sync import FileSynchronizer, SnapshotCorruptError

    # Arrange
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha", encoding="utf-8")
    corrupt_path = tmp_path / "snapshot.pickle"
    corrupt_path.write_bytes(b"not a pickle at all")
    fs = FileSynchronizer(root_dir=str(docs), snapshot_path=str(corrupt_path), auto_load=False)

    # Act / Assert
    with pytest.raises(SnapshotCorruptError):
        fs.load_snapshot()

    missing = FileSynchronizer(
        root_dir=str(docs), snapshot_path=str(tmp_path / "absent.pickle"), auto_load=False
    )
    missing.load_snapshot()
    assert missing.tree is None


def test_unkeyed_build_fails_loud_on_corrupt_snapshot_and_force_resets(tmp_path, monkeypatch):
    from leann.sync import SnapshotCorruptError

    # Arrange: unkeyed build, then corrupt the per-root snapshot
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha", encoding="utf-8")
    recorded_builds: list[list[str]] = []
    loaded_calls: list[set[str]] = []
    cli = _wire_cli(monkeypatch, recorded_builds, loaded_calls)
    asyncio.run(cli.build_index(cli.create_parser().parse_args(_build_args("idx", [str(docs)]))))
    index_dir = tmp_path / ".leann" / "indexes" / "idx"
    snapshots = list(index_dir.glob("sync_*.pickle"))
    assert snapshots
    for snap in snapshots:
        snap.write_bytes(b"not a pickle")

    # Act / Assert: without --force the corruption is a hard error
    with pytest.raises(SnapshotCorruptError, match="--force"):
        asyncio.run(
            cli.build_index(cli.create_parser().parse_args(_build_args("idx", [str(docs)])))
        )

    # Act: --force resets the corrupt snapshot and rebuilds
    asyncio.run(
        cli.build_index(
            cli.create_parser().parse_args(_build_args("idx", [str(docs)], ["--force"]))
        )
    )

    # Assert: snapshot is valid again (a subsequent build sees no changes)
    asyncio.run(cli.build_index(cli.create_parser().parse_args(_build_args("idx", [str(docs)]))))


def test_build_fails_loud_on_corrupt_sync_roots_json(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha", encoding="utf-8")
    recorded_builds: list[list[str]] = []
    loaded_calls: list[set[str]] = []
    cli = _wire_cli(monkeypatch, recorded_builds, loaded_calls)
    asyncio.run(
        cli.build_index(
            cli.create_parser().parse_args(
                _build_args("idx", [str(docs)], ["--sync-key", "corpus-v1"])
            )
        )
    )
    sync_roots = tmp_path / ".leann" / "indexes" / "idx" / "sync_roots.json"
    sync_roots.write_text("{not json", encoding="utf-8")

    # Act / Assert: an unreadable config must not silently unkey the index
    with pytest.raises(ValueError, match=r"sync_roots\.json"):
        asyncio.run(
            cli.build_index(cli.create_parser().parse_args(_build_args("idx", [str(docs)])))
        )

    # Act: --force ignores the corrupt config and rewrites it
    asyncio.run(
        cli.build_index(
            cli.create_parser().parse_args(_build_args("idx", [str(docs)], ["--force"]))
        )
    )
    assert json.loads(sync_roots.read_text(encoding="utf-8"))


def test_build_fails_loud_on_sync_roots_json_with_wrong_json_type(tmp_path, monkeypatch):
    # Arrange: valid JSON that is not an object must not silently unkey the index
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha", encoding="utf-8")
    recorded_builds: list[list[str]] = []
    loaded_calls: list[set[str]] = []
    cli = _wire_cli(monkeypatch, recorded_builds, loaded_calls)
    asyncio.run(
        cli.build_index(
            cli.create_parser().parse_args(
                _build_args("idx", [str(docs)], ["--sync-key", "corpus-v1"])
            )
        )
    )
    sync_roots = tmp_path / ".leann" / "indexes" / "idx" / "sync_roots.json"
    sync_roots.write_text("[]", encoding="utf-8")

    # Act / Assert
    with pytest.raises(ValueError, match=r"sync_roots\.json"):
        asyncio.run(
            cli.build_index(cli.create_parser().parse_args(_build_args("idx", [str(docs)])))
        )

    # --force recovers
    asyncio.run(
        cli.build_index(
            cli.create_parser().parse_args(_build_args("idx", [str(docs)], ["--force"]))
        )
    )
