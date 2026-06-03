import hashlib
import json
import os
import pickle
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import leann.api as leann_api
import pytest
from leann.api import (
    PASSAGE_ID_SCHEME_CONTENT_HASH,
    PASSAGE_ID_SCHEME_SEQUENTIAL,
    LeannBuilder,
)
from leann.cli import LeannCLI


def _content_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _write_basic_meta(index_dir: Path, **overrides):
    meta = {
        "version": "1.0",
        "backend_name": "hnsw",
        "embedding_model": "dummy",
        "dimensions": 3,
        "backend_kwargs": {},
        "embedding_mode": "sentence-transformers",
        "passage_id_scheme": "sequential",
    }
    meta.update(overrides)
    (index_dir / "documents.leann.meta.json").write_text(
        json.dumps(meta),
        encoding="utf-8",
    )


def _write_passages(index_dir: Path, passages: list[dict]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    with open(index_dir / "documents.leann.passages.jsonl", "w", encoding="utf-8") as f:
        for passage in passages:
            offsets[passage["id"]] = f.tell()
            json.dump(passage, f)
            f.write("\n")
    with open(index_dir / "documents.leann.passages.idx", "wb") as f:
        pickle.dump(offsets, f)
    return offsets


def _build_fts5_db(db_path: Path, passages: list[dict]) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE bm25_passages USING fts5("
            "id UNINDEXED, text, tokenize='unicode61 remove_diacritics 2'"
            ")"
        )
        conn.executemany(
            "INSERT INTO bm25_passages(id, text) VALUES (?, ?)",
            ((passage["id"], passage["text"]) for passage in passages),
        )
        conn.commit()
    finally:
        conn.close()


def test_legacy_missing_id_scheme_is_sequential(tmp_path):
    meta_path = tmp_path / "documents.leann.meta.json"
    meta_path.write_text(json.dumps({"version": "1.0"}), encoding="utf-8")

    cli = LeannCLI()

    assert cli._existing_index_id_scheme(str(tmp_path / "documents.leann")) == "sequential"


def test_new_builds_use_backend_aware_default_id_scheme(monkeypatch):
    assert LeannBuilder(backend_name="hnsw").passage_id_scheme == PASSAGE_ID_SCHEME_CONTENT_HASH
    monkeypatch.setitem(leann_api.BACKEND_REGISTRY, "diskann", SimpleNamespace())
    assert LeannBuilder(backend_name="diskann").passage_id_scheme == PASSAGE_ID_SCHEME_SEQUENTIAL

    args = LeannCLI().create_parser().parse_args(["build", "sample", "--docs", "docs"])
    assert args.id_scheme is None
    explicit_args = (
        LeannCLI()
        .create_parser()
        .parse_args(["build", "sample", "--docs", "docs", "--id-scheme", "content-hash"])
    )
    assert explicit_args.id_scheme == PASSAGE_ID_SCHEME_CONTENT_HASH


def test_migrate_ids_rewrites_live_offsets_idmaps_meta_and_fts5(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    index_dir = tmp_path / ".leann" / "indexes" / "sample"
    index_dir.mkdir(parents=True)

    passages = [
        {"id": "0", "text": "alpha beta", "metadata": {"source": "a.txt"}},
        {"id": "1", "text": "gamma delta", "metadata": {"source": "b.txt"}},
    ]
    _write_passages(index_dir, passages)
    (index_dir / "documents.ids.txt").write_text("0\n1\n", encoding="utf-8")
    (index_dir / "documents.ivf_id_map.json").write_text(
        json.dumps(
            {
                "id_to_passage": {"0": "0", "1": "1"},
                "passage_to_id": {"0": 0, "1": 1},
                "next_id": 2,
            }
        ),
        encoding="utf-8",
    )
    bm25_db = index_dir / "documents.leann.bm25.sqlite"
    _build_fts5_db(bm25_db, passages)
    _write_basic_meta(
        index_dir,
        backend_name="ivf",
        bm25_backend="fts5",
        bm25_db="documents.leann.bm25.sqlite",
    )

    LeannCLI().migrate_ids(SimpleNamespace(index_name="sample", dry_run=False, yes=True))

    expected_ids = [_content_id("alpha beta"), _content_id("gamma delta")]
    with open(index_dir / "documents.leann.passages.jsonl", encoding="utf-8") as f:
        migrated_passages = [json.loads(line) for line in f if line.strip()]
    assert [passage["id"] for passage in migrated_passages] == expected_ids

    with open(index_dir / "documents.leann.passages.idx", "rb") as f:
        assert set(pickle.load(f)) == set(expected_ids)
    assert (index_dir / "documents.ids.txt").read_text(
        encoding="utf-8"
    ).splitlines() == expected_ids

    ivf_map = json.loads((index_dir / "documents.ivf_id_map.json").read_text(encoding="utf-8"))
    assert ivf_map == {
        "id_to_passage": {"0": expected_ids[0], "1": expected_ids[1]},
        "passage_to_id": {expected_ids[0]: 0, expected_ids[1]: 1},
        "next_id": 2,
    }

    meta = json.loads((index_dir / "documents.leann.meta.json").read_text(encoding="utf-8"))
    assert meta["version"] == "1.1"
    assert meta["passage_id_scheme"] == "content-hash"
    assert meta["bm25_backend"] == "fts5"
    assert meta["bm25_db"] == "documents.leann.bm25.sqlite"

    conn = sqlite3.connect(bm25_db)
    try:
        rows = conn.execute(
            "SELECT id FROM bm25_passages WHERE bm25_passages MATCH ?",
            ("alpha",),
        ).fetchall()
    finally:
        conn.close()
    assert rows == [(expected_ids[0],)]


def test_migrate_ids_rejects_diskann_indexes(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    index_dir = tmp_path / ".leann" / "indexes" / "sample"
    index_dir.mkdir(parents=True)
    _write_basic_meta(index_dir, backend_name="diskann")

    LeannCLI().migrate_ids(SimpleNamespace(index_name="sample", dry_run=False, yes=True))

    assert "Cannot migrate" in capsys.readouterr().out
    meta = json.loads((index_dir / "documents.leann.meta.json").read_text(encoding="utf-8"))
    assert meta["passage_id_scheme"] == "sequential"


def test_migrate_ids_ignores_stale_passages_absent_from_offset_map(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    index_dir = tmp_path / ".leann" / "indexes" / "sample"
    index_dir.mkdir(parents=True)

    live = {"id": "0", "text": "live text", "metadata": {"source": "live.txt"}}
    stale = {"id": "1", "text": "stale text", "metadata": {"source": "stale.txt"}}
    passages_file = index_dir / "documents.leann.passages.jsonl"
    with open(passages_file, "w", encoding="utf-8") as f:
        live_offset = f.tell()
        json.dump(live, f)
        f.write("\n")
        json.dump(stale, f)
        f.write("\n")
    with open(index_dir / "documents.leann.passages.idx", "wb") as f:
        pickle.dump({"0": live_offset}, f)
    (index_dir / "documents.ids.txt").write_text("0\n1\n", encoding="utf-8")
    _write_basic_meta(index_dir)

    LeannCLI().migrate_ids(SimpleNamespace(index_name="sample", dry_run=False, yes=True))

    expected_id = _content_id("live text")
    with open(passages_file, encoding="utf-8") as f:
        migrated_passages = [json.loads(line) for line in f if line.strip()]
    assert migrated_passages == [
        {"id": expected_id, "text": "live text", "metadata": {"source": "live.txt"}}
    ]
    with open(index_dir / "documents.leann.passages.idx", "rb") as f:
        assert set(pickle.load(f)) == {expected_id}
    assert (index_dir / "documents.ids.txt").read_text(encoding="utf-8").splitlines() == [
        expected_id
    ]


def test_migrate_ids_suffixes_duplicate_text_ids(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    index_dir = tmp_path / ".leann" / "indexes" / "duplicates"
    index_dir.mkdir(parents=True)

    passages = [
        {"id": "0", "text": "repeat me", "metadata": {"source": "a.txt"}},
        {"id": "1", "text": "repeat me", "metadata": {"source": "b.txt"}},
    ]
    _write_passages(index_dir, passages)
    (index_dir / "documents.ids.txt").write_text("0\n1\n", encoding="utf-8")
    _write_basic_meta(index_dir)

    LeannCLI().migrate_ids(SimpleNamespace(index_name="duplicates", dry_run=False, yes=True))

    base_id = _content_id("repeat me")
    with open(index_dir / "documents.leann.passages.jsonl", encoding="utf-8") as f:
        migrated_ids = [json.loads(line)["id"] for line in f if line.strip()]
    assert migrated_ids == [base_id, f"{base_id}-1"]
    assert (index_dir / "documents.ids.txt").read_text(
        encoding="utf-8"
    ).splitlines() == migrated_ids


def test_migrate_ids_rolls_back_when_publish_fails(tmp_path, monkeypatch):
    index_dir = tmp_path / ".leann" / "indexes" / "sample"
    index_dir.mkdir(parents=True)
    index_path = index_dir / "documents.leann"

    passages = [
        {"id": "0", "text": "alpha beta", "metadata": {"source": "a.txt"}},
        {"id": "1", "text": "gamma delta", "metadata": {"source": "b.txt"}},
    ]
    passages_file = index_dir / "documents.leann.passages.jsonl"
    offsets = {}
    with open(passages_file, "w", encoding="utf-8") as f:
        for passage in passages:
            offsets[passage["id"]] = f.tell()
            json.dump(passage, f)
            f.write("\n")

    offset_file = index_dir / "documents.leann.passages.idx"
    with open(offset_file, "wb") as f:
        pickle.dump(offsets, f)

    idmap_file = index_dir / "documents.ids.txt"
    idmap_file.write_text("0\n1\n", encoding="utf-8")
    meta_path = index_dir / "documents.leann.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "backend_name": "hnsw",
                "embedding_model": "dummy",
                "dimensions": 3,
                "backend_kwargs": {},
                "embedding_mode": "sentence-transformers",
                "passage_id_scheme": "sequential",
            }
        ),
        encoding="utf-8",
    )
    original_bytes = {
        path: path.read_bytes() for path in (passages_file, offset_file, idmap_file, meta_path)
    }
    real_replace = os.replace

    def fail_on_meta_publish(src, dst):
        if Path(dst) == meta_path and str(src).endswith(".migrate"):
            raise OSError("simulated publish failure")
        real_replace(src, dst)

    cli = LeannCLI()
    monkeypatch.setattr(cli, "_resolve_index_path", lambda *_args, **_kwargs: str(index_path))
    monkeypatch.setattr("leann.cli.os.replace", fail_on_meta_publish)

    with pytest.raises(OSError, match="simulated publish failure"):
        cli.migrate_ids(SimpleNamespace(index_name="sample", dry_run=False, yes=True))

    assert {
        path: path.read_bytes() for path in (passages_file, offset_file, idmap_file, meta_path)
    } == original_bytes
    assert not list(index_dir.glob("*.migrate"))
    assert not list(index_dir.glob(".*.backup-*"))
