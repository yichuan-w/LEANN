"""Failing tests for `leann verify` (cross-artifact index integrity check)."""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from leann.cli import LeannCLI


def _import_faiss():
    # Same import order as the IVF backend and verify, so writer and reader
    # share one SWIG build (mixing builds breaks invlists access).
    try:
        import faiss
    except ImportError:
        from leann_backend_hnsw import faiss
    return faiss


def _write_faiss_index(path: Path, num_vectors: int, dim: int = 4) -> None:
    # Mirror the real IVF backend: IndexIVFFlat + DirectMap.Hashtable with
    # explicit ids, so verify's type/direct-map checks run against the real shape.
    faiss = _import_faiss()

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, 1, faiss.METRIC_L2)
    index.set_direct_map_type(faiss.DirectMap.Hashtable)
    vectors = np.ascontiguousarray(
        np.random.default_rng(0).random((max(num_vectors, 1), dim), dtype=np.float32)
    )
    ids = np.arange(num_vectors, dtype=np.int64)
    try:
        index.train(vectors)
        if num_vectors:
            index.add_with_ids(vectors[:num_vectors], ids)
    except TypeError:
        index.train(vectors.shape[0], faiss.swig_ptr(vectors))
        if num_vectors:
            index.add_with_ids(num_vectors, faiss.swig_ptr(vectors), faiss.swig_ptr(ids))
    faiss.write_index(index, str(path))


def _make_ivf_index(tmp_path: Path, passage_ids: list[str]) -> Path:
    index_dir = tmp_path / ".leann" / "indexes" / "idx"
    index_dir.mkdir(parents=True)
    prefix = index_dir / "documents.leann"

    Path(str(prefix) + ".meta.json").write_text(
        json.dumps(
            {
                "backend_name": "ivf",
                "embedding_model": "m",
                "embedding_mode": "sentence-transformers",
                "dimensions": 4,
                "backend_kwargs": {},
            }
        ),
        encoding="utf-8",
    )

    offsets: dict[str, int] = {}
    with open(str(prefix) + ".passages.jsonl", "wb") as f:
        for pid in passage_ids:
            offsets[pid] = f.tell()
            line = json.dumps({"id": pid, "text": f"passage {pid}", "metadata": {}}) + "\n"
            f.write(line.encode("utf-8"))
    with open(str(prefix) + ".passages.idx", "wb") as f:
        pickle.dump(offsets, f)

    id_map = {
        "id_to_passage": {str(i): pid for i, pid in enumerate(passage_ids)},
        "passage_to_id": {pid: i for i, pid in enumerate(passage_ids)},
        "next_id": len(passage_ids),
    }
    Path(str(prefix).removesuffix(".leann") + ".ivf_id_map.json").write_text(
        json.dumps(id_map), encoding="utf-8"
    )

    _write_faiss_index(
        Path(str(prefix).removesuffix(".leann") + ".index"), num_vectors=len(passage_ids)
    )
    return prefix


def _make_ivf_index_with_entries(tmp_path: Path, entries: list[tuple[Any, Any]]) -> Path:
    # Like _make_ivf_index but accepts explicit (pid, text) pairs so a pid may
    # repeat (content-hash id scheme). idx and passage_to_id are last-wins.
    index_dir = tmp_path / ".leann" / "indexes" / "idx"
    index_dir.mkdir(parents=True)
    prefix = index_dir / "documents.leann"

    Path(str(prefix) + ".meta.json").write_text(
        json.dumps(
            {
                "backend_name": "ivf",
                "embedding_model": "m",
                "embedding_mode": "sentence-transformers",
                "dimensions": 4,
                "backend_kwargs": {},
            }
        ),
        encoding="utf-8",
    )

    offsets: dict[str, int] = {}
    with open(str(prefix) + ".passages.jsonl", "wb") as f:
        for pid, text in entries:
            offsets[pid] = f.tell()
            line = json.dumps({"id": pid, "text": text, "metadata": {}}) + "\n"
            f.write(line.encode("utf-8"))
    with open(str(prefix) + ".passages.idx", "wb") as f:
        pickle.dump(offsets, f)

    id_map = {
        "id_to_passage": {str(i): pid for i, (pid, _) in enumerate(entries)},
        "passage_to_id": {pid: i for i, (pid, _) in enumerate(entries)},
        "next_id": len(entries),
    }
    Path(str(prefix).removesuffix(".leann") + ".ivf_id_map.json").write_text(
        json.dumps(id_map), encoding="utf-8"
    )

    _write_faiss_index(
        Path(str(prefix).removesuffix(".leann") + ".index"), num_vectors=len(entries)
    )
    return prefix


def _run_verify(index_name: str = "idx") -> int:
    cli = LeannCLI()
    args = cli.create_parser().parse_args(["verify", index_name])
    return int(cli.verify_command(args) or 0)


def test_verify_passes_on_healthy_ivf_index(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.chdir(tmp_path)
    _make_ivf_index(tmp_path, ["0", "1", "2"])

    # Act
    rc = _run_verify()

    # Assert
    assert rc == 0


def test_verify_fails_on_truncated_passages_jsonl(tmp_path, monkeypatch, capsys):
    # Arrange
    monkeypatch.chdir(tmp_path)
    prefix = _make_ivf_index(tmp_path, ["0", "1", "2"])
    jsonl = Path(str(prefix) + ".passages.jsonl")
    lines = jsonl.read_bytes().splitlines(keepends=True)
    jsonl.write_bytes(b"".join(lines[:-1]))

    # Act
    rc = _run_verify()

    # Assert
    assert rc != 0
    captured = capsys.readouterr()
    assert (captured.out + captured.err).strip()


def test_verify_fails_when_id_map_not_exact_inverse(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.chdir(tmp_path)
    prefix = _make_ivf_index(tmp_path, ["0", "1", "2"])
    id_map_path = Path(str(prefix).removesuffix(".leann") + ".ivf_id_map.json")
    id_map = json.loads(id_map_path.read_text(encoding="utf-8"))
    id_map["passage_to_id"]["2"] = 0
    id_map_path.write_text(json.dumps(id_map), encoding="utf-8")

    # Act
    rc = _run_verify()

    # Assert
    assert rc != 0


def test_verify_fails_on_bad_offset_in_passages_idx(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.chdir(tmp_path)
    prefix = _make_ivf_index(tmp_path, ["0", "1", "2"])
    idx_path = Path(str(prefix) + ".passages.idx")
    with open(idx_path, "rb") as f:
        offsets = pickle.load(f)
    offsets["1"] = offsets["1"] + 3
    with open(idx_path, "wb") as f:
        pickle.dump(offsets, f)

    # Act
    rc = _run_verify()

    # Assert
    assert rc != 0


def test_verify_fails_when_id_map_has_id_missing_from_passages(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.chdir(tmp_path)
    prefix = _make_ivf_index(tmp_path, ["0", "1", "2"])
    id_map_path = Path(str(prefix).removesuffix(".leann") + ".ivf_id_map.json")
    id_map = json.loads(id_map_path.read_text(encoding="utf-8"))
    id_map["id_to_passage"]["3"] = "orphan"
    id_map["passage_to_id"]["orphan"] = 3
    id_map["next_id"] = 4
    id_map_path.write_text(json.dumps(id_map), encoding="utf-8")
    _write_faiss_index(Path(str(prefix).removesuffix(".leann") + ".index"), num_vectors=4)

    # Act
    rc = _run_verify()

    # Assert
    assert rc != 0


def test_verify_reports_finding_instead_of_crashing_on_non_numeric_id_keys(
    tmp_path, monkeypatch, capsys
):
    # Arrange: every id_to_passage key is non-numeric (the exact corruption verify flags)
    prefix = _make_ivf_index(tmp_path, ["p0", "p1"])
    id_map_path = prefix.parent / "documents.ivf_id_map.json"
    id_map = json.loads(id_map_path.read_text(encoding="utf-8"))
    id_map["id_to_passage"] = {"abc": "p0", "xyz": "p1"}
    id_map["passage_to_id"] = {"p0": "abc", "p1": "xyz"}
    id_map_path.write_text(json.dumps(id_map), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    # Act
    rc = _run_verify()
    out = capsys.readouterr().out

    # Assert: findings printed, no ValueError traceback
    assert rc == 1
    assert "not an integer" in out


def test_verify_unreadable_idx_does_not_emit_misleading_cross_findings(
    tmp_path, monkeypatch, capsys
):
    # Arrange
    prefix = _make_ivf_index(tmp_path, ["p0", "p1"])
    Path(str(prefix) + ".passages.idx").write_bytes(b"not a pickle")
    monkeypatch.chdir(tmp_path)

    # Act
    rc = _run_verify()
    out = capsys.readouterr().out

    # Assert: the unreadable idx is the finding; no derived mismatch noise
    assert rc == 1
    assert "passages.idx unreadable" in out
    assert "do not match passages.idx" not in out
    assert "passages.jsonl has" not in out


def test_verify_fails_when_index_is_not_ivf_type(tmp_path, monkeypatch, capsys):
    # Arrange: replace the IVF index with a flat index (no direct map)
    faiss = _import_faiss()

    prefix = _make_ivf_index(tmp_path, ["p0", "p1"])
    flat = faiss.IndexFlatL2(4)
    vectors = np.ascontiguousarray(np.random.default_rng(0).random((2, 4), dtype=np.float32))
    try:
        flat.add(vectors)
    except TypeError:
        flat.add(2, faiss.swig_ptr(vectors))
    faiss.write_index(flat, str(prefix.parent / "documents.index"))
    monkeypatch.chdir(tmp_path)

    # Act
    rc = _run_verify()
    out = capsys.readouterr().out

    # Assert
    assert rc == 1
    assert "not an IVF index" in out


def test_verify_reports_findings_on_malformed_artifact_types(tmp_path, monkeypatch, capsys):
    # Arrange: decodable but wrong-shaped artifacts must not traceback
    prefix = _make_ivf_index(tmp_path, ["p0", "p1"])
    with open(str(prefix) + ".passages.idx", "wb") as f:
        pickle.dump(["not", "a", "dict"], f)
    id_map_path = prefix.parent / "documents.ivf_id_map.json"
    id_map = json.loads(id_map_path.read_text(encoding="utf-8"))
    id_map["next_id"] = "two"
    id_map_path.write_text(json.dumps(id_map), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    # Act
    rc = _run_verify()
    out = capsys.readouterr().out

    # Assert
    assert rc == 1
    assert "passages.idx is not a dict" in out
    assert "next_id is not an integer" in out


def test_verify_flags_missing_snapshot_for_recorded_scope(tmp_path, monkeypatch, capsys):
    # Arrange: sync_roots.json records a root but the snapshot pickle is absent
    # (interrupted between index write and snapshot commit)
    prefix = _make_ivf_index(tmp_path, ["p0", "p1"])
    docs = tmp_path / "docs"
    docs.mkdir()
    (prefix.parent / "sync_roots.json").write_text(
        json.dumps({"directories": [str(docs)], "files": []}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    # Act
    rc = _run_verify()
    out = capsys.readouterr().out

    # Assert
    assert rc == 1
    assert "sync snapshot missing" in out


def test_verify_flags_corrupt_snapshot_for_recorded_scope(tmp_path, monkeypatch, capsys):
    # Arrange
    import hashlib as _hashlib

    prefix = _make_ivf_index(tmp_path, ["p0", "p1"])
    docs = tmp_path / "docs"
    docs.mkdir()
    (prefix.parent / "sync_roots.json").write_text(
        json.dumps({"directories": [str(docs)], "files": []}), encoding="utf-8"
    )
    tag = _hashlib.sha256(str(docs).encode()).hexdigest()[:12]
    (prefix.parent / f"sync_{tag}.pickle").write_bytes(b"not a pickle")
    monkeypatch.chdir(tmp_path)

    # Act
    rc = _run_verify()
    out = capsys.readouterr().out

    # Assert
    assert rc == 1
    assert "corrupt" in out


def test_verify_flags_mapped_id_absent_from_index_vectors(tmp_path, monkeypatch, capsys):
    # Arrange: id map references id 5 which the 2-vector index does not contain
    prefix = _make_ivf_index(tmp_path, ["p0", "p1"])
    id_map_path = prefix.parent / "documents.ivf_id_map.json"
    id_map = json.loads(id_map_path.read_text(encoding="utf-8"))
    id_map["id_to_passage"] = {"0": "p0", "5": "p1"}
    id_map["passage_to_id"] = {"p0": 0, "p1": 5}
    id_map["next_id"] = 6
    id_map_path.write_text(json.dumps(id_map), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    # Act
    rc = _run_verify()
    out = capsys.readouterr().out

    # Assert
    assert rc == 1
    assert "missing 1 of 2 mapped ids" in out


def test_verify_passes_on_healthy_index_with_duplicate_content_hash_ids(
    tmp_path, monkeypatch, capsys
):
    # Arrange: byte-identical chunks legitimately share one content-hash id
    _make_ivf_index_with_entries(
        tmp_path, [("dup", "same text"), ("dup", "same text"), ("p1", "other text")]
    )
    monkeypatch.chdir(tmp_path)

    # Act
    rc = _run_verify()
    out = capsys.readouterr().out

    # Assert
    assert rc == 0
    assert "duplicate" not in out
    assert "do not match" not in out


def test_verify_fails_on_duplicate_id_with_different_text(tmp_path, monkeypatch, capsys):
    # Arrange
    _make_ivf_index_with_entries(
        tmp_path, [("dup", "text one"), ("dup", "text two"), ("p1", "other text")]
    )
    monkeypatch.chdir(tmp_path)

    # Act
    rc = _run_verify()
    out = capsys.readouterr().out

    # Assert
    assert rc == 1
    assert out.strip()


def test_verify_fails_on_duplicate_id_with_type_coerced_text(tmp_path, monkeypatch, capsys):
    # Arrange: same id, texts 1 (int) vs "1" (str) must count as different text
    _make_ivf_index_with_entries(tmp_path, [("dup", 1), ("dup", "1")])
    monkeypatch.chdir(tmp_path)

    # Act
    rc = _run_verify()
    out = capsys.readouterr().out

    # Assert
    assert rc == 1
    assert "different text" in out


def test_verify_reports_conflicts_with_heterogeneous_id_types(tmp_path, monkeypatch, capsys):
    # Arrange: conflicting ids of mixed types (int and str) must not crash sorted()
    _make_ivf_index_with_entries(
        tmp_path, [(1, "text a"), (1, "text b"), ("x", "text c"), ("x", "text d")]
    )
    monkeypatch.chdir(tmp_path)

    # Act
    rc = _run_verify()
    out = capsys.readouterr().out

    # Assert
    assert rc == 1
    assert "different text" in out


def test_verify_fails_when_passage_to_id_label_maps_to_other_pid(tmp_path, monkeypatch):
    # Arrange: passage_to_id["p1"] points at a faiss label owned by "dup"
    prefix = _make_ivf_index_with_entries(
        tmp_path, [("dup", "same text"), ("dup", "same text"), ("p1", "other text")]
    )
    id_map_path = prefix.parent / "documents.ivf_id_map.json"
    id_map = json.loads(id_map_path.read_text(encoding="utf-8"))
    id_map["passage_to_id"]["p1"] = 0
    id_map_path.write_text(json.dumps(id_map), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    # Act
    rc = _run_verify()

    # Assert
    assert rc != 0


def test_verify_fails_when_pid_missing_from_passage_to_id(tmp_path, monkeypatch):
    # Arrange
    prefix = _make_ivf_index_with_entries(
        tmp_path, [("dup", "same text"), ("dup", "same text"), ("p1", "other text")]
    )
    id_map_path = prefix.parent / "documents.ivf_id_map.json"
    id_map = json.loads(id_map_path.read_text(encoding="utf-8"))
    del id_map["passage_to_id"]["p1"]
    id_map_path.write_text(json.dumps(id_map), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    # Act
    rc = _run_verify()

    # Assert
    assert rc != 0
