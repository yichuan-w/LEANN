import hashlib
import json
import pickle
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import leann.api as leann_api
import numpy as np
from leann.api import LeannBuilder
from leann.cli import LeannCLI


def _content_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _write_ivf_index(tmp_path, *, passage_id_scheme: str = "content-hash") -> str:
    index_path = tmp_path / "documents.leann"
    existing_id = _content_id("existing text")
    passages_file = tmp_path / "documents.leann.passages.jsonl"
    offset_file = tmp_path / "documents.leann.passages.idx"
    with passages_file.open("w", encoding="utf-8") as f:
        offset = f.tell()
        f.write(
            json.dumps(
                {"id": existing_id, "text": "existing text", "metadata": {"id": existing_id}}
            )
            + "\n"
        )
    with offset_file.open("wb") as f:
        pickle.dump({existing_id: offset}, f)
    (tmp_path / "documents.index").write_bytes(b"fake-index")
    (tmp_path / "documents.leann.meta.json").write_text(
        json.dumps(
            {
                "version": "1.1",
                "backend_name": "ivf",
                "embedding_model": "test-model",
                "embedding_mode": "sentence-transformers",
                "dimensions": 2,
                "backend_kwargs": {},
                "passage_id_scheme": passage_id_scheme,
                "passage_sources": [
                    {
                        "type": "jsonl",
                        "path": passages_file.name,
                        "index_path": offset_file.name,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return str(index_path)


def _patch_ivf_update(monkeypatch):
    calls = []
    fake_ivf = ModuleType("leann_backend_ivf")

    def add_vectors(index_path, embeddings, passage_ids):
        calls.append((index_path, embeddings.copy(), list(passage_ids)))

    cast(Any, fake_ivf).add_vectors = add_vectors
    monkeypatch.setitem(sys.modules, "leann_backend_ivf", fake_ivf)
    monkeypatch.setitem(leann_api.BACKEND_REGISTRY, "ivf", SimpleNamespace())
    monkeypatch.setattr(
        leann_api,
        "compute_embeddings",
        lambda texts, *args, **kwargs: np.ones((len(texts), 2), dtype=np.float32),
    )
    return calls


def test_builder_content_hash_passage_ids_suffix_duplicate_text():
    builder = LeannBuilder(backend_name="hnsw", passage_id_scheme="content-hash")

    builder.add_text("same text", metadata={"source": "a.txt"})
    builder.add_text("same text", metadata={"source": "b.txt"})
    builder.add_text("different text", metadata={"source": "c.txt"})

    same_id = _content_id("same text")
    assert builder.chunks[0]["id"] == same_id
    assert builder.chunks[1]["id"] == f"{same_id}-1"
    assert builder.chunks[2]["id"] == _content_id("different text")


def test_builder_preserves_falsy_explicit_metadata_id():
    builder = LeannBuilder(backend_name="hnsw", passage_id_scheme="content-hash")

    builder.add_text("zero id", metadata={"id": 0, "source": "a.txt"})
    builder.add_text("empty id", metadata={"id": "", "source": "b.txt"})

    assert builder.chunks[0]["id"] == "0"
    assert builder.chunks[1]["id"] == ""


def test_builder_rejects_content_hash_diskann_until_id_map_exists():
    try:
        LeannBuilder(backend_name="diskann", passage_id_scheme="content-hash")
    except ValueError as exc:
        assert "not supported by the DiskANN backend" in str(exc)
    else:
        raise AssertionError("DiskANN content-hash builds must be rejected")


def test_legacy_missing_id_scheme_is_sequential(tmp_path):
    meta_path = tmp_path / "documents.leann.meta.json"
    meta_path.write_text(json.dumps({"version": "1.0"}), encoding="utf-8")

    cli = LeannCLI()

    assert cli._existing_index_id_scheme(str(tmp_path / "documents.leann")) == "sequential"


def test_incremental_content_hash_add_only_does_not_preassign_path_ids(tmp_path, monkeypatch):
    cli = LeannCLI()
    added_chunks = []

    class FakeBuilder:
        passage_id_scheme = "content-hash"

        def add_text(self, text, metadata=None):
            added_chunks.append((text, dict(metadata or {})))

        def update_index(self, index_path):
            assert index_path == str(tmp_path / "documents.leann")

    monkeypatch.setattr(cli, "_make_incremental_builder", lambda _args: FakeBuilder())
    all_texts = [
        {
            "text": "stable content",
            "metadata": {"file_path": str(tmp_path / "doc.txt")},
        }
    ]

    assert cli._incremental_add_only(
        str(tmp_path / "documents.leann"),
        all_texts,
        SimpleNamespace(index_name="demo"),
        {str(tmp_path / "doc.txt")},
    )

    assert added_chunks == [("stable content", {"file_path": str(tmp_path / "doc.txt")})]


def test_update_content_hash_suffixes_existing_id_collision(tmp_path, monkeypatch):
    index_path = _write_ivf_index(tmp_path)
    add_vectors_calls = _patch_ivf_update(monkeypatch)
    builder = LeannBuilder(
        backend_name="ivf",
        dimensions=2,
        passage_id_scheme="content-hash",
    )

    builder.add_text("existing text", metadata={"source": "duplicate.txt"})
    builder.update_index(index_path)

    assert add_vectors_calls[0][2] == [f"{_content_id('existing text')}-1"]
    offset_map = pickle.loads((tmp_path / "documents.leann.passages.idx").read_bytes())
    assert f"{_content_id('existing text')}-1" in offset_map


def test_update_preserves_falsy_explicit_metadata_ids(tmp_path, monkeypatch):
    index_path = _write_ivf_index(tmp_path)
    add_vectors_calls = _patch_ivf_update(monkeypatch)
    builder = LeannBuilder(
        backend_name="ivf",
        dimensions=2,
        passage_id_scheme="content-hash",
    )

    builder.add_text("zero id", metadata={"id": 0})
    builder.add_text("empty id", metadata={"id": ""})
    builder.update_index(index_path)

    assert add_vectors_calls[0][2] == ["0", ""]
    offset_map = pickle.loads((tmp_path / "documents.leann.passages.idx").read_bytes())
    assert "0" in offset_map
    assert "" in offset_map
