import hashlib
import json
import pickle
import sys
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest
from leann.api import LeannBuilder
from leann.registry import BACKEND_REGISTRY


def _content_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@pytest.fixture(autouse=True)
def _register_minimal_backends(monkeypatch):
    registry = dict(BACKEND_REGISTRY)
    registry["hnsw"] = cast(Any, object())
    registry["ivf"] = cast(Any, object())
    monkeypatch.setattr("leann.api.BACKEND_REGISTRY", registry)


def _write_minimal_ivf_index(tmp_path, index_name: str, passages: list[dict]) -> str:
    index_path = tmp_path / index_name
    passages_file = tmp_path / f"{index_name}.passages.jsonl"
    offset_file = tmp_path / f"{index_name}.passages.idx"
    meta_file = tmp_path / f"{index_name}.meta.json"
    index_file = tmp_path / f"{index_path.stem}.index"

    offset_map = {}
    with open(passages_file, "w", encoding="utf-8") as f:
        for passage in passages:
            offset_map[passage["id"]] = f.tell()
            json.dump(passage, f)
            f.write("\n")
    with open(offset_file, "wb") as f:
        pickle.dump(offset_map, f)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "backend_name": "ivf",
                "backend_kwargs": {"distance_metric": "mips"},
                "dimensions": 2,
                "passage_id_scheme": "content-hash",
                "total_passages": len(passages),
            },
            f,
        )
    index_file.write_bytes(b"fake-index")
    return str(index_path)


def _install_fake_ivf_backend(monkeypatch, captured_passage_ids: list[list[str]]) -> None:
    class FakeIvfModule(ModuleType):
        def add_vectors(self, _index_path, _embeddings, passage_ids):
            captured_passage_ids.append(list(passage_ids))

    fake_ivf = FakeIvfModule("leann_backend_ivf")
    monkeypatch.setitem(sys.modules, "leann_backend_ivf", fake_ivf)


def test_builder_content_hash_passage_ids_are_unique_for_duplicate_text():
    builder = LeannBuilder(backend_name="hnsw", passage_id_scheme="content-hash")

    builder.add_text("same text", metadata={"source": "a.txt"})
    builder.add_text("same text", metadata={"source": "b.txt"})
    builder.add_text("same text", metadata={"source": "c.txt"})
    builder.add_text("different text", metadata={"source": "d.txt"})

    same_id = _content_id("same text")
    assert [chunk["id"] for chunk in builder.chunks] == [
        same_id,
        f"{same_id}-1",
        f"{same_id}-2",
        _content_id("different text"),
    ]


def test_builder_respects_explicit_metadata_id_with_content_hash_scheme():
    builder = LeannBuilder(backend_name="hnsw", passage_id_scheme="content-hash")

    builder.add_text("same text", metadata={"id": "explicit-id", "source": "a.txt"})

    assert builder.chunks[0]["id"] == "explicit-id"


def test_builder_preserves_falsy_explicit_metadata_ids():
    builder = LeannBuilder(backend_name="hnsw", passage_id_scheme="content-hash")

    builder.add_text("zero id", metadata={"id": 0, "source": "a.txt"})
    builder.add_text("empty id", metadata={"id": "", "source": "b.txt"})

    assert builder.chunks[0]["id"] == "0"
    assert builder.chunks[1]["id"] == ""


def test_builder_rejects_content_hash_diskann_until_id_map_exists():
    with pytest.raises(ValueError, match="not supported by the DiskANN backend"):
        LeannBuilder(backend_name="diskann", passage_id_scheme="content-hash")


def test_ivf_update_content_hash_suffixes_existing_duplicate(tmp_path, monkeypatch):
    base_id = _content_id("same text")
    index_path = _write_minimal_ivf_index(
        tmp_path,
        "docs.leann",
        [{"id": base_id, "text": "same text", "metadata": {"source": "old.txt"}}],
    )
    captured_passage_ids = []
    _install_fake_ivf_backend(monkeypatch, captured_passage_ids)
    monkeypatch.setattr(
        "leann.api.compute_embeddings",
        lambda *_args, **_kwargs: np.ones((1, 2), dtype=np.float32),
    )

    builder = LeannBuilder(backend_name="ivf", passage_id_scheme="content-hash")
    builder.add_text("same text", metadata={"source": "new.txt"})
    builder.update_index(index_path)

    suffixed_id = f"{base_id}-1"
    assert captured_passage_ids == [[suffixed_id]]
    with open(tmp_path / "docs.leann.passages.idx", "rb") as f:
        offset_map = pickle.load(f)
    assert set(offset_map) == {base_id, suffixed_id}


def test_ivf_update_preserves_falsy_explicit_metadata_ids(tmp_path, monkeypatch):
    index_path = _write_minimal_ivf_index(tmp_path, "docs.leann", [])
    captured_passage_ids = []
    _install_fake_ivf_backend(monkeypatch, captured_passage_ids)
    monkeypatch.setattr(
        "leann.api.compute_embeddings",
        lambda *_args, **_kwargs: np.ones((2, 2), dtype=np.float32),
    )

    builder = LeannBuilder(backend_name="ivf", passage_id_scheme="content-hash")
    builder.add_text("zero id", metadata={"id": 0})
    builder.add_text("empty id", metadata={"id": ""})
    builder.update_index(index_path)

    assert captured_passage_ids == [["0", ""]]
    with open(tmp_path / "docs.leann.passages.idx", "rb") as f:
        offset_map = pickle.load(f)
    assert set(offset_map) == {"0", ""}
