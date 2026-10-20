"""Explicit passage IDs must survive the build and stay reachable after it.

Datasets hand LEANN integer IDs (a row id, a dataframe index), so ``0`` is a valid ID
and ``.passages.idx`` has to be keyed by the same string the backend returns for a
label. ``build_index_from_arrays()`` already stringifies the IDs it is given; this is
the same contract for the text-building path.
"""

import json
import pickle
from unittest.mock import Mock, patch

import leann.api as api
import numpy as np
from leann.api import LeannBuilder, LeannSearcher


def _build(tmp_path, monkeypatch, texts_and_metadata):
    backend = Mock()
    monkeypatch.setitem(api.BACKEND_REGISTRY, "id-test", backend)
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    builder = LeannBuilder(backend_name="id-test", dimensions=2)
    for text, metadata in texts_and_metadata:
        builder.add_text(text, metadata=metadata)
    with patch.object(api, "compute_embeddings", return_value=vectors):
        builder.build_index(str(tmp_path / "docs.leann"))
    return backend, vectors


def _read_offsets(tmp_path):
    with open(tmp_path / "docs.leann.passages.idx", "rb") as f:
        return pickle.load(f)


def test_integer_metadata_id_keys_the_offset_map_as_a_string(tmp_path, monkeypatch):
    """An int ID reaches ``.ids.txt`` as a string, so an int-keyed offset map misses it."""
    _build(tmp_path, monkeypatch, [("alpha", {"id": 7}), ("beta", {})])

    with open(tmp_path / "docs.ids.txt", encoding="utf-8") as f:
        advertised = f.read().split()

    offsets = _read_offsets(tmp_path)
    assert advertised == ["7", "1"]
    assert set(offsets) == {"7", "1"}


def test_zero_metadata_id_keeps_its_own_passage_searchable(tmp_path, monkeypatch):
    """``id=0`` is a real ID. Reusing the insertion position instead of it made the two
    passages share one ID, and the searcher dropped the hit whose ID was not in the map.
    """
    backend, vectors = _build(tmp_path, monkeypatch, [("alpha", {"id": "1"}), ("beta", {"id": 0})])

    with open(tmp_path / "docs.leann.passages.jsonl", encoding="utf-8") as f:
        written = [json.loads(line)["id"] for line in f]

    assert written == ["1", "0"]
    assert set(_read_offsets(tmp_path)) == {"1", "0"}

    # The mock stands in for a backend that resolves its own labels through ``.ids.txt``,
    # which is where a passage ID that was overwritten upstream stops matching.
    backend.searcher.return_value.compute_query_embedding.return_value = vectors[0]
    backend.searcher.return_value.search.return_value = {
        "labels": [written],
        "distances": [[0.9, 0.8]],
    }
    with LeannSearcher(
        str(tmp_path / "docs.leann"), enable_warmup=False, recompute_embeddings=False
    ) as searcher:
        results = searcher.search("doc", top_k=2)

    assert [(result.id, result.text) for result in results] == [("1", "alpha"), ("0", "beta")]
