import json

import numpy as np
from leann.api import PASSAGE_ID_SCHEME_SEQUENTIAL, LeannBuilder


class _FakeBackendFactory:
    class _Builder:
        def build(self, embeddings, ids, index_path, **kwargs):
            return None

    def builder(self, **kwargs):
        return self._Builder()


def test_build_index_records_default_passage_id_scheme(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "leann.api.compute_embeddings",
        lambda chunks, *args, **kwargs: np.ones((len(chunks), 2), dtype=np.float32),
    )
    builder = LeannBuilder(backend_name="hnsw", dimensions=2)
    builder.backend_factory = _FakeBackendFactory()
    builder.add_text("alpha")

    index_path = tmp_path / "documents.leann"
    builder.build_index(str(index_path))

    meta = json.loads((tmp_path / "documents.leann.meta.json").read_text(encoding="utf-8"))
    assert meta["passage_id_scheme"] == PASSAGE_ID_SCHEME_SEQUENTIAL


def test_build_index_from_arrays_records_default_passage_id_scheme(tmp_path):
    builder = LeannBuilder(backend_name="hnsw", dimensions=2)
    builder.backend_factory = _FakeBackendFactory()
    builder.add_text("alpha")

    index_path = tmp_path / "documents.leann"
    builder.build_index_from_arrays(
        str(index_path),
        ids=["0"],
        embeddings=np.ones((1, 2), dtype=np.float32),
    )

    meta = json.loads((tmp_path / "documents.leann.meta.json").read_text(encoding="utf-8"))
    assert meta["passage_id_scheme"] == PASSAGE_ID_SCHEME_SEQUENTIAL
