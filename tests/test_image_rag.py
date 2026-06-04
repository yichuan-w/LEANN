import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps import image_rag
from apps.image_rag import ImageRAG


def _write_image(path, size=(3, 2), color=(255, 0, 0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_discover_image_files_is_deterministic_and_case_insensitive(tmp_path):
    image_dir = tmp_path / "images"
    _write_image(image_dir / "b" / "two.PNG")
    _write_image(image_dir / "A.jpg")
    (image_dir / "notes.txt").write_text("not an image", encoding="utf-8")

    files = image_rag.discover_image_files(image_dir, ["jpg", ".png"])

    assert [path.relative_to(image_dir).as_posix() for path in files] == ["A.jpg", "b/two.PNG"]


def test_image_metadata_has_stable_id_and_dimensions(tmp_path):
    image_dir = tmp_path / "images"
    image_path = image_dir / "nested" / "sample.JPEG"
    _write_image(image_path, size=(7, 5))

    metadata = image_rag.image_record_metadata(image_path, image_dir, "clip-test")

    assert metadata["id"] == image_rag.image_passage_id(image_dir, image_path)
    assert metadata["relative_path"] == "nested/sample.JPEG"
    assert metadata["extension"] == ".jpeg"
    assert metadata["width"] == 7
    assert metadata["height"] == 5
    assert metadata["media_type"] == "image"
    assert metadata["embedding_model"] == "clip-test"


def test_load_images_uses_fake_encoder_and_respects_max_items(tmp_path):
    image_dir = tmp_path / "images"
    _write_image(image_dir / "first.png")
    _write_image(image_dir / "second.png")

    class FakeModel:
        def encode(self, images, **kwargs):
            assert kwargs["normalize_embeddings"] is True
            return np.array([[1.0, 0.0] for _ in images], dtype=np.float32)

    class TestImageRAG(ImageRAG):
        def _load_clip_model(self, embedding_model: str):
            return FakeModel()

    app = TestImageRAG()

    records = app._load_images_and_embeddings(
        SimpleNamespace(
            image_dir=str(image_dir),
            image_extensions=[".png"],
            max_items=1,
            batch_size=8,
            embedding_model="clip-test",
        )
    )

    assert len(records) == 1
    assert records[0]["id"] == records[0]["metadata"]["id"]
    assert records[0]["metadata"]["embedding_model"] == "clip-test"
    assert records[0]["embedding"].dtype == np.float32


def test_load_images_rejects_invalid_batch_size(tmp_path):
    image_dir = tmp_path / "images"
    _write_image(image_dir / "first.png")
    app = ImageRAG()

    with pytest.raises(ValueError, match="batch-size"):
        app._load_images_and_embeddings(
            SimpleNamespace(
                image_dir=str(image_dir),
                image_extensions=[".png"],
                max_items=-1,
                batch_size=0,
                embedding_model="clip-test",
            )
        )


def test_build_index_uses_stable_ids_and_precomputed_arrays(tmp_path, monkeypatch):
    calls = {}

    class FakeBuilder:
        def __init__(self, **kwargs):
            calls["init"] = kwargs
            self.texts = []

        def add_text(self, text, metadata=None):
            self.texts.append((text, metadata))

        def build_index_from_arrays(self, index_path, ids, embeddings):
            calls["build"] = {
                "index_path": index_path,
                "ids": ids,
                "embeddings": embeddings,
                "texts": self.texts,
            }

    monkeypatch.setattr(
        image_rag, "register_project_directory", lambda path: calls.setdefault("registered", path)
    )
    monkeypatch.setattr("leann.api.LeannBuilder", FakeBuilder)

    app = ImageRAG()
    app._image_data = [
        {
            "id": "image-alpha",
            "text": "Image: alpha.png",
            "metadata": {"id": "image-alpha", "relative_path": "alpha.png"},
            "embedding": np.array([1.0, 0.0], dtype=np.float32),
        }
    ]

    index_path = asyncio.run(
        app.build_index(
            SimpleNamespace(
                index_dir=str(tmp_path / "index"),
                backend_name="hnsw",
                graph_degree=16,
                build_complexity=32,
                no_compact=False,
                embedding_model="clip-test",
            ),
            cast(list[dict[str, Any]], ["Image: alpha.png"]),
        ),
    )

    assert index_path.endswith("image_index.leann")
    assert calls["init"]["embedding_model"] == "clip-test"
    assert calls["init"]["embedding_mode"] == "sentence-transformers"
    assert calls["init"]["is_recompute"] is False
    assert calls["init"]["distance_metric"] == "cosine"
    assert calls["build"]["ids"] == ["image-alpha"]
    assert calls["build"]["embeddings"].shape == (1, 2)
    assert calls["build"]["texts"] == [
        ("Image: alpha.png", {"id": "image-alpha", "relative_path": "alpha.png"})
    ]


def test_single_query_disables_recompute_for_clip_index(monkeypatch):
    calls = {}

    class FakeChat:
        def __init__(self, index_path, **kwargs):
            calls["init"] = {"index_path": index_path, **kwargs}

        def ask(self, query, **kwargs):
            calls["ask"] = {"query": query, **kwargs}
            return "found image"

    monkeypatch.setattr(image_rag, "LeannChat", FakeChat)

    app = ImageRAG()
    args = SimpleNamespace(
        llm="simulated",
        llm_model=None,
        llm_host=None,
        llm_api_key=None,
        llm_api_base=None,
        search_complexity=64,
        top_k=3,
        thinking_budget=None,
    )

    asyncio.run(app.run_single_query(args, "image_index.leann", "red car"))

    assert calls["init"]["index_path"] == "image_index.leann"
    assert calls["ask"]["query"] == "red car"
    assert calls["ask"]["top_k"] == 3
    assert calls["ask"]["complexity"] == 64
    assert calls["ask"]["recompute_embeddings"] is False
