import hashlib
import json
from types import SimpleNamespace

from leann.api import LeannBuilder
from leann.cli import LeannCLI


def _content_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


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


def test_incremental_content_hash_add_only_does_not_preassign_path_ids(tmp_path):
    cli = LeannCLI()
    added_chunks = []

    class FakeBuilder:
        passage_id_scheme = "content-hash"

        def add_text(self, text, metadata=None):
            added_chunks.append((text, dict(metadata or {})))

        def update_index(self, index_path):
            assert index_path == str(tmp_path / "documents.leann")

    cli._make_incremental_builder = lambda _args: FakeBuilder()
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
