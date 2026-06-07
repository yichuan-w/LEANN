import hashlib
import json
import pickle
from contextlib import contextmanager
from types import SimpleNamespace

from leann.api import (
    DEFAULT_PASSAGE_ID_SCHEME,
    PASSAGE_ID_SCHEME_CONTENT_HASH,
    PASSAGE_ID_SCHEME_SEQUENTIAL,
    Fts5BM25Index,
    LeannBuilder,
)
from leann.cli import LeannCLI


def _content_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def test_builder_defaults_to_content_hash_passage_ids():
    builder = LeannBuilder(backend_name="hnsw")

    builder.add_text("same text", metadata={"source": "a.txt"})

    assert DEFAULT_PASSAGE_ID_SCHEME == PASSAGE_ID_SCHEME_CONTENT_HASH
    assert builder.chunks[0]["id"] == _content_id("same text")


def test_builder_content_hash_passage_ids_are_content_stable():
    builder = LeannBuilder(backend_name="hnsw", passage_id_scheme="content-hash")

    builder.add_text("same text", metadata={"source": "a.txt"})
    builder.add_text("same text", metadata={"source": "b.txt"})
    builder.add_text("different text", metadata={"source": "c.txt"})

    same_id = _content_id("same text")
    assert builder.chunks[0]["id"] == same_id
    assert builder.chunks[1]["id"] == same_id
    assert builder.chunks[2]["id"] == _content_id("different text")


def test_update_passage_ids_follow_existing_sequential_index_scheme():
    builder = LeannBuilder(backend_name="hnsw")
    builder.add_text("new text", metadata={"source": "new.txt"})
    assert builder.chunks[0]["id"] == _content_id("new text")

    builder._assign_passage_ids_for_existing_scheme(
        builder.chunks,
        PASSAGE_ID_SCHEME_SEQUENTIAL,
        start_index=2,
    )

    assert builder.chunks[0]["id"] == "2"
    assert builder.chunks[0]["metadata"]["id"] == "2"


def test_update_passage_ids_preserve_explicit_api_ids():
    builder = LeannBuilder(backend_name="hnsw")
    builder.add_text("new text", metadata={"id": "caller-id"})

    builder._assign_passage_ids_for_existing_scheme(
        builder.chunks,
        PASSAGE_ID_SCHEME_CONTENT_HASH,
        start_index=2,
    )

    assert builder.chunks[0]["id"] == "caller-id"
    assert builder.chunks[0]["metadata"]["id"] == "caller-id"


def test_existing_index_id_scheme_treats_legacy_meta_as_sequential(tmp_path):
    cli = LeannCLI()
    index_path = tmp_path / "legacy.leann"

    assert cli._existing_index_id_scheme(str(index_path)) is None

    meta_path = tmp_path / "legacy.leann.meta.json"
    meta_path.write_text(json.dumps({"version": "1.0", "backend_name": "hnsw"}), encoding="utf-8")

    assert cli._existing_index_id_scheme(str(index_path)) == "sequential"

    meta_path.write_text(
        json.dumps({"version": "1.1", "backend_name": "hnsw", "passage_id_scheme": "content-hash"}),
        encoding="utf-8",
    )

    assert cli._existing_index_id_scheme(str(index_path)) == "content-hash"


def test_migrate_ids_rewrites_passages_offsets_idmap_meta_and_bm25(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    index_dir = tmp_path / ".leann" / "indexes" / "sample"
    index_dir.mkdir(parents=True)

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

    bm25_db = index_dir / "documents.leann.bm25.sqlite"
    bm25 = Fts5BM25Index(str(bm25_db))
    bm25.fit(passages)
    bm25.close()

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
                "bm25_backend": "fts5",
                "bm25_db": "documents.leann.bm25.sqlite",
                "passage_id_scheme": "sequential",
            }
        ),
        encoding="utf-8",
    )

    lock_paths = []

    @contextmanager
    def fake_index_write_lock(path):
        lock_paths.append(path)
        yield

    monkeypatch.setattr("leann.cli.index_write_lock", fake_index_write_lock)

    cli = LeannCLI()
    cli.migrate_ids(SimpleNamespace(index_name="sample", dry_run=False, yes=True))

    assert lock_paths == [index_dir]

    expected_ids = [_content_id("alpha beta"), _content_id("gamma delta")]
    with open(passages_file, encoding="utf-8") as f:
        migrated_passages = [json.loads(line) for line in f if line.strip()]
    assert [p["id"] for p in migrated_passages] == expected_ids

    with open(offset_file, "rb") as f:
        migrated_offsets = pickle.load(f)
    assert set(migrated_offsets) == set(expected_ids)
    assert idmap_file.read_text(encoding="utf-8").splitlines() == expected_ids

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["version"] == "1.1"
    assert meta["passage_id_scheme"] == "content-hash"
    assert meta["bm25_backend"] == "fts5"
    assert meta["bm25_db"] == "documents.leann.bm25.sqlite"

    migrated_bm25 = Fts5BM25Index(str(bm25_db))
    try:
        bm25_results = migrated_bm25.search("alpha", top_k=1)
    finally:
        migrated_bm25.close()
    assert [result.id for result in bm25_results] == [expected_ids[0]]


def test_migrate_ids_recomputes_passage_plan_under_write_lock(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    index_dir = tmp_path / ".leann" / "indexes" / "sample"
    index_dir.mkdir(parents=True)

    passages_file = index_dir / "documents.leann.passages.jsonl"
    passages_file.write_text(
        json.dumps({"id": "0", "text": "pre lock text", "metadata": {}}) + "\n",
        encoding="utf-8",
    )

    offset_file = index_dir / "documents.leann.passages.idx"
    with open(offset_file, "wb") as f:
        pickle.dump({"0": 0}, f)

    idmap_file = index_dir / "documents.ids.txt"
    idmap_file.write_text("0\n", encoding="utf-8")

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

    @contextmanager
    def fake_index_write_lock(path):
        assert path == index_dir
        passages_file.write_text(
            json.dumps({"id": "0", "text": "post lock text", "metadata": {}}) + "\n",
            encoding="utf-8",
        )
        yield

    monkeypatch.setattr("leann.cli.index_write_lock", fake_index_write_lock)

    cli = LeannCLI()
    cli.migrate_ids(SimpleNamespace(index_name="sample", dry_run=False, yes=True))

    expected_id = _content_id("post lock text")
    with open(passages_file, encoding="utf-8") as f:
        migrated_passages = [json.loads(line) for line in f if line.strip()]
    assert [p["id"] for p in migrated_passages] == [expected_id]
    assert idmap_file.read_text(encoding="utf-8").splitlines() == [expected_id]
