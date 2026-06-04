import json
import pickle

import pytest
from leann import api
from leann.api import LeannSearcher, RegexNgramIndex


class _FakeBackendFactory:
    def searcher(self, *args, **kwargs):
        return _FakeBackendSearcher()


class _FakeBackendSearcher:
    def compute_query_embedding(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("regex search should not compute embeddings")

    def search(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("regex search should not call vector backend")


def _write_test_index(tmp_path, passages, *, build_regex=True):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    index_path = index_dir / "documents.leann"
    passages_file = index_dir / "documents.leann.passages.jsonl"
    offset_file = index_dir / "documents.leann.passages.idx"
    regex_db = index_dir / "documents.leann.regex.sqlite"

    offset_map = {}
    with open(passages_file, "w", encoding="utf-8") as f:
        for passage in passages:
            offset_map[str(passage["id"])] = f.tell()
            json.dump(passage, f)
            f.write("\n")

    with open(offset_file, "wb") as f:
        pickle.dump(offset_map, f)

    meta = {
        "version": "1.0",
        "backend_name": "regex-test",
        "embedding_model": "fake",
        "dimensions": 1,
        "backend_kwargs": {},
        "embedding_mode": "sentence-transformers",
        "passage_sources": [
            {
                "type": "jsonl",
                "path": passages_file.name,
                "index_path": offset_file.name,
            }
        ],
    }
    if build_regex:
        index = RegexNgramIndex(str(regex_db))
        index.fit(passages)
        index.close()
        meta["regex_backend"] = "sqlite_trigram"
        meta["regex_db"] = regex_db.name

    with open(f"{index_path}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    return index_path, regex_db


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch):
    monkeypatch.setitem(api.BACKEND_REGISTRY, "regex-test", _FakeBackendFactory())


def test_required_ngrams_are_conservative_for_regex_constructs():
    assert {"cla", "ret"}.issubset(
        RegexNgramIndex.required_ngrams(r"class .*Retriever", is_regex=True)
    )
    assert RegexNgramIndex.required_ngrams(r"foo|bar", is_regex=True) == set()
    assert RegexNgramIndex.required_ngrams(r"[abc]def", is_regex=True) == set()
    assert RegexNgramIndex.required_ngrams(r"\bfoo", is_regex=True) == {"foo"}


def test_regex_search_uses_indexed_candidates(tmp_path):
    passages = [
        {
            "id": "class",
            "text": "class VectorRetriever:\n    pass",
            "metadata": {"kind": "class"},
        },
        {
            "id": "function",
            "text": "def build_retriever():\n    return None",
            "metadata": {"kind": "function"},
        },
        {
            "id": "notes",
            "text": "retriever = make_component()",
            "metadata": {"kind": "notes"},
        },
    ]
    index_path, _regex_db = _write_test_index(tmp_path, passages)

    searcher = LeannSearcher(str(index_path), enable_warmup=False)

    results = searcher.search(r"class .*Retriever", use_regex=True, top_k=5)
    assert [result.id for result in results] == ["class"]

    filtered = searcher.search(
        "Retriever",
        use_regex=True,
        top_k=5,
        metadata_filters={"kind": {"==": "function"}},
    )
    assert filtered == []


def test_use_grep_is_case_insensitive_literal_search(tmp_path):
    passages = [
        {"id": "one", "text": "class VectorRetriever:\n    pass", "metadata": {}},
        {"id": "two", "text": "class OtherThing:\n    pass", "metadata": {}},
    ]
    index_path, _regex_db = _write_test_index(tmp_path, passages)

    searcher = LeannSearcher(str(index_path), enable_warmup=False)

    results = searcher.search("vectorretriever", use_grep=True, top_k=5)
    assert [result.id for result in results] == ["one"]


def test_invalid_regex_raises_value_error(tmp_path):
    index_path, _regex_db = _write_test_index(
        tmp_path, [{"id": "one", "text": "anything", "metadata": {}}]
    )
    searcher = LeannSearcher(str(index_path), enable_warmup=False)

    with pytest.raises(ValueError, match="Invalid regex query"):
        searcher.search("[unterminated", use_regex=True)


def test_old_index_builds_regex_sidecar_on_demand(tmp_path):
    passages = [{"id": "one", "text": "def exact_match(): pass", "metadata": {}}]
    index_path, regex_db = _write_test_index(tmp_path, passages, build_regex=False)

    assert not regex_db.exists()
    searcher = LeannSearcher(str(index_path), enable_warmup=False)
    results = searcher.search(r"def exact_match", use_regex=True)

    assert [result.id for result in results] == ["one"]
    assert regex_db.exists()
    with open(f"{index_path}.meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["regex_backend"] == "sqlite_trigram"
    assert meta["regex_db"] == regex_db.name
