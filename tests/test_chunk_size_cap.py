"""Chunk sizes are capped (never raised) to fit the embedding model's sequence limit."""

import json

import pytest
from leann import embedding_compute
from leann.chunking_utils import cap_chunk_size, create_text_chunks
from llama_index.core import Document


@pytest.fixture(autouse=True)
def clear_token_limit_cache():
    embedding_compute._token_limit_cache.clear()
    yield
    embedding_compute._token_limit_cache.clear()


@pytest.mark.parametrize(
    "chunk_size,limit,expected",
    [(256, 256, 179), (512, 256, 179), (256, 512, 256), (512, 512, 358), (256, 8192, 256)],
)
def test_cap_chunk_size_only_shrinks(chunk_size, limit, expected):
    assert cap_chunk_size(chunk_size, limit) == expected


def test_large_model_limit_does_not_grow_chunks():
    docs = [Document(text="LEANN recomputes embeddings on demand. " * 400)]
    plain = create_text_chunks(docs, chunk_size=256, chunk_overlap=64)
    with_limit = create_text_chunks(
        docs, chunk_size=256, chunk_overlap=64, max_tokens_per_chunk=2048
    )
    assert len(with_limit) == len(plain)


def test_small_model_limit_shrinks_chunks():
    docs = [Document(text="LEANN recomputes embeddings on demand. " * 400)]
    plain = create_text_chunks(docs, chunk_size=512, chunk_overlap=50)
    capped = create_text_chunks(docs, chunk_size=512, chunk_overlap=50, max_tokens_per_chunk=256)
    assert len(capped) > len(plain)


def test_huggingface_seq_limit_discovery(monkeypatch, tmp_path):
    config = tmp_path / "sentence_bert_config.json"
    config.write_text(json.dumps({"max_seq_length": 256}))
    calls = []

    def fake_download(repo_id, filename):
        calls.append((repo_id, filename))
        return str(config)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    model = "sentence-transformers/all-MiniLM-L6-v2"
    assert embedding_compute.get_model_token_limit(model) == 256
    assert calls == [(model, "sentence_bert_config.json")]
    # Remote servers and bare names never hit the Hugging Face Hub
    embedding_compute.get_model_token_limit(model, base_url="http://localhost:8000/v1")
    embedding_compute.get_model_token_limit("unknown-model-xyz")
    assert len(calls) == 1


def test_huggingface_discovery_failure_uses_default(monkeypatch):
    def fail(repo_id, filename):
        raise OSError("offline")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fail)
    assert embedding_compute.get_model_token_limit("org/no-such-model", default=2048) == 2048
