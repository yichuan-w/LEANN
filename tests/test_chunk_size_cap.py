"""Chunk sizes are capped (never raised) to fit the embedding model's sequence limit."""

import json

import pytest
from leann import chunking_utils, embedding_compute
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


@pytest.mark.parametrize(
    "chunk_size,overlap,limit,expected",
    [
        (256, 64, None, (256, 64)),
        (256, 64, 2048, (256, 64)),  # large limit never grows the chunk
        (512, 50, 256, (179, 50)),
        (256, 128, 256, (179, 89)),  # overlap capped to half the new size
    ],
)
def test_create_text_chunks_caps_splitter_size(monkeypatch, chunk_size, overlap, limit, expected):
    used = []

    class RecordingSplitter:
        def __init__(self, chunk_size, chunk_overlap, **kwargs):
            used.append((chunk_size, chunk_overlap))

        def get_nodes_from_documents(self, docs):
            return []

    monkeypatch.setattr(chunking_utils, "SentenceSplitter", RecordingSplitter)
    docs = [Document(text="LEANN recomputes embeddings on demand.")]
    create_text_chunks(
        docs, chunk_size=chunk_size, chunk_overlap=overlap, max_tokens_per_chunk=limit
    )
    assert used == [expected]


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
