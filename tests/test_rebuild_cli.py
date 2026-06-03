import asyncio
import json
import pickle
from pathlib import Path

import pytest
from leann.cli import LeannCLI


def test_full_rebuild_failure_preserves_existing_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    docs_file = docs_root / "only.md"
    docs_file.write_text("hello", encoding="utf-8")

    cli = LeannCLI()
    index_dir = tmp_path / ".leann" / "indexes" / "sample"
    index_dir.mkdir(parents=True)
    meta_path = index_dir / "documents.leann.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "backend_name": "hnsw",
                "embedding_model": "facebook/contriever",
                "embedding_mode": "sentence-transformers",
                "backend_kwargs": {"is_compact": False, "is_recompute": True},
                "passage_id_scheme": "sequential",
            }
        ),
        encoding="utf-8",
    )
    passages_file = index_dir / "documents.leann.passages.jsonl"
    passages_file.write_text(
        json.dumps({"id": "old", "text": "old text", "metadata": {}}) + "\n",
        encoding="utf-8",
    )
    offset_file = index_dir / "documents.leann.passages.idx"
    with open(offset_file, "wb") as f:
        pickle.dump({"old": 0}, f)
    original_artifacts = {
        path: path.read_bytes() for path in (meta_path, passages_file, offset_file)
    }
    args = cli.create_parser().parse_args(
        [
            "build",
            "sample",
            "--docs",
            str(docs_file),
            "--backend-name",
            "hnsw",
            "--force",
        ]
    )
    built_paths: list[str] = []

    class FakeSynchronizer:
        def create_snapshot(self):
            pass

    class FailingBuilder:
        def __init__(self, **_kwargs):
            pass

        def add_text(self, _text, metadata=None):
            pass

        def build_index(self, index_path):
            built_paths.append(index_path)
            target_dir = Path(index_path).parent
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "documents.leann.passages.jsonl").write_text(
                json.dumps({"id": "new", "text": "new text", "metadata": {}}) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError("simulated build failure")

    monkeypatch.setattr(cli, "_build_synchronizers", lambda *_args, **_kwargs: [FakeSynchronizer()])
    monkeypatch.setattr(
        cli,
        "load_documents",
        lambda *_args, **_kwargs: [{"text": "rebuilt", "metadata": {"file_path": str(docs_file)}}],
    )
    monkeypatch.setattr(cli, "register_project_dir", lambda: None)
    monkeypatch.setattr("leann.cli.LeannBuilder", FailingBuilder)

    with pytest.raises(RuntimeError, match="simulated build failure"):
        asyncio.run(cli.build_index(args))

    assert built_paths
    assert built_paths[0] != str(index_dir / "documents.leann")
    assert {
        path: path.read_bytes() for path in (meta_path, passages_file, offset_file)
    } == original_artifacts
    assert not list(index_dir.parent.glob(".sample.rebuild-*"))
