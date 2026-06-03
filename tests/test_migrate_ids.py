import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
from leann.cli import LeannCLI


def test_migrate_ids_rolls_back_when_publish_fails(tmp_path, monkeypatch):
    index_dir = tmp_path / ".leann" / "indexes" / "sample"
    index_dir.mkdir(parents=True)
    index_path = index_dir / "documents.leann"

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
    original_bytes = {
        path: path.read_bytes() for path in (passages_file, offset_file, idmap_file, meta_path)
    }
    real_replace = os.replace

    def fail_on_meta_publish(src, dst):
        if Path(dst) == meta_path and str(src).endswith(".migrate"):
            raise OSError("simulated publish failure")
        real_replace(src, dst)

    cli = LeannCLI()
    monkeypatch.setattr(cli, "_resolve_index_path", lambda *_args, **_kwargs: str(index_path))
    monkeypatch.setattr("leann.cli.os.replace", fail_on_meta_publish)

    with pytest.raises(OSError, match="simulated publish failure"):
        cli.migrate_ids(SimpleNamespace(index_name="sample", dry_run=False, yes=True))

    assert {
        path: path.read_bytes() for path in (passages_file, offset_file, idmap_file, meta_path)
    } == original_bytes
    assert not list(index_dir.glob("*.migrate"))
    assert not list(index_dir.glob(".*.backup-*"))
