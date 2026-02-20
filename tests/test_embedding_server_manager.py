import json
import os
import time

import pytest
from leann.embedding_server_manager import EmbeddingServerManager


class DummyProcess:
    def __init__(self):
        self.pid = 12345
        self._terminated = False

    def poll(self):
        return 0 if self._terminated else None

    def terminate(self):
        self._terminated = True

    def kill(self):
        self._terminated = True

    def wait(self, timeout=None):
        self._terminated = True
        return 0


@pytest.fixture
def embedding_manager(monkeypatch):
    manager = EmbeddingServerManager("leann_backend_hnsw.hnsw_embedding_server")

    def fake_get_available_port(start_port):
        return start_port

    monkeypatch.setattr(
        "leann.embedding_server_manager._get_available_port",
        fake_get_available_port,
    )

    start_calls = []

    def fake_start_new_server(self, port, model_name, embedding_mode, **kwargs):
        config_signature = kwargs.get("config_signature")
        start_calls.append(config_signature)
        self.server_process = DummyProcess()
        self.server_port = port
        self._server_config = config_signature
        return True, port

    monkeypatch.setattr(
        EmbeddingServerManager,
        "_start_new_server",
        fake_start_new_server,
    )

    # Ensure stop_server doesn't try to operate on real subprocesses
    def fake_stop_server(self):
        self.server_process = None
        self.server_port = None
        self._server_config = None

    monkeypatch.setattr(EmbeddingServerManager, "stop_server", fake_stop_server)

    return manager, start_calls


def _write_meta(meta_path, passages_name, index_name, total):
    meta_path.write_text(
        json.dumps(
            {
                "backend_name": "hnsw",
                "embedding_model": "test-model",
                "embedding_mode": "sentence-transformers",
                "dimensions": 3,
                "backend_kwargs": {},
                "passage_sources": [
                    {
                        "type": "jsonl",
                        "path": passages_name,
                        "index_path": index_name,
                    }
                ],
                "total_passages": total,
            }
        ),
        encoding="utf-8",
    )


def test_server_restarts_when_metadata_changes(tmp_path, embedding_manager):
    manager, start_calls = embedding_manager

    meta_path = tmp_path / "example.meta.json"
    passages_path = tmp_path / "example.passages.jsonl"
    index_path = tmp_path / "example.passages.idx"

    passages_path.write_text("first\n", encoding="utf-8")
    index_path.write_bytes(b"index")
    _write_meta(meta_path, passages_path.name, index_path.name, total=1)

    # Initial start populates signature
    ok, port = manager.start_server(
        port=6000,
        model_name="test-model",
        passages_file=str(meta_path),
        use_daemon=False,
    )
    assert ok
    assert port == 6000
    assert len(start_calls) == 1

    initial_signature = start_calls[0]["passages_signature"]

    # No metadata change => reuse existing server
    ok, port_again = manager.start_server(
        port=6000,
        model_name="test-model",
        passages_file=str(meta_path),
        use_daemon=False,
    )
    assert ok
    assert port_again == 6000
    assert len(start_calls) == 1

    # Modify passage data and metadata to force signature change
    time.sleep(0.01)  # Ensure filesystem timestamps move forward
    passages_path.write_text("second\n", encoding="utf-8")
    _write_meta(meta_path, passages_path.name, index_path.name, total=2)

    ok, port_third = manager.start_server(
        port=6000,
        model_name="test-model",
        passages_file=str(meta_path),
        use_daemon=False,
    )
    assert ok
    assert port_third == 6000
    assert len(start_calls) == 2

    updated_signature = start_calls[1]["passages_signature"]
    assert updated_signature != initial_signature


def test_list_daemons_ignores_stale_records(tmp_path, monkeypatch):
    registry_dir = tmp_path / "servers"
    registry_dir.mkdir()
    monkeypatch.setattr(EmbeddingServerManager, "_registry_dir", staticmethod(lambda: registry_dir))

    stale = registry_dir / "stale.json"
    stale.write_text(
        json.dumps(
            {
                "pid": 999999,
                "port": 65531,
                "backend_module_name": "leann_backend_hnsw.hnsw_embedding_server",
                "config_signature": {"passages_file": "/tmp/a.meta.json"},
            }
        ),
        encoding="utf-8",
    )

    records = EmbeddingServerManager.list_daemons()
    assert records == []
    assert not stale.exists()


def test_stop_daemons_filters_by_backend_and_passages(tmp_path, monkeypatch):
    registry_dir = tmp_path / "servers"
    registry_dir.mkdir()
    monkeypatch.setattr(EmbeddingServerManager, "_registry_dir", staticmethod(lambda: registry_dir))

    meta_path = (tmp_path / "x.meta.json").resolve()
    record = registry_dir / "daemon.json"
    record.write_text(
        json.dumps(
            {
                "pid": 12345,
                "port": 6001,
                "backend_module_name": "leann_backend_hnsw.hnsw_embedding_server",
                "config_signature": {"passages_file": str(meta_path)},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        EmbeddingServerManager,
        "list_daemons",
        classmethod(
            lambda cls: [
                {
                    "pid": 12345,
                    "port": 6001,
                    "backend_module_name": "leann_backend_hnsw.hnsw_embedding_server",
                    "config_signature": {"passages_file": str(meta_path)},
                    "record_path": str(record),
                }
            ]
        ),
    )

    killed: list[tuple[int, int]] = []

    def fake_kill(pid: int, sig: int):
        killed.append((pid, sig))

    monkeypatch.setattr(os, "kill", fake_kill)

    stopped = EmbeddingServerManager.stop_daemons(
        backend_module_name="leann_backend_hnsw.hnsw_embedding_server",
        passages_file=str(meta_path),
    )
    assert stopped == 1
    assert killed == [(12345, 15)]
    assert not record.exists()
