"""Tests for the persistent embedding daemon (#166).

These tests validate the daemon state management, lifecycle, and integration
with EmbeddingServerManager without requiring a real embedding model or the
HNSW C++ backend.
"""

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Patch the heavy C++ import that leann.api triggers at import time.
# This lets us test the pure-Python daemon module without compiling
# leann_backend_hnsw from source.
# ---------------------------------------------------------------------------
def _ensure_importable():
    """Ensure leann.embedding_daemon can be imported.

    If leann_backend_hnsw.convert_to_csr is missing the function referenced by
    api.py, we inject a stub so the rest of the package can load.
    """
    try:
        from leann.embedding_daemon import read_daemon_state  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        # Stub out the missing symbol so ``leann.api`` can be imported.
        _mod = sys.modules.get("leann_backend_hnsw.convert_to_csr")
        if _mod is not None and not hasattr(_mod, "prune_hnsw_embeddings_inplace"):
            _mod.prune_hnsw_embeddings_inplace = lambda *a, **kw: True

        # Also handle the case where the entire module is missing
        if "leann_backend_hnsw" not in sys.modules:
            stub = MagicMock()
            sys.modules["leann_backend_hnsw"] = stub
            sys.modules["leann_backend_hnsw.convert_to_csr"] = stub.convert_to_csr
            stub.convert_to_csr.prune_hnsw_embeddings_inplace = lambda *a, **kw: True


_ensure_importable()

from leann.embedding_daemon import (
    _STALE_THRESHOLD,
    _remove_state_file,
    _write_state,
    daemon_status,
    read_daemon_state,
    stop_daemon,
)


@pytest.fixture(autouse=True)
def isolated_state_dir(tmp_path, monkeypatch):
    """Redirect the daemon state file to a temp directory for test isolation."""
    state_file = tmp_path / "daemon.json"
    monkeypatch.setenv("LEANN_DAEMON_STATE", str(state_file))
    return state_file


class TestDaemonStateManagement:
    """Test the daemon state file read/write/cleanup logic."""

    def test_read_daemon_state_no_file(self):
        assert read_daemon_state() is None

    def test_read_daemon_state_invalid_json(self, isolated_state_dir):
        isolated_state_dir.write_text("not json", encoding="utf-8")
        assert read_daemon_state() is None

    def test_read_daemon_state_missing_fields(self, isolated_state_dir):
        isolated_state_dir.write_text(json.dumps({"pid": 1}), encoding="utf-8")
        assert read_daemon_state() is None

    def test_read_daemon_state_dead_pid(self, isolated_state_dir):
        state = {
            "pid": 999999999,  # Almost certainly not a real PID
            "port": 5557,
            "heartbeat": time.time(),
        }
        isolated_state_dir.write_text(json.dumps(state), encoding="utf-8")
        result = read_daemon_state()
        # PID doesn't exist, should return None and clean up
        assert result is None
        assert not isolated_state_dir.exists()

    def test_read_daemon_state_stale_heartbeat(self, isolated_state_dir):
        state = {
            "pid": os.getpid(),  # Our own PID, definitely alive
            "port": 5557,
            "heartbeat": time.time() - _STALE_THRESHOLD - 10,
        }
        isolated_state_dir.write_text(json.dumps(state), encoding="utf-8")
        result = read_daemon_state()
        assert result is None

    @patch("leann.embedding_daemon._is_port_open", return_value=True)
    def test_read_daemon_state_healthy(self, mock_port, isolated_state_dir):
        state = {
            "pid": os.getpid(),
            "port": 5557,
            "model_name": "facebook/contriever",
            "embedding_mode": "sentence-transformers",
            "heartbeat": time.time(),
        }
        isolated_state_dir.write_text(json.dumps(state), encoding="utf-8")
        result = read_daemon_state()
        assert result is not None
        assert result["port"] == 5557
        assert result["model_name"] == "facebook/contriever"

    def test_write_and_read_state(self, isolated_state_dir):
        state = {"pid": os.getpid(), "port": 9999, "heartbeat": time.time()}
        _write_state(state)

        loaded = json.loads(isolated_state_dir.read_text(encoding="utf-8"))
        assert loaded["port"] == 9999

    def test_remove_state_file(self, isolated_state_dir):
        _write_state({"pid": 1, "port": 1})
        assert isolated_state_dir.exists()
        _remove_state_file()
        assert not isolated_state_dir.exists()


class TestStopDaemon:
    """Test daemon shutdown logic."""

    def test_stop_daemon_no_running(self):
        assert stop_daemon() is False

    @patch("leann.embedding_daemon._is_port_open", return_value=True)
    @patch("leann.embedding_daemon._is_pid_alive", return_value=True)
    @patch("os.kill")
    def test_stop_daemon_sends_sigterm(
        self, mock_kill, mock_alive, mock_port, isolated_state_dir
    ):
        state = {
            "pid": 12345,
            "port": 5557,
            "heartbeat": time.time(),
        }
        isolated_state_dir.write_text(json.dumps(state), encoding="utf-8")

        # After SIGTERM, pretend process dies
        mock_alive.side_effect = [True, True, False]

        result = stop_daemon()
        assert result is True
        mock_kill.assert_called()
        assert not isolated_state_dir.exists()


class TestDaemonStatus:
    """Test status reporting."""

    def test_status_no_daemon(self):
        assert daemon_status() is None

    @patch("leann.embedding_daemon._is_port_open", return_value=True)
    def test_status_healthy_daemon(self, mock_port, isolated_state_dir):
        state = {
            "pid": os.getpid(),
            "port": 5557,
            "model_name": "test-model",
            "embedding_mode": "sentence-transformers",
            "heartbeat": time.time(),
            "started_at": time.time() - 100,
        }
        isolated_state_dir.write_text(json.dumps(state), encoding="utf-8")

        result = daemon_status()
        assert result is not None
        assert result["model_name"] == "test-model"


class TestEmbeddingServerManagerDaemonIntegration:
    """Test that EmbeddingServerManager detects and uses a running daemon."""

    @patch("leann.embedding_daemon._is_port_open", return_value=True)
    def test_manager_uses_daemon(self, mock_port, isolated_state_dir):
        from leann.embedding_server_manager import EmbeddingServerManager

        # Simulate a running daemon
        _write_state(
            {
                "pid": os.getpid(),
                "port": 6789,
                "model_name": "facebook/contriever",
                "embedding_mode": "sentence-transformers",
                "heartbeat": time.time(),
            }
        )

        manager = EmbeddingServerManager(
            backend_module_name="leann_backend_hnsw.hnsw_embedding_server"
        )

        state = manager._try_daemon("facebook/contriever", "sentence-transformers")
        assert state is not None
        assert state["port"] == 6789

    @patch("leann.embedding_daemon._is_port_open", return_value=True)
    def test_manager_ignores_mismatched_daemon(self, mock_port, isolated_state_dir):
        from leann.embedding_server_manager import EmbeddingServerManager

        # Daemon serves a different model
        _write_state(
            {
                "pid": os.getpid(),
                "port": 6789,
                "model_name": "BAAI/bge-large-en-v1.5",
                "embedding_mode": "sentence-transformers",
                "heartbeat": time.time(),
            }
        )

        manager = EmbeddingServerManager(
            backend_module_name="leann_backend_hnsw.hnsw_embedding_server"
        )

        state = manager._try_daemon("facebook/contriever", "sentence-transformers")
        assert state is None


class TestCliServeCommand:
    """Test the CLI serve command parsing and dispatch."""

    def test_serve_status_no_daemon(self, capsys):
        from leann.cli import LeannCLI

        cli = LeannCLI()
        parser = cli.create_parser()
        args = parser.parse_args(["serve", "--status"])
        cli.handle_serve(args)

        captured = capsys.readouterr()
        assert "No embedding daemon is running" in captured.out

    @patch("leann.embedding_daemon._is_port_open", return_value=True)
    def test_serve_status_running(self, mock_port, isolated_state_dir, capsys):
        from leann.cli import LeannCLI

        _write_state(
            {
                "pid": os.getpid(),
                "port": 5557,
                "model_name": "facebook/contriever",
                "embedding_mode": "sentence-transformers",
                "heartbeat": time.time(),
                "started_at": time.time(),
            }
        )

        cli = LeannCLI()
        parser = cli.create_parser()
        args = parser.parse_args(["serve", "--status"])
        cli.handle_serve(args)

        captured = capsys.readouterr()
        assert "Embedding daemon is running" in captured.out
        assert "5557" in captured.out

    def test_serve_stop_no_daemon(self, capsys):
        from leann.cli import LeannCLI

        cli = LeannCLI()
        parser = cli.create_parser()
        args = parser.parse_args(["serve", "--stop"])
        cli.handle_serve(args)

        captured = capsys.readouterr()
        assert "No embedding daemon is running" in captured.out

    def test_serve_parser_has_expected_args(self):
        from leann.cli import LeannCLI

        cli = LeannCLI()
        parser = cli.create_parser()
        args = parser.parse_args([
            "serve",
            "--embedding-model", "BAAI/bge-small-en-v1.5",
            "--embedding-mode", "sentence-transformers",
            "--port", "6000",
            "--foreground",
        ])
        assert args.embedding_model == "BAAI/bge-small-en-v1.5"
        assert args.port == 6000
        assert args.foreground is True
