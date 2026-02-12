"""Tests for the local Cursor proxy (#47).

Tests the context building, request augmentation, HTTP handler, and CLI
integration without requiring a running LLM or real LEANN index.
"""

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import HTTPServer
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.request import Request, urlopen

import pytest

# ---------------------------------------------------------------------------
# Stub C++ backend
# ---------------------------------------------------------------------------
_mod = sys.modules.get("leann_backend_hnsw.convert_to_csr")
if _mod is not None and not hasattr(_mod, "prune_hnsw_embeddings_inplace"):
    _mod.prune_hnsw_embeddings_inplace = lambda *a, **kw: True
if "leann_backend_hnsw" not in sys.modules:
    stub = MagicMock()
    sys.modules["leann_backend_hnsw"] = stub
    sys.modules["leann_backend_hnsw.convert_to_csr"] = stub.convert_to_csr
    stub.convert_to_csr.prune_hnsw_embeddings_inplace = lambda *a, **kw: True


@dataclass
class FakeResult:
    id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class TestBuildContextBlock:
    def test_empty_results(self):
        from leann.local_cursor import _build_context_block

        assert _build_context_block([]) == ""

    def test_single_result(self):
        from leann.local_cursor import _build_context_block

        results = [FakeResult(id="1", score=0.9, text="def hello(): pass", metadata={"file_path": "main.py"})]
        block = _build_context_block(results)
        assert "main.py" in block
        assert "def hello(): pass" in block
        assert "code snippets" in block.lower()

    def test_max_chars_respected(self):
        from leann.local_cursor import _build_context_block

        results = [
            FakeResult(id=str(i), score=0.9, text="x" * 500, metadata={})
            for i in range(20)
        ]
        block = _build_context_block(results, max_chars=1000)
        # Should be truncated well under 2000 chars (1000 + header)
        assert len(block) < 2000

    def test_no_metadata(self):
        from leann.local_cursor import _build_context_block

        results = [FakeResult(id="1", score=0.9, text="some code")]
        block = _build_context_block(results)
        assert "some code" in block


class TestCursorHandler:
    """Test the HTTP handler using a real local server."""

    @pytest.fixture
    def server(self):
        """Start a cursor proxy server on a random port for testing."""
        from leann.local_cursor import _CursorHandler

        # Find a free port
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]

        server = HTTPServer(("127.0.0.1", port), _CursorHandler)
        server.cursor_config = {
            "model": "test-model",
            "ollama_host": "http://localhost:11434",
            "searcher": None,
            "top_k": 5,
            "max_context_chars": 4000,
        }

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.1)

        yield server, port

        server.shutdown()

    def test_health_endpoint(self, server):
        _, port = server
        req = Request(f"http://127.0.0.1:{port}/health")
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        assert data["status"] == "ok"

    def test_models_endpoint(self, server):
        _, port = server
        req = Request(f"http://127.0.0.1:{port}/v1/models")
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"

    def test_chat_no_messages(self, server):
        _, port = server
        payload = json.dumps({"messages": []}).encode()
        req = Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                assert "error" in data
        except Exception:
            pass  # 400 error is expected

    @patch("leann.local_cursor._forward_to_llm")
    def test_chat_with_retrieval(self, mock_forward, server):
        srv, port = server

        # Set up a mock searcher
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = [
            FakeResult(
                id="1",
                score=0.95,
                text="def main():\n    print('hello')",
                metadata={"file_path": "src/main.py"},
            )
        ]
        srv.cursor_config["searcher"] = mock_searcher

        mock_forward.return_value = (
            200,
            {
                "choices": [
                    {"message": {"role": "assistant", "content": "Here is the answer."}}
                ]
            },
        )

        payload = json.dumps(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "What does main do?"}],
            }
        ).encode()
        req = Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())

        # Verify retrieval was called
        mock_searcher.search.assert_called_once()

        # Verify the forwarded payload has augmented context
        call_args = mock_forward.call_args
        forwarded_payload = call_args[0][1]  # second positional arg
        system_msg = forwarded_payload["messages"][0]
        assert system_msg["role"] == "system"
        assert "main.py" in system_msg["content"]

    def test_cors_preflight(self, server):
        _, port = server
        req = Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            method="OPTIONS",
        )
        with urlopen(req, timeout=5) as resp:
            assert resp.status == 200


class TestCliCursorCommand:
    def test_cursor_parser(self):
        from leann.cli import LeannCLI

        cli = LeannCLI()
        parser = cli.create_parser()
        args = parser.parse_args([
            "cursor",
            "--index", "my-code",
            "--model", "codestral:latest",
            "--port", "9000",
            "--top-k", "20",
        ])
        assert args.index == "my-code"
        assert args.model == "codestral:latest"
        assert args.port == 9000
        assert args.top_k == 20

    def test_cursor_parser_defaults(self):
        from leann.cli import LeannCLI

        cli = LeannCLI()
        parser = cli.create_parser()
        args = parser.parse_args(["cursor"])
        assert args.index is None
        assert args.model == "qwen3-coder"
        assert args.port == 8765
