"""Tests for the local Cursor/OpenAI-compatible proxy."""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

if "leann_backend_hnsw" not in sys.modules:
    hnsw_stub = MagicMock()
    sys.modules["leann_backend_hnsw"] = hnsw_stub
    sys.modules["leann_backend_hnsw.convert_to_csr"] = hnsw_stub.convert_to_csr
    hnsw_stub.convert_to_csr.prune_hnsw_embeddings_inplace = lambda *args, **kwargs: True

if "llama_index.core" not in sys.modules:
    llama_index_module = MagicMock()
    llama_core_module = MagicMock()
    llama_node_parser_module = MagicMock()
    llama_core_module.SimpleDirectoryReader = MagicMock()
    llama_node_parser_module.SentenceSplitter = MagicMock()
    sys.modules["llama_index"] = llama_index_module
    sys.modules["llama_index.core"] = llama_core_module
    sys.modules["llama_index.core.node_parser"] = llama_node_parser_module

if "watchfiles" not in sys.modules:
    watchfiles_module = MagicMock()
    watchfiles_module.Change = MagicMock()
    watchfiles_module.awatch = MagicMock()
    sys.modules["watchfiles"] = watchfiles_module


@dataclass
class FakeResult:
    id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@pytest.fixture
def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def cursor_server(free_port):
    from leann.local_cursor import CursorProxyConfig, CursorProxyHandler, CursorProxyServer

    searcher = MagicMock()
    searcher.search.return_value = [
        FakeResult(
            id="1",
            score=0.9,
            text="def main():\n    return 'ok'",
            metadata={"file_path": "src/main.py"},
        )
    ]
    server = CursorProxyServer(("127.0.0.1", free_port), CursorProxyHandler)
    server.cursor_config = CursorProxyConfig(
        model="test-model",
        llm_base_url="http://127.0.0.1:11434",
        searcher=searcher,
        top_k=3,
        complexity=12,
        max_context_chars=4000,
    )

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)

    yield server, free_port

    server.shutdown()
    server.server_close()


def test_build_context_block_includes_sources_and_truncates():
    from leann.local_cursor import build_context_block

    results = [
        FakeResult(id="1", score=0.9, text="x" * 200, metadata={"file_path": "a.py"}),
        FakeResult(id="2", score=0.8, text="y" * 200, metadata={"file_path": "b.py"}),
    ]

    block = build_context_block(results, max_chars=80)

    assert "Relevant local code context" in block
    assert "a.py" in block
    assert "b.py" not in block
    assert len(block) < 220


def test_latest_user_text_accepts_structured_content():
    from leann.local_cursor import latest_user_text

    messages = [
        {"role": "system", "content": "ignore"},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "file://x"}},
                {"type": "text", "text": "Where is main?"},
            ],
        },
    ]

    assert latest_user_text(messages) == "Where is main?"


def test_augment_chat_payload_injects_context_and_overrides_model():
    from leann.local_cursor import CursorProxyConfig, augment_chat_payload

    searcher = MagicMock()
    searcher.search.return_value = [
        FakeResult(
            id="1", score=0.9, text="def helper(): pass", metadata={"file_path": "helper.py"}
        )
    ]
    config = CursorProxyConfig(
        model="local-model",
        llm_base_url="http://127.0.0.1:11434",
        searcher=searcher,
        top_k=4,
        complexity=16,
        recompute_embeddings=False,
    )

    payload = {"model": "client-model", "messages": [{"role": "user", "content": "helper"}]}
    augmented = augment_chat_payload(payload, config)

    assert augmented["model"] == "local-model"
    assert augmented["messages"][0]["role"] == "system"
    assert "helper.py" in augmented["messages"][0]["content"]
    searcher.search.assert_called_once_with(
        "helper", top_k=4, complexity=16, recompute_embeddings=False
    )


def test_health_models_and_cors_headers(cursor_server):
    _, port = cursor_server
    request = Request(
        f"http://127.0.0.1:{port}/v1/models",
        headers={"Origin": "http://localhost:3000"},
    )

    with urlopen(request, timeout=5) as response:
        body = json.loads(response.read())

    assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
    assert body["data"][0]["id"] == "test-model"


def test_disallowed_cors_origin_is_not_echoed(cursor_server):
    _, port = cursor_server
    request = Request(
        f"http://127.0.0.1:{port}/health",
        headers={"Origin": "https://example.com"},
    )

    with urlopen(request, timeout=5) as response:
        json.loads(response.read())

    assert "Access-Control-Allow-Origin" not in response.headers


@patch("leann.local_cursor.forward_json_to_llm")
def test_chat_request_retrieves_and_forwards_augmented_payload(mock_forward, cursor_server):
    server, port = cursor_server
    mock_forward.return_value = (
        200,
        {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
    )
    payload = json.dumps({"messages": [{"role": "user", "content": "What is main?"}]}).encode()
    request = Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=5) as response:
        body = json.loads(response.read())

    assert body["choices"][0]["message"]["content"] == "ok"
    server.cursor_config.searcher.search.assert_called_once()
    forwarded_payload = mock_forward.call_args.args[1]
    assert forwarded_payload["model"] == "test-model"
    assert "src/main.py" in forwarded_payload["messages"][0]["content"]


@patch("leann.local_cursor.forward_stream_to_llm")
def test_streaming_chat_request_is_passed_through(mock_forward, cursor_server):
    _, port = cursor_server
    mock_forward.return_value = (200, "text/event-stream", [b"data: one\n\n", b"data: [DONE]\n\n"])
    payload = json.dumps(
        {"stream": True, "messages": [{"role": "user", "content": "stream it"}]}
    ).encode()
    request = Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=5) as response:
        body = response.read()

    assert response.headers["Content-Type"] == "text/event-stream"
    assert b"data: one" in body


def test_invalid_chat_payload_returns_400(cursor_server):
    _, port = cursor_server
    request = Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"messages": "not-a-list"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with pytest.raises(HTTPError) as exc_info:
        urlopen(request, timeout=5)

    assert exc_info.value.code == 400


def test_cursor_parser_defaults_and_flags():
    from leann.cli import LeannCLI

    parser = LeannCLI().create_parser()
    args = parser.parse_args(
        [
            "cursor",
            "--index",
            "code",
            "--model",
            "codestral",
            "--port",
            "9000",
            "--bind-host",
            "127.0.0.1",
            "--llm-base-url",
            "http://127.0.0.1:1234",
            "--top-k",
            "5",
            "--complexity",
            "20",
            "--max-context",
            "2048",
            "--no-recompute-embeddings",
            "--allow-origin",
            "http://localhost:3000",
        ]
    )

    assert args.index == "code"
    assert args.model == "codestral"
    assert args.port == 9000
    assert args.bind_host == "127.0.0.1"
    assert args.llm_base_url == "http://127.0.0.1:1234"
    assert args.top_k == 5
    assert args.complexity == 20
    assert args.max_context == 2048
    assert args.recompute_embeddings is False
    assert args.allow_origin == ["http://localhost:3000"]


def test_cursor_command_rejects_missing_named_index(capsys):
    from leann.cli import LeannCLI

    cli = LeannCLI()
    args = cli.create_parser().parse_args(["cursor", "--index", "missing"])

    with pytest.raises(SystemExit) as exc_info:
        cli.handle_cursor(args)

    assert exc_info.value.code == 2
    assert "Index 'missing' not found" in capsys.readouterr().out
