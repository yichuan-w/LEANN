"""
Tests for cold start latency and ZMQ reliability fixes (issues #177, #182).

Covers:
- test_warmup_sends_dummy_request: enable_warmup triggers a dummy embedding call
- test_retry_on_zmq_failure: _compute_embedding_via_server retries on transient failures
- test_no_double_ensure_server: compute_query_embedding does not call _ensure_server_running
"""

import json
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Bootstrap: load leann.searcher_base without triggering the heavy
# leann/__init__.py (which pulls in torch, backend modules, etc.)
# ---------------------------------------------------------------------------
_LEANN_SRC = Path(__file__).resolve().parent.parent / "packages" / "leann-core" / "src"

if "leann" not in sys.modules:
    _fake_leann = ModuleType("leann")
    _fake_leann.__path__ = [str(_LEANN_SRC / "leann")]
    _fake_leann.__package__ = "leann"
    sys.modules["leann"] = _fake_leann

from leann.searcher_base import BaseSearcher  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers to construct a minimal BaseSearcher instance without hitting disk
# ---------------------------------------------------------------------------


def _make_searcher():
    """Create a minimal concrete BaseSearcher for testing, bypassing __init__."""

    class _TestSearcher(BaseSearcher):
        def search(
            self,
            query: np.ndarray,
            top_k: int,
            complexity: int = 64,
            beam_width: int = 1,
            prune_ratio: float = 0.0,
            recompute_embeddings: bool = False,
            pruning_strategy: Literal["global", "local", "proportional"] = "global",
            zmq_port: Optional[int] = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            return {"labels": [], "distances": []}

    # Build a temp directory with a valid meta file so paths resolve
    tmp = tempfile.mkdtemp()
    index_name = "test_index.leann"
    meta = {
        "dimensions": 384,
        "embedding_model": "test-model",
        "embedding_mode": "sentence-transformers",
        "embedding_options": {},
    }
    meta_path = Path(tmp) / f"{index_name}.meta.json"
    meta_path.write_text(json.dumps(meta))
    # Create a dummy index file so the path exists
    (Path(tmp) / index_name).touch()

    searcher = object.__new__(_TestSearcher)
    searcher.index_path = Path(tmp) / index_name
    searcher.index_dir = searcher.index_path.parent
    searcher.meta = meta
    searcher.dimensions = meta["dimensions"]
    searcher.embedding_model = meta["embedding_model"]
    searcher.embedding_mode = meta["embedding_mode"]
    searcher.embedding_options = meta["embedding_options"]
    searcher.embedding_server_manager = MagicMock()

    return searcher


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWarmupSendsDummyRequest:
    """Fix A: enable_warmup is popped from kwargs and triggers a dummy embedding."""

    def test_warmup_sends_dummy_request(self):
        searcher = _make_searcher()

        # Mock start_server to return success
        searcher.embedding_server_manager.start_server.return_value = (True, 5560)

        # Track calls to _compute_embedding_via_server
        warmup_calls = []

        def mock_compute(chunks, port):
            warmup_calls.append((chunks, port))
            return np.zeros((len(chunks), 384), dtype=np.float32)

        searcher._compute_embedding_via_server = mock_compute

        # Call _ensure_server_running with enable_warmup=True
        port = searcher._ensure_server_running(
            "/tmp/fake.meta.json", 5557, enable_warmup=True
        )

        assert port == 5560
        # Verify warmup request was sent
        assert len(warmup_calls) == 1
        assert warmup_calls[0][0] == ["warmup"]
        assert warmup_calls[0][1] == 5560

    def test_warmup_disabled_by_default(self):
        searcher = _make_searcher()
        searcher.embedding_server_manager.start_server.return_value = (True, 5560)

        warmup_calls = []

        def mock_compute(chunks, port):
            warmup_calls.append((chunks, port))
            return np.zeros((len(chunks), 384), dtype=np.float32)

        searcher._compute_embedding_via_server = mock_compute

        # Call without enable_warmup (defaults to False)
        port = searcher._ensure_server_running("/tmp/fake.meta.json", 5557)

        assert port == 5560
        assert len(warmup_calls) == 0  # No warmup call

    def test_warmup_failure_is_silent(self):
        """Warmup failure should not propagate -- it is best effort."""
        searcher = _make_searcher()
        searcher.embedding_server_manager.start_server.return_value = (True, 5560)

        def mock_compute_failing(chunks, port):
            raise RuntimeError("ZMQ connection refused")

        searcher._compute_embedding_via_server = mock_compute_failing

        # Should NOT raise even though warmup fails
        port = searcher._ensure_server_running(
            "/tmp/fake.meta.json", 5557, enable_warmup=True
        )
        assert port == 5560

    def test_enable_warmup_not_forwarded_to_start_server(self):
        """enable_warmup should be popped from kwargs, not forwarded to start_server."""
        searcher = _make_searcher()
        searcher.embedding_server_manager.start_server.return_value = (True, 5560)
        searcher._compute_embedding_via_server = MagicMock(
            return_value=np.zeros((1, 384), dtype=np.float32)
        )

        searcher._ensure_server_running(
            "/tmp/fake.meta.json", 5557, enable_warmup=True
        )

        # Verify that start_server was called WITHOUT enable_warmup
        call_kwargs = searcher.embedding_server_manager.start_server.call_args
        assert "enable_warmup" not in call_kwargs.kwargs
        # Also not in positional args as a keyword
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        assert "enable_warmup" not in all_kwargs


class TestRetryOnZmqFailure:
    """Fix C: _compute_embedding_via_server retries with backoff on transient failures."""

    def test_retry_succeeds_on_second_attempt(self):
        searcher = _make_searcher()

        call_count = 0

        # We need to mock zmq and msgpack
        mock_socket = MagicMock()
        mock_context = MagicMock()
        mock_context.socket.return_value = mock_socket

        # First call fails, second succeeds
        embedding_data = [[0.1, 0.2, 0.3]]
        import msgpack as real_msgpack

        def recv_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Connection refused")
            return real_msgpack.packb(embedding_data)

        mock_socket.recv.side_effect = recv_side_effect

        with patch("zmq.Context", return_value=mock_context), \
             patch("time.sleep"):  # Don't actually sleep in tests
            result = searcher._compute_embedding_via_server(["test query"], 5557)

        assert call_count == 2
        np.testing.assert_array_almost_equal(
            result, np.array(embedding_data, dtype=np.float32)
        )

    def test_retry_exhaustion_raises(self):
        searcher = _make_searcher()

        mock_socket = MagicMock()
        mock_context = MagicMock()
        mock_context.socket.return_value = mock_socket

        mock_socket.recv.side_effect = Exception("Connection refused")

        with patch("zmq.Context", return_value=mock_context), \
             patch("time.sleep"):
            with pytest.raises(RuntimeError, match="after 3 attempts"):
                searcher._compute_embedding_via_server(["test query"], 5557)

        # Verify all 3 attempts were made
        assert mock_socket.recv.call_count == 3

    def test_retry_backoff_delays(self):
        searcher = _make_searcher()

        mock_socket = MagicMock()
        mock_context = MagicMock()
        mock_context.socket.return_value = mock_socket

        mock_socket.recv.side_effect = Exception("timeout")

        sleep_calls = []

        def track_sleep(seconds):
            sleep_calls.append(seconds)

        with patch("zmq.Context", return_value=mock_context), \
             patch("time.sleep", side_effect=track_sleep):
            with pytest.raises(RuntimeError):
                searcher._compute_embedding_via_server(["test"], 5557)

        # 3 attempts = 2 sleeps (no sleep after last failure)
        assert len(sleep_calls) == 2
        assert sleep_calls[0] == 0.5
        assert sleep_calls[1] == 1.0

    def test_sndtimeo_is_set(self):
        """Verify SNDTIMEO is configured on the ZMQ socket."""
        searcher = _make_searcher()

        mock_socket = MagicMock()
        mock_context = MagicMock()
        mock_context.socket.return_value = mock_socket

        import msgpack as real_msgpack

        mock_socket.recv.return_value = real_msgpack.packb([[0.1, 0.2]])

        import zmq

        with patch("zmq.Context", return_value=mock_context):
            searcher._compute_embedding_via_server(["test"], 5557)

        # Check that setsockopt was called with SNDTIMEO
        setsockopt_calls = mock_socket.setsockopt.call_args_list
        sndtimeo_calls = [c for c in setsockopt_calls if c[0][0] == zmq.SNDTIMEO]
        assert len(sndtimeo_calls) == 1
        assert sndtimeo_calls[0][0][1] == 10000  # 10 second send timeout

    def test_linger_is_set(self):
        """Verify LINGER=0 is set for clean socket teardown."""
        searcher = _make_searcher()

        mock_socket = MagicMock()
        mock_context = MagicMock()
        mock_context.socket.return_value = mock_socket

        import msgpack as real_msgpack

        mock_socket.recv.return_value = real_msgpack.packb([[0.1, 0.2]])

        import zmq

        with patch("zmq.Context", return_value=mock_context):
            searcher._compute_embedding_via_server(["test"], 5557)

        setsockopt_calls = mock_socket.setsockopt.call_args_list
        linger_calls = [c for c in setsockopt_calls if c[0][0] == zmq.LINGER]
        assert len(linger_calls) == 1
        assert linger_calls[0][0][1] == 0


class TestNoDoubleEnsureServer:
    """Fix B: compute_query_embedding no longer calls _ensure_server_running."""

    def test_no_ensure_server_called(self):
        """compute_query_embedding should NOT call _ensure_server_running internally."""
        searcher = _make_searcher()

        ensure_calls = []

        def mock_ensure(*args, **kwargs):
            ensure_calls.append((args, kwargs))
            return 5557

        def mock_compute(chunks, port):
            return np.random.rand(len(chunks), 384).astype(np.float32)

        searcher._ensure_server_running = mock_ensure
        searcher._compute_embedding_via_server = mock_compute

        result = searcher.compute_query_embedding(
            "test query",
            use_server_if_available=True,
            zmq_port=5560,
        )

        # _ensure_server_running should NOT have been called
        assert len(ensure_calls) == 0
        assert result.shape == (1, 384)

    def test_server_path_uses_provided_port(self):
        """compute_query_embedding should use the port provided by the caller."""
        searcher = _make_searcher()

        used_ports = []

        def mock_compute(chunks, port):
            used_ports.append(port)
            return np.random.rand(len(chunks), 384).astype(np.float32)

        searcher._compute_embedding_via_server = mock_compute

        searcher.compute_query_embedding(
            "test query",
            use_server_if_available=True,
            zmq_port=9999,
        )

        assert used_ports == [9999]

    def test_fallback_when_server_disabled(self):
        """When use_server_if_available=False, should go directly to fallback."""
        searcher = _make_searcher()

        server_calls = []

        def mock_compute(chunks, port):
            server_calls.append(True)
            return np.random.rand(len(chunks), 384).astype(np.float32)

        searcher._compute_embedding_via_server = mock_compute

        # Create a mock embedding_compute module for the fallback path
        mock_ec = MagicMock()
        mock_ec.compute_embeddings = MagicMock(
            return_value=np.random.rand(1, 384).astype(np.float32)
        )
        with patch.dict(sys.modules, {"leann.embedding_compute": mock_ec}):
            result = searcher.compute_query_embedding(
                "test query",
                use_server_if_available=False,
            )

        # Server should not have been called
        assert len(server_calls) == 0
        assert result.shape == (1, 384)

    def test_fallback_on_server_failure(self):
        """When server fails, should fall back to direct computation."""
        searcher = _make_searcher()

        def mock_compute_failing(chunks, port):
            raise RuntimeError("Server down")

        searcher._compute_embedding_via_server = mock_compute_failing

        mock_ec = MagicMock()
        mock_ec.compute_embeddings = MagicMock(
            return_value=np.random.rand(1, 384).astype(np.float32)
        )
        with patch.dict(sys.modules, {"leann.embedding_compute": mock_ec}):
            result = searcher.compute_query_embedding(
                "test query",
                use_server_if_available=True,
                zmq_port=5557,
            )

        mock_ec.compute_embeddings.assert_called_once()
        assert result.shape == (1, 384)
