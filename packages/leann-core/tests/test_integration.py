"""Integration tests for LEANN HTTP server.

These tests verify end-to-end functionality including:
- HTTP server startup and shutdown
- WebSocket streaming
- MCP protocol integration
- Search and Q&A workflows
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from leann.http_server import app


@pytest.fixture
def mock_index_path(tmp_path):
    """Create mock index structure."""
    index_dir = tmp_path / ".leann" / "indexes" / "test-index"
    index_dir.mkdir(parents=True)

    # Create mock metadata file
    meta_file = index_dir / "documents.leann.meta.json"
    meta_data = {
        "backend_name": "diskann",
        "embedding_model": "test-model",
        "dimensions": 384,
    }
    meta_file.write_text(json.dumps(meta_data))

    # Create mock index file
    index_file = index_dir / "documents.leann"
    index_file.touch()

    return str(index_file)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestEndToEndWorkflows:
    """Test complete user workflows."""

    def test_health_check_workflow(self, client):
        """Test health check provides server status."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "index_loaded" in data
        assert "llm_type" in data

    def test_list_indexes_workflow(self, client):
        """Test listing available indexes."""
        response = client.get("/indexes")
        assert response.status_code == 200

        data = response.json()
        assert "indexes" in data
        assert isinstance(data["indexes"], list)

    def test_mcp_initialization_workflow(self, client):
        """Test complete MCP initialization sequence."""
        # Step 1: Initialize
        init_response = client.post("/", json={"method": "initialize", "id": 1, "jsonrpc": "2.0"})
        assert init_response.status_code == 200

        init_data = init_response.json()
        assert init_data["jsonrpc"] == "2.0"
        assert "result" in init_data
        assert init_data["result"]["protocolVersion"] == "2024-11-05"

        # Step 2: List tools
        tools_response = client.post("/", json={"method": "tools/list", "id": 2, "jsonrpc": "2.0"})
        assert tools_response.status_code == 200

        tools_data = tools_response.json()
        assert "result" in tools_data
        assert len(tools_data["result"]["tools"]) == 3

        # Verify tool names and schemas
        tools = {t["name"]: t for t in tools_data["result"]["tools"]}
        assert "leann_search" in tools
        assert "leann_list" in tools
        assert "leann_ask" in tools

        # Verify schemas
        assert "inputSchema" in tools["leann_search"]
        assert "query" in tools["leann_search"]["inputSchema"]["properties"]


class TestSearchWorkflows:
    """Test search-related workflows."""

    @patch("leann.http_server.state")
    def test_search_with_metadata_filters(self, mock_state, client):
        """Test search with metadata filtering."""
        # Setup mock
        mock_searcher = Mock()
        mock_result = Mock()
        mock_result.id = "test-1"
        mock_result.score = 0.95
        mock_result.text = "Test content"
        mock_result.metadata = {"source": "test.pdf", "page": 1}
        mock_searcher.search.return_value = [mock_result]

        mock_state.searcher = mock_searcher

        # Execute search with filters
        response = client.post(
            "/search",
            json={
                "query": "test query",
                "top_k": 5,
                "show_metadata": True,
                "metadata_filters": {"source": "test.pdf"},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) > 0

        # Verify metadata was included
        result = data["results"][0]
        assert "metadata" in result
        assert result["metadata"]["source"] == "test.pdf"

    @patch("leann.http_server.state")
    def test_search_performance_tracking(self, mock_state, client):
        """Test that search timing is tracked."""
        mock_searcher = Mock()
        mock_result = Mock()
        mock_result.id = "test-1"
        mock_result.score = 0.95
        mock_result.text = "Test content"
        mock_searcher.search.return_value = [mock_result]

        mock_state.searcher = mock_searcher

        response = client.post("/search", json={"query": "test", "top_k": 5})

        assert response.status_code == 200
        data = response.json()
        assert "search_time_ms" in data
        assert isinstance(data["search_time_ms"], (int, float))
        assert data["search_time_ms"] >= 0


class TestWebSocketWorkflows:
    """Test WebSocket streaming workflows."""

    @patch("leann.http_server.state")
    def test_complete_qa_workflow(self, mock_state, client):
        """Test complete Q&A workflow with streaming."""
        # Setup mocks
        mock_searcher = Mock()
        mock_result = Mock()
        mock_result.text = "Context text"
        mock_result.score = 0.95
        mock_result.id = "test-1"
        mock_result.metadata = {}
        mock_searcher.search.return_value = [mock_result]

        mock_llm = Mock()
        mock_llm.ask_stream.return_value = iter(["Hello", " ", "World"])

        mock_state.searcher = mock_searcher
        mock_state.llm = mock_llm

        with client.websocket_connect("/ws/ask") as websocket:
            # Send question
            websocket.send_json(
                {"question": "What is this?", "top_k": 5, "complexity": 64, "llm_params": {}}
            )

            # Receive search results
            search_msg = websocket.receive_json()
            assert search_msg["type"] == "search_results"
            assert "results" in search_msg
            assert len(search_msg["results"]) > 0

            # Collect streaming tokens
            tokens = []
            while True:
                msg = websocket.receive_json()
                if msg["type"] == "done":
                    break
                elif msg["type"] == "token":
                    tokens.append(msg["content"])
                elif msg["type"] == "error":
                    pytest.fail(f"Received error: {msg['message']}")

            # Verify we received tokens
            assert len(tokens) > 0

    @patch("leann.http_server.state")
    def test_websocket_error_handling(self, mock_state, client):
        """Test WebSocket error handling."""
        mock_searcher = Mock()
        mock_searcher.search.side_effect = Exception("Search failed")

        mock_state.searcher = mock_searcher
        mock_state.llm = Mock()

        with client.websocket_connect("/ws/ask") as websocket:
            websocket.send_json({"question": "test"})

            # Should receive error message
            msg = websocket.receive_json()
            assert msg["type"] == "error"
            assert "failed" in msg["message"].lower()


class TestMCPWorkflows:
    """Test MCP protocol workflows."""

    @patch("leann.http_server.state")
    def test_mcp_search_workflow(self, mock_state, client):
        """Test MCP search tool workflow."""
        mock_searcher = Mock()
        mock_result = Mock()
        mock_result.text = "Result text"
        mock_result.score = 0.92
        mock_searcher.search.return_value = [mock_result]

        mock_state.searcher = mock_searcher
        mock_state.index_path = "/test/path"

        response = client.post(
            "/",
            json={
                "method": "tools/call",
                "id": 1,
                "params": {"name": "leann_search", "arguments": {"query": "test query"}},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert "content" in data["result"]
        assert "Result text" in data["result"]["content"][0]["text"]

        # Verify search was called with fixed params
        mock_searcher.search.assert_called_once_with(query="test query", top_k=10, complexity=32)

    @patch("leann.http_server.state")
    def test_mcp_ask_workflow(self, mock_state, client):
        """Test MCP ask tool workflow."""
        mock_searcher = Mock()
        mock_result = Mock()
        mock_result.text = "Context"
        mock_searcher.search.return_value = [mock_result]

        mock_llm = Mock()
        mock_llm.ask.return_value = "Answer to question"

        mock_state.searcher = mock_searcher
        mock_state.llm = mock_llm

        response = client.post(
            "/",
            json={
                "method": "tools/call",
                "id": 1,
                "params": {"name": "leann_ask", "arguments": {"question": "What is this?"}},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert data["result"]["content"][0]["text"] == "Answer to question"

        # Verify both search and LLM were called
        mock_searcher.search.assert_called_once()
        mock_llm.ask.assert_called_once()

    @patch("leann.http_server.state")
    def test_mcp_list_workflow(self, mock_state, client):
        """Test MCP list tool workflow."""
        mock_state.index_path = "/home/user/.leann/indexes/my-docs/documents.leann"

        # Create temporary metadata file
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir) / "my-docs"
            index_dir.mkdir()

            meta_file = index_dir / "documents.leann.meta.json"
            meta_data = {
                "backend_name": "diskann",
                "embedding_model": "sentence-transformers",
                "dimensions": 384,
            }
            meta_file.write_text(json.dumps(meta_data))

            mock_state.index_path = str(index_dir / "documents.leann")

            response = client.post(
                "/",
                json={"method": "tools/call", "id": 1, "params": {"name": "leann_list"}},
            )

            assert response.status_code == 200
            data = response.json()
            assert "result" in data
            text = data["result"]["content"][0]["text"]
            assert "my-docs" in text
            assert "diskann" in text
            assert "sentence-transformers" in text


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_json_request(self, client):
        """Test handling of invalid JSON."""
        response = client.post("/", data="not json", headers={"Content-Type": "application/json"})
        # Server returns 500 for JSON parse errors in MCP endpoint
        assert response.status_code == 500
        data = response.json()
        assert "error" in data

    @patch("leann.http_server.state")
    def test_search_with_invalid_parameters(self, mock_state, client):
        """Test search with invalid parameters."""
        mock_state.searcher = Mock()

        # Invalid top_k (too large)
        response = client.post("/search", json={"query": "test", "top_k": 1000})
        assert response.status_code == 422

        # Missing query
        response = client.post("/search", json={"top_k": 5})
        assert response.status_code == 422

    def test_mcp_invalid_tool_name(self, client):
        """Test MCP call with invalid tool name."""
        response = client.post(
            "/",
            json={
                "method": "tools/call",
                "id": 1,
                "params": {"name": "invalid_tool", "arguments": {}},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert "Unknown tool" in data["error"]["message"]


@pytest.mark.asyncio
class TestConcurrency:
    """Test concurrent request handling."""

    @patch("leann.http_server.state")
    async def test_concurrent_searches(self, mock_state):
        """Test handling multiple concurrent search requests."""
        mock_searcher = Mock()
        mock_result = Mock()
        mock_result.text = "Result"
        mock_result.score = 0.95
        mock_result.id = "test"
        mock_searcher.search.return_value = [mock_result]

        mock_state.searcher = mock_searcher

        client = TestClient(app)

        # Send multiple concurrent requests
        async def make_request():
            response = client.post("/search", json={"query": "test", "top_k": 5})
            return response.status_code

        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All requests should succeed
        assert all(status == 200 for status in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
