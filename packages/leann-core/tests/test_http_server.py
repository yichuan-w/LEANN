"""Tests for LEANN HTTP server."""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

# Import the app and related functions
from leann.http_server import (
    ServerState,
    _mcp_initialize,
    _mcp_tool_ask,
    _mcp_tool_list,
    _mcp_tool_search,
    _mcp_tools_list,
    app,
    handle_mcp_request,
    state,
)


@pytest.fixture
def mock_searcher():
    """Mock LeannSearcher."""
    searcher = Mock()
    result = Mock()
    result.score = 0.95
    result.text = "Test result text"
    result.id = "test-id"
    result.metadata = {"source": "test"}
    searcher.search.return_value = [result]
    return searcher


@pytest.fixture
def mock_llm():
    """Mock LLM interface."""
    llm = Mock()
    llm.ask.return_value = "Test answer"
    llm.ask_stream.return_value = iter(["Test ", "answer"])
    return llm


@pytest.fixture
def setup_state(mock_searcher, mock_llm):
    """Setup server state with mocks."""
    original_searcher = state.searcher
    original_llm = state.llm
    original_index_path = state.index_path

    state.searcher = mock_searcher
    state.llm = mock_llm
    state.index_path = "/path/to/test.leann"

    yield state

    # Cleanup
    state.searcher = original_searcher
    state.llm = original_llm
    state.index_path = original_index_path


class TestMCPHandlers:
    """Tests for MCP request handlers."""

    def test_mcp_initialize(self):
        """Test MCP initialize request."""
        request = {"method": "initialize", "id": 1}
        response = _mcp_initialize(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert response["result"]["protocolVersion"] == "2024-11-05"
        assert response["result"]["serverInfo"]["name"] == "leann-http"

    def test_mcp_tools_list(self):
        """Test MCP tools/list request."""
        request = {"method": "tools/list", "id": 2}
        response = _mcp_tools_list(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 2
        assert "result" in response
        assert len(response["result"]["tools"]) == 3

        tool_names = [t["name"] for t in response["result"]["tools"]]
        assert "leann_search" in tool_names
        assert "leann_list" in tool_names
        assert "leann_ask" in tool_names

    def test_mcp_tool_search_success(self, setup_state):
        """Test successful MCP search tool call."""
        request = {"method": "tools/call", "id": 3}
        args = {"query": "test query"}

        response = _mcp_tool_search(request, args)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 3
        assert "result" in response
        assert "content" in response["result"]
        assert len(response["result"]["content"]) > 0
        assert "Test result text" in response["result"]["content"][0]["text"]

        # Verify search was called with correct params
        setup_state.searcher.search.assert_called_once_with(
            query="test query", top_k=10, complexity=32
        )

    def test_mcp_tool_search_missing_query(self):
        """Test MCP search with missing query parameter."""
        request = {"method": "tools/call", "id": 4}
        args = {}

        response = _mcp_tool_search(request, args)

        assert response["jsonrpc"] == "2.0"
        assert "result" in response
        assert "Error: query parameter is required" in response["result"]["content"][0]["text"]

    def test_mcp_tool_search_no_searcher(self):
        """Test MCP search when searcher is not initialized."""
        original_searcher = state.searcher
        state.searcher = None

        request = {"method": "tools/call", "id": 5}
        args = {"query": "test"}

        response = _mcp_tool_search(request, args)

        assert "Error: Searcher not initialized" in response["result"]["content"][0]["text"]

        state.searcher = original_searcher

    def test_mcp_tool_list_no_index(self):
        """Test MCP list tool when no index is loaded."""
        original_path = state.index_path
        state.index_path = ""

        request = {"method": "tools/call", "id": 6}
        response = _mcp_tool_list(request)

        assert "No index loaded" in response["result"]["content"][0]["text"]

        state.index_path = original_path

    def test_mcp_tool_ask_success(self, setup_state):
        """Test successful MCP ask tool call."""
        request = {"method": "tools/call", "id": 7}
        args = {"question": "What is this?"}

        response = _mcp_tool_ask(request, args)

        assert response["jsonrpc"] == "2.0"
        assert "result" in response
        assert response["result"]["content"][0]["text"] == "Test answer"

        # Verify search and LLM were called
        setup_state.searcher.search.assert_called_once()
        setup_state.llm.ask.assert_called_once()

    def test_mcp_tool_ask_missing_question(self):
        """Test MCP ask with missing question parameter."""
        request = {"method": "tools/call", "id": 8}
        args = {}

        response = _mcp_tool_ask(request, args)

        assert "Error: question parameter is required" in response["result"]["content"][0]["text"]

    def test_mcp_tool_ask_no_llm(self, mock_searcher):
        """Test MCP ask when LLM is not initialized."""
        original_searcher = state.searcher
        original_llm = state.llm

        state.searcher = mock_searcher
        state.llm = None

        request = {"method": "tools/call", "id": 9}
        args = {"question": "test"}

        response = _mcp_tool_ask(request, args)

        assert "Error: LLM not initialized" in response["result"]["content"][0]["text"]

        state.searcher = original_searcher
        state.llm = original_llm

    def test_handle_mcp_request_unknown_method(self):
        """Test handling unknown MCP method."""
        request = {"method": "unknown_method", "id": 10}
        response = handle_mcp_request(request)

        assert response["jsonrpc"] == "2.0"
        assert "error" in response
        assert response["error"]["code"] == -32601
        assert "Method not found" in response["error"]["message"]

    def test_handle_mcp_request_unknown_tool(self, setup_state):
        """Test handling unknown tool in tools/call."""
        request = {
            "method": "tools/call",
            "id": 11,
            "params": {"name": "unknown_tool", "arguments": {}},
        }
        response = handle_mcp_request(request)

        assert "error" in response
        assert "Unknown tool" in response["error"]["message"]


class TestAPIEndpoints:
    """Tests for REST API endpoints."""

    @pytest.fixture
    def client(self):
        """Create FastAPI test client."""
        return TestClient(app)

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "index_loaded" in data

    def test_mcp_root_initialize(self, client):
        """Test MCP POST endpoint with initialize."""
        response = client.post("/", json={"method": "initialize", "id": 1})
        assert response.status_code == 200

        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert "result" in data

    def test_mcp_root_tools_list(self, client):
        """Test MCP POST endpoint with tools/list."""
        response = client.post("/", json={"method": "tools/list", "id": 2})
        assert response.status_code == 200

        data = response.json()
        assert "result" in data
        assert "tools" in data["result"]

    def test_search_endpoint_no_searcher(self, client):
        """Test search endpoint when searcher is not initialized."""
        original_searcher = state.searcher
        state.searcher = None

        response = client.post("/search", json={"query": "test"})
        assert response.status_code == 503

        state.searcher = original_searcher

    def test_search_endpoint_success(self, client, setup_state):
        """Test successful search endpoint call."""
        response = client.post(
            "/search", json={"query": "test query", "top_k": 5, "complexity": 64}
        )
        assert response.status_code == 200

        data = response.json()
        assert "results" in data
        assert "search_time_ms" in data
        assert len(data["results"]) > 0

    def test_list_indexes_endpoint(self, client):
        """Test list indexes endpoint."""
        response = client.get("/indexes")
        assert response.status_code == 200

        data = response.json()
        assert "indexes" in data


class TestWebSocket:
    """Tests for WebSocket endpoints."""

    @pytest.fixture
    def client(self):
        """Create FastAPI test client."""
        return TestClient(app)

    def test_websocket_ask_no_searcher(self, client):
        """Test WebSocket when searcher is not initialized."""
        original_searcher = state.searcher
        state.searcher = None

        with client.websocket_connect("/ws/ask") as websocket:
            data = websocket.receive_json()
            assert data["type"] == "error"
            assert "not initialized" in data["message"].lower()

        state.searcher = original_searcher

    def test_websocket_ask_success(self, client, setup_state):
        """Test successful WebSocket Q&A."""
        with client.websocket_connect("/ws/ask") as websocket:
            # Send question
            websocket.send_json({"question": "What is this?", "top_k": 5})

            # Receive search results
            data = websocket.receive_json()
            assert data["type"] == "search_results"
            assert "results" in data

            # Receive LLM response
            data = websocket.receive_json()
            assert data["type"] in ["token", "done"]

    def test_websocket_ask_missing_question(self, client, setup_state):
        """Test WebSocket with missing question field."""
        with client.websocket_connect("/ws/ask") as websocket:
            # Send invalid request
            websocket.send_json({})

            # Receive error
            data = websocket.receive_json()
            assert data["type"] == "error"
            assert "question" in data["message"].lower()


@pytest.mark.asyncio
class TestSSE:
    """Tests for SSE endpoints."""

    @pytest.fixture
    def client(self):
        """Create FastAPI test client."""
        return TestClient(app)

    def test_mcp_sse_endpoint(self, client):
        """Test MCP SSE endpoint creates session."""
        # Note: Testing SSE with TestClient is limited
        # This is a basic smoke test
        # In production, use a proper SSE client for testing
        response = client.get("/sse")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
