"""Test the LEANN MCP server JSON-RPC protocol handling."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "leann-core" / "src"))
from leann.mcp import handle_request


def test_initialize():
    """MCP initialize should return server info and capabilities."""
    req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    resp = handle_request(req)
    assert resp["id"] == 1
    result = resp["result"]
    assert result["protocolVersion"] == "2024-11-05"
    assert result["serverInfo"]["name"] == "leann-mcp"
    assert "tools" in result["capabilities"]


def test_tools_list():
    """MCP tools/list should expose leann_search and leann_list."""
    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    resp = handle_request(req)
    tools = resp["result"]["tools"]
    names = {t["name"] for t in tools}
    assert "leann_search" in names
    assert "leann_list" in names


def test_tools_list_search_schema():
    """leann_search tool must declare index_name and query as required params."""
    req = {"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}}
    resp = handle_request(req)
    search_tool = next(t for t in resp["result"]["tools"] if t["name"] == "leann_search")
    schema = search_tool["inputSchema"]
    assert "index_name" in schema["properties"]
    assert "query" in schema["properties"]
    assert "index_name" in schema["required"]
    assert "query" in schema["required"]


def test_search_missing_params():
    """leann_search should return an error when required params are missing."""
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {"name": "leann_search", "arguments": {}},
    }
    resp = handle_request(req)
    text = resp["result"]["content"][0]["text"]
    assert "Error" in text or "error" in text


def test_search_missing_query():
    """leann_search should error when query is empty."""
    req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {"name": "leann_search", "arguments": {"index_name": "test", "query": ""}},
    }
    resp = handle_request(req)
    text = resp["result"]["content"][0]["text"]
    assert "Error" in text or "error" in text


def test_jsonrpc_envelope():
    """All responses must follow JSON-RPC 2.0 format."""
    req = {"jsonrpc": "2.0", "id": 42, "method": "initialize", "params": {}}
    resp = handle_request(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 42
    serialized = json.dumps(resp)
    parsed = json.loads(serialized)
    assert parsed == resp
