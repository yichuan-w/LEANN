"""Exercise the real MCP reader/transport against an HTTP protocol fixture."""

import asyncio
import importlib
import json
import sys
from types import SimpleNamespace

import pytest

# The opt-in reader has its own installation; do not add SDK requirements to
# collection of LEANN's ordinary test suite.
httpx = pytest.importorskip("httpx")
pytest.importorskip("mcp")
reader = importlib.import_module("apps.parallel_data.parallel_mcp_reader")
export = importlib.import_module("apps.parallel_data.__main__")


def install_server(monkeypatch, search, extraction=None, tool_error=False, tools=None):
    """Install an HTTP server fixture that observes every actual SDK request."""
    requests = []
    calls = []

    def handle(request):
        requests.append(request)
        assert str(request.url) == reader.MCP_URL
        assert request.headers["User-Agent"] == reader.USER_AGENT
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        payload = json.loads(request.content)
        if "id" not in payload:
            return httpx.Response(202)
        method = payload["method"]
        if method == "initialize":
            result = {
                "protocolVersion": payload["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {"name": name, "inputSchema": {"type": "object"}}
                    for name in (tools if tools is not None else ["web_search", "web_fetch"])
                ]
            }
        else:
            assert method == "tools/call"
            calls.append(payload["params"])
            data = search if payload["params"]["name"] == "web_search" else extraction
            result = {
                "content": [{"type": "text", "text": json.dumps(data)}],
                "isError": tool_error,
            }
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result})

    client_class = httpx.AsyncClient
    monkeypatch.setattr(
        reader.httpx,
        "AsyncClient",
        lambda **kwargs: client_class(transport=httpx.MockTransport(handle), **kwargs),
    )
    return requests, calls


def item(url, text="Search excerpt", title="A source"):
    return {"url": url, "title": title, "excerpts": [text], "publish_date": "2026-10-01"}


def test_search_fetch_real_transport(monkeypatch, caplog):
    first = item("https://example.com/one")
    second = item("https://example.com/two", "Fallback excerpt")
    requests, calls = install_server(
        monkeypatch,
        {"results": [first, first, second, item("https://example.com/three")]},
        {
            "results": [item(first["url"], "Fetched excerpt", "Fetched title")],
            "errors": [{"url": second["url"], "error_type": "HTTP_ERROR"}],
        },
    )
    monkeypatch.setenv("PARALLEL_API_KEY", "must-not-be-sent")
    instance = reader.ParallelMCPReader()
    documents = asyncio.run(instance.read_web_data("Research", ["research web sources"], 2, True))
    assert len(requests) >= 5  # initialization, discovery, search, and fetch
    assert [call["name"] for call in calls] == ["web_search", "web_fetch"]
    assert calls[1]["arguments"]["urls"] == [first["url"], second["url"]]
    assert calls[0]["arguments"]["session_id"] == calls[1]["arguments"]["session_id"]
    assert calls[0]["arguments"]["session_id"] == instance.session_id
    assert calls[1]["arguments"]["search_queries"] == ["research web sources"]
    assert documents[0]["text"] == "Fetched excerpt"
    assert documents[0]["metadata"]["title"] == "Fetched title"
    assert documents[1]["text"] == "Fallback excerpt"
    assert "retaining search excerpts" in caplog.text


def test_search_only_and_blank_excerpts(monkeypatch):
    _, calls = install_server(
        monkeypatch,
        {"results": [item("https://example.com/empty", "  "), item("https://example.com")]},
    )
    documents = asyncio.run(reader.ParallelMCPReader().read_web_data("Research", ["web sources"]))
    assert len(documents) == 1
    assert documents[0]["metadata"]["url"] == "https://example.com"
    assert [call["name"] for call in calls] == ["web_search"]


def test_empty_search_does_not_fetch(monkeypatch):
    _, calls = install_server(monkeypatch, {"results": []})
    assert (
        asyncio.run(reader.ParallelMCPReader().read_web_data("Research", ["sources"], fetch=True))
        == []
    )
    assert len(calls) == 1


def test_tool_error_propagates(monkeypatch):
    install_server(monkeypatch, {"error": "rate limited"}, tool_error=True)
    with pytest.raises(Exception) as error:
        asyncio.run(reader.ParallelMCPReader().read_web_data("Research", ["sources"]))
    assert "Parallel MCP tool failed" in repr(error.value)


def test_missing_tool_propagates(monkeypatch):
    install_server(monkeypatch, {"results": []}, tools=["web_search"])
    with pytest.raises(Exception) as error:
        asyncio.run(reader.ParallelMCPReader().read_web_data("Research", ["sources"], fetch=True))
    assert "missing tools" in repr(error.value)


def test_structured_output_and_invalid_json():
    result = SimpleNamespace(isError=False, structuredContent={"results": []}, content=[])
    assert reader._tool_payload(result) == {"results": []}
    result.structuredContent = None
    result.content = [SimpleNamespace(text="invalid")]
    with pytest.raises(RuntimeError, match="invalid JSON"):
        reader._tool_payload(result)


@pytest.mark.parametrize(
    "objective,queries,limit", [("", ["x"], 1), ("x", [], 1), ("x", [" "], 1), ("x", ["y"], 21)]
)
def test_invalid_arguments(objective, queries, limit):
    with pytest.raises(ValueError):
        asyncio.run(reader.ParallelMCPReader().read_web_data(objective, queries, limit))


def test_overall_timeout(monkeypatch):
    async def stalled(*args):
        await asyncio.Event().wait()

    instance = reader.ParallelMCPReader(timeout=0.01)
    monkeypatch.setattr(instance, "_read", stalled)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(instance.read_web_data("Research", ["sources"]))


def test_export_command_preserves_sources(monkeypatch, tmp_path, capsys):
    async def read_data(self, objective, queries, limit, fetch):
        assert (objective, queries, limit, fetch) == ("Research", ["web sources"], 5, False)
        return [
            {"text": "Useful café excerpt", "metadata": item("https://example.com/../../source")}
        ]

    monkeypatch.setattr(reader.ParallelMCPReader, "read_web_data", read_data)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parallel_data",
            "--objective",
            "Research",
            "--search-query",
            "web sources",
            "--output-dir",
            str(tmp_path),
        ],
    )
    export.main()
    paths = list(tmp_path.glob("*.md"))
    assert len(paths) == 1
    assert paths[0].parent == tmp_path
    assert paths[0].name.startswith("example-com-source-")
    text = paths[0].read_text()
    assert "Source: https://example.com/../../source" in text
    assert "Published: 2026-10-01" in text
    assert "Useful café excerpt" in text
    assert "Exported 1" in capsys.readouterr().out


@pytest.mark.parametrize(
    "url,prefix",
    [
        ("https://arxiv.org/abs/2506.08276", "arxiv-org-abs-2506-08276-"),
        ("https://yichuan-w.github.io/", "yichuan-w-github-io-"),
        ("https://example.com/" + "a" * 200, "example-com-" + "a" * 68 + "-"),
    ],
)
def test_document_filename_is_readable_and_unique(url, prefix):
    name = export.document_filename(url)
    assert name.startswith(prefix) and name.endswith(".md")
    assert export.document_filename(url) == name
    assert export.document_filename(url + "?v=2") != name


def test_rag_app_chunks_web_documents_with_sources(monkeypatch):
    pytest.importorskip("leann")
    pytest.importorskip("llama_index.core")
    rag = importlib.import_module("apps.parallel_rag")

    async def read_data(self, objective, queries, limit, fetch):
        assert (objective, queries, limit, fetch) == ("Research", ["web sources"], 5, True)
        return [{"text": "LEANN recomputes embeddings. " * 40, "metadata": item("https://a.io/x")}]

    monkeypatch.setattr(reader.ParallelMCPReader, "read_web_data", read_data)
    app = rag.ParallelWebRAG()
    args = app.parser.parse_args(
        ["--objective", "Research", "--search-query", "web sources", "--fetch"]
    )
    texts = asyncio.run(app.load_data(args))
    assert len(texts) > 1
    assert texts[0]["text"].startswith("A source\nSource: https://a.io/x")
    assert all(text["metadata"]["url"] == "https://a.io/x" for text in texts)
