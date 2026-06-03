from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from leann import server


def test_web_ui_routes_serve_packaged_assets(monkeypatch):
    monkeypatch.setattr(server, "_list_current_project_indexes", lambda: [])
    client = TestClient(server.create_app())

    response = client.get("/")
    assert response.status_code == 200
    assert "LEANN" in response.text
    assert "/ui/app.css" in response.text
    assert "/ui/app.js" in response.text

    css_response = client.get("/ui/app.css")
    assert css_response.status_code == 200
    assert ".workspace" in css_response.text

    js_response = client.get("/ui/app.js")
    assert js_response.status_code == 200
    assert "fetchJson" in js_response.text


def test_web_ui_rejects_missing_or_escaped_assets(monkeypatch):
    monkeypatch.setattr(server, "_list_current_project_indexes", lambda: [])
    client = TestClient(server.create_app())

    assert client.get("/ui/missing.js").status_code == 404
    assert client.get("/ui/../server.py").status_code == 404


def test_search_endpoint_still_uses_existing_api_shape(monkeypatch):
    calls = []

    class FakeSearcher:
        def __init__(self, index_path):
            calls.append(("init", index_path))

        def search(self, **kwargs):
            calls.append(("search", kwargs))
            return [
                SimpleNamespace(
                    id="doc-1",
                    score=0.75,
                    text="Storage-efficient vector search",
                    metadata={"source": "docs/web_ui.md", "page": 1},
                )
            ]

    monkeypatch.setattr(server, "_resolve_index_path", lambda index_name: f"/tmp/{index_name}")
    monkeypatch.setattr(server, "LeannSearcher", FakeSearcher)
    client = TestClient(server.create_app())

    response = client.post(
        "/indexes/docs/search",
        json={"query": "storage", "top_k": 3, "complexity": 32, "use_grep": True},
    )

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": "doc-1",
            "score": 0.75,
            "text": "Storage-efficient vector search",
            "metadata": {"source": "docs/web_ui.md", "page": 1},
        }
    ]
    assert calls == [
        ("init", "/tmp/docs"),
        (
            "search",
            {
                "query": "storage",
                "top_k": 3,
                "complexity": 32,
                "beam_width": 1,
                "prune_ratio": 0.0,
                "recompute_embeddings": True,
                "pruning_strategy": "global",
                "use_grep": True,
            },
        ),
    ]
