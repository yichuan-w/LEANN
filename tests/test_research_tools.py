from unittest.mock import MagicMock

from leann.api import SearchResult
from leann.research_tools import (
    LeannSearchTool,
    VisitPageTool,
    WebSearchTool,
    build_research_tools,
)


def test_build_research_tools_respects_source_policy_without_web_key():
    searcher = MagicMock()
    web_searcher = MagicMock()
    web_searcher.api_key = None

    tools = build_research_tools(searcher, web_searcher, "both")

    assert sorted(tools) == ["leann_search"]


def test_build_research_tools_adds_web_tools_when_allowed_and_configured():
    searcher = MagicMock()
    web_searcher = MagicMock()
    web_searcher.api_key = "key"

    tools = build_research_tools(searcher, web_searcher, "both")

    assert sorted(tools) == ["leann_search", "visit_page", "web_search"]


def test_leann_search_tool_returns_normalized_result():
    searcher = MagicMock()
    searcher.search.return_value = [
        SearchResult(id="p1", score=0.9, text="local result", metadata={"source": "docs"})
    ]

    result = LeannSearchTool(searcher).run("query", top_k=1)

    searcher.search.assert_called_once_with("query", top_k=1)
    assert result.source == "local"
    assert result.results_count == 1
    assert "local result" in result.observation


def test_web_search_tool_returns_normalized_result():
    web_searcher = MagicMock()
    web_searcher.search.return_value = [
        {"title": "Title", "link": "https://example.com", "snippet": "snippet"}
    ]

    result = WebSearchTool(web_searcher).run("query", top_k=1)

    web_searcher.search.assert_called_once_with("query", top_k=1)
    assert result.source == "web"
    assert result.results_count == 1
    assert "Title" in result.observation


def test_visit_page_tool_truncates_content():
    web_searcher = MagicMock()
    web_searcher.get_page_content.return_value = "x" * 20000

    result = VisitPageTool(web_searcher).run("https://example.com", top_k=5)

    web_searcher.get_page_content.assert_called_once_with("https://example.com")
    assert result.source == "web"
    assert result.results_count == 1
    assert "x" * 15000 in result.observation
    assert "x" * 15001 not in result.observation
