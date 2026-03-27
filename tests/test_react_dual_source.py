"""
Tests for ReAct agent dual-source routing (issue #283).
Covers: prompt adaptation, web fallback, visit_page errors, routing, and pipeline.
"""

from unittest.mock import MagicMock, patch

from leann.api import SearchResult
from leann.react_agent import ReActAgent
from leann.web_search import WebSearcher


def _make_searcher() -> MagicMock:
    """Create a lightweight mocked searcher for deterministic unit tests."""
    searcher = MagicMock()
    searcher.search.return_value = [
        SearchResult(
            id="1",
            score=0.91,
            text="LEANN achieves 97% storage reduction via graph pruning.",
            metadata={"source": "docs"},
        ),
        SearchResult(
            id="2",
            score=0.83,
            text="The search function uses HNSW for approximate nearest neighbors.",
            metadata={"source": "code"},
        ),
    ]
    return searcher


# ── 1. Dual-source presence ──────────────────────────────────────────


def test_prompt_includes_web_tools_when_key_present():
    """When SERPER_API_KEY is set, prompt should list all three tools."""
    searcher = _make_searcher()
    agent = ReActAgent(searcher=searcher, llm=MagicMock(), serper_api_key="test-key")
    prompt = agent._create_react_prompt("test question", 1, [])
    assert "web_search" in prompt
    assert "visit_page" in prompt
    assert "leann_search" in prompt
    assert agent.web_search_available is True


def test_prompt_excludes_web_tools_when_no_key():
    """When no SERPER_API_KEY, prompt should only show leann_search."""
    searcher = _make_searcher()
    with patch.dict("os.environ", {}, clear=False):
        agent = ReActAgent(searcher=searcher, llm=MagicMock(), serper_api_key=None)
    prompt = agent._create_react_prompt("test question", 1, [])
    assert "leann_search" in prompt
    assert "web_search" not in prompt or "not available" in prompt
    assert agent.web_search_available is False


# ── 2. Routing behavior ─────────────────────────────────────────────


def test_local_only_routing():
    """Agent uses leann_search for local queries."""
    searcher = _make_searcher()
    mock_llm = MagicMock()
    mock_llm.ask.side_effect = [
        'Thought: This is about our codebase.\nAction: leann_search("LEANN storage reduction")',
        "Thought: I have the answer.\nAction: Final Answer: LEANN saves 97% storage.",
    ]
    agent = ReActAgent(searcher=searcher, llm=mock_llm, max_iterations=3)
    agent.run("How does LEANN reduce storage?", top_k=2)

    assert len(agent.search_history) >= 1
    assert agent.search_history[0]["source"] == "local"
    assert "leann_search" in agent.search_history[0]["action"]


def test_web_only_routing():
    """Agent uses web_search for web queries (mocked)."""
    searcher = _make_searcher()

    with patch.object(WebSearcher, "search") as mock_web:
        mock_web.return_value = [
            {
                "title": "Python 3.13 News",
                "link": "https://python.org",
                "snippet": "New features in 3.13",
            }
        ]
        mock_llm = MagicMock()
        mock_llm.ask.side_effect = [
            'Thought: Need current web info.\nAction: web_search("Python 3.13 features")',
            "Thought: Got it.\nAction: Final Answer: Python 3.13 has new features.",
        ]
        agent = ReActAgent(
            searcher=searcher, llm=mock_llm, max_iterations=3, serper_api_key="test-key"
        )
        agent.run("What's new in Python 3.13?", top_k=2)

        assert len(agent.search_history) >= 1
        assert agent.search_history[0]["source"] == "web"
        mock_web.assert_called_once()


def test_mixed_routing_local_then_web():
    """Agent uses leann_search first, then web_search."""
    searcher = _make_searcher()

    with patch.object(WebSearcher, "search") as mock_web:
        mock_web.return_value = [
            {
                "title": "Best Practices",
                "link": "https://example.com",
                "snippet": "Current best practices...",
            }
        ]
        mock_llm = MagicMock()
        mock_llm.ask.side_effect = [
            'Thought: First check our code.\nAction: leann_search("search implementation")',
            'Thought: Now check best practices.\nAction: web_search("vector DB best practices")',
            "Thought: I can compare now.\nAction: Final Answer: Our search is good but could improve.",
        ]
        agent = ReActAgent(
            searcher=searcher, llm=mock_llm, max_iterations=5, serper_api_key="test-key"
        )
        agent.run("Compare our search with best practices", top_k=2)

        assert len(agent.search_history) >= 2
        assert agent.search_history[0]["source"] == "local"
        assert agent.search_history[1]["source"] == "web"


# ── 3. Pipeline correctness ─────────────────────────────────────────


def test_web_results_formatted_as_observations():
    """Web search results are properly formatted and passed to the next iteration."""
    searcher = _make_searcher()

    with patch.object(WebSearcher, "search") as mock_web:
        mock_web.return_value = [
            {"title": "Result A", "link": "https://a.com", "snippet": "Snippet A"},
            {"title": "Result B", "link": "https://b.com", "snippet": "Snippet B"},
        ]
        mock_llm = MagicMock()
        mock_llm.ask.side_effect = [
            'Thought: Search the web.\nAction: web_search("test query")',
            "Thought: Done.\nAction: Final Answer: Found it.",
        ]
        agent = ReActAgent(
            searcher=searcher, llm=mock_llm, max_iterations=3, serper_api_key="test-key"
        )
        agent.run("test", top_k=2)

        # The second LLM call should contain the formatted web results
        second_prompt = mock_llm.ask.call_args_list[1][0][0]
        assert "Result A" in second_prompt or "Snippet A" in second_prompt


def test_visit_page_content_truncated():
    """visit_page content is truncated to 15k chars."""
    searcher = _make_searcher()

    with patch.object(WebSearcher, "get_page_content") as mock_fetch:
        mock_fetch.return_value = "x" * 20000
        mock_llm = MagicMock()
        mock_llm.ask.side_effect = [
            'Thought: Read this page.\nAction: visit_page("https://docs.python.org/3")',
            "Thought: Done.\nAction: Final Answer: Got the content.",
        ]
        agent = ReActAgent(
            searcher=searcher, llm=mock_llm, max_iterations=3, serper_api_key="test-key"
        )
        agent.run("read docs", top_k=2)

        second_prompt = mock_llm.ask.call_args_list[1][0][0]
        # Content should be truncated — observation contains at most 15000 chars of content
        assert "x" * 15000 in second_prompt
        assert "x" * 15001 not in second_prompt


def test_leann_search_results_include_scores():
    """Local search results include scores and text snippets."""
    searcher = _make_searcher()
    mock_llm = MagicMock()
    mock_llm.ask.side_effect = [
        'Thought: Search locally.\nAction: leann_search("LEANN")',
        "Thought: Done.\nAction: Final Answer: Found results.",
    ]
    agent = ReActAgent(searcher=searcher, llm=mock_llm, max_iterations=3)
    agent.run("tell me about LEANN", top_k=2)

    second_prompt = mock_llm.ask.call_args_list[1][0][0]
    assert "Score:" in second_prompt
    assert "Result" in second_prompt


# ── 4. Edge cases ────────────────────────────────────────────────────


def test_web_search_no_api_key_graceful():
    """web_search without API key returns clear fallback message, not a crash."""
    searcher = _make_searcher()

    mock_llm = MagicMock()
    mock_llm.ask.side_effect = [
        'Thought: Try web.\nAction: web_search("Python 3.13")',
        'Thought: Web failed, try local.\nAction: leann_search("search")',
        "Thought: Done.\nAction: Final Answer: Here's what I found locally.",
    ]
    with patch.dict("os.environ", {}, clear=False):
        agent = ReActAgent(searcher=searcher, llm=mock_llm, max_iterations=5, serper_api_key=None)
    answer = agent.run("test", top_k=2)

    assert agent.search_history[0]["results_count"] == 0
    assert agent.search_history[0]["source"] == "web"
    # Agent should not crash and should produce an answer
    assert answer is not None and len(answer) > 0


def test_web_search_invalid_key_graceful():
    """Invalid Serper key returns error, agent continues."""
    searcher = _make_searcher()

    with patch.object(WebSearcher, "search") as mock_web:
        mock_web.return_value = [
            {"title": "Error", "link": "", "snippet": "Web Search failed:401 Unauthorized"}
        ]
        mock_llm = MagicMock()
        mock_llm.ask.side_effect = [
            'Thought: Try web.\nAction: web_search("test")',
            "Thought: Web failed.\nAction: Final Answer: Could not search web.",
        ]
        agent = ReActAgent(
            searcher=searcher, llm=mock_llm, max_iterations=3, serper_api_key="bad-key"
        )
        answer = agent.run("test", top_k=2)

        assert agent.search_history[0]["results_count"] == 0
        assert answer is not None


def test_visit_page_404_graceful():
    """visit_page on a 404 URL returns error string, agent continues."""
    searcher = _make_searcher()

    with patch.object(WebSearcher, "get_page_content") as mock_fetch:
        mock_fetch.return_value = "Error fetching content: 404 Not Found"
        mock_llm = MagicMock()
        mock_llm.ask.side_effect = [
            'Thought: Read page.\nAction: visit_page("https://example.com/404")',
            "Thought: Page not found.\nAction: Final Answer: Page was not accessible.",
        ]
        agent = ReActAgent(
            searcher=searcher, llm=mock_llm, max_iterations=3, serper_api_key="test-key"
        )
        answer = agent.run("read page", top_k=2)

        assert agent.search_history[0]["results_count"] == 0
        assert "not accessible" in answer.lower() or len(answer) > 0


def test_max_iterations_with_mixed_sources():
    """Agent hitting max iterations with mixed sources still produces an answer."""
    searcher = _make_searcher()

    with patch.object(WebSearcher, "search") as mock_web:
        mock_web.return_value = [
            {"title": "Web Result", "link": "https://example.com", "snippet": "Some web info"}
        ]
        mock_llm = MagicMock()
        # Never gives a Final Answer — forces max iterations
        mock_llm.ask.side_effect = [
            'Thought: Search locally.\nAction: leann_search("storage")',
            'Thought: Now web.\nAction: web_search("vector databases")',
            # Max iterations reached — the agent will ask for a final answer
            "Based on all searches, LEANN is a storage-efficient vector DB.",
        ]
        agent = ReActAgent(
            searcher=searcher, llm=mock_llm, max_iterations=2, serper_api_key="test-key"
        )
        answer = agent.run("Compare approaches", top_k=2)

        assert len(agent.search_history) == 2
        sources = {h["source"] for h in agent.search_history}
        assert "local" in sources
        assert "web" in sources
        assert answer is not None and len(answer) > 0


def test_search_history_has_source_field():
    """search_history entries include 'source' field for easy inspection."""
    searcher = _make_searcher()
    mock_llm = MagicMock()
    mock_llm.ask.side_effect = [
        'Thought: Local search.\nAction: leann_search("test")',
        "Thought: Done.\nAction: Final Answer: Done.",
    ]
    agent = ReActAgent(searcher=searcher, llm=mock_llm, max_iterations=3)
    agent.run("test", top_k=2)

    assert "source" in agent.search_history[0]
    assert agent.search_history[0]["source"] in ("local", "web")
