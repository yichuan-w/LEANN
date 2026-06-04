"""Provider-neutral research tools for LEANN agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from .api import LeannSearcher, SearchResult
from .web_search import WebSearcher

SearchSource = Literal["local", "web"]
SearchSourcePolicy = Literal["local", "web", "both"]


@dataclass(frozen=True)
class ResearchToolResult:
    """Normalized result returned by an agent research tool."""

    observation: str
    results_count: int
    source: SearchSource


class ResearchTool(Protocol):
    """Minimal contract that SDK adapters and built-in agents can share."""

    name: str
    description: str
    source: SearchSource

    def run(self, query: str, top_k: int = 5) -> ResearchToolResult:
        """Run the tool and return a normalized observation."""


def format_search_results(results: list[SearchResult]) -> str:
    """Format LEANN search results for model observation text."""
    if not results:
        return "No results found."

    formatted = []
    for i, result in enumerate(results, 1):
        formatted.append(f"[Result {i}] (Score: {result.score:.3f})\n{result.text[:500]}...")
        if result.metadata.get("source"):
            formatted[-1] += f"\nSource: {result.metadata['source']}"
    return "\n\n".join(formatted)


class LeannSearchTool:
    name = "leann_search"
    description = "Search the local private LEANN knowledge base."
    source: SearchSource = "local"

    def __init__(self, searcher: LeannSearcher):
        self.searcher = searcher

    def run(self, query: str, top_k: int = 5) -> ResearchToolResult:
        results = self.searcher.search(query, top_k=top_k)
        return ResearchToolResult(
            observation=format_search_results(results),
            results_count=len(results),
            source=self.source,
        )


class WebSearchTool:
    name = "web_search"
    description = "Search the public internet for up-to-date information."
    source: SearchSource = "web"

    def __init__(self, web_searcher: WebSearcher):
        self.web_searcher = web_searcher

    def run(self, query: str, top_k: int = 5) -> ResearchToolResult:
        web_results = self.web_searcher.search(query, top_k=top_k)

        is_error = len(web_results) == 1 and web_results[0].get("title") == "Error"
        if is_error:
            return ResearchToolResult(
                observation=f"Web search failed: {web_results[0].get('snippet', 'Unknown error')}.",
                results_count=0,
                source=self.source,
            )
        if not web_results:
            return ResearchToolResult(
                observation="No web results found.",
                results_count=0,
                source=self.source,
            )

        formatted = []
        for i, res in enumerate(web_results, 1):
            formatted.append(
                f"[Web Result {i}]\nTitle: {res['title']}\n"
                f"Link: {res['link']}\nSnippet: {res['snippet']}"
            )
        return ResearchToolResult(
            observation="\n\n".join(formatted),
            results_count=len(web_results),
            source=self.source,
        )


class VisitPageTool:
    name = "visit_page"
    description = "Read the full content of a specific HTTP(S) URL."
    source: SearchSource = "web"

    def __init__(self, web_searcher: WebSearcher):
        self.web_searcher = web_searcher

    def run(self, query: str, top_k: int = 5) -> ResearchToolResult:
        content = self.web_searcher.get_page_content(query)
        return ResearchToolResult(
            observation=f"Content of {query}:\n{content[:15000]}",
            results_count=1 if not content.startswith("Error") else 0,
            source=self.source,
        )


def build_research_tools(
    searcher: LeannSearcher,
    web_searcher: WebSearcher,
    source_policy: SearchSourcePolicy,
) -> dict[str, ResearchTool]:
    """Build the available research-tool registry for a source policy."""
    tools: dict[str, ResearchTool] = {}
    if source_policy in ("local", "both"):
        tools[LeannSearchTool.name] = LeannSearchTool(searcher)
    if source_policy in ("web", "both") and web_searcher.api_key:
        tools[WebSearchTool.name] = WebSearchTool(web_searcher)
        tools[VisitPageTool.name] = VisitPageTool(web_searcher)
    return tools
