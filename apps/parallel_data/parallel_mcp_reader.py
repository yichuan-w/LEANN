"""Read source-attributed web documents using anonymous Parallel Search MCP."""

import asyncio
import json
import logging
from typing import Any
from uuid import uuid4

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

logger = logging.getLogger(__name__)
MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "LEANN-Parallel-MCP/0.1 (https://github.com/StarTrail-org/LEANN)"


def _tool_payload(result: Any) -> dict[str, Any]:
    """Decode structured MCP output, rejecting tool errors and invalid responses."""
    if result.isError:
        detail = "\n".join(getattr(item, "text", "") for item in result.content)
        raise RuntimeError(f"Parallel MCP tool failed: {detail}")
    if result.structuredContent is not None:
        payload = result.structuredContent
    else:
        text = "\n".join(getattr(item, "text", "") for item in result.content)
        try:
            payload = json.loads(text)
        except ValueError as exc:
            raise RuntimeError("Parallel MCP returned invalid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise RuntimeError("Parallel MCP response is missing its results list")
    for warning in payload.get("warnings") or []:
        logger.warning("Parallel MCP: %s", warning.get("message", warning))
    return payload


class ParallelMCPReader:
    """Search the web and optionally fetch excerpts for local LEANN ingestion.

    This reader sends objectives and queries to a remote service. It does not
    read saved credentials, load dotenv files, or send a Parallel API key.
    """

    def __init__(self, timeout: float = 90.0):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout
        self.session_id = str(uuid4())

    async def read_web_data(
        self,
        objective: str,
        search_queries: list[str],
        max_results: int = 5,
        fetch: bool = False,
    ) -> list[dict[str, Any]]:
        """Return text/metadata documents, bounded to 1-20 selected search results.

        With fetch enabled, fetch the selected URLs in one call. Per-URL fetch
        failures retain their search excerpts and emit warnings. Tool/transport
        failures propagate; the reader does not retry rate-limited requests.
        The timeout bounds the entire operation, including connection setup.
        """
        if (
            not objective.strip()
            or not search_queries
            or any(not query.strip() for query in search_queries)
        ):
            raise ValueError("An objective and nonempty search queries are required")
        if not 1 <= max_results <= 20:
            raise ValueError("max_results must be between 1 and 20")
        return await asyncio.wait_for(
            self._read(objective, search_queries, max_results, fetch), timeout=self.timeout
        )

    async def _read(
        self, objective: str, search_queries: list[str], max_results: int, fetch: bool
    ) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT}, timeout=self.timeout
        ) as client:
            async with streamable_http_client(MCP_URL, http_client=client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = {tool.name for tool in (await session.list_tools()).tools}
                    required = {"web_search", "web_fetch"} if fetch else {"web_search"}
                    if not required.issubset(tools):
                        raise RuntimeError(
                            f"Parallel MCP is missing tools: {sorted(required - tools)}"
                        )
                    search = _tool_payload(
                        await session.call_tool(
                            "web_search",
                            {
                                "objective": objective,
                                "search_queries": search_queries,
                                "session_id": self.session_id,
                            },
                        )
                    )
                    # Deduplicate before selecting URLs for the optional fetch call.
                    selected = {}
                    for item in search["results"]:
                        selected.setdefault(item["url"], item)
                        if len(selected) >= max_results:
                            break
                    fetched = {}
                    if fetch and selected:
                        extraction = _tool_payload(
                            await session.call_tool(
                                "web_fetch",
                                {
                                    "urls": list(selected),
                                    "search_queries": search_queries,
                                    "session_id": self.session_id,
                                },
                            )
                        )
                        fetched = {item["url"]: item for item in extraction["results"]}
                        for error in extraction.get("errors", []):
                            logger.warning(
                                "Fetch failed for %s (%s); retaining search excerpts",
                                error.get("url"),
                                error.get("error_type"),
                            )
        documents = []
        for url, search_item in selected.items():
            item = fetched.get(url, search_item)
            text = "\n\n".join(item.get("excerpts", [])).strip()
            if not text:
                text = "\n\n".join(search_item.get("excerpts", [])).strip()
            if text:
                documents.append(
                    {
                        "text": text,
                        "metadata": {
                            "url": url,
                            "title": item.get("title") or search_item.get("title") or url,
                            "publish_date": item.get("publish_date")
                            or search_item.get("publish_date"),
                            "source": "parallel-mcp",
                        },
                    }
                )
        return documents
