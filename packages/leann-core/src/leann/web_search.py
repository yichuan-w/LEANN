import logging
import os
from typing import Any
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


class WebSearcher:
    """Helper class for performing web searches via Serper API."""

    def __init__(
        self,
        api_key: str | None = None,
        jina_api_key: str | None = None,
        timeout_seconds: float = 10.0,
    ):
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        self.jina_api_key = jina_api_key or os.getenv("JINA_API_KEY")
        self.timeout_seconds = timeout_seconds
        if not self.api_key:
            logger.warning("No SERPER_API_KEY found. Web search will not be available.")

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.api_key:
            return [{"title": "Error", "link": "", "snippet": "Missing SERPER_API_KEY env var"}]

        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}

        try:
            response = requests.post(
                url,
                headers=headers,
                json={"q": query, "num": top_k},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()

            results = []

            if "organic" in data:
                for item in data["organic"][:top_k]:
                    results.append(
                        {
                            "title": item.get("title", ""),
                            "link": item.get("link", ""),
                            "snippet": item.get("snippet", ""),
                        }
                    )

            return results
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return [{"title": "Error", "link": "", "snippet": f"Web search failed: {e!s}"}]

    def get_page_content(self, url: str) -> str:
        """Fetch page content using Jina AI Reader (https://jina.ai/reader)."""
        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"http", "https"}:
            return "Error fetching content: only http and https URLs can be fetched."

        jina_url = f"https://r.jina.ai/{url}"

        headers = {"X-Return-Format": "markdown"}
        if self.jina_api_key:
            headers["Authorization"] = f"Bearer {self.jina_api_key}"

        try:
            logger.info(f"Fetching content from: {url}")
            response = requests.get(jina_url, headers=headers, timeout=self.timeout_seconds)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Failed to fetch content from {url}: {e}")
            return f"Error fetching content: {e!s}"
