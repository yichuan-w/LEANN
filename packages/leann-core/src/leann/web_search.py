import json
import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)


class WebSearcher:
    """Helper class for performing web searches via Serper API."""

    def __init__(self, api_key: str | None = None, jina_api_key: str | None = None):
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        self.jina_api_key = jina_api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            logger.warning("No SERPER_API_KEY found. Web seach will fail")

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.api_key:
            return [{"title": "Error", "link": "", "snippet": "Missing SERPER_API_KEY env var"}]

        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "num": top_k})
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}

        try:
            response = requests.request("POST", url, headers=headers, data=payload)
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
            return [{"title": "Error", "link": "", "snippet": f"Web Search failed:{e!s}"}]

    def get_page_content(self, url: str) -> str:
        """
        Fetch page content using Jina AI (https://jina.ai/reader).
        """
        jina_url = f"https://r.jina.ai/{url}"

        # Use Jina API key if available for higher limits, but works without it
        headers = {"X-Return-Format": "markdown"}
        if self.jina_api_key:
            headers["Authorization"] = f"Bearer {self.jina_api_key}"

        try:
            logger.info(f"Fetching content from: {url}")
            response = requests.get(jina_url, headers=headers)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Failed to fetch content from {url}: {e}")
            return f"Error fetching content: {e!s}"
