#!/usr/bin/env python3
"""
Twitter MCP Reader for LEANN

This module provides functionality to connect to Twitter MCP servers and fetch bookmark data
for indexing in LEANN. It supports various Twitter MCP server implementations and provides
flexible bookmark processing options.
"""

import asyncio
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TwitterMCPReader:
    """
    Reader for Twitter bookmark data via MCP (Model Context Protocol) servers.

    This class connects to Twitter MCP servers to fetch bookmark data and convert it
    into a format suitable for LEANN indexing.
    """

    def __init__(
        self,
        mcp_server_command: str,
        username: Optional[str] = None,
        include_tweet_content: bool = True,
        include_metadata: bool = True,
        max_bookmarks: int = 1000,
    ):
        """
        Initialize the Twitter MCP Reader.

        Args:
            mcp_server_command: Command to start the MCP server (e.g., 'twitter-mcp-server')
            username: Optional Twitter username to filter bookmarks
            include_tweet_content: Whether to include full tweet content
            include_metadata: Whether to include tweet metadata (likes, retweets, etc.)
            max_bookmarks: Maximum number of bookmarks to fetch
        """
        self.mcp_server_command = mcp_server_command
        self.username = username
        self.include_tweet_content = include_tweet_content
        self.include_metadata = include_metadata
        self.max_bookmarks = max_bookmarks
        self.mcp_process: asyncio.subprocess.Process | None = None

    async def start_mcp_server(self):
        """Start the MCP server process."""
        try:
            self.mcp_process = await asyncio.create_subprocess_exec(
                *self.mcp_server_command.split(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            logger.info(f"Started MCP server: {self.mcp_server_command}")
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise

    async def stop_mcp_server(self):
        """Stop the MCP server process."""
        if self.mcp_process:
            self.mcp_process.terminate()
            await self.mcp_process.wait()
            logger.info("Stopped MCP server")

    async def send_mcp_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a request to the MCP server and get response."""
        proc = self.mcp_process
        if proc is None:
            raise RuntimeError("MCP server not started")
        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError("MCP server stdio not available")

        request_json = json.dumps(request) + "\n"
        proc.stdin.write(request_json.encode())
        await proc.stdin.drain()

        response_line = await proc.stdout.readline()
        if not response_line:
            raise RuntimeError("No response from MCP server")

        return json.loads(response_line.decode().strip())

    async def initialize_mcp_connection(self):
        """Initialize the MCP connection."""
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "leann-twitter-reader", "version": "1.0.0"},
            },
        }

        response = await self.send_mcp_request(init_request)
        if "error" in response:
            raise RuntimeError(f"MCP initialization failed: {response['error']}")

        logger.info("MCP connection initialized successfully")

    async def list_available_tools(self) -> list[dict[str, Any]]:
        """List available tools from the MCP server."""
        list_request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}

        response = await self.send_mcp_request(list_request)
        if "error" in response:
            raise RuntimeError(f"Failed to list tools: {response['error']}")

        return response.get("result", {}).get("tools", [])

    async def fetch_twitter_bookmarks(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """
        Fetch Twitter bookmarks using MCP tools.

        Args:
            limit: Maximum number of bookmarks to fetch

        Returns:
            List of bookmark dictionaries
        """
        tools = await self.list_available_tools()
        bookmark_tool = None

        # Look for a tool that can fetch bookmarks
        for tool in tools:
            tool_name = tool.get("name", "").lower()
            if any(keyword in tool_name for keyword in ["bookmark", "saved", "favorite"]):
                bookmark_tool = tool
                break

        if not bookmark_tool:
            raise RuntimeError("No bookmark fetching tool found in MCP server")

        # Prepare tool call parameters
        tool_params = {}
        if limit or self.max_bookmarks:
            tool_params["limit"] = limit or self.max_bookmarks
        if self.username:
            tool_params["username"] = self.username

        fetch_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": bookmark_tool["name"], "arguments": tool_params},
        }

        response = await self.send_mcp_request(fetch_request)
        if "error" in response:
            raise RuntimeError(f"Failed to fetch bookmarks: {response['error']}")

        # Extract bookmarks from response
        result = response.get("result", {})
        if "content" in result and isinstance(result["content"], list):
            content = result["content"][0] if result["content"] else {}
            if "text" in content:
                try:
                    bookmarks = json.loads(content["text"])
                except json.JSONDecodeError:
                    # If not JSON, treat as plain text
                    bookmarks = [{"text": content["text"], "source": "twitter"}]
            else:
                bookmarks = result["content"]
        else:
            bookmarks = result.get("bookmarks", result.get("tweets", [result]))

        return bookmarks if isinstance(bookmarks, list) else [bookmarks]

    def _format_bookmark(self, bookmark: dict[str, Any]) -> str:
        """Format a single bookmark for indexing."""
        # Extract tweet information
        text = bookmark.get("text", bookmark.get("content", ""))
        author = bookmark.get(
            "author", bookmark.get("username", bookmark.get("user", {}).get("username", "Unknown"))
        )
        timestamp = bookmark.get("created_at", bookmark.get("timestamp", ""))
        url = bookmark.get("url", bookmark.get("tweet_url", ""))

        # Extract metadata if available
        likes = bookmark.get("likes", bookmark.get("favorite_count", 0))
        retweets = bookmark.get("retweets", bookmark.get("retweet_count", 0))
        replies = bookmark.get("replies", bookmark.get("reply_count", 0))

        # Build formatted bookmark
        parts = []

        # Header
        parts.append("=== Twitter Bookmark ===")

        if author:
            parts.append(f"Author: @{author}")

        if timestamp:
            # Format timestamp if it's a standard format
            try:
                import datetime

                if "T" in str(timestamp):  # ISO format
                    dt = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    formatted_time = str(timestamp)
                parts.append(f"Date: {formatted_time}")
            except (ValueError, TypeError):
                parts.append(f"Date: {timestamp}")

        if url:
            parts.append(f"URL: {url}")

        # Tweet content
        if text and self.include_tweet_content:
            parts.append("")
            parts.append("Content:")
            parts.append(text)

        # Metadata
        if self.include_metadata and any([likes, retweets, replies]):
            parts.append("")
            parts.append("Engagement:")
            if likes:
                parts.append(f"  Likes: {likes}")
            if retweets:
                parts.append(f"  Retweets: {retweets}")
            if replies:
                parts.append(f"  Replies: {replies}")

        # Extract hashtags and mentions if available
        hashtags = bookmark.get("hashtags", [])
        mentions = bookmark.get("mentions", [])

        if hashtags or mentions:
            parts.append("")
            if hashtags:
                parts.append(f"Hashtags: {', '.join(hashtags)}")
            if mentions:
                parts.append(f"Mentions: {', '.join(mentions)}")

        return "\n".join(parts)

    async def read_twitter_bookmarks(self) -> list[str]:
        """
        Read Twitter bookmark data and return formatted text chunks.

        Returns:
            List of formatted text chunks ready for LEANN indexing
        """
        try:
            await self.start_mcp_server()
            await self.initialize_mcp_connection()

            print(f"Fetching up to {self.max_bookmarks} bookmarks...")
            if self.username:
                print(f"Filtering for user: @{self.username}")

            bookmarks = await self.fetch_twitter_bookmarks()

            if not bookmarks:
                print("No bookmarks found")
                return []

            print(f"Processing {len(bookmarks)} bookmarks...")

            all_texts = []
            processed_count = 0

            for bookmark in bookmarks:
                try:
                    formatted_bookmark = self._format_bookmark(bookmark)
                    if formatted_bookmark.strip():
                        all_texts.append(formatted_bookmark)
                        processed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to format bookmark: {e}")
                    continue

            print(f"Successfully processed {processed_count} bookmarks")
            return all_texts

        finally:
            await self.stop_mcp_server()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_mcp_server()
        await self.initialize_mcp_connection()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_mcp_server()
