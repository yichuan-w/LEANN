#!/usr/bin/env python3
"""
Slack MCP Reader for LEANN

This module provides functionality to connect to Slack MCP servers and fetch message data
for indexing in LEANN. It supports various Slack MCP server implementations and provides
flexible message processing options.
"""

import ast
import asyncio
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SlackMCPReader:
    """
    Reader for Slack data via MCP (Model Context Protocol) servers.

    This class connects to Slack MCP servers to fetch message data and convert it
    into a format suitable for LEANN indexing.
    """

    def __init__(
        self,
        mcp_server_command: str,
        workspace_name: Optional[str] = None,
        concatenate_conversations: bool = True,
        max_messages_per_conversation: int = 100,
        max_retries: int = 5,
        retry_delay: float = 2.0,
    ):
        """
        Initialize the Slack MCP Reader.

        Args:
            mcp_server_command: Command to start the MCP server (e.g., 'slack-mcp-server')
            workspace_name: Optional workspace name to filter messages
            concatenate_conversations: Whether to group messages by channel/thread
            max_messages_per_conversation: Maximum messages to include per conversation
            max_retries: Maximum number of retries for failed operations
            retry_delay: Initial delay between retries in seconds
        """
        self.mcp_server_command = mcp_server_command
        self.workspace_name = workspace_name
        self.concatenate_conversations = concatenate_conversations
        self.max_messages_per_conversation = max_messages_per_conversation
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.mcp_process = None

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
        if not self.mcp_process:
            raise RuntimeError("MCP server not started")

        request_json = json.dumps(request) + "\n"
        self.mcp_process.stdin.write(request_json.encode())
        await self.mcp_process.stdin.drain()

        response_line = await self.mcp_process.stdout.readline()
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
                "clientInfo": {"name": "leann-slack-reader", "version": "1.0.0"},
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

    def _is_cache_sync_error(self, error: dict) -> bool:
        """Check if the error is related to users cache not being ready."""
        if isinstance(error, dict):
            message = error.get("message", "").lower()
            return (
                "users cache is not ready" in message or "sync process is still running" in message
            )
        return False

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry a function with exponential backoff, especially for cache sync issues."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                # Check if this is a cache sync error
                error_dict = {}
                if hasattr(e, "args") and e.args and isinstance(e.args[0], dict):
                    error_dict = e.args[0]
                elif "Failed to fetch messages" in str(e):
                    # Try to extract error from the exception message
                    import re

                    match = re.search(r"'error':\s*(\{[^}]+\})", str(e))
                    if match:
                        try:
                            error_dict = ast.literal_eval(match.group(1))
                        except (ValueError, SyntaxError):
                            pass
                    else:
                        # Try alternative format
                        match = re.search(r"Failed to fetch messages:\s*(\{[^}]+\})", str(e))
                        if match:
                            try:
                                error_dict = ast.literal_eval(match.group(1))
                            except (ValueError, SyntaxError):
                                pass

                if self._is_cache_sync_error(error_dict):
                    if attempt < self.max_retries:
                        delay = self.retry_delay * (2**attempt)  # Exponential backoff
                        logger.info(
                            f"Cache sync not ready, waiting {delay:.1f}s before retry {attempt + 1}/{self.max_retries}"
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.warning(
                            f"Cache sync still not ready after {self.max_retries} retries, giving up"
                        )
                        break
                else:
                    # Not a cache sync error, don't retry
                    break

        # If we get here, all retries failed or it's not a retryable error
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Unexpected error: no exception captured during retry loop")

    async def fetch_slack_messages(
        self, channel: Optional[str] = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """
        Fetch Slack messages using MCP tools with retry logic for cache sync issues.

        Args:
            channel: Optional channel name to filter messages
            limit: Maximum number of messages to fetch

        Returns:
            List of message dictionaries
        """
        return await self._retry_with_backoff(self._fetch_slack_messages_impl, channel, limit)

    async def _fetch_slack_messages_impl(
        self, channel: Optional[str] = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """
        Internal implementation of fetch_slack_messages without retry logic.
        """
        # This is a generic implementation - specific MCP servers may have different tool names
        # Common tool names might be: 'get_messages', 'list_messages', 'fetch_channel_history'

        tools = await self.list_available_tools()
        logger.info(f"Available tools: {[tool.get('name') for tool in tools]}")
        message_tool = None

        # Look for a tool that can fetch messages - prioritize conversations_history
        message_tool = None

        # First, try to find conversations_history specifically
        for tool in tools:
            tool_name = tool.get("name", "").lower()
            if "conversations_history" in tool_name:
                message_tool = tool
                logger.info(f"Found conversations_history tool: {tool}")
                break

        # If not found, look for other message-fetching tools
        if not message_tool:
            for tool in tools:
                tool_name = tool.get("name", "").lower()
                if any(
                    keyword in tool_name
                    for keyword in ["conversations_search", "message", "history"]
                ):
                    message_tool = tool
                    break

        if not message_tool:
            raise RuntimeError("No message fetching tool found in MCP server")

        # Prepare tool call parameters
        tool_params = {"limit": "180d"}  # Use 180 days to get older messages
        if channel:
            # For conversations_history, use channel_id parameter
            if message_tool["name"] == "conversations_history":
                tool_params["channel_id"] = channel
            else:
                # Try common parameter names for channel specification
                for param_name in ["channel", "channel_id", "channel_name"]:
                    tool_params[param_name] = channel
                    break

        logger.info(f"Tool parameters: {tool_params}")

        fetch_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": message_tool["name"], "arguments": tool_params},
        }

        response = await self.send_mcp_request(fetch_request)
        if "error" in response:
            raise RuntimeError(f"Failed to fetch messages: {response['error']}")

        # Extract messages from response - format may vary by MCP server
        result = response.get("result", {})
        if "content" in result and isinstance(result["content"], list):
            # Some MCP servers return content as a list
            content = result["content"][0] if result["content"] else {}
            if "text" in content:
                try:
                    messages = json.loads(content["text"])
                except json.JSONDecodeError:
                    # If not JSON, try to parse as CSV format (Slack MCP server format)
                    text_content = content.get("text", "")
                    messages = self._parse_csv_messages(
                        text_content if text_content else "", channel or "unknown"
                    )
            else:
                messages = result["content"]
        else:
            # Direct message format
            messages = result.get("messages", [result])

        return messages if isinstance(messages, list) else [messages]

    def _parse_csv_messages(self, csv_text: str, channel: str) -> list[dict[str, Any]]:
        """Parse CSV format messages from Slack MCP server."""
        import csv
        import io

        messages = []
        try:
            # Split by lines and process each line as a CSV row
            lines = csv_text.strip().split("\n")
            if not lines:
                return messages

            # Skip header line if it exists
            start_idx = 0
            if lines[0].startswith("MsgID,UserID,UserName"):
                start_idx = 1

            for line in lines[start_idx:]:
                if not line.strip():
                    continue

                # Parse CSV line
                reader = csv.reader(io.StringIO(line))
                try:
                    row = next(reader)
                    if len(row) >= 7:  # Ensure we have enough columns
                        message = {
                            "ts": row[0],
                            "user": row[1],
                            "username": row[2],
                            "real_name": row[3],
                            "channel": row[4],
                            "thread_ts": row[5],
                            "text": row[6],
                            "time": row[7] if len(row) > 7 else "",
                            "reactions": row[8] if len(row) > 8 else "",
                            "cursor": row[9] if len(row) > 9 else "",
                        }
                        messages.append(message)
                except Exception as e:
                    logger.warning(f"Failed to parse CSV line: {line[:100]}... Error: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Failed to parse CSV messages: {e}")
            # Fallback: treat entire text as one message
            messages = [{"text": csv_text, "channel": channel or "unknown"}]

        return messages

    def _format_message(self, message: dict[str, Any]) -> str:
        """Format a single message for indexing."""
        text = message.get("text", "")
        user = message.get("user", message.get("username", "Unknown"))
        channel = message.get("channel", message.get("channel_name", "Unknown"))
        timestamp = message.get("ts", message.get("timestamp", ""))

        # Format timestamp if available
        formatted_time = ""
        if timestamp:
            try:
                import datetime

                if isinstance(timestamp, str) and "." in timestamp:
                    dt = datetime.datetime.fromtimestamp(float(timestamp))
                    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(timestamp, (int, float)):
                    dt = datetime.datetime.fromtimestamp(timestamp)
                    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    formatted_time = str(timestamp)
            except (ValueError, TypeError):
                formatted_time = str(timestamp)

        # Build formatted message
        parts = []
        if channel:
            parts.append(f"Channel: #{channel}")
        if user:
            parts.append(f"User: {user}")
        if formatted_time:
            parts.append(f"Time: {formatted_time}")
        if text:
            parts.append(f"Message: {text}")

        return "\n".join(parts)

    def _create_concatenated_content(self, messages: list[dict[str, Any]], channel: str) -> str:
        """Create concatenated content from multiple messages in a channel."""
        if not messages:
            return ""

        # Sort messages by timestamp if available
        try:
            messages.sort(key=lambda x: float(x.get("ts", x.get("timestamp", 0))))
        except (ValueError, TypeError):
            pass  # Keep original order if timestamps aren't numeric

        # Limit messages per conversation
        if len(messages) > self.max_messages_per_conversation:
            messages = messages[-self.max_messages_per_conversation :]

        # Create header
        content_parts = [
            f"Slack Channel: #{channel}",
            f"Message Count: {len(messages)}",
            f"Workspace: {self.workspace_name or 'Unknown'}",
            "=" * 50,
            "",
        ]

        # Add messages
        for message in messages:
            formatted_msg = self._format_message(message)
            if formatted_msg.strip():
                content_parts.append(formatted_msg)
                content_parts.append("-" * 30)
                content_parts.append("")

        return "\n".join(content_parts)

    async def get_all_channels(self) -> list[str]:
        """Get list of all available channels."""
        try:
            channels_list_request = {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "channels_list", "arguments": {}},
            }
            channels_response = await self.send_mcp_request(channels_list_request)
            if "result" in channels_response:
                result = channels_response["result"]
                if "content" in result and isinstance(result["content"], list):
                    content = result["content"][0] if result["content"] else {}
                    if "text" in content:
                        # Parse the channels from the response
                        channels = []
                        lines = content["text"].split("\n")
                        for line in lines:
                            if line.strip() and ("#" in line or "C" in line[:10]):
                                # Extract channel ID or name
                                parts = line.split()
                                for part in parts:
                                    if part.startswith("C") and len(part) > 5:
                                        channels.append(part)
                                    elif part.startswith("#"):
                                        channels.append(part[1:])  # Remove #
                        logger.info(f"Found {len(channels)} channels: {channels}")
                        return channels
            return []
        except Exception as e:
            logger.warning(f"Failed to get channels list: {e}")
            return []

    async def read_slack_data(self, channels: Optional[list[str]] = None) -> list[str]:
        """
        Read Slack data and return formatted text chunks.

        Args:
            channels: Optional list of channel names to fetch. If None, fetches from all available channels.

        Returns:
            List of formatted text chunks ready for LEANN indexing
        """
        try:
            await self.start_mcp_server()
            await self.initialize_mcp_connection()

            all_texts = []

            if channels:
                # Fetch specific channels
                for channel in channels:
                    try:
                        messages = await self.fetch_slack_messages(channel=channel, limit=1000)
                        if messages:
                            if self.concatenate_conversations:
                                text_content = self._create_concatenated_content(messages, channel)
                                if text_content.strip():
                                    all_texts.append(text_content)
                            else:
                                # Process individual messages
                                for message in messages:
                                    formatted_msg = self._format_message(message)
                                    if formatted_msg.strip():
                                        all_texts.append(formatted_msg)
                    except Exception as e:
                        logger.warning(f"Failed to fetch messages from channel {channel}: {e}")
                        continue
            else:
                # Fetch from all available channels
                logger.info("Fetching from all available channels...")
                all_channels = await self.get_all_channels()

                if not all_channels:
                    # Fallback to common channel names if we can't get the list
                    all_channels = ["general", "random", "announcements", "C0GN5BX0F"]
                    logger.info(f"Using fallback channels: {all_channels}")

                for channel in all_channels:
                    try:
                        logger.info(f"Searching channel: {channel}")
                        messages = await self.fetch_slack_messages(channel=channel, limit=1000)
                        if messages:
                            if self.concatenate_conversations:
                                text_content = self._create_concatenated_content(messages, channel)
                                if text_content.strip():
                                    all_texts.append(text_content)
                            else:
                                # Process individual messages
                                for message in messages:
                                    formatted_msg = self._format_message(message)
                                    if formatted_msg.strip():
                                        all_texts.append(formatted_msg)
                    except Exception as e:
                        logger.warning(f"Failed to fetch messages from channel {channel}: {e}")
                        continue

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
