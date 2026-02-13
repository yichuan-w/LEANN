#!/usr/bin/env python3
"""
Twitter RAG Application with MCP Support

This application enables RAG (Retrieval-Augmented Generation) on Twitter bookmarks
by connecting to Twitter MCP servers to fetch live data and index it in LEANN.

Usage:
    python -m apps.twitter_rag --mcp-server "twitter-mcp-server" --query "What articles did I bookmark about AI?"
"""

import argparse
import asyncio
from typing import Any

from apps.base_rag_example import BaseRAGExample
from apps.twitter_data.twitter_mcp_reader import TwitterMCPReader


class TwitterMCPRAG(BaseRAGExample):
    """
    RAG application for Twitter bookmarks via MCP servers.

    This class provides a complete RAG pipeline for Twitter bookmark data, including
    MCP server connection, data fetching, indexing, and interactive chat.
    """

    def __init__(self):
        super().__init__(
            name="Twitter MCP RAG",
            description="RAG application for Twitter bookmarks via MCP servers",
            default_index_name="twitter_bookmarks",
        )

    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        """Add Twitter MCP-specific arguments."""
        parser.add_argument(
            "--mcp-server",
            type=str,
            required=True,
            help="Command to start the Twitter MCP server (e.g., 'twitter-mcp-server' or 'npx twitter-mcp-server')",
        )

        parser.add_argument(
            "--username", type=str, help="Twitter username to filter bookmarks (without @)"
        )

        parser.add_argument(
            "--max-bookmarks",
            type=int,
            default=1000,
            help="Maximum number of bookmarks to fetch (default: 1000)",
        )

        parser.add_argument(
            "--no-tweet-content",
            action="store_true",
            help="Exclude tweet content, only include metadata",
        )

        parser.add_argument(
            "--no-metadata",
            action="store_true",
            help="Exclude engagement metadata (likes, retweets, etc.)",
        )

        parser.add_argument(
            "--test-connection",
            action="store_true",
            help="Test MCP server connection and list available tools without indexing",
        )

    async def test_mcp_connection(self, args) -> bool:
        """Test the MCP server connection and display available tools."""
        print(f"Testing connection to MCP server: {args.mcp_server}")

        try:
            reader = TwitterMCPReader(
                mcp_server_command=args.mcp_server,
                username=args.username,
                include_tweet_content=not args.no_tweet_content,
                include_metadata=not args.no_metadata,
                max_bookmarks=args.max_bookmarks,
            )

            async with reader:
                tools = await reader.list_available_tools()

                print("\n✅ Successfully connected to MCP server!")
                print(f"Available tools ({len(tools)}):")

                for i, tool in enumerate(tools, 1):
                    name = tool.get("name", "Unknown")
                    description = tool.get("description", "No description available")
                    print(f"\n{i}. {name}")
                    print(
                        f"   Description: {description[:100]}{'...' if len(description) > 100 else ''}"
                    )

                    # Show input schema if available
                    schema = tool.get("inputSchema", {})
                    if schema.get("properties"):
                        props = list(schema["properties"].keys())[:3]  # Show first 3 properties
                        print(
                            f"   Parameters: {', '.join(props)}{'...' if len(schema['properties']) > 3 else ''}"
                        )

                return True

        except Exception as e:
            print(f"\n❌ Failed to connect to MCP server: {e}")
            print("\nTroubleshooting tips:")
            print("1. Make sure the Twitter MCP server is installed and accessible")
            print("2. Check if the server command is correct")
            print("3. Ensure you have proper Twitter API credentials configured")
            print("4. Verify your Twitter account has bookmarks to fetch")
            print("5. Try running the MCP server command directly to test it")
            return False

    async def load_data(self, args) -> list[dict[str, Any]]:
        """Load Twitter bookmarks via MCP server."""
        print(f"Connecting to Twitter MCP server: {args.mcp_server}")

        if args.username:
            print(f"Username filter: @{args.username}")

        print(f"Max bookmarks: {args.max_bookmarks}")
        print(f"Include tweet content: {not args.no_tweet_content}")
        print(f"Include metadata: {not args.no_metadata}")

        try:
            reader = TwitterMCPReader(
                mcp_server_command=args.mcp_server,
                username=args.username,
                include_tweet_content=not args.no_tweet_content,
                include_metadata=not args.no_metadata,
                max_bookmarks=args.max_bookmarks,
            )

            texts = await reader.read_twitter_bookmarks()

            if not texts:
                print("❌ No bookmarks found! This could mean:")
                print("- You don't have any bookmarks on Twitter")
                print("- The MCP server couldn't access your bookmarks")
                print("- Authentication issues with Twitter API")
                print("- The username filter didn't match any bookmarks")
                return []

            print(f"✅ Successfully loaded {len(texts)} bookmarks from Twitter")

            # Show sample of what was loaded
            if texts:
                sample_text = texts[0][:300] + "..." if len(texts[0]) > 300 else texts[0]
                print("\nSample bookmark:")
                print("-" * 50)
                print(sample_text)
                print("-" * 50)

            # Convert strings to dict format expected by base class
            return [{"text": text, "metadata": {"source": "twitter"}} for text in texts]

        except Exception as e:
            print(f"❌ Error loading Twitter bookmarks: {e}")
            print("\nThis might be due to:")
            print("- MCP server connection issues")
            print("- Twitter API authentication problems")
            print("- Network connectivity issues")
            print("- Rate limiting from Twitter API")
            raise

    async def run(self):
        """Main entry point with MCP connection testing."""
        args = self.parser.parse_args()

        # Test connection if requested
        if args.test_connection:
            success = await self.test_mcp_connection(args)
            if not success:
                return
            print(
                "\n🎉 MCP server is working! You can now run without --test-connection to start indexing."
            )
            return

        # Run the standard RAG pipeline
        await super().run()


async def main():
    """Main entry point for the Twitter MCP RAG application."""
    app = TwitterMCPRAG()
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())
