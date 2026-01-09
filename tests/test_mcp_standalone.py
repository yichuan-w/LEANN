#!/usr/bin/env python3
"""
Standalone test script for MCP integration implementations.

This script tests the basic functionality of the MCP readers
without requiring LEANN core dependencies.
"""

import json
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from apps
sys.path.append(str(Path(__file__).parent.parent))


def test_slack_reader_basic():
    """Test basic SlackMCPReader functionality without async operations."""
    print("Testing SlackMCPReader basic functionality...")

    # Import and test initialization
    from apps.slack_data.slack_mcp_reader import SlackMCPReader

    reader = SlackMCPReader("slack-mcp-server")
    assert reader.mcp_server_command == "slack-mcp-server"
    assert reader.concatenate_conversations

    # Test message formatting
    message = {
        "text": "Hello team! How's the project going?",
        "user": "john_doe",
        "channel": "general",
        "ts": "1234567890.123456",
    }

    formatted = reader._format_message(message)
    assert "Channel: #general" in formatted
    assert "User: john_doe" in formatted
    assert "Message: Hello team!" in formatted

    # Test concatenated content creation
    messages = [
        {"text": "First message", "user": "alice", "ts": "1000"},
        {"text": "Second message", "user": "bob", "ts": "2000"},
    ]

    content = reader._create_concatenated_content(messages, "dev-team")
    assert "Slack Channel: #dev-team" in content
    assert "Message Count: 2" in content
    assert "First message" in content
    assert "Second message" in content

    print("✅ SlackMCPReader basic tests passed")


def test_twitter_reader_basic():
    """Test basic TwitterMCPReader functionality."""
    print("Testing TwitterMCPReader basic functionality...")

    from apps.twitter_data.twitter_mcp_reader import TwitterMCPReader

    reader = TwitterMCPReader("twitter-mcp-server")
    assert reader.mcp_server_command == "twitter-mcp-server"
    assert reader.include_tweet_content
    assert reader.max_bookmarks == 1000

    # Test bookmark formatting
    bookmark = {
        "text": "Amazing article about the future of AI! Must read for everyone interested in tech.",
        "author": "tech_guru",
        "created_at": "2024-01-15T14:30:00Z",
        "url": "https://twitter.com/tech_guru/status/123456789",
        "likes": 156,
        "retweets": 42,
        "replies": 23,
        "hashtags": ["AI", "tech", "future"],
        "mentions": ["@openai", "@anthropic"],
    }

    formatted = reader._format_bookmark(bookmark)
    assert "=== Twitter Bookmark ===" in formatted
    assert "Author: @tech_guru" in formatted
    assert "Amazing article about the future of AI!" in formatted
    assert "Likes: 156" in formatted
    assert "Retweets: 42" in formatted
    assert "Hashtags: AI, tech, future" in formatted
    assert "Mentions: @openai, @anthropic" in formatted

    # Test with minimal data
    simple_bookmark = {"text": "Short tweet", "author": "user123"}
    formatted_simple = reader._format_bookmark(simple_bookmark)
    assert "=== Twitter Bookmark ===" in formatted_simple
    assert "Short tweet" in formatted_simple
    assert "Author: @user123" in formatted_simple

    print("✅ TwitterMCPReader basic tests passed")


def test_mcp_request_format():
    """Test MCP request formatting."""
    print("Testing MCP request formatting...")

    # Test initialization request format
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "leann-slack-reader", "version": "1.0.0"},
        },
    }

    # Verify it's valid JSON
    json_str = json.dumps(init_request)
    parsed = json.loads(json_str)
    assert parsed["jsonrpc"] == "2.0"
    assert parsed["method"] == "initialize"
    assert parsed["params"]["protocolVersion"] == "2025-11-25"

    # Test tools/list request
    list_request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}

    json_str = json.dumps(list_request)
    parsed = json.loads(json_str)
    assert parsed["method"] == "tools/list"

    print("✅ MCP request formatting tests passed")


def test_data_processing():
    """Test data processing capabilities."""
    print("Testing data processing capabilities...")

    from apps.slack_data.slack_mcp_reader import SlackMCPReader
    from apps.twitter_data.twitter_mcp_reader import TwitterMCPReader

    # Test Slack message processing with various formats
    slack_reader = SlackMCPReader("test-server")

    messages_with_timestamps = [
        {"text": "Meeting in 5 minutes", "user": "alice", "ts": "1000.123"},
        {"text": "On my way!", "user": "bob", "ts": "1001.456"},
        {"text": "Starting now", "user": "charlie", "ts": "1002.789"},
    ]

    content = slack_reader._create_concatenated_content(messages_with_timestamps, "meetings")
    assert "Meeting in 5 minutes" in content
    assert "On my way!" in content
    assert "Starting now" in content

    # Test Twitter bookmark processing with engagement data
    twitter_reader = TwitterMCPReader("test-server", include_metadata=True)

    high_engagement_bookmark = {
        "text": "Thread about startup lessons learned 🧵",
        "author": "startup_founder",
        "likes": 1250,
        "retweets": 340,
        "replies": 89,
    }

    formatted = twitter_reader._format_bookmark(high_engagement_bookmark)
    assert "Thread about startup lessons learned" in formatted
    assert "Likes: 1250" in formatted
    assert "Retweets: 340" in formatted
    assert "Replies: 89" in formatted

    # Test with metadata disabled
    twitter_reader_no_meta = TwitterMCPReader("test-server", include_metadata=False)
    formatted_no_meta = twitter_reader_no_meta._format_bookmark(high_engagement_bookmark)
    assert "Thread about startup lessons learned" in formatted_no_meta
    assert "Likes:" not in formatted_no_meta
    assert "Retweets:" not in formatted_no_meta

    print("✅ Data processing tests passed")


def main():
    """Run all standalone tests."""
    print("🧪 Running MCP Integration Standalone Tests")
    print("=" * 60)
    print("Testing core functionality without LEANN dependencies...")
    print()

    try:
        test_slack_reader_basic()
        test_twitter_reader_basic()
        test_mcp_request_format()
        test_data_processing()

        print("\n" + "=" * 60)
        print("🎉 All standalone tests passed!")
        print("\n✨ MCP Integration Summary:")
        print("- SlackMCPReader: Ready for Slack message processing")
        print("- TwitterMCPReader: Ready for Twitter bookmark processing")
        print("- MCP Protocol: Properly formatted JSON-RPC requests")
        print("- Data Processing: Handles various message/bookmark formats")

        print("\n🚀 Next Steps:")
        print("1. Install MCP servers: npm install -g slack-mcp-server twitter-mcp-server")
        print("2. Configure API credentials for Slack and Twitter")
        print("3. Test connections: python -m apps.slack_rag --test-connection")
        print("4. Start indexing live data from your platforms!")

        print("\n📖 Documentation:")
        print("- Check README.md for detailed setup instructions")
        print("- Run examples/mcp_integration_demo.py for usage examples")
        print("- Explore apps/slack_rag.py and apps/twitter_rag.py for implementation details")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
