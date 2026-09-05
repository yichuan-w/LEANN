"""
Tests for MiniMax provider integration.

These tests validate MiniMax provider settings, chat class, and factory integration.
We import from leann.settings and leann.chat directly to avoid triggering
the full leann.__init__ import chain which requires C++ backend builds.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add the leann-core source to sys.path so we can import submodules
# without triggering __init__.py's backend imports.
_LEANN_SRC = os.path.join(os.path.dirname(__file__), "..", "packages", "leann-core", "src")
if _LEANN_SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_LEANN_SRC))

# Prevent leann.__init__ from running its heavy imports by pre-registering
# a lightweight stub in sys.modules (if not already present).
if "leann" not in sys.modules:
    import types

    _stub = types.ModuleType("leann")
    _stub.__path__ = [os.path.join(os.path.abspath(_LEANN_SRC), "leann")]
    sys.modules["leann"] = _stub

# Now we can safely import the modules we actually test.
from leann.settings import (  # noqa: E402
    resolve_minimax_api_key,
    resolve_minimax_base_url,
)


class TestMiniMaxSettings:
    """Test MiniMax settings resolver functions."""

    def test_resolve_minimax_api_key_explicit(self):
        assert resolve_minimax_api_key("test-key") == "test-key"

    def test_resolve_minimax_api_key_from_env(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key"}):
            assert resolve_minimax_api_key() == "env-key"

    def test_resolve_minimax_api_key_none(self):
        with patch.dict(os.environ, {}, clear=True):
            assert resolve_minimax_api_key() is None

    def test_resolve_minimax_base_url_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert resolve_minimax_base_url() == "https://api.minimax.io/v1"

    def test_resolve_minimax_base_url_explicit(self):
        assert resolve_minimax_base_url("https://custom.url/v1") == "https://custom.url/v1"

    def test_resolve_minimax_base_url_from_env(self):
        with patch.dict(os.environ, {"MINIMAX_BASE_URL": "https://env.url/v1"}):
            assert resolve_minimax_base_url() == "https://env.url/v1"

    def test_resolve_minimax_base_url_leann_env(self):
        with patch.dict(os.environ, {"LEANN_MINIMAX_BASE_URL": "https://leann.url/v1"}):
            assert resolve_minimax_base_url() == "https://leann.url/v1"

    def test_resolve_minimax_base_url_strips_trailing_slash(self):
        assert resolve_minimax_base_url("https://api.minimax.io/v1/") == "https://api.minimax.io/v1"


class TestMiniMaxChat:
    """Test MiniMaxChat class."""

    def test_init_requires_api_key(self):
        from leann.chat import MiniMaxChat

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="MiniMax API key is required"):
                MiniMaxChat(api_key=None)

    @patch("openai.OpenAI")
    def test_init_with_api_key(self, mock_openai_cls):
        from leann.chat import MiniMaxChat

        chat = MiniMaxChat(api_key="test-key")
        assert chat.model == "MiniMax-M2.5"
        assert chat.api_key == "test-key"
        assert chat.base_url == "https://api.minimax.io/v1"
        mock_openai_cls.assert_called_once_with(
            api_key="test-key", base_url="https://api.minimax.io/v1"
        )

    @patch("openai.OpenAI")
    def test_init_custom_model(self, mock_openai_cls):
        from leann.chat import MiniMaxChat

        chat = MiniMaxChat(model="MiniMax-M2.5-highspeed", api_key="test-key")
        assert chat.model == "MiniMax-M2.5-highspeed"

    @patch("openai.OpenAI")
    def test_init_custom_base_url(self, mock_openai_cls):
        from leann.chat import MiniMaxChat

        chat = MiniMaxChat(api_key="test-key", base_url="https://api.minimaxi.com/v1")
        assert chat.base_url == "https://api.minimaxi.com/v1"

    @patch("openai.OpenAI")
    def test_ask_returns_response(self, mock_openai_cls):
        from leann.chat import MiniMaxChat

        # Mock the response chain
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello from MiniMax!"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.total_tokens = 100
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 50
        mock_client.chat.completions.create.return_value = mock_response

        chat = MiniMaxChat(api_key="test-key")
        result = chat.ask("Hello")

        assert result == "Hello from MiniMax!"
        mock_client.chat.completions.create.assert_called_once()

    @patch("openai.OpenAI")
    def test_ask_with_kwargs(self, mock_openai_cls):
        from leann.chat import MiniMaxChat

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.total_tokens = 50
        mock_response.usage.prompt_tokens = 25
        mock_response.usage.completion_tokens = 25
        mock_client.chat.completions.create.return_value = mock_response

        chat = MiniMaxChat(api_key="test-key")
        chat.ask("Hello", temperature=0.5, max_tokens=500, top_p=0.9)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 500
        assert call_kwargs["top_p"] == 0.9

    @patch("openai.OpenAI")
    def test_ask_handles_error(self, mock_openai_cls):
        from leann.chat import MiniMaxChat

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        chat = MiniMaxChat(api_key="test-key")
        result = chat.ask("Hello")

        assert "Error" in result
        assert "MiniMax" in result


class TestGetLLMFactory:
    """Test get_llm factory function with minimax type."""

    @patch("openai.OpenAI")
    def test_get_llm_minimax(self, mock_openai_cls):
        from leann.chat import MiniMaxChat, get_llm

        llm = get_llm({"type": "minimax", "api_key": "test-key"})
        assert isinstance(llm, MiniMaxChat)
        assert llm.model == "MiniMax-M2.5"

    @patch("openai.OpenAI")
    def test_get_llm_minimax_custom_model(self, mock_openai_cls):
        from leann.chat import MiniMaxChat, get_llm

        llm = get_llm({"type": "minimax", "model": "MiniMax-M2.5-highspeed", "api_key": "test-key"})
        assert isinstance(llm, MiniMaxChat)
        assert llm.model == "MiniMax-M2.5-highspeed"

    @patch("openai.OpenAI")
    def test_get_llm_minimax_custom_base_url(self, mock_openai_cls):
        from leann.chat import MiniMaxChat, get_llm

        llm = get_llm(
            {
                "type": "minimax",
                "api_key": "test-key",
                "base_url": "https://api.minimaxi.com/v1",
            }
        )
        assert isinstance(llm, MiniMaxChat)
        assert llm.base_url == "https://api.minimaxi.com/v1"


@pytest.mark.skipif(
    not os.getenv("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set; skipping live API test",
)
class TestMiniMaxLiveAPI:
    """Live API tests for MiniMax provider (requires MINIMAX_API_KEY)."""

    def test_minimax_m25_live(self):
        from leann.chat import MiniMaxChat

        chat = MiniMaxChat(model="MiniMax-M2.5")
        response = chat.ask("Say hello in one word.", max_tokens=10)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_minimax_m25_highspeed_live(self):
        from leann.chat import MiniMaxChat

        chat = MiniMaxChat(model="MiniMax-M2.5-highspeed")
        response = chat.ask("Say hello in one word.", max_tokens=10)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_minimax_via_get_llm_live(self):
        from leann.chat import get_llm

        llm = get_llm({"type": "minimax"})
        response = llm.ask("What is 1+1? Reply with just the number.", max_tokens=10)
        assert isinstance(response, str)
        assert len(response) > 0
