"""
Tests for Novita AI provider integration.

These tests validate Novita provider settings, chat class, and factory integration.
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
    resolve_novita_api_key,
    resolve_novita_base_url,
)


class TestNovitaSettings:
    """Test Novita settings resolver functions."""

    def test_resolve_novita_api_key_explicit(self):
        assert resolve_novita_api_key("test-key") == "test-key"

    def test_resolve_novita_api_key_from_novita_env(self):
        with patch.dict(os.environ, {"NOVITA_API_KEY": "novita-key"}, clear=True):
            assert resolve_novita_api_key() == "novita-key"

    def test_resolve_novita_api_key_fallback_to_openai(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True):
            assert resolve_novita_api_key() == "openai-key"

    def test_resolve_novita_api_key_novita_takes_precedence(self):
        with patch.dict(
            os.environ,
            {"NOVITA_API_KEY": "novita-key", "OPENAI_API_KEY": "openai-key"},
        ):
            assert resolve_novita_api_key() == "novita-key"

    def test_resolve_novita_api_key_none(self):
        with patch.dict(os.environ, {}, clear=True):
            assert resolve_novita_api_key() is None

    def test_resolve_novita_base_url_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert resolve_novita_base_url() == "https://api.novita.ai/openai"

    def test_resolve_novita_base_url_explicit(self):
        assert resolve_novita_base_url("https://custom.url/v1") == "https://custom.url/v1"

    def test_resolve_novita_base_url_from_novita_env(self):
        with patch.dict(os.environ, {"NOVITA_BASE_URL": "https://env.url/v1"}):
            assert resolve_novita_base_url() == "https://env.url/v1"

    def test_resolve_novita_base_url_from_leann_env(self):
        with patch.dict(os.environ, {"LEANN_NOVITA_BASE_URL": "https://leann.url/v1"}):
            assert resolve_novita_base_url() == "https://leann.url/v1"

    def test_resolve_novita_base_url_fallback_to_openai_env(self):
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://openai.url/v1"}, clear=True):
            assert resolve_novita_base_url() == "https://openai.url/v1"

    def test_resolve_novita_base_url_strips_trailing_slash(self):
        assert (
            resolve_novita_base_url("https://api.novita.ai/openai/")
            == "https://api.novita.ai/openai"
        )


class TestNovitaChat:
    """Test NovitaChat class."""

    def test_init_requires_api_key(self):
        from leann.chat import NovitaChat

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Novita API key is required"):
                NovitaChat(api_key=None)

    @patch("openai.OpenAI")
    def test_init_with_api_key(self, mock_openai_cls):
        from leann.chat import NovitaChat

        chat = NovitaChat(api_key="test-key")
        assert chat.model == "moonshotai/kimi-k2.5"
        assert chat.api_key == "test-key"
        assert chat.base_url == "https://api.novita.ai/openai"
        mock_openai_cls.assert_called_once_with(
            api_key="test-key", base_url="https://api.novita.ai/openai"
        )

    @patch("openai.OpenAI")
    def test_init_custom_model(self, mock_openai_cls):
        from leann.chat import NovitaChat

        chat = NovitaChat(model="zai-org/glm-5", api_key="test-key")
        assert chat.model == "zai-org/glm-5"

    @patch("openai.OpenAI")
    def test_init_custom_base_url(self, mock_openai_cls):
        from leann.chat import NovitaChat

        chat = NovitaChat(api_key="test-key", base_url="https://custom.novita.url/v1")
        assert chat.base_url == "https://custom.novita.url/v1"

    @patch("openai.OpenAI")
    def test_ask_returns_response(self, mock_openai_cls):
        from leann.chat import NovitaChat

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello from Novita!"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.total_tokens = 100
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 50
        mock_client.chat.completions.create.return_value = mock_response

        chat = NovitaChat(api_key="test-key")
        result = chat.ask("Hello")

        assert result == "Hello from Novita!"
        mock_client.chat.completions.create.assert_called_once()

    @patch("openai.OpenAI")
    def test_ask_with_kwargs(self, mock_openai_cls):
        from leann.chat import NovitaChat

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

        chat = NovitaChat(api_key="test-key")
        chat.ask("Hello", temperature=0.5, max_tokens=500, top_p=0.9)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 500
        assert call_kwargs["top_p"] == 0.9

    @patch("openai.OpenAI")
    def test_ask_handles_error(self, mock_openai_cls):
        from leann.chat import NovitaChat

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        chat = NovitaChat(api_key="test-key")
        result = chat.ask("Hello")

        assert "Error" in result
        assert "Novita" in result


class TestGetLLMFactory:
    """Test get_llm factory function with novita type."""

    @patch("openai.OpenAI")
    def test_get_llm_novita(self, mock_openai_cls):
        from leann.chat import NovitaChat, get_llm

        llm = get_llm({"type": "novita", "api_key": "test-key"})
        assert isinstance(llm, NovitaChat)
        assert llm.model == "moonshotai/kimi-k2.5"

    @patch("openai.OpenAI")
    def test_get_llm_novita_custom_model(self, mock_openai_cls):
        from leann.chat import NovitaChat, get_llm

        llm = get_llm(
            {
                "type": "novita",
                "model": "zai-org/glm-5",
                "api_key": "test-key",
            }
        )
        assert isinstance(llm, NovitaChat)
        assert llm.model == "zai-org/glm-5"

    @patch("openai.OpenAI")
    def test_get_llm_novita_custom_base_url(self, mock_openai_cls):
        from leann.chat import NovitaChat, get_llm

        llm = get_llm(
            {
                "type": "novita",
                "api_key": "test-key",
                "base_url": "https://custom.novita.url/v1",
            }
        )
        assert isinstance(llm, NovitaChat)
        assert llm.base_url == "https://custom.novita.url/v1"


@pytest.mark.skipif(
    not os.getenv("NOVITA_API_KEY"),
    reason="NOVITA_API_KEY not set; skipping live API test",
)
class TestNovitaLiveAPI:
    """Live API tests for Novita provider (requires NOVITA_API_KEY)."""

    def test_novita_kimi_k25_live(self):
        from leann.chat import NovitaChat

        chat = NovitaChat(model="moonshotai/kimi-k2.5")
        response = chat.ask("Say hello in one word.", max_tokens=10)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_novita_via_get_llm_live(self):
        from leann.chat import get_llm

        llm = get_llm({"type": "novita"})
        response = llm.ask("What is 1+1? Reply with just the number.", max_tokens=10)
        assert isinstance(response, str)
        assert len(response) > 0
