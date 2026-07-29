"""
Tests for Atlas Cloud provider integration.

These tests validate Atlas Cloud provider settings, chat class, and factory
integration without triggering LEANN's compiled backend imports.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

_LEANN_SRC = os.path.join(os.path.dirname(__file__), "..", "packages", "leann-core", "src")
if _LEANN_SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_LEANN_SRC))

if "leann" not in sys.modules:
    import types

    _stub = types.ModuleType("leann")
    _stub.__path__ = [os.path.join(os.path.abspath(_LEANN_SRC), "leann")]
    sys.modules["leann"] = _stub

from leann.settings import (  # noqa: E402
    resolve_atlascloud_api_key,
    resolve_atlascloud_base_url,
)


class TestAtlasCloudSettings:
    """Test Atlas Cloud settings resolver functions."""

    def test_resolve_atlascloud_api_key_explicit(self):
        assert resolve_atlascloud_api_key("test-key") == "test-key"

    def test_resolve_atlascloud_api_key_from_primary_env(self):
        with patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "atlas-key"}, clear=True):
            assert resolve_atlascloud_api_key() == "atlas-key"

    def test_resolve_atlascloud_api_key_from_spaced_env(self):
        with patch.dict(os.environ, {"ATLAS_CLOUD_API_KEY": "atlas-cloud-key"}, clear=True):
            assert resolve_atlascloud_api_key() == "atlas-cloud-key"

    def test_resolve_atlascloud_api_key_does_not_fallback_to_openai(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True):
            assert resolve_atlascloud_api_key() is None

    def test_resolve_atlascloud_base_url_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert resolve_atlascloud_base_url() == "https://api.atlascloud.ai/v1"

    def test_resolve_atlascloud_base_url_explicit(self):
        assert resolve_atlascloud_base_url("https://custom.url/v1") == "https://custom.url/v1"

    def test_resolve_atlascloud_base_url_env_precedence(self):
        with patch.dict(
            os.environ,
            {
                "LEANN_ATLASCLOUD_BASE_URL": "https://leann.url/v1",
                "ATLAS_CLOUD_BASE_URL": "https://fallback.url/v1",
            },
            clear=True,
        ):
            assert resolve_atlascloud_base_url() == "https://leann.url/v1"

    def test_resolve_atlascloud_base_url_strips_trailing_slash(self):
        assert (
            resolve_atlascloud_base_url("https://api.atlascloud.ai/v1/")
            == "https://api.atlascloud.ai/v1"
        )


class TestAtlasCloudChat:
    """Test AtlasCloudChat class."""

    def test_init_requires_api_key(self):
        from leann.chat import AtlasCloudChat

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Atlas Cloud API key is required"):
                AtlasCloudChat(api_key=None)

    @patch("openai.OpenAI")
    def test_init_with_api_key(self, mock_openai_cls):
        from leann.chat import AtlasCloudChat

        chat = AtlasCloudChat(api_key="test-key")
        assert chat.model == "deepseek-ai/deepseek-v4-pro"
        assert chat.api_key == "test-key"
        assert chat.base_url == "https://api.atlascloud.ai/v1"
        mock_openai_cls.assert_called_once_with(
            api_key="test-key", base_url="https://api.atlascloud.ai/v1"
        )

    @patch("openai.OpenAI")
    def test_init_custom_model(self, mock_openai_cls):
        from leann.chat import AtlasCloudChat

        chat = AtlasCloudChat(model="qwen/qwen3.5-27b", api_key="test-key")
        assert chat.model == "qwen/qwen3.5-27b"

    @patch("openai.OpenAI")
    def test_init_custom_base_url(self, mock_openai_cls):
        from leann.chat import AtlasCloudChat

        chat = AtlasCloudChat(api_key="test-key", base_url="https://custom.atlas.url/v1")
        assert chat.base_url == "https://custom.atlas.url/v1"

    @patch("openai.OpenAI")
    def test_ask_returns_response(self, mock_openai_cls):
        from leann.chat import AtlasCloudChat

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello from Atlas Cloud!"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.total_tokens = 100
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 50
        mock_client.chat.completions.create.return_value = mock_response

        chat = AtlasCloudChat(api_key="test-key")
        result = chat.ask("Hello")

        assert result == "Hello from Atlas Cloud!"
        mock_client.chat.completions.create.assert_called_once()

    @patch("openai.OpenAI")
    def test_ask_with_kwargs(self, mock_openai_cls):
        from leann.chat import AtlasCloudChat

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

        chat = AtlasCloudChat(api_key="test-key")
        chat.ask("Hello", temperature=0.5, max_tokens=500, top_p=0.9)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 500
        assert call_kwargs["top_p"] == 0.9

    @patch("openai.OpenAI")
    def test_ask_handles_error(self, mock_openai_cls):
        from leann.chat import AtlasCloudChat

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        chat = AtlasCloudChat(api_key="test-key")
        result = chat.ask("Hello")

        assert "Error" in result
        assert "Atlas Cloud" in result


class TestGetLLMFactory:
    """Test get_llm factory function with Atlas Cloud types."""

    @patch("openai.OpenAI")
    def test_get_llm_atlascloud(self, mock_openai_cls):
        from leann.chat import AtlasCloudChat, get_llm

        llm = get_llm({"type": "atlascloud", "api_key": "test-key"})
        assert isinstance(llm, AtlasCloudChat)
        assert llm.model == "deepseek-ai/deepseek-v4-pro"

    @patch("openai.OpenAI")
    @pytest.mark.parametrize("provider_type", ["atlascloud", "atlas-cloud", "atlas"])
    def test_get_llm_atlascloud_aliases(self, mock_openai_cls, provider_type):
        from leann.chat import AtlasCloudChat, get_llm

        llm = get_llm({"type": provider_type, "api_key": "test-key"})
        assert isinstance(llm, AtlasCloudChat)

    @patch("openai.OpenAI")
    def test_get_llm_atlascloud_custom_model(self, mock_openai_cls):
        from leann.chat import AtlasCloudChat, get_llm

        llm = get_llm(
            {
                "type": "atlascloud",
                "model": "qwen/qwen3.5-27b",
                "api_key": "test-key",
            }
        )
        assert isinstance(llm, AtlasCloudChat)
        assert llm.model == "qwen/qwen3.5-27b"

    @patch("openai.OpenAI")
    def test_get_llm_atlascloud_custom_base_url(self, mock_openai_cls):
        from leann.chat import AtlasCloudChat, get_llm

        llm = get_llm(
            {
                "type": "atlascloud",
                "api_key": "test-key",
                "base_url": "https://custom.atlas.url/v1",
            }
        )
        assert isinstance(llm, AtlasCloudChat)
        assert llm.base_url == "https://custom.atlas.url/v1"


@pytest.mark.skipif(
    not (os.getenv("ATLASCLOUD_API_KEY") or os.getenv("ATLAS_CLOUD_API_KEY")),
    reason="ATLASCLOUD_API_KEY or ATLAS_CLOUD_API_KEY not set; skipping live API test",
)
class TestAtlasCloudLiveAPI:
    """Live API tests for Atlas Cloud provider."""

    def test_atlascloud_deepseek_live(self):
        from leann.chat import AtlasCloudChat

        chat = AtlasCloudChat(model="deepseek-ai/deepseek-v4-pro")
        response = chat.ask("Say hello in one word.", max_tokens=10)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_atlascloud_via_get_llm_live(self):
        from leann.chat import get_llm

        llm = get_llm({"type": "atlascloud"})
        response = llm.ask("What is 1+1? Reply with just the number.", max_tokens=10)
        assert isinstance(response, str)
        assert len(response) > 0
