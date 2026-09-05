"""
Tests for LiteLLM provider integration.

These tests validate LiteLLM provider settings, chat class, and factory
integration. We import from leann.settings and leann.chat directly to avoid
triggering the full leann.__init__ import chain which requires C++ backend
builds. A lightweight fake ``litellm`` module is installed so the provider
imports cleanly even when the optional dependency isn't present.
"""

import os
import sys
import types
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
    _stub = types.ModuleType("leann")
    _stub.__path__ = [os.path.join(os.path.abspath(_LEANN_SRC), "leann")]
    sys.modules["leann"] = _stub

# Install a fake `litellm` module so `import litellm` succeeds even when the
# optional dependency isn't installed. Real installs are left untouched.
if "litellm" not in sys.modules:
    _fake_litellm = types.ModuleType("litellm")
    # Assign through __dict__ so static type checkers don't flag attribute
    # writes on a bare ModuleType (the attributes are what `import litellm`
    # then resolves).
    _fake_litellm.__dict__["completion"] = MagicMock(name="litellm.completion")
    _fake_litellm.__dict__["supports_reasoning"] = MagicMock(name="litellm.supports_reasoning")
    sys.modules["litellm"] = _fake_litellm

# Now we can safely import the modules we actually test.
from leann.settings import (  # noqa: E402
    resolve_litellm_api_key,
    resolve_litellm_base_url,
)


def _make_response(content: str, finish_reason: str = "stop"):
    """Build an OpenAI-shaped response mock like litellm.completion returns."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.choices[0].finish_reason = finish_reason
    response.usage.total_tokens = 100
    response.usage.prompt_tokens = 50
    response.usage.completion_tokens = 50
    return response


class TestLiteLLMSettings:
    """Test LiteLLM settings resolver functions."""

    def test_resolve_api_key_explicit(self):
        assert resolve_litellm_api_key("test-key") == "test-key"

    def test_resolve_api_key_from_litellm_env(self):
        with patch.dict(os.environ, {"LITELLM_API_KEY": "lite-key"}, clear=True):
            assert resolve_litellm_api_key() == "lite-key"

    def test_resolve_api_key_from_leann_env(self):
        with patch.dict(os.environ, {"LEANN_LITELLM_API_KEY": "leann-key"}, clear=True):
            assert resolve_litellm_api_key() == "leann-key"

    def test_resolve_api_key_none(self):
        with patch.dict(os.environ, {}, clear=True):
            assert resolve_litellm_api_key() is None

    def test_resolve_base_url_default_none(self):
        # Unlike single-provider resolvers, LiteLLM has no default base URL:
        # it routes directly to the provider inferred from the model string.
        with patch.dict(os.environ, {}, clear=True):
            assert resolve_litellm_base_url() is None

    def test_resolve_base_url_explicit(self):
        assert resolve_litellm_base_url("https://proxy.url/v1") == "https://proxy.url/v1"

    def test_resolve_base_url_from_litellm_env(self):
        with patch.dict(os.environ, {"LITELLM_BASE_URL": "https://env.url/v1"}, clear=True):
            assert resolve_litellm_base_url() == "https://env.url/v1"

    def test_resolve_base_url_from_leann_env(self):
        with patch.dict(os.environ, {"LEANN_LITELLM_BASE_URL": "https://leann.url/v1"}, clear=True):
            assert resolve_litellm_base_url() == "https://leann.url/v1"

    def test_resolve_base_url_from_api_base_env(self):
        with patch.dict(os.environ, {"LITELLM_API_BASE": "https://apibase.url/v1"}, clear=True):
            assert resolve_litellm_base_url() == "https://apibase.url/v1"

    def test_resolve_base_url_strips_trailing_slash(self):
        assert resolve_litellm_base_url("https://proxy.url/v1/") == "https://proxy.url/v1"


class TestLiteLLMChat:
    """Test LiteLLMChat class."""

    def test_init_does_not_require_api_key(self):
        # LiteLLM reads each provider's own env var, so no key is needed at init.
        from leann.chat import LiteLLMChat

        with patch.dict(os.environ, {}, clear=True):
            chat = LiteLLMChat()
            assert chat.model == "gpt-4o"
            assert chat.api_key is None
            assert chat.base_url is None

    def test_init_custom_model(self):
        from leann.chat import LiteLLMChat

        chat = LiteLLMChat(model="anthropic/claude-haiku-4-5")
        assert chat.model == "anthropic/claude-haiku-4-5"

    def test_init_proxy_config(self):
        from leann.chat import LiteLLMChat

        chat = LiteLLMChat(api_key="proxy-key", base_url="https://proxy.url/v1")
        assert chat.api_key == "proxy-key"
        assert chat.base_url == "https://proxy.url/v1"

    def test_ask_returns_response(self):
        from leann.chat import LiteLLMChat

        with patch("litellm.completion", return_value=_make_response("Hello from LiteLLM!")) as m:
            chat = LiteLLMChat(model="gpt-4o")
            result = chat.ask("Hello")

        assert result == "Hello from LiteLLM!"
        m.assert_called_once()

    def test_ask_sets_drop_params_by_default(self):
        from leann.chat import LiteLLMChat

        with patch("litellm.completion", return_value=_make_response("ok")) as m:
            LiteLLMChat(model="anthropic/claude-haiku-4-5").ask("Hello")

        call_kwargs = m.call_args[1]
        assert call_kwargs["drop_params"] is True
        assert call_kwargs["model"] == "anthropic/claude-haiku-4-5"
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]

    def test_ask_forwards_credentials_only_when_set(self):
        from leann.chat import LiteLLMChat

        # No proxy config -> api_key/api_base omitted so LiteLLM uses env vars.
        with patch("litellm.completion", return_value=_make_response("ok")) as m:
            with patch.dict(os.environ, {}, clear=True):
                LiteLLMChat(model="gpt-4o").ask("Hello")
        call_kwargs = m.call_args[1]
        assert "api_key" not in call_kwargs
        assert "api_base" not in call_kwargs

        # Proxy config -> forwarded as api_key/api_base.
        with patch("litellm.completion", return_value=_make_response("ok")) as m:
            LiteLLMChat(api_key="proxy-key", base_url="https://proxy.url/v1").ask("Hello")
        call_kwargs = m.call_args[1]
        assert call_kwargs["api_key"] == "proxy-key"
        assert call_kwargs["api_base"] == "https://proxy.url/v1"

    def test_ask_with_kwargs(self):
        from leann.chat import LiteLLMChat

        with patch("litellm.completion", return_value=_make_response("Response")) as m:
            LiteLLMChat(model="gpt-4o").ask("Hello", temperature=0.5, max_tokens=500, top_p=0.9)

        call_kwargs = m.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 500
        assert call_kwargs["top_p"] == 0.9

    def test_ask_maps_thinking_budget_for_reasoning_model(self):
        from leann.chat import LiteLLMChat

        with (
            patch("litellm.completion", return_value=_make_response("Response")) as m,
            patch("litellm.supports_reasoning", return_value=True),
        ):
            LiteLLMChat(model="anthropic/claude-haiku-4-5").ask("Hello", thinking_budget="high")

        call_kwargs = m.call_args[1]
        assert call_kwargs["reasoning_effort"] == "high"
        # thinking_budget itself must not leak through to the API call.
        assert "thinking_budget" not in call_kwargs

    def test_ask_skips_thinking_budget_for_non_reasoning_model(self):
        from leann.chat import LiteLLMChat

        # Non-reasoning models must NOT get reasoning_effort: a LiteLLM proxy
        # rejects the unsupported param (client-side drop_params isn't honored).
        with (
            patch("litellm.completion", return_value=_make_response("Response")) as m,
            patch("litellm.supports_reasoning", return_value=False),
        ):
            LiteLLMChat(model="gpt-4.1-mini").ask("Hello", thinking_budget="high")

        call_kwargs = m.call_args[1]
        assert "reasoning_effort" not in call_kwargs
        assert "thinking_budget" not in call_kwargs

    def test_ask_handles_error(self):
        from leann.chat import LiteLLMChat

        with patch("litellm.completion", side_effect=Exception("API error")):
            result = LiteLLMChat(model="gpt-4o").ask("Hello")

        assert "Error" in result
        assert "LiteLLM" in result


class TestGetLLMFactory:
    """Test get_llm factory function with litellm type."""

    def test_get_llm_litellm(self):
        from leann.chat import LiteLLMChat, get_llm

        llm = get_llm({"type": "litellm"})
        assert isinstance(llm, LiteLLMChat)
        assert llm.model == "gpt-4o"

    def test_get_llm_litellm_custom_model(self):
        from leann.chat import LiteLLMChat, get_llm

        llm = get_llm({"type": "litellm", "model": "gemini/gemini-2.5-flash"})
        assert isinstance(llm, LiteLLMChat)
        assert llm.model == "gemini/gemini-2.5-flash"

    def test_get_llm_litellm_proxy(self):
        from leann.chat import LiteLLMChat, get_llm

        llm = get_llm(
            {
                "type": "litellm",
                "api_key": "proxy-key",
                "base_url": "https://proxy.url/v1",
            }
        )
        assert isinstance(llm, LiteLLMChat)
        assert llm.api_key == "proxy-key"
        assert llm.base_url == "https://proxy.url/v1"


@pytest.mark.skipif(
    not os.getenv("LITELLM_LIVE_TEST"),
    reason="LITELLM_LIVE_TEST not set; skipping live API test",
)
class TestLiteLLMLiveAPI:
    """Live API tests for LiteLLM provider.

    Requires a real provider key (e.g. OPENAI_API_KEY or ANTHROPIC_API_KEY)
    and LITELLM_LIVE_TEST=1. Set --model via LITELLM_LIVE_MODEL (default gpt-4o).
    """

    def test_litellm_live(self):
        from leann.chat import LiteLLMChat

        model = os.getenv("LITELLM_LIVE_MODEL", "gpt-4o")
        chat = LiteLLMChat(model=model)
        response = chat.ask("What is 1+1? Reply with just the number.", max_tokens=10)
        assert isinstance(response, str)
        assert len(response) > 0
