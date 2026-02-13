"""Runtime configuration helpers for LEANN."""

from __future__ import annotations

import json
import os
from typing import Any

# Default fallbacks to preserve current behaviour while keeping them in one place.
_DEFAULT_OLLAMA_HOST = "http://localhost:11434"
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"


def _clean_url(value: str) -> str:
    """Normalize URL strings by stripping trailing slashes."""

    return value.rstrip("/") if value else value


def resolve_ollama_host(explicit: str | None = None) -> str:
    """Resolve the Ollama-compatible endpoint to use."""

    candidates = (
        explicit,
        os.getenv("LEANN_LOCAL_LLM_HOST"),
        os.getenv("LEANN_OLLAMA_HOST"),
        os.getenv("OLLAMA_HOST"),
        os.getenv("LOCAL_LLM_ENDPOINT"),
    )

    for candidate in candidates:
        if candidate:
            return _clean_url(candidate)

    return _clean_url(_DEFAULT_OLLAMA_HOST)


def resolve_openai_base_url(explicit: str | None = None) -> str:
    """Resolve the base URL for OpenAI-compatible services."""

    candidates = (
        explicit,
        os.getenv("LEANN_OPENAI_BASE_URL"),
        os.getenv("OPENAI_BASE_URL"),
        os.getenv("LOCAL_OPENAI_BASE_URL"),
    )

    for candidate in candidates:
        if candidate:
            return _clean_url(candidate)

    return _clean_url(_DEFAULT_OPENAI_BASE_URL)


def resolve_anthropic_base_url(explicit: str | None = None) -> str:
    """Resolve the base URL for Anthropic-compatible services."""

    candidates = (
        explicit,
        os.getenv("LEANN_ANTHROPIC_BASE_URL"),
        os.getenv("ANTHROPIC_BASE_URL"),
        os.getenv("LOCAL_ANTHROPIC_BASE_URL"),
    )

    for candidate in candidates:
        if candidate:
            return _clean_url(candidate)

    return _clean_url(_DEFAULT_ANTHROPIC_BASE_URL)


def resolve_openai_api_key(explicit: str | None = None) -> str | None:
    """Resolve the API key for OpenAI-compatible services."""

    if explicit:
        return explicit

    return os.getenv("OPENAI_API_KEY")


def resolve_anthropic_api_key(explicit: str | None = None) -> str | None:
    """Resolve the API key for Anthropic services."""

    if explicit:
        return explicit

    return os.getenv("ANTHROPIC_API_KEY")


def encode_provider_options(options: dict[str, Any] | None) -> str | None:
    """Serialize provider options for child processes."""

    if not options:
        return None

    try:
        return json.dumps(options)
    except (TypeError, ValueError):
        # Fall back to empty payload if serialization fails
        return None
