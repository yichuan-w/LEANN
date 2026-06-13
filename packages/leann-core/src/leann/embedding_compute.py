"""
Unified embedding computation module
Consolidates all embedding computation logic using SentenceTransformer
Preserves all optimization parameters to ensure performance
"""

import json
import logging
import os
from typing import Any, Optional, Protocol

import numpy as np

# torch and tiktoken are imported lazily inside the functions that use them, so
# `import leann` (e.g. for MCP search over an existing index, BM25-only flows,
# or non-embedding utilities) doesn't pull torch's ~1 GB of state into memory.

# Set up logger with proper level
logger = logging.getLogger(__name__)
LOG_LEVEL = os.getenv("LEANN_LOG_LEVEL", "WARNING").upper()
log_level = getattr(logging, LOG_LEVEL, logging.WARNING)
logger.setLevel(log_level)


class _SentenceTransformerLike(Protocol):
    def eval(self) -> Any: ...
    def parameters(self) -> Any: ...
    def encode(self, *args: Any, **kwargs: Any) -> Any: ...
    def half(self) -> Any: ...


# Token limit registry for embedding models
# Used as fallback when dynamic discovery fails (e.g., LM Studio, OpenAI)
# Ollama models use dynamic discovery via /api/show
EMBEDDING_MODEL_LIMITS = {
    # Nomic models (common across servers)
    "nomic-embed-text": 2048,  # Corrected from 512 - verified via /api/show
    "nomic-embed-text-v1.5": 2048,
    "nomic-embed-text-v2": 512,
    # Other embedding models
    "mxbai-embed-large": 512,
    "all-minilm": 512,
    "bge-m3": 8192,
    "snowflake-arctic-embed": 512,
    # OpenAI models
    "text-embedding-3-small": 8192,
    "text-embedding-3-large": 8192,
    "text-embedding-ada-002": 8192,
}

# Runtime cache for dynamically discovered token limits
# Key: (model_name, base_url), Value: token_limit
# Prevents repeated SDK/API calls for the same model
_token_limit_cache: dict[tuple[str, str], int] = {}


def get_model_token_limit(
    model_name: str,
    base_url: Optional[str] = None,
    default: int = 2048,
) -> int:
    """
    Get token limit for a given embedding model.
    Uses hybrid approach: dynamic discovery for Ollama, registry fallback for others.
    Caches discovered limits to prevent repeated API/SDK calls.

    Args:
        model_name: Name of the embedding model
        base_url: Base URL of the embedding server (for dynamic discovery)
        default: Default token limit if model not found

    Returns:
        Token limit for the model in tokens
    """
    # Check cache first to avoid repeated SDK/API calls
    cache_key = (model_name, base_url or "")
    if cache_key in _token_limit_cache:
        cached_limit = _token_limit_cache[cache_key]
        logger.debug(f"Using cached token limit for {model_name}: {cached_limit}")
        return cached_limit

    # Try Ollama dynamic discovery if base_url provided
    if base_url:
        # Detect Ollama servers by port or "ollama" in URL
        if "11434" in base_url or "ollama" in base_url.lower():
            limit = _query_ollama_context_limit(model_name, base_url)
            if limit:
                _token_limit_cache[cache_key] = limit
                return limit

        # Try LM Studio SDK discovery
        if "1234" in base_url or "lmstudio" in base_url.lower() or "lm.studio" in base_url.lower():
            # Convert HTTP to WebSocket URL
            ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
            # Remove /v1 suffix if present
            if ws_url.endswith("/v1"):
                ws_url = ws_url[:-3]

            limit = _query_lmstudio_context_limit(model_name, ws_url)
            if limit:
                _token_limit_cache[cache_key] = limit
                return limit

    # Fallback to known model registry with version handling (from PR #154)
    # Handle versioned model names (e.g., "nomic-embed-text:latest" -> "nomic-embed-text")
    base_model_name = model_name.split(":")[0]

    # Check exact match first
    if model_name in EMBEDDING_MODEL_LIMITS:
        limit = EMBEDDING_MODEL_LIMITS[model_name]
        _token_limit_cache[cache_key] = limit
        return limit

    # Check base name match
    if base_model_name in EMBEDDING_MODEL_LIMITS:
        limit = EMBEDDING_MODEL_LIMITS[base_model_name]
        _token_limit_cache[cache_key] = limit
        return limit

    # Check partial matches for common patterns
    for known_model, registry_limit in EMBEDDING_MODEL_LIMITS.items():
        if known_model in base_model_name or base_model_name in known_model:
            _token_limit_cache[cache_key] = registry_limit
            return registry_limit

    # Default fallback
    logger.warning(f"Unknown model '{model_name}', using default {default} token limit")
    _token_limit_cache[cache_key] = default
    return default


def truncate_to_token_limit(texts: list[str], token_limit: int) -> list[str]:
    """
    Truncate texts to fit within token limit using tiktoken.

    Args:
        texts: List of text strings to truncate
        token_limit: Maximum number of tokens allowed

    Returns:
        List of truncated texts (same length as input)
    """
    if not texts:
        return []

    import tiktoken

    # Use tiktoken with cl100k_base encoding
    enc = tiktoken.get_encoding("cl100k_base")

    truncated_texts = []
    truncation_count = 0
    total_tokens_removed = 0
    max_original_length = 0

    for i, text in enumerate(texts):
        tokens = enc.encode(text)
        original_length = len(tokens)

        if original_length <= token_limit:
            # Text is within limit, keep as is
            truncated_texts.append(text)
        else:
            # Truncate to token_limit
            truncated_tokens = tokens[:token_limit]
            truncated_text = enc.decode(truncated_tokens)
            truncated_texts.append(truncated_text)

            # Track truncation statistics
            truncation_count += 1
            tokens_removed = original_length - token_limit
            total_tokens_removed += tokens_removed
            max_original_length = max(max_original_length, original_length)

            # Log individual truncation at WARNING level (first few only)
            if truncation_count <= 3:
                logger.warning(
                    f"Text {i + 1} truncated: {original_length} → {token_limit} tokens "
                    f"({tokens_removed} tokens removed)"
                )
            elif truncation_count == 4:
                logger.warning("Further truncation warnings suppressed...")

    # Log summary at INFO level
    if truncation_count > 0:
        logger.warning(
            f"Truncation summary: {truncation_count}/{len(texts)} texts truncated "
            f"(removed {total_tokens_removed} tokens total, longest was {max_original_length} tokens)"
        )
    else:
        logger.debug(
            f"No truncation needed - all {len(texts)} texts within {token_limit} token limit"
        )

    return truncated_texts


_model_cache: dict[str, Any] = {}


# ── Provider registry ──────────────────────────────────────────────────

_providers: dict[str, "Any"] = {}  # mode_name -> compute function


def _query_ollama_context_limit(model_name: str, base_url: str) -> Optional[int]:
    """Query Ollama API for model context window size."""
    try:
        import urllib.request

        url = f"http://{base_url}/api/show"
        data = json.dumps({"name": model_name}).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            info = json.loads(resp.read())
        for key in (
            "context_length",
            "num_ctx",
            "max_seq_len",
            "max_position_embeddings",
        ):
            if key in info:
                return int(info[key])
        params = info.get("parameters", "")
        if isinstance(params, str) and "num_ctx" in params:
            import re

            m = re.search(r"num_ctx\s+(\d+)", params)
            if m:
                return int(m.group(1))
        return None
    except Exception:
        return None


def _query_lmstudio_context_limit(model_name: str, base_url: str) -> Optional[int]:
    """Query LM Studio WebSocket API for model context window size."""
    try:
        import json as _json

        import websocket

        ws = websocket.create_connection(base_url, timeout=10)
        try:
            request = _json.dumps(
                {
                    "type": "modelInfo",
                    "modelKey": model_name,
                }
            )
            ws.send(request)
            response_str = ws.recv()
            response = _json.loads(response_str)
            if (
                isinstance(response, dict)
                and response.get("type") == "modelInfo"
                and "modelInfo" in response
            ):
                model_info = response["modelInfo"]
                for key in (
                    "contextLength",
                    "context_length",
                    "maxContextLength",
                    "maxSeqLen",
                ):
                    val = model_info.get(key)
                    if isinstance(val, (int, float)):
                        return int(val)
        finally:
            ws.close()
        return None
    except Exception:
        return None


def _init_providers() -> None:
    """Lazy-import and register all embedding providers."""
    global compute_embeddings_sentence_transformers, compute_embeddings_openai
    global compute_embeddings_mlx, compute_embeddings_ollama, compute_embeddings_gemini

    from .providers.gemini import compute_embeddings_gemini as _gem
    from .providers.mlx import compute_embeddings_mlx as _mlx
    from .providers.ollama import compute_embeddings_ollama as _oll
    from .providers.openai import compute_embeddings_openai as _oa
    from .providers.sentence_transformers import compute_embeddings_sentence_transformers as _st

    compute_embeddings_sentence_transformers = _st  # type: ignore[assignment]
    compute_embeddings_openai = _oa  # type: ignore[assignment]
    compute_embeddings_mlx = _mlx  # type: ignore[assignment]
    compute_embeddings_ollama = _oll  # type: ignore[assignment]
    compute_embeddings_gemini = _gem  # type: ignore[assignment]

    _providers["sentence-transformers"] = _st
    _providers["openai"] = _oa
    _providers["mlx"] = _mlx
    _providers["ollama"] = _oll
    _providers["gemini"] = _gem


# ── Backward-compat stubs (set at module load, replaced by _init_providers) ──


def _sentinel(*args: Any, **kwargs: Any) -> "np.ndarray":
    _init_providers()
    raise RuntimeError("Provider not initialized — this is a bug.")


compute_embeddings_sentence_transformers = _sentinel  # type: ignore[assignment]
compute_embeddings_openai = _sentinel  # type: ignore[assignment]
compute_embeddings_mlx = _sentinel  # type: ignore[assignment]
compute_embeddings_ollama = _sentinel  # type: ignore[assignment]
compute_embeddings_gemini = _sentinel  # type: ignore[assignment]


# ── Public API ─────────────────────────────────────────────────────────


def compute_embeddings(
    texts: list[str],
    model_name: str,
    mode: str = "sentence-transformers",
    is_build: bool = False,
    batch_size: int = 32,
    adaptive_optimization: bool = True,
    manual_tokenize: bool = False,
    max_length: int = 512,
    provider_options: Optional[dict[str, Any]] = None,
) -> "np.ndarray":
    """
    Unified embedding computation entry point.

    Args:
        texts: List of texts to compute embeddings for.
        model_name: Model name / identifier.
        mode: Computation mode:
            ``"sentence-transformers"`` | ``"openai"`` | ``"mlx"`` |
            ``"ollama"`` | ``"gemini"``.
        is_build: Whether this is a build operation (shows progress bar).
        batch_size: Batch size for processing.
        adaptive_optimization: Whether to use adaptive optimization.
        manual_tokenize: Use manual HF tokenizer path (experimental).
        max_length: Maximum sequence length.
        provider_options: Provider-specific options dict.

    Returns:
        Embeddings array of shape ``(len(texts), embedding_dim)``.
    """
    _init_providers()

    provider_options = provider_options or {}

    # Allow batch_size override from provider_options
    if "batch_size" in provider_options:
        batch_size = provider_options["batch_size"]
        adaptive_optimization = False

    fn = _providers.get(mode)
    if fn is None:
        raise ValueError(
            f"Unsupported embedding mode: {mode!r}. Available: {sorted(_providers.keys())}"
        )

    if mode == "sentence-transformers":
        return fn(
            texts,
            model_name,
            is_build=is_build,
            batch_size=batch_size,
            adaptive_optimization=adaptive_optimization,
            manual_tokenize=manual_tokenize,
            max_length=max_length,
        )
    elif mode == "openai":
        return fn(
            texts,
            model_name,
            base_url=provider_options.get("base_url"),
            api_key=provider_options.get("api_key"),
            provider_options=provider_options,
        )
    elif mode == "mlx":
        return fn(texts, model_name)
    elif mode == "ollama":
        return fn(
            texts,
            model_name,
            is_build=is_build,
            host=provider_options.get("host"),
            provider_options=provider_options,
        )
    elif mode == "gemini":
        return fn(texts, model_name, is_build=is_build)
    else:
        raise ValueError(f"Unsupported embedding mode: {mode!r}")


# Register a custom provider (public API for third-party backends)
def register_provider(name: str, fn: "Any") -> None:
    """Register a custom embedding provider function.

    Args:
        name: Mode name (used as the ``mode=`` argument).
        fn: Callable ``(texts, model_name, **kwargs) -> np.ndarray``.
    """
    _providers[name] = fn
