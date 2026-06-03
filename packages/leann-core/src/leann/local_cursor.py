"""OpenAI-compatible local code proxy backed by LEANN retrieval.

The proxy exposes ``/v1/models`` and ``/v1/chat/completions`` for local
OpenAI-compatible clients such as Cursor configured with a custom base URL.
For each chat request it retrieves relevant snippets from a LEANN code index,
injects those snippets into the system prompt, and forwards the request to a
local OpenAI-compatible model server such as Ollama or LM Studio.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .settings import resolve_ollama_host

logger = logging.getLogger(__name__)

DEFAULT_CURSOR_HOST = "127.0.0.1"
DEFAULT_CURSOR_PORT = 8765
DEFAULT_CURSOR_MODEL = "qwen3-coder"
DEFAULT_CURSOR_TOP_K = 10
DEFAULT_CURSOR_COMPLEXITY = 32
DEFAULT_CURSOR_MAX_CONTEXT_CHARS = 8000
DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost",
    "http://127.0.0.1",
    "http://[::1]",
)
_LOCAL_ORIGIN_HOSTS = {"localhost", "127.0.0.1", "::1"}
MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class CursorProxyConfig:
    """Runtime configuration shared by all proxy request handlers."""

    model: str
    llm_base_url: str
    searcher: Any | None = None
    top_k: int = DEFAULT_CURSOR_TOP_K
    complexity: int = DEFAULT_CURSOR_COMPLEXITY
    max_context_chars: int = DEFAULT_CURSOR_MAX_CONTEXT_CHARS
    recompute_embeddings: bool = True
    allowed_origins: tuple[str, ...] = DEFAULT_ALLOWED_ORIGINS


class CursorProxyServer(ThreadingHTTPServer):
    """HTTP server carrying typed Cursor proxy configuration."""

    cursor_config: CursorProxyConfig


def _origin_allowed(origin: str | None, allowed_origins: Sequence[str]) -> bool:
    if not origin:
        return False

    parsed = urlparse(origin)
    if parsed.scheme not in {"http", "https"}:
        return False

    normalized = f"{parsed.scheme}://{parsed.netloc}"
    if normalized in allowed_origins:
        return True

    if parsed.hostname not in _LOCAL_ORIGIN_HOSTS:
        return False

    for allowed_origin in allowed_origins:
        allowed = urlparse(allowed_origin)
        if allowed.scheme == parsed.scheme and allowed.hostname == parsed.hostname:
            return True
    return False


def _response_body(message: str, error_type: str = "proxy_error") -> dict[str, dict[str, str]]:
    return {"error": {"message": message, "type": error_type}}


def _result_text(result: Any) -> str:
    text = getattr(result, "text", "")
    return text if isinstance(text, str) else str(text)


def _result_source(result: Any) -> str:
    metadata = getattr(result, "metadata", {})
    if not isinstance(metadata, dict):
        return ""

    source = (
        metadata.get("file_path") or metadata.get("relative_path") or metadata.get("source") or ""
    )
    return source if isinstance(source, str) else str(source)


def build_context_block(
    results: Iterable[Any], max_chars: int = DEFAULT_CURSOR_MAX_CONTEXT_CHARS
) -> str:
    """Format LEANN search results into a bounded prompt context block."""
    snippets: list[str] = []
    used_chars = 0

    for result in results:
        text = _result_text(result).strip()
        if not text:
            continue

        source = _result_source(result)
        header = f"--- {source} ---\n" if source else ""
        snippet = f"{header}{text}\n"
        if snippets and used_chars + len(snippet) > max_chars:
            break
        if not snippets and len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "\n"

        snippets.append(snippet)
        used_chars += len(snippet)

    if not snippets:
        return ""

    return (
        "Relevant local code context retrieved by LEANN. Use it when it helps, "
        "and ignore it when it is unrelated.\n\n" + "\n".join(snippets)
    )


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return " ".join(part for part in text_parts if isinstance(part, str))
    return str(content) if content is not None else ""


def latest_user_text(messages: Sequence[Any]) -> str:
    """Return the latest user text from OpenAI-style chat messages."""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _message_text(message.get("content", "")).strip()
        if text:
            return text
    return ""


def augment_messages(messages: Sequence[Any], context_block: str) -> list[Any]:
    """Return chat messages with retrieved context injected into the system prompt."""
    if not context_block:
        return list(messages)

    augmented = list(messages)
    if augmented and isinstance(augmented[0], dict) and augmented[0].get("role") == "system":
        existing = _message_text(augmented[0].get("content", "")).strip()
        content = f"{existing}\n\n{context_block}" if existing else context_block
        augmented[0] = {**augmented[0], "content": content}
    else:
        augmented.insert(0, {"role": "system", "content": context_block})
    return augmented


def augment_chat_payload(payload: dict[str, Any], config: CursorProxyConfig) -> dict[str, Any]:
    """Inject LEANN context into an OpenAI chat-completions payload."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    if not messages:
        raise ValueError("messages must not be empty")

    user_text = latest_user_text(messages)
    context_block = ""
    if user_text and config.searcher is not None:
        results = config.searcher.search(
            user_text,
            top_k=config.top_k,
            complexity=config.complexity,
            recompute_embeddings=config.recompute_embeddings,
        )
        context_block = build_context_block(results, config.max_context_chars)
        if context_block:
            logger.info("Retrieved %d chars of local code context", len(context_block))

    return {
        **payload,
        "model": config.model,
        "messages": augment_messages(messages, context_block),
    }


def _llm_chat_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/v1/chat/completions"


def forward_json_to_llm(base_url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Forward a non-streaming chat request to the local LLM."""
    request = Request(
        _llm_chat_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            body = response.read()
            return response.status, cast(dict[str, Any], json.loads(body))
    except HTTPError as exc:
        try:
            return exc.code, cast(dict[str, Any], json.loads(exc.read()))
        except Exception:
            return exc.code, _response_body(str(exc), "upstream_error")
    except (OSError, URLError) as exc:
        logger.error("Local LLM forwarding failed: %s", exc)
        return 502, _response_body(str(exc))


def forward_stream_to_llm(
    base_url: str, payload: dict[str, Any]
) -> tuple[int, str, Iterable[bytes]]:
    """Forward a streaming chat request to the local LLM."""
    request = Request(
        _llm_chat_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        response = urlopen(request, timeout=120)
    except HTTPError as exc:
        try:
            body = exc.read()
        except Exception:
            body = json.dumps(_response_body(str(exc), "upstream_error")).encode("utf-8")
        return exc.code, "application/json", (body,)
    except (OSError, URLError) as exc:
        body = json.dumps(_response_body(str(exc))).encode("utf-8")
        return 502, "application/json", (body,)

    content_type = response.headers.get("Content-Type", "text/event-stream")

    def chunks() -> Iterable[bytes]:
        with response:
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                yield chunk

    return response.status, content_type, chunks()


class CursorProxyHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible request handler for local Cursor-style clients."""

    server: CursorProxyServer

    def log_message(self, format: str, *args: Any) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(format, *args)

    def _set_cors_headers(self) -> None:
        origin = self.headers.get("Origin")
        if not _origin_allowed(origin, self.server.cursor_config.allowed_origins):
            return

        assert origin is not None
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(raw)

    def _send_stream(self, status: int, content_type: str, chunks: Iterable[bytes]) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self._set_cors_headers()
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(chunk)
            self.wfile.flush()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/health"}:
            self._send_json(200, {"status": "ok", "service": "leann-local-cursor"})
        elif path in {"/models", "/v1/models"}:
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.cursor_config.model,
                            "object": "model",
                            "owned_by": "local",
                        }
                    ],
                },
            )
        else:
            self._send_json(404, _response_body("Not found", "not_found"))

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/chat/completions", "/v1/chat/completions"}:
            self._send_json(404, _response_body("Not found", "not_found"))
            return
        self._handle_chat()

    def _read_json_body(self) -> dict[str, Any] | None:
        raw_content_length = self.headers.get("Content-Length", "0")
        try:
            content_length = int(raw_content_length)
        except ValueError:
            self._send_json(400, _response_body("Invalid Content-Length", "invalid_request"))
            return None

        if content_length < 0:
            self._send_json(400, _response_body("Invalid Content-Length", "invalid_request"))
            return None
        if content_length > MAX_REQUEST_BODY_BYTES:
            self._send_json(413, _response_body("Request body too large", "invalid_request"))
            return None

        try:
            payload = json.loads(self.rfile.read(content_length))
        except json.JSONDecodeError:
            self._send_json(400, _response_body("Invalid JSON", "invalid_request"))
            return None

        if not isinstance(payload, dict):
            self._send_json(
                400, _response_body("Request body must be a JSON object", "invalid_request")
            )
            return None
        return cast(dict[str, Any], payload)

    def _handle_chat(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return

        try:
            augmented_payload = augment_chat_payload(payload, self.server.cursor_config)
        except ValueError as exc:
            self._send_json(400, _response_body(str(exc), "invalid_request"))
            return
        except Exception as exc:
            logger.warning("LEANN retrieval failed: %s", exc)
            augmented_payload = {**payload, "model": self.server.cursor_config.model}

        if augmented_payload.get("stream") is True:
            status, content_type, chunks = forward_stream_to_llm(
                self.server.cursor_config.llm_base_url, augmented_payload
            )
            self._send_stream(status, content_type, chunks)
            return

        status, response = forward_json_to_llm(
            self.server.cursor_config.llm_base_url, augmented_payload
        )
        self._send_json(status, response)


def start_cursor_server(
    index_path: str | None = None,
    model: str = DEFAULT_CURSOR_MODEL,
    port: int = DEFAULT_CURSOR_PORT,
    bind_host: str = DEFAULT_CURSOR_HOST,
    llm_base_url: str | None = None,
    top_k: int = DEFAULT_CURSOR_TOP_K,
    complexity: int = DEFAULT_CURSOR_COMPLEXITY,
    max_context_chars: int = DEFAULT_CURSOR_MAX_CONTEXT_CHARS,
    recompute_embeddings: bool = True,
    allowed_origins: Sequence[str] = DEFAULT_ALLOWED_ORIGINS,
) -> None:
    """Start the local Cursor proxy server."""
    resolved_llm_base_url = resolve_ollama_host(llm_base_url)

    searcher = None
    if index_path:
        from .api import LeannSearcher

        searcher = LeannSearcher(index_path)
        logger.info("Loaded LEANN code index from %s", index_path)

    config = CursorProxyConfig(
        model=model,
        llm_base_url=resolved_llm_base_url,
        searcher=searcher,
        top_k=top_k,
        complexity=complexity,
        max_context_chars=max_context_chars,
        recompute_embeddings=recompute_embeddings,
        allowed_origins=tuple(allowed_origins),
    )

    server = CursorProxyServer((bind_host, port), CursorProxyHandler)
    server.cursor_config = config

    print("LEANN local Cursor proxy started")
    print(f"  Endpoint:  http://{bind_host}:{port}/v1/chat/completions")
    print(f"  Model:     {model}")
    print(f"  LLM host:  {resolved_llm_base_url}")
    print(f"  Index:     {index_path or '(none)'}")
    print(f"  Top-K:     {top_k}")
    print()
    print("Configure OpenAI-compatible clients with:")
    print(f"  OPENAI_BASE_URL=http://{bind_host}:{port}/v1")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        print("\nLEANN local Cursor proxy stopped.")
