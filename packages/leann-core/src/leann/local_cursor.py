"""Local Cursor — OpenAI-compatible proxy with LEANN code retrieval (#47).

Starts a lightweight HTTP server that exposes ``/v1/chat/completions`` and
``/v1/models``.  When a chat request arrives, the proxy:

1. Extracts the user's latest message.
2. Runs a LEANN search against a configured code index.
3. Prepends the retrieved context to the system prompt.
4. Forwards the augmented request to a local LLM (Ollama, LM Studio, etc.)
   via the OpenAI-compatible API.

This gives you a **fully local** Cursor-like experience: local model + local
code retrieval, zero data leaves your machine.

Usage::

    leann cursor --index my-code --model qwen3-coder:30b
    leann cursor --index my-code --model codestral:latest --port 8080

Then point your editor / Claude Code / Cursor CLI at::

    ANTHROPIC_BASE_URL=http://localhost:8765
    # or
    OPENAI_BASE_URL=http://localhost:8765/v1
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .settings import resolve_ollama_host

logger = logging.getLogger(__name__)

# Default configuration
_DEFAULT_PORT = 8765
_DEFAULT_MODEL = "qwen3-coder"
_DEFAULT_TOP_K = 10
_DEFAULT_MAX_CONTEXT_CHARS = 8000  # Stay within typical 8K context window budget
_MAX_REQUEST_BODY = 10 * 1024 * 1024  # 10 MB — reject oversized requests


def _build_context_block(results: list, max_chars: int = _DEFAULT_MAX_CONTEXT_CHARS) -> str:
    """Format LEANN search results as a code context block for the LLM."""
    if not results:
        return ""

    parts = []
    total = 0
    for r in results:
        meta = r.metadata if hasattr(r, "metadata") and isinstance(r.metadata, dict) else {}
        source = meta.get("file_path", meta.get("source", ""))
        header = f"--- {source} ---\n" if source else ""
        block = f"{header}{r.text}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)

    if not parts:
        return ""

    return (
        "The following code snippets are retrieved from the local codebase and may "
        "be relevant to the user's question:\n\n" + "\n".join(parts)
    )


def _forward_to_llm(
    ollama_host: str,
    payload: dict,
) -> tuple[int, dict]:
    """Forward the augmented request to the local LLM.

    Returns ``(status_code, response_body_dict)``.
    """
    url = f"{ollama_host}/v1/chat/completions"
    data = json.dumps(payload).encode("utf-8")

    req = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=120) as resp:
            body = resp.read()
            return resp.status, json.loads(body)
    except HTTPError as e:
        # Preserve the upstream error body so callers get useful diagnostics.
        try:
            err_body = json.loads(e.read())
        except Exception:
            err_body = {"error": {"message": str(e), "type": "upstream_error"}}
        logger.error("LLM returned HTTP %d: %s", e.code, err_body)
        return e.code, err_body
    except (URLError, OSError) as e:
        logger.error("LLM forwarding failed (network): %s", e)
        return 502, {"error": {"message": str(e), "type": "proxy_error"}}


class _CursorHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the local cursor proxy."""

    # Suppress default access logging
    def log_message(self, format, *args):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(format, *args)

    def _send_json(self, status: int, body: dict) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path in ("/v1/models", "/models"):
            self._handle_models()
        elif path in ("/health", "/"):
            self._send_json(200, {"status": "ok", "service": "leann-cursor"})
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        path = urlparse(self.path).path

        if path in ("/v1/chat/completions", "/chat/completions"):
            self._handle_chat()
        else:
            self._send_json(404, {"error": "Not found"})

    def _handle_models(self):
        """List models — returns the configured model."""
        cfg = self.server.cursor_config
        self._send_json(
            200,
            {
                "object": "list",
                "data": [
                    {
                        "id": cfg["model"],
                        "object": "model",
                        "owned_by": "local",
                    }
                ],
            },
        )

    def _handle_chat(self):
        """Handle a chat completion request with LEANN retrieval augmentation."""
        cfg = self.server.cursor_config

        # Read request body (with size limit to prevent OOM)
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length < 0:
            self._send_json(400, {"error": "Invalid Content-Length"})
            return
        if content_length > _MAX_REQUEST_BODY:
            self._send_json(413, {"error": "Request body too large"})
            return
        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        messages = payload.get("messages", [])
        if not messages:
            self._send_json(400, {"error": "No messages provided"})
            return

        # Extract the last user message for retrieval
        user_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Handle structured content (e.g., [{"type": "text", "text": "..."}])
                    text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                    user_msg = " ".join(text_parts)
                else:
                    user_msg = str(content)
                break

        if not user_msg:
            # No user message — forward as-is
            status, resp = _forward_to_llm(cfg["ollama_host"], payload)
            self._send_json(status, resp)
            return

        # Perform LEANN retrieval
        context_block = ""
        if cfg.get("searcher") is not None:
            try:
                results = cfg["searcher"].search(
                    user_msg,
                    top_k=cfg.get("top_k", _DEFAULT_TOP_K),
                    complexity=32,
                    recompute_embeddings=True,
                )
                context_block = _build_context_block(
                    results, cfg.get("max_context_chars", _DEFAULT_MAX_CONTEXT_CHARS)
                )
                if context_block:
                    logger.info(
                        "Retrieved %d snippets (%d chars) for query",
                        len(results),
                        len(context_block),
                    )
            except Exception as e:
                logger.warning("LEANN retrieval failed: %s", e)

        # Augment the system prompt with retrieved context
        if context_block:
            augmented_messages = list(messages)  # shallow copy
            # Prepend or merge into system message
            if augmented_messages and augmented_messages[0].get("role") == "system":
                existing = augmented_messages[0].get("content", "")
                # Content can be a string or a list (multimodal format) — coerce to str.
                if not isinstance(existing, str):
                    existing = str(existing)
                augmented_messages[0] = {
                    **augmented_messages[0],
                    "content": existing + "\n\n" + context_block,
                }
            else:
                augmented_messages.insert(0, {"role": "system", "content": context_block})
            payload = {**payload, "messages": augmented_messages}

        # Override model to configured local model
        payload["model"] = cfg["model"]

        # Forward to LLM
        status, resp = _forward_to_llm(cfg["ollama_host"], payload)
        self._send_json(status, resp)


def start_cursor_server(
    index_path: Optional[str] = None,
    model: str = _DEFAULT_MODEL,
    port: int = _DEFAULT_PORT,
    ollama_host: Optional[str] = None,
    top_k: int = _DEFAULT_TOP_K,
    max_context_chars: int = _DEFAULT_MAX_CONTEXT_CHARS,
) -> None:
    """Start the local cursor proxy server.

    Args:
        index_path: Path to a LEANN index. If None, runs without retrieval.
        model: Local LLM model name (default: qwen3-coder).
        port: HTTP port to listen on (default: 8765).
        ollama_host: Ollama API base URL (auto-detected from env).
        top_k: Number of code snippets to retrieve per query.
        max_context_chars: Maximum characters of context to inject.
    """
    resolved_host = resolve_ollama_host(ollama_host)

    # Load LEANN searcher if index provided
    searcher = None
    if index_path:
        from .api import LeannSearcher

        searcher = LeannSearcher(index_path)
        logger.info("LEANN index loaded: %s", index_path)

    config: dict[str, Any] = {
        "model": model,
        "ollama_host": resolved_host,
        "searcher": searcher,
        "top_k": top_k,
        "max_context_chars": max_context_chars,
    }

    server = ThreadingHTTPServer(("0.0.0.0", port), _CursorHandler)
    server.cursor_config = config  # type: ignore[attr-defined]

    print("LEANN Local Cursor proxy started:")
    print(f"  Endpoint:  http://localhost:{port}/v1/chat/completions")
    print(f"  Model:     {model}")
    print(f"  LLM host:  {resolved_host}")
    if searcher:
        print(f"  Index:     {index_path}")
        print(f"  Top-K:     {top_k}")
    else:
        print("  Index:     (none — running without code retrieval)")
    print()
    print("To use with Claude Code:")
    print(f'  ANTHROPIC_BASE_URL="http://localhost:{port}" claude --model {model}')
    print()
    print("To use with Cursor CLI:")
    print(f'  OPENAI_BASE_URL="http://localhost:{port}/v1" cursor')
    print()
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        print("\nLocal Cursor proxy stopped.")
