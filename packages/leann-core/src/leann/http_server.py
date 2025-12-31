#!/usr/bin/env python3
"""
FastAPI-based HTTP server for LEANN with WebSocket streaming support.

This server provides REST API endpoints for semantic search and WebSocket
endpoints for streaming Q&A with LLM integration.
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from .api import LeannSearcher
from .chat import LLMInterface, OpenAIChat, OllamaChat, AnthropicChat
from .settings import (
    resolve_openai_api_key,
    resolve_openai_base_url,
    resolve_ollama_host,
    resolve_anthropic_api_key,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for request validation
class SearchRequest(BaseModel):
    """Request model for semantic code search."""

    query: str = Field(..., description="Search query")
    top_k: int = Field(5, ge=1, le=100, description="Number of results to return")
    complexity: int = Field(64, ge=16, le=512, description="Search complexity level")
    show_metadata: bool = Field(False, description="Include metadata in results")
    metadata_filters: Optional[dict] = Field(None, description="Metadata filters for search")


class AskRequest(BaseModel):
    """Request model for Q&A with LLM."""

    question: str = Field(..., description="Question to ask")
    top_k: int = Field(5, ge=1, le=100, description="Number of search results to use as context")
    complexity: int = Field(
        64, ge=16, le=512, description="Search complexity level for context retrieval"
    )
    llm_params: dict[str, Any] = Field(
        default_factory=dict, description="Additional parameters for LLM (temperature, max_tokens, etc.)"
    )


# Server state management
class ServerState:
    """Global state for the HTTP server."""

    def __init__(self):
        self.searcher: Optional[LeannSearcher] = None
        self.llm: Optional[LLMInterface] = None
        self.index_path: str = ""
        self.embedding_model: Optional[str] = None
        self.llm_config: dict[str, Any] = {}


state = ServerState()

# FastAPI app initialization
app = FastAPI(
    title="LEANN HTTP Server",
    version="1.0.0",
    description="HTTP API for LEANN semantic code search and Q&A with LLM streaming",
)


# Startup and shutdown handlers
@app.on_event("startup")
async def startup_event():
    """Initialize searcher and LLM on server startup."""
    logger.info(f"Initializing LEANN with index: {state.index_path}")

    try:
        # Initialize searcher
        state.searcher = LeannSearcher(state.index_path, enable_warmup=True)
        logger.info("Searcher initialized successfully")

        # Initialize LLM based on config
        llm_type = state.llm_config.get("type", "openai")

        if llm_type == "openai":
            state.llm = OpenAIChat(
                model=state.llm_config["model"],
                api_key=state.llm_config.get("api_key"),
                base_url=state.llm_config.get("base_url"),
                ssl_cert=state.llm_config.get("ssl_cert"),
            )
            logger.info(f"OpenAI LLM initialized with model: {state.llm_config['model']}")

        elif llm_type == "ollama":
            state.llm = OllamaChat(
                model=state.llm_config["model"], host=state.llm_config.get("host")
            )
            logger.info(f"Ollama LLM initialized with model: {state.llm_config['model']}")

        elif llm_type == "anthropic":
            state.llm = AnthropicChat(
                model=state.llm_config["model"], api_key=state.llm_config.get("api_key")
            )
            logger.info(f"Anthropic LLM initialized with model: {state.llm_config['model']}")

        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")

        logger.info("LEANN HTTP server ready")

    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on server shutdown."""
    logger.info("Shutting down LEANN HTTP server")
    # Cleanup resources if needed


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "index_loaded": state.searcher is not None,
        "index_path": state.index_path,
        "llm_type": state.llm_config.get("type"),
        "llm_model": state.llm_config.get("model"),
    }


@app.get("/indexes")
async def list_indexes():
    """List available indexes."""
    try:
        # Find .leann directory
        home = Path.home()
        leann_dir = home / ".leann" / "indexes"

        if not leann_dir.exists():
            return {"indexes": []}

        indexes = []
        for index_dir in leann_dir.iterdir():
            if index_dir.is_dir():
                meta_file = index_dir / "documents.leann.meta.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r") as f:
                            meta = json.load(f)
                        indexes.append(
                            {
                                "name": index_dir.name,
                                "path": str(index_dir / "documents.leann"),
                                "backend": meta.get("backend_name"),
                                "embedding_model": meta.get("embedding_model"),
                                "dimensions": meta.get("dimensions"),
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to read metadata for {index_dir.name}: {e}")

        return {"indexes": indexes}

    except Exception as e:
        logger.error(f"Error listing indexes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search(request: SearchRequest):
    """Semantic code search endpoint."""
    if not state.searcher:
        raise HTTPException(status_code=503, detail="Searcher not initialized")

    try:
        start_time = time.time()

        # Perform search
        results = state.searcher.search(
            query=request.query,
            top_k=request.top_k,
            complexity=request.complexity,
            metadata_filters=request.metadata_filters,
        )

        search_time_ms = (time.time() - start_time) * 1000

        # Convert results to dict
        result_dicts = []
        for r in results:
            result_dict = {
                "id": getattr(r, "id", None),
                "score": float(getattr(r, "score", 0.0)),  # Convert numpy.float32 to Python float
                "text": getattr(r, "text", ""),
            }

            if request.show_metadata and hasattr(r, "metadata"):
                result_dict["metadata"] = r.metadata

            result_dicts.append(result_dict)

        return {"results": result_dicts, "search_time_ms": search_time_ms}

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/ask")
async def websocket_ask(websocket: WebSocket):
    """WebSocket endpoint for streaming Q&A with LLM."""
    await websocket.accept()

    if not state.searcher or not state.llm:
        await websocket.send_json(
            {"type": "error", "message": "Searcher or LLM not initialized"}
        )
        await websocket.close()
        return

    try:
        while True:
            # Receive request
            data = await websocket.receive_text()
            request_data = json.loads(data)

            question = request_data.get("question")
            if not question:
                await websocket.send_json({"type": "error", "message": "Missing 'question' field"})
                continue

            top_k = request_data.get("top_k", 5)
            complexity = request_data.get("complexity", 64)
            llm_params = request_data.get("llm_params", {})

            # Perform search
            search_start = time.time()
            try:
                results = state.searcher.search(
                    query=question, top_k=top_k, complexity=complexity
                )
                search_time_ms = (time.time() - search_start) * 1000

                # Send search results
                result_dicts = []
                for r in results:
                    result_dicts.append(
                        {
                            "id": getattr(r, "id", None),
                            "score": float(getattr(r, "score", 0.0)),  # Convert numpy.float32 to Python float
                            "text": getattr(r, "text", ""),
                            "metadata": getattr(r, "metadata", {}),
                        }
                    )

                await websocket.send_json(
                    {"type": "search_results", "results": result_dicts, "search_time_ms": search_time_ms}
                )

            except Exception as e:
                logger.error(f"Search error in WebSocket: {e}")
                await websocket.send_json({"type": "error", "message": f"Search failed: {e}"})
                continue

            # Build prompt with context
            context = "\n\n".join([r.text for r in results])
            prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

            # Stream LLM response
            try:
                # Set default max_tokens if not provided
                if "max_tokens" not in llm_params:
                    llm_params["max_tokens"] = 10000

                # Check if streaming is supported
                if hasattr(state.llm, "ask_stream"):
                    for token in state.llm.ask_stream(prompt, **llm_params):
                        await websocket.send_json({"type": "token", "content": token})
                else:
                    # Fallback: send complete response
                    response = state.llm.ask(prompt, **llm_params)
                    await websocket.send_json({"type": "token", "content": response})

                # Send completion
                await websocket.send_json({"type": "done"})

            except Exception as e:
                logger.error(f"LLM error in WebSocket: {e}")
                await websocket.send_json({"type": "error", "message": f"LLM failed: {e}"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass


# Main entry point
def main():
    """Main entry point for the HTTP server."""
    parser = argparse.ArgumentParser(
        description="LEANN HTTP Server with WebSocket streaming support"
    )

    # Server options
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument(
        "--server-ssl-cert", help="Path to SSL certificate for HTTPS server (optional)"
    )
    parser.add_argument(
        "--server-ssl-key", help="Path to SSL private key for HTTPS server (optional)"
    )

    # Index options
    parser.add_argument(
        "--index-path", required=True, help="Path to LEANN index (e.g., path/to/documents.leann)"
    )

    # LLM options
    parser.add_argument("--model", default="gpt-4o", help="LLM model name (default: gpt-4o)")
    parser.add_argument(
        "--llm-type",
        default="openai",
        choices=["openai", "ollama", "anthropic"],
        help="LLM provider type (default: openai)",
    )
    parser.add_argument(
        "--llm-ssl-cert",
        help="Path to SSL certificate for LLM client connection (SSL_CERT_FILE)",
    )

    # Embedding options
    parser.add_argument(
        "--embedding-model",
        help="Embedding model name (e.g., q-embedding) - stored for reference",
    )

    args = parser.parse_args()

    # Configure state (index_path will be set after resolution)
    state.embedding_model = args.embedding_model

    # Configure LLM
    state.llm_config = {
        "type": args.llm_type,
        "model": args.model,
        "ssl_cert": args.llm_ssl_cert,
    }

    # Add provider-specific configuration
    if args.llm_type == "openai":
        state.llm_config["api_key"] = resolve_openai_api_key()
        state.llm_config["base_url"] = resolve_openai_base_url()
    elif args.llm_type == "ollama":
        state.llm_config["host"] = resolve_ollama_host()
    elif args.llm_type == "anthropic":
        state.llm_config["api_key"] = resolve_anthropic_api_key()

    # Resolve index name to path if needed
    index_path = Path(args.index_path)

    # If the path doesn't exist as-is, try to resolve it as an index name
    if not index_path.exists():
        # Try CLI format: .leann/indexes/<index_name>/documents.index
        # First try current directory, then home directory
        current_dir = Path.cwd()
        search_dirs = [current_dir, Path.home()]

        found = False
        for search_dir in search_dirs:
            cli_index_dir = search_dir / ".leann" / "indexes" / args.index_path
            if cli_index_dir.exists():
                # Look for documents.leann in the directory (CLI format)
                potential_index = cli_index_dir / "documents.leann"
                if potential_index.exists() or (cli_index_dir / "documents.leann.meta.json").exists():
                    # Use documents.leann path (meta file will be documents.leann.meta.json)
                    index_path = potential_index
                    logger.info(f"Resolved index name '{args.index_path}' to {index_path}")
                    found = True
                    break

        if not found:
            logger.error(f"Index not found: {args.index_path}")
            logger.error(f"Tried as path: {args.index_path}")
            logger.error(f"Tried in: {[str(d / '.leann' / 'indexes' / args.index_path) for d in search_dirs]}")
            raise FileNotFoundError(f"Index not found: {args.index_path}")

    # Update state with resolved path
    state.index_path = str(index_path)

    # Run with optional server-side HTTPS
    uvicorn_config = {"host": args.host, "port": args.port}

    if args.server_ssl_cert and args.server_ssl_key:
        uvicorn_config["ssl_certfile"] = args.server_ssl_cert
        uvicorn_config["ssl_keyfile"] = args.server_ssl_key
        logger.info(f"Starting HTTPS server on {args.host}:{args.port}")
    else:
        logger.info(f"Starting HTTP server on {args.host}:{args.port}")

    logger.info(f"Index: {args.index_path}")
    logger.info(f"LLM: {args.llm_type} / {args.model}")

    uvicorn.run(app, **uvicorn_config)


if __name__ == "__main__":
    main()
