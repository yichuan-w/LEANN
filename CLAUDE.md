# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LEANN is a lightweight vector database and RAG (Retrieval-Augmented Generation) system that achieves 97% storage reduction compared to traditional vector databases through graph-based selective recomputation. It enables semantic search across various data sources (emails, browser history, chat history, code, documents) on a single laptop without cloud dependencies.

## Build & Development Commands

```bash
# Install uv first (required package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Development setup (from source)
git submodule update --init --recursive

# macOS
brew install libomp boost protobuf zeromq pkgconf
uv sync --extra diskann

# Ubuntu/Debian
sudo apt-get install libomp-dev libboost-all-dev protobuf-compiler \
    libabsl-dev libmkl-full-dev libaio-dev libzmq3-dev
uv sync --extra diskann

# Install lint tools
uv sync --group lint

# Install test tools
uv sync --group test
```

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_basic.py

# Run with coverage
uv run pytest --cov=leann

# Skip slow tests
uv run pytest -m "not slow"

# Skip tests requiring OpenAI API
uv run pytest -m "not openai"
```

Test markers: `slow`, `openai`, `integration`

## Code Quality

```bash
# Format code
ruff format

# Lint with auto-fix
ruff check --fix

# Pre-commit hooks (install once)
pre-commit install

# Run pre-commit manually
uv run pre-commit run --all-files
```

## Architecture

### Core API Layer (`packages/leann-core/src/leann/`)

- `api.py`: Main APIs - `LeannBuilder`, `LeannSearcher`, `LeannChat`
- `react_agent.py`: `ReActAgent` for multi-turn reasoning
- `cli.py`: CLI implementation (`leann build`, `leann search`, `leann ask`)
- `chat.py`: LLM provider integrations (OpenAI, Ollama, HuggingFace, Anthropic)
- `embedding_compute.py`: Embedding computation (sentence-transformers, MLX, OpenAI)
- `metadata_filter.py`: Search result filtering by metadata

### Backend Layer (`packages/`)

- `leann-backend-hnsw/`: Default backend using FAISS HNSW for fast in-memory search
- `leann-backend-diskann/`: DiskANN backend for larger-than-memory datasets
- `leann-mcp/`: MCP server for Claude Code integration

Backends are auto-discovered via `leann-backend-*` naming convention and registered in `registry.py`.

### RAG Applications (`apps/`)

Example applications demonstrating RAG on various data sources:
- `document_rag.py`: PDF/TXT/MD documents
- `email_rag.py`: Apple Mail
- `browser_rag.py`: Chrome browser history
- `wechat_rag.py`, `imessage_rag.py`: Chat history
- `code_rag.py`: Codebase search with AST-aware chunking
- `slack_rag.py`, `twitter_rag.py`: MCP-based live data

## Key Design Patterns

### Threading Environment Variables

LEANN sets critical threading environment variables in `__init__.py` to prevent FAISS/ZMQ hangs:
- macOS: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `KMP_DUPLICATE_LIB_OK=TRUE`
- Linux: `OMP_NUM_THREADS=1`, `FAISS_NUM_THREADS=1`, `OMP_WAIT_POLICY=PASSIVE`

### Index Structure

A LEANN index consists of:
- `<name>.meta.json`: Metadata (backend, embedding model, dimensions)
- `<name>.passages.jsonl`: Raw text chunks with metadata
- `<name>.passages.idx`: Offset map for fast passage lookup
- `<name>.index`: Backend-specific vector index

### Embedding Recomputation

The core storage optimization: instead of storing embeddings, LEANN stores a pruned graph and recomputes embeddings on-demand during search via ZMQ server communication.

## CLI Usage

```bash
# Build index
leann build my-docs --docs ./documents/

# Search
leann search my-docs "query"

# Interactive chat
leann ask my-docs --interactive

# List indexes
leann list

# Remove index
leann remove my-docs
```

## Common Development Tasks

Running example RAG applications:
```bash
# Document RAG (easiest to test)
python -m apps.document_rag --query "What is LEANN?"

# Code RAG
python -m apps.code_rag --repo-dir ./src --query "How does search work?"
```

## Python Version

Requires Python 3.10+ (uses PEP 604 union syntax `X | Y`).
