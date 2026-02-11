# Getting Started with LEANN

Welcome! This tutorial will guide you through installing LEANN and performing your first semantic search in under 5 minutes.

## 1. Prerequisites

LEANN requires **Python 3.10+** and [uv](https://docs.astral.sh/uv/) for the best installation experience.

**For macOS Users:**
Install system dependencies for the high-performance backends:
```bash
brew install libomp boost protobuf zeromq pkgconf
```

## 2. Global Installation

Install LEANN as a global tool with all supported backends (HNSW, DiskANN, and AST-aware chunking):

```bash
uv tool install leann \
  --with leann-backend-hnsw \
  --with leann-backend-diskann \
  --with astchunk-leann
```

CPU-only install (Linux):

```bash
uv tool install "leann[cpu]" \
  --default-index https://download.pytorch.org/whl/cpu \
  --index https://pypi.org/simple \
  --index-strategy first-index \
  --with leann-backend-hnsw \
  --with leann-backend-diskann \
  --with astchunk-leann
```

Verify the installation:
```bash
leann --help
```

## 3. Your First Index

Pick a folder containing some text files, PDFs, or Markdown notes and create an index named `my-notes`:

```bash
leann build my-notes --docs ~/Documents/Notes
```

## 4. Semantic Search

Find information in your notes using natural language instead of keywords:

```bash
leann search my-notes "What were the main points of the last meeting?"
```

## 5. Interactive RAG

Chat with your data using a local LLM via Ollama (ensure Ollama is running):

```bash
leann ask my-notes "Summarize the project goals mentioned in my notes" --interactive
```

Next: Check out the [Ingestion Guides](ingestion-guides.md) to learn how to index your emails, calendar, and code.
