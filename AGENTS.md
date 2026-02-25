# AGENTS.md

## Cursor Cloud specific instructions

### Overview

LEANN is a Python monorepo with C++ native extensions (FAISS-HNSW, DiskANN backends). The primary package manager is `uv`. See `CLAUDE.md` for build/test/lint commands and architecture details.

### Environment caveats

- **C++ compiler**: The default system compiler on Ubuntu 24.04 is Clang 18, which cannot find libstdc++ headers. You must export `CC=gcc CXX=g++` before running `uv sync` or any build that compiles the C++ backends. These are set in `~/.bashrc`.
- **libstdc++ symlink**: A symlink `/usr/lib/x86_64-linux-gnu/libstdc++.so -> /usr/lib/gcc/x86_64-linux-gnu/13/libstdc++.so` is required for the linker. If missing, create it with `sudo ln -sf /usr/lib/gcc/x86_64-linux-gnu/13/libstdc++.so /usr/lib/x86_64-linux-gnu/libstdc++.so`.
- **DiskANN backend**: Not built by default. Use `uv sync --extra diskann` if DiskANN tests are needed. DiskANN test failures are expected when only the HNSW backend is installed.
- **Model downloads**: First-time test runs download the `facebook/contriever` embedding model (~500MB). Some tests may timeout on first run due to this. Subsequent runs use the cached model.

### Running services

- No external services (databases, Docker, etc.) are required for core functionality.
- `uv run leann build <name> --docs <path>` builds an index; `uv run leann search <name> "query"` searches it.
- Tests requiring OpenAI API, Ollama, or LM Studio are gated behind `@pytest.mark.openai` / `@pytest.mark.integration` markers and skipped by default.

### Quick reference

| Task | Command |
|------|---------|
| Install deps | `uv sync --group lint --group test --group dev` |
| Lint | `uv run ruff check .` and `uv run ruff format --check .` |
| Test (fast) | `uv run pytest -m "not openai and not integration and not slow"` |
| Test (all local) | `uv run pytest -m "not openai and not integration"` |
| Build index | `uv run leann build <name> --docs <path>` |
| Search index | `uv run leann search <name> "query"` |
