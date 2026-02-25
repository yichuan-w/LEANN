# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

LEANN is a lightweight vector database and RAG system. See `CLAUDE.md` for architecture details and standard commands (build, test, lint, run).

### Services

| Service | How to run | Notes |
|---------|-----------|-------|
| LEANN (core + HNSW backend) | `uv run python -c "from leann import LeannBuilder, LeannSearcher; ..."` or `uv run leann build/search/ask` | Fully self-contained; no external services needed for index build + search |
| ZMQ embedding server | Auto-spawned by `EmbeddingServerManager` | No manual setup required |

### Non-obvious caveats

- **C++ compiler**: The default `c++` on Ubuntu 24.04 in this environment is Clang 18, which fails to find `<vector>` and other C++ standard library headers when building the HNSW backend. The system must use GCC (`g++`) as the default C++ compiler: `sudo update-alternatives --set c++ /usr/bin/g++` and `sudo update-alternatives --set cc /usr/bin/gcc`. The `libstdc++.so` dev symlink may also be missing; create it with `sudo ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/libstdc++.so`.
- **DiskANN backend is optional**: It requires `libmkl-full-dev` (Intel MKL), which is very large. For development, the HNSW backend alone is sufficient. Tests marked with `diskann` will fail without it—this is expected.
- **Model downloads on first run**: The first `uv run pytest` or `uv run leann build` downloads the `facebook/contriever` embedding model (~440 MB). Some subprocess-based tests may timeout on the first run while this download occurs.
- **LLM for chat/ask**: `leann ask` and `LeannChat` default to OpenAI and require `OPENAI_API_KEY`. For testing without an API key, use `--llm hf --llm-model Qwen/Qwen3-0.6B` or just use `leann search` (no LLM needed).
- **Tests**: Run `uv run pytest -m "not slow and not openai and not integration"` for the standard test suite. DiskANN and subprocess-timeout failures are expected in this environment.
- **Lint**: `uv run ruff check` and `uv run ruff format --check`. See `CLAUDE.md` for details.
