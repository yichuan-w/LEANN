# Web documents from Parallel Search MCP

Use the opt-in reader to search the web and save source-attributed Markdown for
LEANN's document ingestion. [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
provides free anonymous search in Fast mode and page fetching, with no account or
API key. Free access has rate limits; this reader does not retry failed requests.

Search objectives, queries, and selected URLs are sent to Parallel. This is a
remote data source, unlike LEANN's local retrieval. Avoid including private data
in your search. No existing LEANN application or provider default changes.

## Install and export

From the repository root, create a separate environment for the reader (Python
3.10 or newer). It does not require the LEANN runtime or an LLM:

```bash
uv venv /tmp/leann-parallel
uv pip install --python /tmp/leann-parallel/bin/python -r apps/parallel_data/requirements.txt
/tmp/leann-parallel/bin/python -m apps.parallel_data \
  --objective "Find how LEANN reduces vector index storage" \
  --search-query "LEANN low storage vector index" \
  --max-results 3 \
  --output-dir ./web-data
```

Add `--fetch` to fetch excerpts from the selected URLs after searching. This
uses MCP Streamable HTTP at `https://search.parallel.ai/mcp`. The reader retains
search excerpts when a particular URL cannot be fetched and prints a warning.
It exports excerpts, not guaranteed full pages. `--max-results` limits local
selection and fetching, not the server's search result count. `--timeout` bounds
the entire operation (90 seconds by default).

Each file is named after its URL (for example
`arxiv-org-abs-2506-08276-<hash>.md`) and contains its title, source URL,
publication date when available, and excerpts. Repeating a search refreshes files for matching URLs; other files remain.
Use a fresh directory for a separate collection.

## Search, index, and ask in one command

Inside your LEANN environment, install the reader requirements and run the RAG
app. It searches, chunks the excerpts, builds a LEANN index, and answers with the
configured LLM (`--llm`, see the other `apps/*_rag.py` examples):

```bash
uv pip install -r apps/parallel_data/requirements.txt
python -m apps.parallel_rag \
  --objective "Find how LEANN reduces vector index storage" \
  --search-query "LEANN low storage vector index" \
  --fetch --query "How does LEANN save storage?"
```

Each chunk keeps the page title and source URL so answers can cite them. The
index is reused on later runs; pass `--force-rebuild` after changing the search.

## Use the exported documents

In your existing [LEANN installation](../README.md#installation), ingest the
exported directory using the normal document workflow:

```bash
leann build web-notes --docs ./web-data
leann search web-notes "How does LEANN save storage?"
```

Embedding configuration follows your existing LEANN setup. The export command
only collects documents; it does not build an index or run an agent loop.

For Python callers, `ParallelMCPReader.read_web_data()` returns dictionaries with
`text` and `metadata` fields, including `url`, `title`, `publish_date`, and `source`.

## Reader tests

```bash
uv pip install --python /tmp/leann-parallel/bin/python pytest
/tmp/leann-parallel/bin/python -m pytest -o addopts= tests/test_parallel_mcp_reader.py
```
