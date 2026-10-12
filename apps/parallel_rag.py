"""
Web RAG example using Parallel Search MCP and the unified interface.

Searches the web (optionally fetching the selected pages) through anonymous
Parallel Search MCP, then chunks and indexes the excerpts in LEANN.

Usage:
    python -m apps.parallel_rag --objective "Find Yichuan Wang's MLSys research" \\
        --search-query "Yichuan Wang MLsys" --fetch --query "What did he publish at MLSys?"
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from base_rag_example import BaseRAGExample
from chunking import create_text_chunks
from llama_index.core import Document

from .parallel_data.parallel_mcp_reader import ParallelMCPReader


class ParallelWebRAG(BaseRAGExample):
    """RAG example for web documents from Parallel Search MCP."""

    def __init__(self):
        super().__init__(
            name="Parallel Web",
            description="Search the web with Parallel Search MCP and run RAG over the results",
            default_index_name="parallel_web_index",
        )

    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        web_group = parser.add_argument_group("Parallel Web Parameters")
        web_group.add_argument(
            "--objective", required=True, help="What information to find on the web"
        )
        web_group.add_argument(
            "--search-query", action="append", required=True, help="Repeat for related queries"
        )
        web_group.add_argument(
            "--max-results", type=int, default=5, help="Select up to 1-20 results locally"
        )
        web_group.add_argument(
            "--fetch", action="store_true", help="Fetch excerpts from selected result URLs"
        )
        web_group.add_argument(
            "--timeout", type=float, default=90, help="Overall web timeout in seconds"
        )
        web_group.add_argument(
            "--chunk-size", type=int, default=256, help="Text chunk size (default: 256)"
        )
        web_group.add_argument(
            "--chunk-overlap", type=int, default=128, help="Text chunk overlap (default: 128)"
        )

    async def load_data(self, args) -> list[dict[str, Any]]:
        print(f"Searching the web via Parallel Search MCP: {args.search_query}")
        web_documents = await ParallelMCPReader(timeout=args.timeout).read_web_data(
            args.objective, args.search_query, args.max_results, args.fetch
        )
        for document in web_documents:
            print(f"  - {document['metadata']['title']} ({document['metadata']['url']})")
        documents = [
            # Keep title and URL in the text so the LLM can cite its sources.
            Document(
                text=f"{doc['metadata']['title']}\nSource: {doc['metadata']['url']}\n\n{doc['text']}",
                metadata=doc["metadata"],
            )
            for doc in web_documents
        ]
        texts = create_text_chunks(
            documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        )
        print(f"Created {len(texts)} text chunks from {len(documents)} web documents")
        return texts


if __name__ == "__main__":
    asyncio.run(ParallelWebRAG().run())
