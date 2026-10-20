"""
Gemini CLI RAG example.
Indexes and searches Gemini CLI history (~/.gemini).
"""

import sys
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from base_rag_example import BaseRAGExample
from chunking import create_text_chunks

from .gemini_data.gemini_reader import GeminiReader


class GeminiRAG(BaseRAGExample):
    """RAG example for Gemini CLI history."""

    def __init__(self):
        super().__init__(
            name="Gemini CLI",
            description="Process and query Gemini CLI history with LEANN",
            default_index_name="gemini_index",
        )

    def _add_specific_arguments(self, parser):
        """Add Gemini-specific arguments."""
        group = parser.add_argument_group("Gemini Parameters")
        group.add_argument(
            "--gemini-path",
            type=str,
            default="~/.gemini",
            help="Path to .gemini directory (default: ~/.gemini)",
        )

    async def load_data(self, args) -> list[dict[str, Any]]:
        """Load Gemini history and convert to text chunks."""
        print(f"Loading Gemini history from: {args.gemini_path}")

        reader = GeminiReader()
        documents = reader.load_data(history_dir=args.gemini_path, max_count=args.max_items)

        if not documents:
            print("No documents found! Check if ~/.gemini exists and has history.")
            return []

        # Convert dicts to Document objects for chunking
        from llama_index.core import Document

        docs = [Document(text=d["text"], metadata=d["metadata"]) for d in documents]

        # Convert to text chunks
        print(f"splitting {len(documents)} documents into chunks...")
        chunks = create_text_chunks(docs)

        return chunks


if __name__ == "__main__":
    import asyncio

    print("\n✨ Gemini CLI RAG")
    print("=" * 50)

    rag = GeminiRAG()
    asyncio.run(rag.run())
