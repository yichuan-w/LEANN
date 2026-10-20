"""
Qwen Code RAG example.
Indexes and searches Qwen Code CLI history (~/.qwen-code).
"""

import sys
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from base_rag_example import BaseRAGExample
from chunking import create_text_chunks

from .qwen_data.qwen_reader import QwenReader


class QwenRAG(BaseRAGExample):
    """RAG example for Qwen Code CLI history."""

    def __init__(self):
        super().__init__(
            name="Qwen Code",
            description="Process and query Qwen Code CLI history with LEANN",
            default_index_name="qwen_index",
        )

    def _add_specific_arguments(self, parser):
        """Add Qwen-specific arguments."""
        group = parser.add_argument_group("Qwen Parameters")
        group.add_argument(
            "--qwen-path",
            type=str,
            default="~/.qwen-code",
            help="Path to .qwen-code directory (default: ~/.qwen-code)",
        )

    async def load_data(self, args) -> list[dict[str, Any]]:
        """Load Qwen history and convert to text chunks."""
        print(f"Loading Qwen history from: {args.qwen_path}")

        reader = QwenReader()
        documents = reader.load_data(history_dir=args.qwen_path, max_count=args.max_items)

        if not documents:
            print("No documents found! Check if ~/.qwen-code exists and has history.")
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

    print("\n✨ Qwen Code RAG")
    print("=" * 50)

    rag = QwenRAG()
    asyncio.run(rag.run())
