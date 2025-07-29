"""
Document RAG example using the unified interface.
Supports PDF, TXT, MD, and other document formats.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from base_rag_example import BaseRAGExample, create_text_chunks
from llama_index.core import SimpleDirectoryReader


class DocumentRAG(BaseRAGExample):
    """RAG example for document processing (PDF, TXT, MD, etc.)."""

    def __init__(self):
        super().__init__(
            name="Document",
            description="Process and query documents (PDF, TXT, MD, etc.) with LEANN",
            default_index_name="test_doc_files",  # Match original main_cli_example.py default
        )

    def _add_specific_arguments(self, parser):
        """Add document-specific arguments."""
        doc_group = parser.add_argument_group("Document Parameters")
        doc_group.add_argument(
            "--data-dir",
            type=str,
            default="examples/data",
            help="Directory containing documents to index (default: examples/data)",
        )
        doc_group.add_argument(
            "--file-types",
            nargs="+",
            default=[".pdf", ".txt", ".md"],
            help="File types to process (default: .pdf .txt .md)",
        )
        doc_group.add_argument(
            "--chunk-size", type=int, default=256, help="Text chunk size (default: 256)"
        )
        doc_group.add_argument(
            "--chunk-overlap", type=int, default=128, help="Text chunk overlap (default: 128)"
        )

    async def load_data(self, args) -> list[str]:
        """Load documents and convert to text chunks."""
        print(f"Loading documents from: {args.data_dir}")
        print(f"File types: {args.file_types}")

        # Check if data directory exists
        data_path = Path(args.data_dir)
        if not data_path.exists():
            raise ValueError(f"Data directory not found: {args.data_dir}")

        # Load documents
        documents = SimpleDirectoryReader(
            args.data_dir,
            recursive=True,
            encoding="utf-8",
            required_exts=args.file_types,
        ).load_data(show_progress=True)

        if not documents:
            print(f"No documents found in {args.data_dir} with extensions {args.file_types}")
            return []

        print(f"Loaded {len(documents)} documents")

        # Convert to text chunks
        all_texts = create_text_chunks(
            documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        )

        # Apply max_items limit if specified
        if args.max_items > 0 and len(all_texts) > args.max_items:
            print(f"Limiting to {args.max_items} chunks (from {len(all_texts)})")
            all_texts = all_texts[: args.max_items]

        return all_texts


if __name__ == "__main__":
    import asyncio

    # Example queries for document RAG
    print("\n📄 Document RAG Example")
    print("=" * 50)
    print("\nExample queries you can try:")
    print("- 'What are the main techniques LEANN uses?'")
    print("- 'Summarize the key findings in these papers'")
    print("- 'What is the storage reduction achieved by LEANN?'")
    print("\nOr run without --query for interactive mode\n")

    rag = DocumentRAG()
    asyncio.run(rag.run())
