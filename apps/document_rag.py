"""
Document RAG example using the unified interface.
Supports PDF, TXT, MD, and other document formats.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from base_rag_example import BaseRAGExample
from chunking import create_text_chunks
from llama_index.core import SimpleDirectoryReader


class DocumentRAG(BaseRAGExample):
    """RAG example for document processing (PDF, TXT, MD, etc.)."""

    def __init__(self):
        super().__init__(
            name="Document",
            description="Process and query documents (PDF, TXT, MD, etc.) with LEANN",
            default_index_name="test_doc_files",
        )

    def _add_specific_arguments(self, parser):
        """Add document-specific arguments."""
        doc_group = parser.add_argument_group("Document Parameters")
        doc_group.add_argument(
            "--data-dir",
            type=str,
            default="data",
            help="Directory containing documents to index (default: data)",
        )
        doc_group.add_argument(
            "--file-types",
            nargs="+",
            default=None,
            help="Filter by file types (e.g., .pdf .txt .md). If not specified, all supported types are processed",
        )
        doc_group.add_argument(
            "--chunk-size", type=int, default=256, help="Text chunk size (default: 256)"
        )
        doc_group.add_argument(
            "--chunk-overlap", type=int, default=128, help="Text chunk overlap (default: 128)"
        )
        doc_group.add_argument(
            "--enable-code-chunking",
            action="store_true",
            help="Enable AST-aware chunking for code files in the data directory",
        )

    async def load_data(self, args) -> list[str]:
        """Load documents and convert to text chunks."""
        print(f"Loading documents from: {args.data_dir}")
        if args.file_types:
            print(f"Filtering by file types: {args.file_types}")
        else:
            print("Processing all supported file types")

        # Check if data directory exists
        data_path = Path(args.data_dir)
        if not data_path.exists():
            raise ValueError(f"Data directory not found: {args.data_dir}")

        # Load documents
        reader_kwargs = {
            "recursive": True,
            "encoding": "utf-8",
        }
        if args.file_types:
            reader_kwargs["required_exts"] = args.file_types

        documents = SimpleDirectoryReader(args.data_dir, **reader_kwargs).load_data(
            show_progress=True
        )

        if not documents:
            print(f"No documents found in {args.data_dir} with extensions {args.file_types}")
            return []

        print(f"Loaded {len(documents)} documents")

        # Determine chunking strategy
        use_ast = args.enable_code_chunking or getattr(args, "use_ast_chunking", False)

        if use_ast:
            print("Using AST-aware chunking for code files")

        # Convert to text chunks with optional AST support
        all_texts = create_text_chunks(
            documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            use_ast_chunking=use_ast,
            ast_chunk_size=getattr(args, "ast_chunk_size", 512),
            ast_chunk_overlap=getattr(args, "ast_chunk_overlap", 64),
            code_file_extensions=getattr(args, "code_file_extensions", None),
            ast_fallback_traditional=getattr(args, "ast_fallback_traditional", True),
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
    print("- 'What is the technique DLPM?'")
    print("- 'Who does Elizabeth Bennet marry?'")
    print(
        "- 'What is the problem of developing pan gu model Huawei meets? (盘古大模型开发中遇到什么问题?)'"
    )
    print("\n🚀 NEW: Code-aware chunking available!")
    print("- Use --enable-code-chunking to enable AST-aware chunking for code files")
    print("- Supports Python, Java, C#, TypeScript files")
    print("- Better semantic understanding of code structure")
    print("\nOr run without --query for interactive mode\n")

    rag = DocumentRAG()
    asyncio.run(rag.run())
