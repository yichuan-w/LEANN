"""
Code RAG example using AST-aware chunking for optimal code understanding.
Specialized for code repositories with automatic language detection and
optimized chunking parameters.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from base_rag_example import BaseRAGExample
from chunking import CODE_EXTENSIONS, create_text_chunks
from llama_index.core import SimpleDirectoryReader


class CodeRAG(BaseRAGExample):
    """Specialized RAG example for code repositories with AST-aware chunking."""

    def __init__(self):
        super().__init__(
            name="Code",
            description="Process and query code repositories with AST-aware chunking",
            default_index_name="code_index",
        )
        # Override defaults for code-specific usage
        self.embedding_model_default = "facebook/contriever"  # Good for code
        self.max_items_default = -1  # Process all code files by default

    def _add_specific_arguments(self, parser):
        """Add code-specific arguments."""
        code_group = parser.add_argument_group("Code Repository Parameters")

        code_group.add_argument(
            "--repo-dir",
            type=str,
            default=".",
            help="Code repository directory to index (default: current directory)",
        )
        code_group.add_argument(
            "--include-extensions",
            nargs="+",
            default=list(CODE_EXTENSIONS.keys()),
            help="File extensions to include (default: supported code extensions)",
        )
        code_group.add_argument(
            "--exclude-dirs",
            nargs="+",
            default=[
                ".git",
                "__pycache__",
                "node_modules",
                "venv",
                ".venv",
                "build",
                "dist",
                "target",
            ],
            help="Directories to exclude from indexing",
        )
        code_group.add_argument(
            "--max-file-size",
            type=int,
            default=1000000,  # 1MB
            help="Maximum file size in bytes to process (default: 1MB)",
        )
        code_group.add_argument(
            "--include-comments",
            action="store_true",
            help="Include comments in chunking (useful for documentation)",
        )
        code_group.add_argument(
            "--preserve-imports",
            action="store_true",
            default=True,
            help="Try to preserve import statements in chunks (default: True)",
        )

    async def load_data(self, args) -> list[str]:
        """Load code files and convert to AST-aware chunks."""
        print(f"🔍 Scanning code repository: {args.repo_dir}")
        print(f"📁 Including extensions: {args.include_extensions}")
        print(f"🚫 Excluding directories: {args.exclude_dirs}")

        # Check if repository directory exists
        repo_path = Path(args.repo_dir)
        if not repo_path.exists():
            raise ValueError(f"Repository directory not found: {args.repo_dir}")

        # Load code files with filtering
        reader_kwargs = {
            "recursive": True,
            "encoding": "utf-8",
            "required_exts": args.include_extensions,
            "exclude_hidden": True,
        }

        # Create exclusion filter
        def file_filter(file_path: str) -> bool:
            """Filter out unwanted files and directories."""
            path = Path(file_path)

            # Check file size
            try:
                if path.stat().st_size > args.max_file_size:
                    print(f"⚠️ Skipping large file: {path.name} ({path.stat().st_size} bytes)")
                    return False
            except Exception:
                return False

            # Check if in excluded directory
            for exclude_dir in args.exclude_dirs:
                if exclude_dir in path.parts:
                    return False

            return True

        try:
            # Load documents with file filtering
            documents = SimpleDirectoryReader(
                args.repo_dir,
                file_extractor=None,  # Use default extractors
                **reader_kwargs,
            ).load_data(show_progress=True)

            # Apply custom filtering
            filtered_docs = []
            for doc in documents:
                file_path = doc.metadata.get("file_path", "")
                if file_filter(file_path):
                    filtered_docs.append(doc)

            documents = filtered_docs

        except Exception as e:
            print(f"❌ Error loading code files: {e}")
            return []

        if not documents:
            print(
                f"❌ No code files found in {args.repo_dir} with extensions {args.include_extensions}"
            )
            return []

        print(f"✅ Loaded {len(documents)} code files")

        # Show breakdown by language/extension
        ext_counts = {}
        for doc in documents:
            file_path = doc.metadata.get("file_path", "")
            if file_path:
                ext = Path(file_path).suffix.lower()
                ext_counts[ext] = ext_counts.get(ext, 0) + 1

        print("📊 Files by extension:")
        for ext, count in sorted(ext_counts.items()):
            print(f"   {ext}: {count} files")

        # Use AST-aware chunking by default for code
        print(
            f"🧠 Using AST-aware chunking (chunk_size: {args.ast_chunk_size}, overlap: {args.ast_chunk_overlap})"
        )

        all_texts = create_text_chunks(
            documents,
            chunk_size=256,  # Fallback for non-code files
            chunk_overlap=64,
            use_ast_chunking=True,  # Always use AST for code RAG
            ast_chunk_size=args.ast_chunk_size,
            ast_chunk_overlap=args.ast_chunk_overlap,
            code_file_extensions=args.include_extensions,
            ast_fallback_traditional=True,
        )

        # Apply max_items limit if specified
        if args.max_items > 0 and len(all_texts) > args.max_items:
            print(f"⏳ Limiting to {args.max_items} chunks (from {len(all_texts)})")
            all_texts = all_texts[: args.max_items]

        print(f"✅ Generated {len(all_texts)} code chunks")
        return all_texts


if __name__ == "__main__":
    import asyncio

    # Example queries for code RAG
    print("\n💻 Code RAG Example")
    print("=" * 50)
    print("\nExample queries you can try:")
    print("- 'How does the embedding computation work?'")
    print("- 'What are the main classes in this codebase?'")
    print("- 'Show me the search implementation'")
    print("- 'How is error handling implemented?'")
    print("- 'What design patterns are used?'")
    print("- 'Explain the chunking logic'")
    print("\n🚀 Features:")
    print("- ✅ AST-aware chunking preserves code structure")
    print("- ✅ Automatic language detection")
    print("- ✅ Smart filtering of large files and common excludes")
    print("- ✅ Optimized for code understanding")
    print("\nUsage examples:")
    print("  python -m apps.code_rag --repo-dir ./my_project")
    print(
        "  python -m apps.code_rag --include-extensions .py .js --query 'How does authentication work?'"
    )
    print("\nOr run without --query for interactive mode\n")

    rag = CodeRAG()
    asyncio.run(rag.run())
