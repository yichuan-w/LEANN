import argparse
import asyncio
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from .api import LeannBuilder, LeannChat, LeannSearcher


def extract_pdf_text_with_pymupdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF for better quality."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except ImportError:
        # Fallback to default reader
        return None


def extract_pdf_text_with_pdfplumber(file_path: str) -> str:
    """Extract text from PDF using pdfplumber for better quality."""
    try:
        import pdfplumber

        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except ImportError:
        # Fallback to default reader
        return None


class LeannCLI:
    def __init__(self):
        # Always use project-local .leann directory (like .git)
        self.indexes_dir = Path.cwd() / ".leann" / "indexes"
        self.indexes_dir.mkdir(parents=True, exist_ok=True)

        # Default parser for documents
        self.node_parser = SentenceSplitter(
            chunk_size=256, chunk_overlap=128, separator=" ", paragraph_separator="\n\n"
        )

        # Code-optimized parser
        self.code_parser = SentenceSplitter(
            chunk_size=512,  # Larger chunks for code context
            chunk_overlap=50,  # Less overlap to preserve function boundaries
            separator="\n",  # Split by lines for code
            paragraph_separator="\n\n",  # Preserve logical code blocks
        )

    def get_index_path(self, index_name: str) -> str:
        index_dir = self.indexes_dir / index_name
        return str(index_dir / "documents.leann")

    def index_exists(self, index_name: str) -> bool:
        index_dir = self.indexes_dir / index_name
        meta_file = index_dir / "documents.leann.meta.json"
        return meta_file.exists()

    def create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="leann",
            description="LEANN - Local Enhanced AI Navigation",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  leann build my-docs --docs ./documents                    # Build index named my-docs
  leann build my-ppts --docs ./ --file-types .pptx,.pdf    # Index only PowerPoint and PDF files
  leann search my-docs "query"                             # Search in my-docs index
  leann ask my-docs "question"                             # Ask my-docs index
  leann list                                              # List all stored indexes
            """,
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Build command
        build_parser = subparsers.add_parser("build", help="Build document index")
        build_parser.add_argument("index_name", help="Index name")
        build_parser.add_argument(
            "--docs", type=str, default=".", help="Documents directory (default: current directory)"
        )
        build_parser.add_argument(
            "--backend", type=str, default="hnsw", choices=["hnsw", "diskann"]
        )
        build_parser.add_argument("--embedding-model", type=str, default="facebook/contriever")
        build_parser.add_argument(
            "--embedding-mode",
            type=str,
            default="sentence-transformers",
            choices=["sentence-transformers", "openai", "mlx", "ollama"],
            help="Embedding backend mode (default: sentence-transformers)",
        )
        build_parser.add_argument("--force", "-f", action="store_true", help="Force rebuild")
        build_parser.add_argument("--graph-degree", type=int, default=32)
        build_parser.add_argument("--complexity", type=int, default=64)
        build_parser.add_argument("--num-threads", type=int, default=1)
        build_parser.add_argument("--compact", action="store_true", default=True)
        build_parser.add_argument("--recompute", action="store_true", default=True)
        build_parser.add_argument(
            "--file-types",
            type=str,
            help="Comma-separated list of file extensions to include (e.g., '.txt,.pdf,.pptx'). If not specified, uses default supported types.",
        )

        # Search command
        search_parser = subparsers.add_parser("search", help="Search documents")
        search_parser.add_argument("index_name", help="Index name")
        search_parser.add_argument("query", help="Search query")
        search_parser.add_argument("--top-k", type=int, default=5)
        search_parser.add_argument("--complexity", type=int, default=64)
        search_parser.add_argument("--beam-width", type=int, default=1)
        search_parser.add_argument("--prune-ratio", type=float, default=0.0)
        search_parser.add_argument(
            "--recompute-embeddings",
            action="store_true",
            default=True,
            help="Recompute embeddings (default: True)",
        )
        search_parser.add_argument(
            "--pruning-strategy",
            choices=["global", "local", "proportional"],
            default="global",
        )

        # Ask command
        ask_parser = subparsers.add_parser("ask", help="Ask questions")
        ask_parser.add_argument("index_name", help="Index name")
        ask_parser.add_argument(
            "--llm",
            type=str,
            default="ollama",
            choices=["simulated", "ollama", "hf", "openai"],
        )
        ask_parser.add_argument("--model", type=str, default="qwen3:8b")
        ask_parser.add_argument("--host", type=str, default="http://localhost:11434")
        ask_parser.add_argument("--interactive", "-i", action="store_true")
        ask_parser.add_argument("--top-k", type=int, default=20)
        ask_parser.add_argument("--complexity", type=int, default=32)
        ask_parser.add_argument("--beam-width", type=int, default=1)
        ask_parser.add_argument("--prune-ratio", type=float, default=0.0)
        ask_parser.add_argument(
            "--recompute-embeddings",
            action="store_true",
            default=True,
            help="Recompute embeddings (default: True)",
        )
        ask_parser.add_argument(
            "--pruning-strategy",
            choices=["global", "local", "proportional"],
            default="global",
        )
        ask_parser.add_argument(
            "--thinking-budget",
            type=str,
            choices=["low", "medium", "high"],
            default=None,
            help="Thinking budget for reasoning models (low/medium/high). Supported by GPT-Oss:20b and other reasoning models.",
        )

        # List command
        subparsers.add_parser("list", help="List all indexes")

        return parser

    def register_project_dir(self):
        """Register current project directory in global registry"""
        global_registry = Path.home() / ".leann" / "projects.json"
        global_registry.parent.mkdir(exist_ok=True)

        current_dir = str(Path.cwd())

        # Load existing registry
        projects = []
        if global_registry.exists():
            try:
                import json

                with open(global_registry) as f:
                    projects = json.load(f)
            except Exception:
                projects = []

        # Add current directory if not already present
        if current_dir not in projects:
            projects.append(current_dir)

        # Save registry
        import json

        with open(global_registry, "w") as f:
            json.dump(projects, f, indent=2)

    def list_indexes(self):
        print("Stored LEANN indexes:")

        # Get all project directories with .leann
        global_registry = Path.home() / ".leann" / "projects.json"
        all_projects = []

        if global_registry.exists():
            try:
                import json

                with open(global_registry) as f:
                    all_projects = json.load(f)
            except Exception:
                pass

        # Filter to only existing directories with .leann
        valid_projects = []
        for project_dir in all_projects:
            project_path = Path(project_dir)
            if project_path.exists() and (project_path / ".leann" / "indexes").exists():
                valid_projects.append(project_path)

        # Add current project if it has .leann but not in registry
        current_path = Path.cwd()
        if (current_path / ".leann" / "indexes").exists() and current_path not in valid_projects:
            valid_projects.append(current_path)

        if not valid_projects:
            print("No indexes found. Use 'leann build <name> --docs <dir>' to create one.")
            return

        total_indexes = 0
        current_dir = Path.cwd()

        for project_path in valid_projects:
            indexes_dir = project_path / ".leann" / "indexes"
            if not indexes_dir.exists():
                continue

            index_dirs = [d for d in indexes_dir.iterdir() if d.is_dir()]
            if not index_dirs:
                continue

            # Show project header
            if project_path == current_dir:
                print(f"\n📁 Current project ({project_path}):")
            else:
                print(f"\n📂 {project_path}:")

            for index_dir in index_dirs:
                total_indexes += 1
                index_name = index_dir.name
                meta_file = index_dir / "documents.leann.meta.json"
                status = "✓" if meta_file.exists() else "✗"

                print(f"  {total_indexes}. {index_name} [{status}]")
                if status == "✓":
                    size_mb = sum(f.stat().st_size for f in index_dir.iterdir() if f.is_file()) / (
                        1024 * 1024
                    )
                    print(f"     Size: {size_mb:.1f} MB")

        if total_indexes > 0:
            print(f"\nTotal: {total_indexes} indexes across {len(valid_projects)} projects")
            print("\nUsage (current project only):")

            # Show example from current project
            current_indexes_dir = current_dir / ".leann" / "indexes"
            if current_indexes_dir.exists():
                current_index_dirs = [d for d in current_indexes_dir.iterdir() if d.is_dir()]
                if current_index_dirs:
                    example_name = current_index_dirs[0].name
                    print(f'  leann search {example_name} "your query"')
                    print(f"  leann ask {example_name} --interactive")

    def load_documents(self, docs_dir: str, custom_file_types: str | None = None):
        print(f"Loading documents from {docs_dir}...")
        if custom_file_types:
            print(f"Using custom file types: {custom_file_types}")

        # Try to use better PDF parsers first
        documents = []
        docs_path = Path(docs_dir)

        for file_path in docs_path.rglob("*.pdf"):
            print(f"Processing PDF: {file_path}")

            # Try PyMuPDF first (best quality)
            text = extract_pdf_text_with_pymupdf(str(file_path))
            if text is None:
                # Try pdfplumber
                text = extract_pdf_text_with_pdfplumber(str(file_path))

            if text:
                # Create a simple document structure
                from llama_index.core import Document

                doc = Document(text=text, metadata={"source": str(file_path)})
                documents.append(doc)
            else:
                # Fallback to default reader
                print(f"Using default reader for {file_path}")
                default_docs = SimpleDirectoryReader(
                    str(file_path.parent),
                    filename_as_id=True,
                    required_exts=[file_path.suffix],
                ).load_data()
                documents.extend(default_docs)

        # Load other file types with default reader
        if custom_file_types:
            # Parse custom file types from comma-separated string
            code_extensions = [ext.strip() for ext in custom_file_types.split(",") if ext.strip()]
            # Ensure extensions start with a dot
            code_extensions = [ext if ext.startswith(".") else f".{ext}" for ext in code_extensions]
        else:
            # Use default supported file types
            code_extensions = [
                # Original document types
                ".txt",
                ".md",
                ".docx",
                ".pptx",
                # Code files for Claude Code integration
                ".py",
                ".js",
                ".ts",
                ".jsx",
                ".tsx",
                ".java",
                ".cpp",
                ".c",
                ".h",
                ".hpp",
                ".cs",
                ".go",
                ".rs",
                ".rb",
                ".php",
                ".swift",
                ".kt",
                ".scala",
                ".r",
                ".sql",
                ".sh",
                ".bash",
                ".zsh",
                ".fish",
                ".ps1",
                ".bat",
                # Config and markup files
                ".json",
                ".yaml",
                ".yml",
                ".xml",
                ".toml",
                ".ini",
                ".cfg",
                ".conf",
                ".html",
                ".css",
                ".scss",
                ".less",
                ".vue",
                ".svelte",
                # Data science
                ".ipynb",
                ".R",
                ".py",
                ".jl",
            ]
        # Try to load other file types, but don't fail if none are found
        try:
            other_docs = SimpleDirectoryReader(
                docs_dir,
                recursive=True,
                encoding="utf-8",
                required_exts=code_extensions,
            ).load_data(show_progress=True)
            documents.extend(other_docs)
        except ValueError as e:
            if "No files found" in str(e):
                print("No additional files found for other supported types.")
            else:
                raise e

        all_texts = []

        # Define code file extensions for intelligent chunking
        code_file_exts = {
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".hpp",
            ".cs",
            ".go",
            ".rs",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".scala",
            ".r",
            ".sql",
            ".sh",
            ".bash",
            ".zsh",
            ".fish",
            ".ps1",
            ".bat",
            ".json",
            ".yaml",
            ".yml",
            ".xml",
            ".toml",
            ".ini",
            ".cfg",
            ".conf",
            ".html",
            ".css",
            ".scss",
            ".less",
            ".vue",
            ".svelte",
            ".ipynb",
            ".R",
            ".jl",
        }

        for doc in documents:
            # Check if this is a code file based on source path
            source_path = doc.metadata.get("source", "")
            is_code_file = any(source_path.endswith(ext) for ext in code_file_exts)

            # Use appropriate parser based on file type
            parser = self.code_parser if is_code_file else self.node_parser
            nodes = parser.get_nodes_from_documents([doc])

            for node in nodes:
                all_texts.append(node.get_content())

        print(f"Loaded {len(documents)} documents, {len(all_texts)} chunks")
        return all_texts

    async def build_index(self, args):
        docs_dir = args.docs
        index_name = args.index_name
        index_dir = self.indexes_dir / index_name
        index_path = self.get_index_path(index_name)

        print(f"📂 Indexing: {Path(docs_dir).resolve()}")

        if index_dir.exists() and not args.force:
            print(f"Index '{index_name}' already exists. Use --force to rebuild.")
            return

        all_texts = self.load_documents(docs_dir, args.file_types)
        if not all_texts:
            print("No documents found")
            return

        index_dir.mkdir(parents=True, exist_ok=True)

        print(f"Building index '{index_name}' with {args.backend} backend...")

        builder = LeannBuilder(
            backend_name=args.backend,
            embedding_model=args.embedding_model,
            embedding_mode=args.embedding_mode,
            graph_degree=args.graph_degree,
            complexity=args.complexity,
            is_compact=args.compact,
            is_recompute=args.recompute,
            num_threads=args.num_threads,
        )

        for chunk_text in all_texts:
            builder.add_text(chunk_text)

        builder.build_index(index_path)
        print(f"Index built at {index_path}")

        # Register this project directory in global registry
        self.register_project_dir()

    async def search_documents(self, args):
        index_name = args.index_name
        query = args.query
        index_path = self.get_index_path(index_name)

        if not self.index_exists(index_name):
            print(
                f"Index '{index_name}' not found. Use 'leann build {index_name} --docs <dir>' to create it."
            )
            return

        searcher = LeannSearcher(index_path=index_path)
        results = searcher.search(
            query,
            top_k=args.top_k,
            complexity=args.complexity,
            beam_width=args.beam_width,
            prune_ratio=args.prune_ratio,
            recompute_embeddings=args.recompute_embeddings,
            pruning_strategy=args.pruning_strategy,
        )

        print(f"Search results for '{query}' (top {len(results)}):")
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result.score:.3f}")
            print(f"   {result.text[:200]}...")
            print()

    async def ask_questions(self, args):
        index_name = args.index_name
        index_path = self.get_index_path(index_name)

        if not self.index_exists(index_name):
            print(
                f"Index '{index_name}' not found. Use 'leann build {index_name} --docs <dir>' to create it."
            )
            return

        print(f"Starting chat with index '{index_name}'...")
        print(f"Using {args.model} ({args.llm})")

        llm_config = {"type": args.llm, "model": args.model}
        if args.llm == "ollama":
            llm_config["host"] = args.host

        chat = LeannChat(index_path=index_path, llm_config=llm_config)

        if args.interactive:
            print("LEANN Assistant ready! Type 'quit' to exit")
            print("=" * 40)

            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                if not user_input:
                    continue

                # Prepare LLM kwargs with thinking budget if specified
                llm_kwargs = {}
                if args.thinking_budget:
                    llm_kwargs["thinking_budget"] = args.thinking_budget

                response = chat.ask(
                    user_input,
                    top_k=args.top_k,
                    complexity=args.complexity,
                    beam_width=args.beam_width,
                    prune_ratio=args.prune_ratio,
                    recompute_embeddings=args.recompute_embeddings,
                    pruning_strategy=args.pruning_strategy,
                    llm_kwargs=llm_kwargs,
                )
                print(f"LEANN: {response}")
        else:
            query = input("Enter your question: ").strip()
            if query:
                # Prepare LLM kwargs with thinking budget if specified
                llm_kwargs = {}
                if args.thinking_budget:
                    llm_kwargs["thinking_budget"] = args.thinking_budget

                response = chat.ask(
                    query,
                    top_k=args.top_k,
                    complexity=args.complexity,
                    beam_width=args.beam_width,
                    prune_ratio=args.prune_ratio,
                    recompute_embeddings=args.recompute_embeddings,
                    pruning_strategy=args.pruning_strategy,
                    llm_kwargs=llm_kwargs,
                )
                print(f"LEANN: {response}")

    async def run(self, args=None):
        parser = self.create_parser()

        if args is None:
            args = parser.parse_args()

        if not args.command:
            parser.print_help()
            return

        if args.command == "list":
            self.list_indexes()
        elif args.command == "build":
            await self.build_index(args)
        elif args.command == "search":
            await self.search_documents(args)
        elif args.command == "ask":
            await self.ask_questions(args)
        else:
            parser.print_help()


def main():
    import dotenv

    dotenv.load_dotenv()

    cli = LeannCLI()
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
