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
        self.indexes_dir = Path.home() / ".leann" / "indexes"
        self.indexes_dir.mkdir(parents=True, exist_ok=True)

        self.node_parser = SentenceSplitter(
            chunk_size=256, chunk_overlap=128, separator=" ", paragraph_separator="\n\n"
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
  leann build my-docs --docs ./documents    # Build index named my-docs
  leann search my-docs "query"             # Search in my-docs index
  leann ask my-docs "question"             # Ask my-docs index
  leann list                              # List all stored indexes
            """,
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Build command
        build_parser = subparsers.add_parser("build", help="Build document index")
        build_parser.add_argument("index_name", help="Index name")
        build_parser.add_argument("--docs", type=str, required=True, help="Documents directory")
        build_parser.add_argument(
            "--backend", type=str, default="hnsw", choices=["hnsw", "diskann"]
        )
        build_parser.add_argument("--embedding-model", type=str, default="facebook/contriever")
        build_parser.add_argument("--force", "-f", action="store_true", help="Force rebuild")
        build_parser.add_argument("--graph-degree", type=int, default=32)
        build_parser.add_argument("--complexity", type=int, default=64)
        build_parser.add_argument("--num-threads", type=int, default=1)
        build_parser.add_argument("--compact", action="store_true", default=True)
        build_parser.add_argument("--recompute", action="store_true", default=True)

        # Search command
        search_parser = subparsers.add_parser("search", help="Search documents")
        search_parser.add_argument("index_name", help="Index name")
        search_parser.add_argument("query", help="Search query")
        search_parser.add_argument("--top-k", type=int, default=5)
        search_parser.add_argument("--complexity", type=int, default=64)
        search_parser.add_argument("--beam-width", type=int, default=1)
        search_parser.add_argument("--prune-ratio", type=float, default=0.0)
        search_parser.add_argument("--recompute-embeddings", action="store_true")
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
        ask_parser.add_argument("--recompute-embeddings", action="store_true")
        ask_parser.add_argument(
            "--pruning-strategy",
            choices=["global", "local", "proportional"],
            default="global",
        )

        # List command
        subparsers.add_parser("list", help="List all indexes")

        return parser

    def list_indexes(self):
        print("Stored LEANN indexes:")

        if not self.indexes_dir.exists():
            print("No indexes found. Use 'leann build <name> --docs <dir>' to create one.")
            return

        index_dirs = [d for d in self.indexes_dir.iterdir() if d.is_dir()]

        if not index_dirs:
            print("No indexes found. Use 'leann build <name> --docs <dir>' to create one.")
            return

        print(f"Found {len(index_dirs)} indexes:")
        for i, index_dir in enumerate(index_dirs, 1):
            index_name = index_dir.name
            status = "✓" if self.index_exists(index_name) else "✗"

            print(f"  {i}. {index_name} [{status}]")
            if self.index_exists(index_name):
                index_dir / "documents.leann.meta.json"
                size_mb = sum(f.stat().st_size for f in index_dir.iterdir() if f.is_file()) / (
                    1024 * 1024
                )
                print(f"     Size: {size_mb:.1f} MB")

        if index_dirs:
            example_name = index_dirs[0].name
            print("\nUsage:")
            print(f'  leann search {example_name} "your query"')
            print(f"  leann ask {example_name} --interactive")

    def load_documents(self, docs_dir: str):
        print(f"Loading documents from {docs_dir}...")

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
        other_docs = SimpleDirectoryReader(
            docs_dir,
            recursive=True,
            encoding="utf-8",
            required_exts=[".txt", ".md", ".docx"],
        ).load_data(show_progress=True)
        documents.extend(other_docs)

        all_texts = []
        for doc in documents:
            nodes = self.node_parser.get_nodes_from_documents([doc])
            for node in nodes:
                all_texts.append(node.get_content())

        print(f"Loaded {len(documents)} documents, {len(all_texts)} chunks")
        return all_texts

    async def build_index(self, args):
        docs_dir = args.docs
        index_name = args.index_name
        index_dir = self.indexes_dir / index_name
        index_path = self.get_index_path(index_name)

        if index_dir.exists() and not args.force:
            print(f"Index '{index_name}' already exists. Use --force to rebuild.")
            return

        all_texts = self.load_documents(docs_dir)
        if not all_texts:
            print("No documents found")
            return

        index_dir.mkdir(parents=True, exist_ok=True)

        print(f"Building index '{index_name}' with {args.backend} backend...")

        builder = LeannBuilder(
            backend_name=args.backend,
            embedding_model=args.embedding_model,
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

                response = chat.ask(
                    user_input,
                    top_k=args.top_k,
                    complexity=args.complexity,
                    beam_width=args.beam_width,
                    prune_ratio=args.prune_ratio,
                    recompute_embeddings=args.recompute_embeddings,
                    pruning_strategy=args.pruning_strategy,
                )
                print(f"LEANN: {response}")
        else:
            query = input("Enter your question: ").strip()
            if query:
                response = chat.ask(
                    query,
                    top_k=args.top_k,
                    complexity=args.complexity,
                    beam_width=args.beam_width,
                    prune_ratio=args.prune_ratio,
                    recompute_embeddings=args.recompute_embeddings,
                    pruning_strategy=args.pruning_strategy,
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
