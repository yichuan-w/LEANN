"""
Base class for unified RAG examples interface.
Provides common parameters and functionality for all RAG examples.
"""

import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import dotenv
from leann.api import LeannBuilder, LeannChat
from leann.registry import register_project_directory

dotenv.load_dotenv()


class BaseRAGExample(ABC):
    """Base class for all RAG examples with unified interface."""

    def __init__(
        self,
        name: str,
        description: str,
        default_index_name: str,
    ):
        self.name = name
        self.description = description
        self.default_index_name = default_index_name
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with common parameters."""
        parser = argparse.ArgumentParser(
            description=self.description, formatter_class=argparse.RawDescriptionHelpFormatter
        )

        # Core parameters (all examples share these)
        core_group = parser.add_argument_group("Core Parameters")
        core_group.add_argument(
            "--index-dir",
            type=str,
            default=f"./{self.default_index_name}",
            help=f"Directory to store the index (default: ./{self.default_index_name})",
        )
        core_group.add_argument(
            "--query",
            type=str,
            default=None,
            help="Query to run (if not provided, will run in interactive mode)",
        )
        # Allow subclasses to override default max_items
        max_items_default = getattr(self, "max_items_default", -1)
        core_group.add_argument(
            "--max-items",
            type=int,
            default=max_items_default,
            help="Maximum number of items to process  -1 for all, means index all documents, and you should set it to a reasonable number if you have a large dataset and try at the first time)",
        )
        core_group.add_argument(
            "--force-rebuild", action="store_true", help="Force rebuild index even if it exists"
        )

        # Embedding parameters
        embedding_group = parser.add_argument_group("Embedding Parameters")
        # Allow subclasses to override default embedding_model
        embedding_model_default = getattr(self, "embedding_model_default", "facebook/contriever")
        embedding_group.add_argument(
            "--embedding-model",
            type=str,
            default=embedding_model_default,
            help=f"Embedding model to use (default: {embedding_model_default}), we provide facebook/contriever, text-embedding-3-small,mlx-community/Qwen3-Embedding-0.6B-8bit or nomic-embed-text",
        )
        embedding_group.add_argument(
            "--embedding-mode",
            type=str,
            default="sentence-transformers",
            choices=["sentence-transformers", "openai", "mlx", "ollama"],
            help="Embedding backend mode (default: sentence-transformers), we provide sentence-transformers, openai, mlx, or ollama",
        )

        # LLM parameters
        llm_group = parser.add_argument_group("LLM Parameters")
        llm_group.add_argument(
            "--llm",
            type=str,
            default="openai",
            choices=["openai", "ollama", "hf", "simulated"],
            help="LLM backend: openai, ollama, or hf (default: openai)",
        )
        llm_group.add_argument(
            "--llm-model",
            type=str,
            default=None,
            help="Model name (default: gpt-4o) e.g., gpt-4o-mini, llama3.2:1b, Qwen/Qwen2.5-1.5B-Instruct",
        )
        llm_group.add_argument(
            "--llm-host",
            type=str,
            default="http://localhost:11434",
            help="Host for Ollama API (default: http://localhost:11434)",
        )
        llm_group.add_argument(
            "--thinking-budget",
            type=str,
            choices=["low", "medium", "high"],
            default=None,
            help="Thinking budget for reasoning models (low/medium/high). Supported by GPT-Oss:20b and other reasoning models.",
        )

        # AST Chunking parameters
        ast_group = parser.add_argument_group("AST Chunking Parameters")
        ast_group.add_argument(
            "--use-ast-chunking",
            action="store_true",
            help="Enable AST-aware chunking for code files (requires astchunk)",
        )
        ast_group.add_argument(
            "--ast-chunk-size",
            type=int,
            default=512,
            help="Maximum characters per AST chunk (default: 512)",
        )
        ast_group.add_argument(
            "--ast-chunk-overlap",
            type=int,
            default=64,
            help="Overlap between AST chunks (default: 64)",
        )
        ast_group.add_argument(
            "--code-file-extensions",
            nargs="+",
            default=None,
            help="Additional code file extensions to process with AST chunking (e.g., .py .java .cs .ts)",
        )
        ast_group.add_argument(
            "--ast-fallback-traditional",
            action="store_true",
            default=True,
            help="Fall back to traditional chunking if AST chunking fails (default: True)",
        )

        # Search parameters
        search_group = parser.add_argument_group("Search Parameters")
        search_group.add_argument(
            "--top-k", type=int, default=20, help="Number of results to retrieve (default: 20)"
        )
        search_group.add_argument(
            "--search-complexity",
            type=int,
            default=32,
            help="Search complexity for graph traversal (default: 64)",
        )

        # Index building parameters
        index_group = parser.add_argument_group("Index Building Parameters")
        index_group.add_argument(
            "--backend-name",
            type=str,
            default="hnsw",
            choices=["hnsw", "diskann"],
            help="Backend to use for index (default: hnsw)",
        )
        index_group.add_argument(
            "--graph-degree",
            type=int,
            default=32,
            help="Graph degree for index construction (default: 32)",
        )
        index_group.add_argument(
            "--build-complexity",
            type=int,
            default=64,
            help="Build complexity for index construction (default: 64)",
        )
        index_group.add_argument(
            "--no-compact",
            action="store_true",
            help="Disable compact index storage",
        )
        index_group.add_argument(
            "--no-recompute",
            action="store_true",
            help="Disable embedding recomputation",
        )

        # Add source-specific parameters
        self._add_specific_arguments(parser)

        return parser

    @abstractmethod
    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        """Add source-specific arguments. Override in subclasses."""
        pass

    @abstractmethod
    async def load_data(self, args) -> list[str]:
        """Load data from the source. Returns list of text chunks."""
        pass

    def get_llm_config(self, args) -> dict[str, Any]:
        """Get LLM configuration based on arguments."""
        config = {"type": args.llm}

        if args.llm == "openai":
            config["model"] = args.llm_model or "gpt-4o"
        elif args.llm == "ollama":
            config["model"] = args.llm_model or "llama3.2:1b"
            config["host"] = args.llm_host
        elif args.llm == "hf":
            config["model"] = args.llm_model or "Qwen/Qwen2.5-1.5B-Instruct"
        elif args.llm == "simulated":
            # Simulated LLM doesn't need additional configuration
            pass

        return config

    async def build_index(self, args, texts: list[str]) -> str:
        """Build LEANN index from texts."""
        index_path = str(Path(args.index_dir) / f"{self.default_index_name}.leann")

        print(f"\n[Building Index] Creating {self.name} index...")
        print(f"Total text chunks: {len(texts)}")

        builder = LeannBuilder(
            backend_name=args.backend_name,
            embedding_model=args.embedding_model,
            embedding_mode=args.embedding_mode,
            graph_degree=args.graph_degree,
            complexity=args.build_complexity,
            is_compact=not args.no_compact,
            is_recompute=not args.no_recompute,
            num_threads=1,  # Force single-threaded mode
        )

        # Add texts in batches for better progress tracking
        batch_size = 1000
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                builder.add_text(text)
            print(f"Added {min(i + batch_size, len(texts))}/{len(texts)} texts...")

        print("Building index structure...")
        builder.build_index(index_path)
        print(f"Index saved to: {index_path}")

        # Register project directory so leann list can discover this index
        # The index is saved as args.index_dir/index_name.leann
        # We want to register the current working directory where the app is run
        register_project_directory(Path.cwd())

        return index_path

    async def run_interactive_chat(self, args, index_path: str):
        """Run interactive chat with the index."""
        chat = LeannChat(
            index_path,
            llm_config=self.get_llm_config(args),
            system_prompt=f"You are a helpful assistant that answers questions about {self.name} data.",
            complexity=args.search_complexity,
        )

        print(f"\n[Interactive Mode] Chat with your {self.name} data!")
        print("Type 'quit' or 'exit' to stop.\n")

        while True:
            try:
                query = input("You: ").strip()
                if query.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                if not query:
                    continue

                # Prepare LLM kwargs with thinking budget if specified
                llm_kwargs = {}
                if hasattr(args, "thinking_budget") and args.thinking_budget:
                    llm_kwargs["thinking_budget"] = args.thinking_budget

                response = chat.ask(
                    query,
                    top_k=args.top_k,
                    complexity=args.search_complexity,
                    llm_kwargs=llm_kwargs,
                )
                print(f"\nAssistant: {response}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    async def run_single_query(self, args, index_path: str, query: str):
        """Run a single query against the index."""
        chat = LeannChat(
            index_path,
            llm_config=self.get_llm_config(args),
            system_prompt=f"You are a helpful assistant that answers questions about {self.name} data.",
            complexity=args.search_complexity,
        )

        print(f"\n[Query]: \033[36m{query}\033[0m")

        # Prepare LLM kwargs with thinking budget if specified
        llm_kwargs = {}
        if hasattr(args, "thinking_budget") and args.thinking_budget:
            llm_kwargs["thinking_budget"] = args.thinking_budget

        response = chat.ask(
            query, top_k=args.top_k, complexity=args.search_complexity, llm_kwargs=llm_kwargs
        )
        print(f"\n[Response]: \033[36m{response}\033[0m")

    async def run(self):
        """Main entry point for the example."""
        args = self.parser.parse_args()

        # Check if index exists
        index_path = str(Path(args.index_dir) / f"{self.default_index_name}.leann")
        index_exists = Path(args.index_dir).exists()

        if not index_exists or args.force_rebuild:
            # Load data and build index
            print(f"\n{'Rebuilding' if index_exists else 'Building'} index...")
            texts = await self.load_data(args)

            if not texts:
                print("No data found to index!")
                return

            index_path = await self.build_index(args, texts)
        else:
            print(f"\nUsing existing index in {args.index_dir}")

        # Run query or interactive mode
        if args.query:
            await self.run_single_query(args, index_path, args.query)
        else:
            await self.run_interactive_chat(args, index_path)
