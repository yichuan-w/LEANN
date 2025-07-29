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
from llama_index.core.node_parser import SentenceSplitter

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
        max_items_default = getattr(self, "max_items_default", 1000)
        core_group.add_argument(
            "--max-items",
            type=int,
            default=max_items_default,
            help=f"Maximum number of items to process (default: {max_items_default}, -1 for all)",
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
            help=f"Embedding model to use (default: {embedding_model_default})",
        )
        embedding_group.add_argument(
            "--embedding-mode",
            type=str,
            default="sentence-transformers",
            choices=["sentence-transformers", "openai", "mlx"],
            help="Embedding backend mode (default: sentence-transformers)",
        )

        # LLM parameters
        llm_group = parser.add_argument_group("LLM Parameters")
        llm_group.add_argument(
            "--llm",
            type=str,
            default="openai",
            choices=["openai", "ollama", "hf"],
            help="LLM backend to use (default: openai)",
        )
        llm_group.add_argument(
            "--llm-model",
            type=str,
            default=None,
            help="LLM model name (default: gpt-4o for openai, llama3.2:1b for ollama)",
        )
        llm_group.add_argument(
            "--llm-host",
            type=str,
            default="http://localhost:11434",
            help="Host for Ollama API (default: http://localhost:11434)",
        )

        # Search parameters
        search_group = parser.add_argument_group("Search Parameters")
        search_group.add_argument(
            "--top-k", type=int, default=20, help="Number of results to retrieve (default: 20)"
        )
        search_group.add_argument(
            "--search-complexity",
            type=int,
            default=64,
            help="Search complexity for graph traversal (default: 64)",
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

        return config

    async def build_index(self, args, texts: list[str]) -> str:
        """Build LEANN index from texts."""
        index_path = str(Path(args.index_dir) / f"{self.default_index_name}.leann")

        print(f"\n[Building Index] Creating {self.name} index...")
        print(f"Total text chunks: {len(texts)}")

        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model=args.embedding_model,
            embedding_mode=args.embedding_mode,
            graph_degree=32,
            complexity=64,
            is_compact=True,
            is_recompute=True,
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

        return index_path

    async def run_interactive_chat(self, args, index_path: str):
        """Run interactive chat with the index."""
        chat = LeannChat(
            index_path,
            llm_config=self.get_llm_config(args),
            system_prompt=f"You are a helpful assistant that answers questions about {self.name} data.",
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

                response = chat.ask(query, top_k=args.top_k, complexity=args.search_complexity)
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
        )

        print(f"\n[Query] {query}")
        response = chat.ask(query, top_k=args.top_k, complexity=args.search_complexity)
        print(f"\n[Response] {response}\n")

    async def run(self):
        """Main entry point for the example."""
        args = self.parser.parse_args()

        # Check if index exists
        index_path = str(Path(args.index_dir) / f"{self.default_index_name}.leann")
        index_exists = Path(index_path).exists()

        if not index_exists or args.force_rebuild:
            # Load data and build index
            print(f"\n{'Rebuilding' if index_exists else 'Building'} index...")
            texts = await self.load_data(args)

            if not texts:
                print("No data found to index!")
                return

            index_path = await self.build_index(args, texts)
        else:
            print(f"\nUsing existing index: {index_path}")

        # Run query or interactive mode
        if args.query:
            await self.run_single_query(args, index_path, args.query)
        else:
            await self.run_interactive_chat(args, index_path)


def create_text_chunks(documents, chunk_size=256, chunk_overlap=25) -> list[str]:
    """Helper function to create text chunks from documents."""
    node_parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=" ",
        paragraph_separator="\n\n",
    )

    all_texts = []
    for doc in documents:
        nodes = node_parser.get_nodes_from_documents([doc])
        if nodes:
            all_texts.extend(node.get_content() for node in nodes)

    return all_texts
