import argparse
import asyncio
import os

try:
    import dotenv

    dotenv.load_dotenv()
except ModuleNotFoundError:
    # python-dotenv is not installed; skip loading environment variables
    dotenv = None
from pathlib import Path

from leann.api import LeannBuilder, LeannChat
from llama_index.core.node_parser import SentenceSplitter

# dotenv.load_dotenv()  # handled above if python-dotenv is available

# Default Chrome profile path
DEFAULT_CHROME_PROFILE = os.path.expanduser("~/Library/Application Support/Google/Chrome/Default")


def create_leann_index_from_multiple_chrome_profiles(
    profile_dirs: list[Path],
    index_path: str = "chrome_history_index.leann",
    max_count: int = -1,
    embedding_model: str = "facebook/contriever",
    embedding_mode: str = "sentence-transformers",
):
    """
    Create LEANN index from multiple Chrome profile data sources.

    Args:
        profile_dirs: List of Path objects pointing to Chrome profile directories
        index_path: Path to save the LEANN index
        max_count: Maximum number of history entries to process per profile
        embedding_model: The embedding model to use
        embedding_mode: The embedding backend mode
    """
    print("Creating LEANN index from multiple Chrome profile data sources...")

    # Load documents using ChromeHistoryReader from history_data
    from history_data.history import ChromeHistoryReader

    reader = ChromeHistoryReader()

    INDEX_DIR = Path(index_path).parent

    if not INDEX_DIR.exists():
        print("--- Index directory not found, building new index ---")
        all_documents = []
        total_processed = 0

        # Process each Chrome profile directory
        for i, profile_dir in enumerate(profile_dirs):
            print(f"\nProcessing Chrome profile {i + 1}/{len(profile_dirs)}: {profile_dir}")

            try:
                documents = reader.load_data(
                    chrome_profile_path=str(profile_dir), max_count=max_count
                )
                if documents:
                    print(f"Loaded {len(documents)} history documents from {profile_dir}")
                    all_documents.extend(documents)
                    total_processed += len(documents)

                    # Check if we've reached the max count
                    if max_count > 0 and total_processed >= max_count:
                        print(f"Reached max count of {max_count} documents")
                        break
                else:
                    print(f"No documents loaded from {profile_dir}")
            except Exception as e:
                print(f"Error processing {profile_dir}: {e}")
                continue

        if not all_documents:
            print("No documents loaded from any source. Exiting.")
            # highlight info that you need to close all chrome browser before running this script and high light the instruction!!
            print(
                "\033[91mYou need to close or quit all chrome browser before running this script\033[0m"
            )
            return None

        print(
            f"\nTotal loaded {len(all_documents)} history documents from {len(profile_dirs)} profiles"
        )

        # Create text splitter with 256 chunk size
        text_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=128)

        # Convert Documents to text strings and chunk them
        all_texts = []
        for doc in all_documents:
            # Split the document into chunks
            nodes = text_splitter.get_nodes_from_documents([doc])
            for node in nodes:
                text = node.get_content()
                # text = '[Title] ' + doc.metadata["title"] + '\n' + text
                all_texts.append(text)

        print(f"Created {len(all_texts)} text chunks from {len(all_documents)} documents")

        # Create LEANN index directory
        print("--- Index directory not found, building new index ---")
        INDEX_DIR.mkdir(exist_ok=True)

        print("--- Building new LEANN index ---")

        print("\n[PHASE 1] Building Leann index...")

        # Use HNSW backend for better macOS compatibility
        # LeannBuilder will automatically detect normalized embeddings and set appropriate distance metric
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model=embedding_model,
            embedding_mode=embedding_mode,
            graph_degree=32,
            complexity=64,
            is_compact=True,
            is_recompute=True,
            num_threads=1,  # Force single-threaded mode
        )

        print(f"Adding {len(all_texts)} history chunks to index...")
        for chunk_text in all_texts:
            builder.add_text(chunk_text)

        builder.build_index(index_path)
        print(f"\nLEANN index built at {index_path}!")
    else:
        print(f"--- Using existing index at {INDEX_DIR} ---")

    return index_path


def create_leann_index(
    profile_path: str | None = None,
    index_path: str = "chrome_history_index.leann",
    max_count: int = 1000,
    embedding_model: str = "facebook/contriever",
    embedding_mode: str = "sentence-transformers",
):
    """
    Create LEANN index from Chrome history data.

    Args:
        profile_path: Path to the Chrome profile directory (optional, uses default if None)
        index_path: Path to save the LEANN index
        max_count: Maximum number of history entries to process
        embedding_model: The embedding model to use
        embedding_mode: The embedding backend mode
    """
    print("Creating LEANN index from Chrome history data...")
    INDEX_DIR = Path(index_path).parent

    if not INDEX_DIR.exists():
        print("--- Index directory not found, building new index ---")
        INDEX_DIR.mkdir(exist_ok=True)

        print("--- Building new LEANN index ---")

        print("\n[PHASE 1] Building Leann index...")

        # Load documents using ChromeHistoryReader from history_data
        from history_data.history import ChromeHistoryReader

        reader = ChromeHistoryReader()

        documents = reader.load_data(chrome_profile_path=profile_path, max_count=max_count)

        if not documents:
            print("No documents loaded. Exiting.")
            return None

        print(f"Loaded {len(documents)} history documents")

        # Create text splitter with 256 chunk size
        text_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=25)

        # Convert Documents to text strings and chunk them
        all_texts = []
        for doc in documents:
            # Split the document into chunks
            nodes = text_splitter.get_nodes_from_documents([doc])
            for node in nodes:
                all_texts.append(node.get_content())

        print(f"Created {len(all_texts)} text chunks from {len(documents)} documents")

        # Create LEANN index directory
        print("--- Index directory not found, building new index ---")
        INDEX_DIR.mkdir(exist_ok=True)

        print("--- Building new LEANN index ---")

        print("\n[PHASE 1] Building Leann index...")

        # Use HNSW backend for better macOS compatibility
        # LeannBuilder will automatically detect normalized embeddings and set appropriate distance metric
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model=embedding_model,
            embedding_mode=embedding_mode,
            graph_degree=32,
            complexity=64,
            is_compact=True,
            is_recompute=True,
            num_threads=1,  # Force single-threaded mode
        )

        print(f"Adding {len(all_texts)} history chunks to index...")
        for chunk_text in all_texts:
            builder.add_text(chunk_text)

        builder.build_index(index_path)
        print(f"\nLEANN index built at {index_path}!")
    else:
        print(f"--- Using existing index at {INDEX_DIR} ---")

    return index_path


async def query_leann_index(index_path: str, query: str):
    """
    Query the LEANN index.

    Args:
        index_path: Path to the LEANN index
        query: The query string
    """
    print("\n[PHASE 2] Starting Leann chat session...")
    chat = LeannChat(index_path=index_path)

    print(f"You: {query}")
    chat_response = chat.ask(
        query,
        top_k=10,
        recompute_beighbor_embeddings=True,
        complexity=32,
        beam_width=1,
        llm_config={
            "type": "openai",
            "model": "gpt-4o",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        llm_kwargs={"temperature": 0.0, "max_tokens": 1000},
    )

    print(f"Leann chat response: \033[36m{chat_response}\033[0m")


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="LEANN Chrome History Reader - Create and query browser history index"
    )
    parser.add_argument(
        "--chrome-profile",
        type=str,
        default=DEFAULT_CHROME_PROFILE,
        help=f"Path to Chrome profile directory (default: {DEFAULT_CHROME_PROFILE}), usually you dont need to change this",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="./google_history_index",
        help="Directory to store the LEANN index (default: ./chrome_history_index_leann_test)",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=1000,
        help="Maximum number of history entries to process (default: 1000)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to run (default: runs example queries)",
    )
    parser.add_argument(
        "--auto-find-profiles",
        action="store_true",
        default=True,
        help="Automatically find all Chrome profiles (default: True)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="facebook/contriever",
        help="The embedding model to use (e.g., 'facebook/contriever', 'text-embedding-3-small')",
    )
    parser.add_argument(
        "--embedding-mode",
        type=str,
        default="sentence-transformers",
        choices=["sentence-transformers", "openai", "mlx"],
        help="The embedding backend mode",
    )
    parser.add_argument(
        "--use-existing-index",
        action="store_true",
        help="Use existing index without rebuilding",
    )

    args = parser.parse_args()

    INDEX_DIR = Path(args.index_dir)
    INDEX_PATH = str(INDEX_DIR / "chrome_history.leann")

    print(f"Using Chrome profile: {args.chrome_profile}")
    print(f"Index directory: {INDEX_DIR}")
    print(f"Max entries: {args.max_entries}")

    if args.use_existing_index:
        # Use existing index without rebuilding
        if not Path(INDEX_PATH).exists():
            print(f"Error: Index file not found at {INDEX_PATH}")
            return
        print(f"Using existing index at {INDEX_PATH}")
        index_path = INDEX_PATH
    else:
        # Find Chrome profile directories
        from history_data.history import ChromeHistoryReader

        if args.auto_find_profiles:
            profile_dirs = ChromeHistoryReader.find_chrome_profiles()
            if not profile_dirs:
                print("No Chrome profiles found automatically. Exiting.")
                return
        else:
            # Use single specified profile
            profile_path = Path(args.chrome_profile)
            if not profile_path.exists():
                print(f"Chrome profile not found: {profile_path}")
                return
            profile_dirs = [profile_path]

        # Create or load the LEANN index from all sources
        index_path = create_leann_index_from_multiple_chrome_profiles(
            profile_dirs, INDEX_PATH, args.max_entries, args.embedding_model, args.embedding_mode
        )

    if index_path:
        if args.query:
            # Run single query
            await query_leann_index(index_path, args.query)
        else:
            # Example queries
            queries = [
                "What websites did I visit about machine learning?",
                "Find my search history about programming",
            ]

            for query in queries:
                print("\n" + "=" * 60)
                await query_leann_index(index_path, query)


if __name__ == "__main__":
    asyncio.run(main())
