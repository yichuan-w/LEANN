import os
import asyncio
import dotenv
import argparse
from pathlib import Path
from typing import List, Any, Optional
from leann.api import LeannBuilder, LeannSearcher, LeannChat
from llama_index.core.node_parser import SentenceSplitter
import requests
import time

dotenv.load_dotenv()

# Default WeChat export directory
DEFAULT_WECHAT_EXPORT_DIR = "./wechat_export_direct"


def create_leann_index_from_multiple_wechat_exports(
    export_dirs: List[Path],
    index_path: str = "wechat_history_index.leann",
    max_count: int = -1,
):
    """
    Create LEANN index from multiple WeChat export data sources.

    Args:
        export_dirs: List of Path objects pointing to WeChat export directories
        index_path: Path to save the LEANN index
        max_count: Maximum number of chat entries to process per export
    """
    print("Creating LEANN index from multiple WeChat export data sources...")

    # Load documents using WeChatHistoryReader from history_data
    from history_data.wechat_history import WeChatHistoryReader

    reader = WeChatHistoryReader()

    INDEX_DIR = Path(index_path).parent

    if not INDEX_DIR.exists():
        print(f"--- Index directory not found, building new index ---")
        all_documents = []
        total_processed = 0

        # Process each WeChat export directory
        for i, export_dir in enumerate(export_dirs):
            print(
                f"\nProcessing WeChat export {i + 1}/{len(export_dirs)}: {export_dir}"
            )

            try:
                documents = reader.load_data(
                    wechat_export_dir=str(export_dir),
                    max_count=max_count,
                    concatenate_messages=True,  # Disable concatenation - one message per document
                )
                if documents:
                    print(f"Loaded {len(documents)} chat documents from {export_dir}")
                    all_documents.extend(documents)
                    total_processed += len(documents)

                    # Check if we've reached the max count
                    if max_count > 0 and total_processed >= max_count:
                        print(f"Reached max count of {max_count} documents")
                        break
                else:
                    print(f"No documents loaded from {export_dir}")
            except Exception as e:
                print(f"Error processing {export_dir}: {e}")
                continue

        if not all_documents:
            print("No documents loaded from any source. Exiting.")
            return None

        print(
            f"\nTotal loaded {len(all_documents)} chat documents from {len(export_dirs)} exports and starting to split them into chunks"
        )

        # Create text splitter with 256 chunk size
        text_splitter = SentenceSplitter(chunk_size=192, chunk_overlap=64)

        # Convert Documents to text strings and chunk them
        all_texts = []
        for doc in all_documents:
            # Split the document into chunks
            nodes = text_splitter.get_nodes_from_documents([doc])
            for node in nodes:
                text = '[Contact] means the message is from: ' + doc.metadata["contact_name"] + '\n' + node.get_content()
                all_texts.append(text)

        print(
            f"Finished splitting {len(all_documents)} documents into {len(all_texts)} text chunks"
        )

        # Create LEANN index directory
        print(f"--- Index directory not found, building new index ---")
        INDEX_DIR.mkdir(exist_ok=True)

        print(f"--- Building new LEANN index ---")

        print(f"\n[PHASE 1] Building Leann index...")

        # Use HNSW backend for better macOS compatibility
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="Qwen/Qwen3-Embedding-0.6B",
            graph_degree=32,
            complexity=64,
            is_compact=True,
            is_recompute=True,
            num_threads=1,  # Force single-threaded mode
        )

        print(f"Adding {len(all_texts)} chat chunks to index...")
        for chunk_text in all_texts:
            builder.add_text(chunk_text)

        builder.build_index(index_path)
        print(f"\nLEANN index built at {index_path}!")
    else:
        print(f"--- Using existing index at {INDEX_DIR} ---")

    return index_path


def create_leann_index(
    export_dir: str = None,
    index_path: str = "wechat_history_index.leann",
    max_count: int = 1000,
):
    """
    Create LEANN index from WeChat chat history data.

    Args:
        export_dir: Path to the WeChat export directory (optional, uses default if None)
        index_path: Path to save the LEANN index
        max_count: Maximum number of chat entries to process
    """
    print("Creating LEANN index from WeChat chat history data...")
    INDEX_DIR = Path(index_path).parent

    if not INDEX_DIR.exists():
        print(f"--- Index directory not found, building new index ---")
        INDEX_DIR.mkdir(exist_ok=True)

        print(f"--- Building new LEANN index ---")

        print(f"\n[PHASE 1] Building Leann index...")

        # Load documents using WeChatHistoryReader from history_data
        from history_data.wechat_history import WeChatHistoryReader

        reader = WeChatHistoryReader()

        documents = reader.load_data(
            wechat_export_dir=export_dir,
            max_count=max_count,
            concatenate_messages=False,  # Disable concatenation - one message per document
        )

        if not documents:
            print("No documents loaded. Exiting.")
            return None

        print(f"Loaded {len(documents)} chat documents")

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
        print(f"--- Index directory not found, building new index ---")
        INDEX_DIR.mkdir(exist_ok=True)

        print(f"--- Building new LEANN index ---")

        print(f"\n[PHASE 1] Building Leann index...")

        # Use HNSW backend for better macOS compatibility
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",  # MLX-optimized model
            graph_degree=32,
            complexity=64,
            is_compact=True,
            is_recompute=True,
            num_threads=1,  # Force single-threaded mode
        )

        print(f"Adding {len(all_texts)} chat chunks to index...")
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
    print(f"\n[PHASE 2] Starting Leann chat session...")
    chat = LeannChat(index_path=index_path)

    print(f"You: {query}")
    chat_response = chat.ask(
        query,
        top_k=20,
        recompute_beighbor_embeddings=True,
        complexity=16,
        beam_width=1,
        llm_config={
            "type": "openai",
            "model": "gpt-4o",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        llm_kwargs={"temperature": 0.0, "max_tokens": 1000},
    )
    print(f"Leann: {chat_response}")


async def main():
    """Main function with integrated WeChat export functionality."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="LEANN WeChat History Reader - Create and query WeChat chat history index"
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=DEFAULT_WECHAT_EXPORT_DIR,
        help=f"Directory to store WeChat exports (default: {DEFAULT_WECHAT_EXPORT_DIR})",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="./wechat_history_magic_test_11Debug_new",
        help="Directory to store the LEANN index (default: ./wechat_history_index_leann_test)",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=50,
        help="Maximum number of chat entries to process (default: 5000)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to run (default: runs example queries)",
    )
    parser.add_argument(
        "--force-export",
        action="store_true",
        default=False,
        help="Force re-export of WeChat data even if exports exist",
    )

    args = parser.parse_args()

    INDEX_DIR = Path(args.index_dir)
    INDEX_PATH = str(INDEX_DIR / "wechat_history.leann")

    print(f"Using WeChat export directory: {args.export_dir}")
    print(f"Index directory: {INDEX_DIR}")
    print(f"Max entries: {args.max_entries}")

    # Initialize WeChat reader with export capabilities
    from history_data.wechat_history import WeChatHistoryReader

    reader = WeChatHistoryReader()

    # Find existing exports or create new ones using the centralized method
    export_dirs = reader.find_or_export_wechat_data(args.export_dir)
    if not export_dirs:
        print("Failed to find or export WeChat data. Exiting.")
        return

    # Create or load the LEANN index from all sources
    index_path = create_leann_index_from_multiple_wechat_exports(
        export_dirs, INDEX_PATH, max_count=args.max_entries
    )

    if index_path:
        if args.query:
            # Run single query
            await query_leann_index(index_path, args.query)
        else:
            # Example queries
            queries = [
                "我想买魔术师约翰逊的球衣，给我一些对应聊天记录?",
            ]

            for query in queries:
                print("\n" + "=" * 60)
                await query_leann_index(index_path, query)


if __name__ == "__main__":
    asyncio.run(main())
