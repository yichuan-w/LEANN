"""
iMessage RAG Example.

This example demonstrates how to build a RAG system on your iMessage conversation history.
"""

import asyncio
from pathlib import Path
from typing import Any

from leann.chunking_utils import create_text_chunks

from apps.base_rag_example import BaseRAGExample
from apps.imessage_data.imessage_reader import IMessageReader


class IMessageRAG(BaseRAGExample):
    """RAG example for iMessage conversation history."""

    def __init__(self):
        super().__init__(
            name="iMessage",
            description="RAG on your iMessage conversation history",
            default_index_name="imessage_index",
        )

    def _add_specific_arguments(self, parser):
        """Add iMessage-specific arguments."""
        imessage_group = parser.add_argument_group("iMessage Parameters")
        imessage_group.add_argument(
            "--db-path",
            type=str,
            default=None,
            help="Path to iMessage chat.db file (default: ~/Library/Messages/chat.db)",
        )
        imessage_group.add_argument(
            "--concatenate-conversations",
            action="store_true",
            default=True,
            help="Concatenate messages within conversations for better context (default: True)",
        )
        imessage_group.add_argument(
            "--no-concatenate-conversations",
            action="store_true",
            help="Process each message individually instead of concatenating by conversation",
        )
        imessage_group.add_argument(
            "--chunk-size",
            type=int,
            default=1000,
            help="Maximum characters per text chunk (default: 1000)",
        )
        imessage_group.add_argument(
            "--chunk-overlap",
            type=int,
            default=200,
            help="Overlap between text chunks (default: 200)",
        )

    async def load_data(self, args) -> list[dict[str, Any]]:
        """Load iMessage history and convert to text chunks."""
        print("Loading iMessage conversation history...")

        # Determine concatenation setting
        concatenate = args.concatenate_conversations and not args.no_concatenate_conversations

        # Initialize iMessage reader
        reader = IMessageReader(concatenate_conversations=concatenate)

        # Load documents
        try:
            if args.db_path:
                # Use custom database path
                db_dir = str(Path(args.db_path).parent)
                documents = reader.load_data(input_dir=db_dir)
            else:
                # Use default macOS location
                documents = reader.load_data()

        except Exception as e:
            print(f"Error loading iMessage data: {e}")
            print("\nTroubleshooting tips:")
            print("1. Make sure you have granted Full Disk Access to your terminal/IDE")
            print("2. Check that the iMessage database exists at ~/Library/Messages/chat.db")
            print("3. Try specifying a custom path with --db-path if you have a backup")
            return []

        if not documents:
            print("No iMessage conversations found!")
            return []

        print(f"Loaded {len(documents)} iMessage documents")

        # Show some statistics
        total_messages = sum(doc.metadata.get("message_count", 1) for doc in documents)
        print(f"Total messages: {total_messages}")

        if concatenate:
            # Show chat statistics
            chat_names = [doc.metadata.get("chat_name", "Unknown") for doc in documents]
            unique_chats = len(set(chat_names))
            print(f"Unique conversations: {unique_chats}")

        # Convert to text chunks
        all_texts = create_text_chunks(
            documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        # Apply max_items limit if specified
        if args.max_items > 0:
            all_texts = all_texts[: args.max_items]
            print(f"Limited to {len(all_texts)} text chunks (max_items={args.max_items})")

        return all_texts


async def main():
    """Main entry point."""
    app = IMessageRAG()
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())
