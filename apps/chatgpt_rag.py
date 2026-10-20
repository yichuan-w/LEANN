"""
ChatGPT RAG example using the unified interface.
Supports ChatGPT export data from chat.html files.
"""

import sys
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from base_rag_example import BaseRAGExample
from chunking import create_text_chunks

from .chatgpt_data.chatgpt_reader import ChatGPTReader


class ChatGPTRAG(BaseRAGExample):
    """RAG example for ChatGPT conversation data."""

    def __init__(self):
        # Set default values BEFORE calling super().__init__
        self.max_items_default = -1  # Process all conversations by default
        self.embedding_model_default = (
            "sentence-transformers/all-MiniLM-L6-v2"  # Fast 384-dim model
        )

        super().__init__(
            name="ChatGPT",
            description="Process and query ChatGPT conversation exports with LEANN",
            default_index_name="chatgpt_conversations_index",
        )

    def _add_specific_arguments(self, parser):
        """Add ChatGPT-specific arguments."""
        chatgpt_group = parser.add_argument_group("ChatGPT Parameters")
        chatgpt_group.add_argument(
            "--export-path",
            type=str,
            default="./chatgpt_export",
            help="Path to ChatGPT export file (.zip or .html) or directory containing exports (default: ./chatgpt_export)",
        )
        chatgpt_group.add_argument(
            "--concatenate-conversations",
            action="store_true",
            default=True,
            help="Concatenate messages within conversations for better context (default: True)",
        )
        chatgpt_group.add_argument(
            "--separate-messages",
            action="store_true",
            help="Process each message as a separate document (overrides --concatenate-conversations)",
        )
        chatgpt_group.add_argument(
            "--chunk-size", type=int, default=512, help="Text chunk size (default: 512)"
        )
        chatgpt_group.add_argument(
            "--chunk-overlap", type=int, default=128, help="Text chunk overlap (default: 128)"
        )

    def _find_chatgpt_exports(self, export_path: Path) -> list[Path]:
        """
        Find ChatGPT export files in the given path.

        Args:
            export_path: Path to search for exports

        Returns:
            List of paths to ChatGPT export files
        """
        export_files = []

        if export_path.is_file():
            if export_path.suffix.lower() in [".zip", ".html"]:
                export_files.append(export_path)
        elif export_path.is_dir():
            # Look for zip and html files
            export_files.extend(export_path.glob("*.zip"))
            export_files.extend(export_path.glob("*.html"))

        return export_files

    async def load_data(self, args) -> list[dict[str, Any]]:
        """Load ChatGPT export data and convert to text chunks."""
        export_path = Path(args.export_path)

        if not export_path.exists():
            print(f"ChatGPT export path not found: {export_path}")
            print(
                "Please ensure you have exported your ChatGPT data and placed it in the correct location."
            )
            print("\nTo export your ChatGPT data:")
            print("1. Sign in to ChatGPT")
            print("2. Click on your profile icon → Settings → Data Controls")
            print("3. Click 'Export' under Export Data")
            print("4. Download the zip file from the email link")
            print("5. Extract or place the file/directory at the specified path")
            return []

        # Find export files
        export_files = self._find_chatgpt_exports(export_path)

        if not export_files:
            print(f"No ChatGPT export files (.zip or .html) found in: {export_path}")
            return []

        print(f"Found {len(export_files)} ChatGPT export files")

        # Create reader with appropriate settings
        concatenate = args.concatenate_conversations and not args.separate_messages
        reader = ChatGPTReader(concatenate_conversations=concatenate)

        # Process each export file
        all_documents = []
        total_processed = 0

        for i, export_file in enumerate(export_files):
            print(f"\nProcessing export file {i + 1}/{len(export_files)}: {export_file.name}")

            try:
                # Apply max_items limit per file
                max_per_file = -1
                if args.max_items > 0:
                    remaining = args.max_items - total_processed
                    if remaining <= 0:
                        break
                    max_per_file = remaining

                # Load conversations
                documents = reader.load_data(
                    chatgpt_export_path=str(export_file),
                    max_count=max_per_file,
                    include_metadata=True,
                )

                if documents:
                    all_documents.extend(documents)
                    total_processed += len(documents)
                    print(f"Processed {len(documents)} conversations from this file")
                else:
                    print(f"No conversations loaded from {export_file}")

            except Exception as e:
                print(f"Error processing {export_file}: {e}")
                continue

        if not all_documents:
            print("No conversations found to process!")
            print("\nTroubleshooting:")
            print("- Ensure the export file is a valid ChatGPT export")
            print("- Check that the HTML file contains conversation data")
            print("- Try extracting the zip file and pointing to the HTML file directly")
            return []

        print(f"\nTotal conversations processed: {len(all_documents)}")
        print("Now starting to split into text chunks... this may take some time")

        # Convert to text chunks
        all_texts = create_text_chunks(
            all_documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        )

        print(f"Created {len(all_texts)} text chunks from {len(all_documents)} conversations")
        return all_texts


if __name__ == "__main__":
    import asyncio

    # Example queries for ChatGPT RAG
    print("\n🤖 ChatGPT RAG Example")
    print("=" * 50)
    print("\nExample queries you can try:")
    print("- 'What did I ask about Python programming?'")
    print("- 'Show me conversations about machine learning'")
    print("- 'Find discussions about travel planning'")
    print("- 'What advice did ChatGPT give me about career development?'")
    print("- 'Search for conversations about cooking recipes'")
    print("\nTo get started:")
    print("1. Export your ChatGPT data from Settings → Data Controls → Export")
    print("2. Place the downloaded zip file or extracted HTML in ./chatgpt_export/")
    print("3. Run this script to build your personal ChatGPT knowledge base!")
    print("\nOr run without --query for interactive mode\n")

    rag = ChatGPTRAG()
    asyncio.run(rag.run())
