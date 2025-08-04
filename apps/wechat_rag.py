"""
WeChat History RAG example using the unified interface.
Supports WeChat chat history export and search.
"""

import subprocess
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from base_rag_example import BaseRAGExample

from .history_data.wechat_history import WeChatHistoryReader


class WeChatRAG(BaseRAGExample):
    """RAG example for WeChat chat history."""

    def __init__(self):
        # Set default values BEFORE calling super().__init__
        self.max_items_default = -1  # Match original default
        self.embedding_model_default = (
            "sentence-transformers/all-MiniLM-L6-v2"  # Fast 384-dim model
        )

        super().__init__(
            name="WeChat History",
            description="Process and query WeChat chat history with LEANN",
            default_index_name="wechat_history_magic_test_11Debug_new",
        )

    def _add_specific_arguments(self, parser):
        """Add WeChat-specific arguments."""
        wechat_group = parser.add_argument_group("WeChat Parameters")
        wechat_group.add_argument(
            "--export-dir",
            type=str,
            default="./wechat_export",
            help="Directory to store WeChat exports (default: ./wechat_export)",
        )
        wechat_group.add_argument(
            "--force-export",
            action="store_true",
            help="Force re-export of WeChat data even if exports exist",
        )
        wechat_group.add_argument(
            "--chunk-size", type=int, default=192, help="Text chunk size (default: 192)"
        )
        wechat_group.add_argument(
            "--chunk-overlap", type=int, default=64, help="Text chunk overlap (default: 64)"
        )

    def _export_wechat_data(self, export_dir: Path) -> bool:
        """Export WeChat data using wechattweak-cli."""
        print("Exporting WeChat data...")

        # Check if WeChat is running
        try:
            result = subprocess.run(["pgrep", "WeChat"], capture_output=True, text=True)
            if result.returncode != 0:
                print("WeChat is not running. Please start WeChat first.")
                return False
        except Exception:
            pass  # pgrep might not be available on all systems

        # Create export directory
        export_dir.mkdir(parents=True, exist_ok=True)

        # Run export command
        cmd = ["packages/wechat-exporter/wechattweak-cli", "export", str(export_dir)]

        try:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print("WeChat data exported successfully!")
                return True
            else:
                print(f"Export failed: {result.stderr}")
                return False

        except FileNotFoundError:
            print("\nError: wechattweak-cli not found!")
            print("Please install it first:")
            print("  sudo packages/wechat-exporter/wechattweak-cli install")
            return False
        except Exception as e:
            print(f"Export error: {e}")
            return False

    async def load_data(self, args) -> list[str]:
        """Load WeChat history and convert to text chunks."""
        # Initialize WeChat reader with export capabilities
        reader = WeChatHistoryReader()

        # Find existing exports or create new ones using the centralized method
        export_dirs = reader.find_or_export_wechat_data(args.export_dir)
        if not export_dirs:
            print("Failed to find or export WeChat data. Trying to find any existing exports...")
            # Try to find any existing exports in common locations
            export_dirs = reader.find_wechat_export_dirs()
            if not export_dirs:
                print("No WeChat data found. Please ensure WeChat exports exist.")
                return []

        # Load documents from all found export directories
        all_documents = []
        total_processed = 0

        for i, export_dir in enumerate(export_dirs):
            print(f"\nProcessing WeChat export {i + 1}/{len(export_dirs)}: {export_dir}")

            try:
                # Apply max_items limit per export
                max_per_export = -1
                if args.max_items > 0:
                    remaining = args.max_items - total_processed
                    if remaining <= 0:
                        break
                    max_per_export = remaining

                documents = reader.load_data(
                    wechat_export_dir=str(export_dir),
                    max_count=max_per_export,
                    concatenate_messages=True,  # Enable message concatenation for better context
                )

                if documents:
                    print(f"Loaded {len(documents)} chat documents from {export_dir}")
                    all_documents.extend(documents)
                    total_processed += len(documents)
                else:
                    print(f"No documents loaded from {export_dir}")

            except Exception as e:
                print(f"Error processing {export_dir}: {e}")
                continue

        if not all_documents:
            print("No documents loaded from any source. Exiting.")
            return []

        print(f"\nTotal loaded {len(all_documents)} chat documents from {len(export_dirs)} exports")
        print("now starting to split into text chunks ... take some time")

        # Convert to text chunks with contact information
        all_texts = []
        for doc in all_documents:
            # Split the document into chunks
            from llama_index.core.node_parser import SentenceSplitter

            text_splitter = SentenceSplitter(
                chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
            )
            nodes = text_splitter.get_nodes_from_documents([doc])

            for node in nodes:
                # Add contact information to each chunk
                contact_name = doc.metadata.get("contact_name", "Unknown")
                text = f"[Contact] means the message is from: {contact_name}\n" + node.get_content()
                all_texts.append(text)

        print(f"Created {len(all_texts)} text chunks from {len(all_documents)} documents")
        return all_texts


if __name__ == "__main__":
    import asyncio

    # Check platform
    if sys.platform != "darwin":
        print("\n⚠️  Warning: WeChat export is only supported on macOS")
        print("   You can still query existing exports on other platforms\n")

    # Example queries for WeChat RAG
    print("\n💬 WeChat History RAG Example")
    print("=" * 50)
    print("\nExample queries you can try:")
    print("- 'Show me conversations about travel plans'")
    print("- 'Find group chats about weekend activities'")
    print("- '我想买魔术师约翰逊的球衣,给我一些对应聊天记录?'")
    print("- 'What did we discuss about the project last month?'")
    print("\nNote: WeChat must be running for export to work\n")

    rag = WeChatRAG()
    asyncio.run(rag.run())
