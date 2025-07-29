"""
WeChat History RAG example using the unified interface.
Supports WeChat chat history export and search.
"""

import subprocess
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from base_rag_example import BaseRAGExample, create_text_chunks
from history_data.wechat_history import WeChatHistoryReader


class WeChatRAG(BaseRAGExample):
    """RAG example for WeChat chat history."""

    def __init__(self):
        # Set default values BEFORE calling super().__init__
        self.max_items_default = 50  # Match original default
        self.embedding_model_default = "Qwen/Qwen3-Embedding-0.6B"  # Match original default

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
        export_path = Path(args.export_dir)

        # Check if we need to export
        need_export = (
            args.force_export or not export_path.exists() or not any(export_path.iterdir())
        )

        if need_export:
            if sys.platform != "darwin":
                print("\n⚠️  Error: WeChat export is only supported on macOS")
                return []

            success = self._export_wechat_data(export_path)
            if not success:
                print("Failed to export WeChat data")
                return []
        else:
            print(f"Using existing WeChat export: {export_path}")

        # Load WeChat data
        reader = WeChatHistoryReader()

        try:
            print("\nLoading WeChat history...")
            documents = reader.load_data(
                wechat_export_dir=str(export_path),
                max_count=args.max_items if args.max_items > 0 else -1,
            )

            if not documents:
                print("No WeChat data found!")
                return []

            print(f"Loaded {len(documents)} chat entries")

            # Convert to text chunks
            all_texts = create_text_chunks(documents)

            return all_texts

        except Exception as e:
            print(f"Error loading WeChat data: {e}")
            return []


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
