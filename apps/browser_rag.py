"""
Browser History RAG example using the unified interface.
Supports Chrome browser history.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from base_rag_example import BaseRAGExample
from chunking import create_text_chunks

from .history_data.history import ChromeHistoryReader


class BrowserRAG(BaseRAGExample):
    """RAG example for Chrome browser history."""

    def __init__(self):
        # Set default values BEFORE calling super().__init__
        self.embedding_model_default = (
            "sentence-transformers/all-MiniLM-L6-v2"  # Fast 384-dim model
        )

        super().__init__(
            name="Browser History",
            description="Process and query Chrome browser history with LEANN",
            default_index_name="google_history_index",
        )

    def _add_specific_arguments(self, parser):
        """Add browser-specific arguments."""
        browser_group = parser.add_argument_group("Browser Parameters")
        browser_group.add_argument(
            "--chrome-profile",
            type=str,
            default=None,
            help="Path to Chrome profile directory (auto-detected if not specified)",
        )
        browser_group.add_argument(
            "--auto-find-profiles",
            action="store_true",
            default=True,
            help="Automatically find all Chrome profiles (default: True)",
        )
        browser_group.add_argument(
            "--chunk-size", type=int, default=256, help="Text chunk size (default: 256)"
        )
        browser_group.add_argument(
            "--chunk-overlap", type=int, default=128, help="Text chunk overlap (default: 128)"
        )

    def _get_chrome_base_path(self) -> Path:
        """Get the base Chrome profile path based on OS."""
        if sys.platform == "darwin":
            return Path.home() / "Library" / "Application Support" / "Google" / "Chrome"
        elif sys.platform.startswith("linux"):
            return Path.home() / ".config" / "google-chrome"
        elif sys.platform == "win32":
            return Path(os.environ["LOCALAPPDATA"]) / "Google" / "Chrome" / "User Data"
        else:
            raise ValueError(f"Unsupported platform: {sys.platform}")

    def _find_chrome_profiles(self) -> list[Path]:
        """Auto-detect all Chrome profiles."""
        base_path = self._get_chrome_base_path()
        if not base_path.exists():
            return []

        profiles = []

        # Check Default profile
        default_profile = base_path / "Default"
        if default_profile.exists() and (default_profile / "History").exists():
            profiles.append(default_profile)

        # Check numbered profiles
        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith("Profile "):
                if (item / "History").exists():
                    profiles.append(item)

        return profiles

    async def load_data(self, args) -> list[str]:
        """Load browser history and convert to text chunks."""
        # Determine Chrome profiles
        if args.chrome_profile and not args.auto_find_profiles:
            profile_dirs = [Path(args.chrome_profile)]
        else:
            print("Auto-detecting Chrome profiles...")
            profile_dirs = self._find_chrome_profiles()

            # If specific profile given, filter to just that one
            if args.chrome_profile:
                profile_path = Path(args.chrome_profile)
                profile_dirs = [p for p in profile_dirs if p == profile_path]

        if not profile_dirs:
            print("No Chrome profiles found!")
            print("Please specify --chrome-profile manually")
            return []

        print(f"Found {len(profile_dirs)} Chrome profiles")

        # Create reader
        reader = ChromeHistoryReader()

        # Process each profile
        all_documents = []
        total_processed = 0

        for i, profile_dir in enumerate(profile_dirs):
            print(f"\nProcessing profile {i + 1}/{len(profile_dirs)}: {profile_dir.name}")

            try:
                # Apply max_items limit per profile
                max_per_profile = -1
                if args.max_items > 0:
                    remaining = args.max_items - total_processed
                    if remaining <= 0:
                        break
                    max_per_profile = remaining

                # Load history
                documents = reader.load_data(
                    chrome_profile_path=str(profile_dir),
                    max_count=max_per_profile,
                )

                if documents:
                    all_documents.extend(documents)
                    total_processed += len(documents)
                    print(f"Processed {len(documents)} history entries from this profile")

            except Exception as e:
                print(f"Error processing {profile_dir}: {e}")
                continue

        if not all_documents:
            print("No browser history found to process!")
            return []

        print(f"\nTotal history entries processed: {len(all_documents)}")

        # Convert to text chunks
        all_texts = create_text_chunks(
            all_documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        )

        return all_texts


if __name__ == "__main__":
    import asyncio

    # Example queries for browser history RAG
    print("\n🌐 Browser History RAG Example")
    print("=" * 50)
    print("\nExample queries you can try:")
    print("- 'What websites did I visit about machine learning?'")
    print("- 'Find my search history about programming'")
    print("- 'What YouTube videos did I watch recently?'")
    print("- 'Show me websites about travel planning'")
    print("\nNote: Make sure Chrome is closed before running\n")

    rag = BrowserRAG()
    asyncio.run(rag.run())
