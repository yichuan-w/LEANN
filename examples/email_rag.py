"""
Email RAG example using the unified interface.
Supports Apple Mail on macOS.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from base_rag_example import BaseRAGExample, create_text_chunks
from email_data.LEANN_email_reader import EmlxReader


class EmailRAG(BaseRAGExample):
    """RAG example for Apple Mail processing."""

    def __init__(self):
        super().__init__(
            name="Email",
            description="Process and query Apple Mail emails with LEANN",
            default_index_name="mail_index",  # Match original: "./mail_index"
            include_embedding_mode=False,  # Original mail_reader_leann.py doesn't have embedding_mode
        )

    def _add_specific_arguments(self, parser):
        """Add email-specific arguments."""
        email_group = parser.add_argument_group("Email Parameters")
        email_group.add_argument(
            "--mail-path",
            type=str,
            default=None,
            help="Path to Apple Mail directory (auto-detected if not specified)",
        )
        email_group.add_argument(
            "--include-html", action="store_true", help="Include HTML content in email processing"
        )

    def _find_mail_directories(self) -> list[Path]:
        """Auto-detect all Apple Mail directories."""
        mail_base = Path.home() / "Library" / "Mail"
        if not mail_base.exists():
            return []

        # Find all Messages directories
        messages_dirs = []
        for item in mail_base.rglob("Messages"):
            if item.is_dir():
                messages_dirs.append(item)

        return messages_dirs

    async def load_data(self, args) -> list[str]:
        """Load emails and convert to text chunks."""
        # Determine mail directories
        if args.mail_path:
            messages_dirs = [Path(args.mail_path)]
        else:
            print("Auto-detecting Apple Mail directories...")
            messages_dirs = self._find_mail_directories()

        if not messages_dirs:
            print("No Apple Mail directories found!")
            print("Please specify --mail-path manually")
            return []

        print(f"Found {len(messages_dirs)} mail directories")

        # Create reader
        reader = EmlxReader()

        # Process each directory
        all_documents = []
        total_processed = 0

        for i, messages_dir in enumerate(messages_dirs):
            print(f"\nProcessing directory {i + 1}/{len(messages_dirs)}: {messages_dir}")

            try:
                # Count emlx files
                emlx_files = list(messages_dir.glob("*.emlx"))
                print(f"Found {len(emlx_files)} email files")

                # Apply max_items limit per directory
                max_per_dir = -1
                if args.max_items > 0:
                    remaining = args.max_items - total_processed
                    if remaining <= 0:
                        break
                    max_per_dir = remaining

                # Load emails
                documents = reader.load_data(
                    file_path=str(messages_dir),
                    max_count=max_per_dir,
                    include_html=args.include_html,
                )

                if documents:
                    all_documents.extend(documents)
                    total_processed += len(documents)
                    print(f"Processed {len(documents)} emails from this directory")

            except Exception as e:
                print(f"Error processing {messages_dir}: {e}")
                continue

        if not all_documents:
            print("No emails found to process!")
            return []

        print(f"\nTotal emails processed: {len(all_documents)}")

        # Convert to text chunks
        # Email reader uses chunk_overlap=25 as in original
        all_texts = create_text_chunks(all_documents, chunk_overlap=25)

        return all_texts


if __name__ == "__main__":
    import asyncio

    # Check platform
    if sys.platform != "darwin":
        print("\n⚠️  Warning: This example is designed for macOS (Apple Mail)")
        print("   Windows/Linux support coming soon!\n")

    # Example queries for email RAG
    print("\n📧 Email RAG Example")
    print("=" * 50)
    print("\nExample queries you can try:")
    print("- 'What did my boss say about deadlines?'")
    print("- 'Find emails about travel expenses'")
    print("- 'Show me emails from last month about the project'")
    print("- 'What food did I order from DoorDash?'")
    print("\nNote: You may need to grant Full Disk Access to your terminal\n")

    rag = EmailRAG()
    asyncio.run(rag.run())
