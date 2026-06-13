"""
Shared base for chat export RAG apps (ChatGPT, Claude, etc.).
Unifies: find export files → load with reader → chunk → index.
"""
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).parent))

from base_rag_example import BaseRAGExample
from chunking import create_text_chunks


class ChatExportRAG(BaseRAGExample):
    """Generic RAG app for chat export data (ChatGPT, Claude, etc.).

    No method overrides needed — just provide constructor args.
    """

    def __init__(
        self,
        name: str,
        description: str,
        default_index_name: str,
        reader_factory: Callable[[bool], Any],
        export_keyword: str,
        file_extensions: list[str],
        default_export_dir: str,
        example_queries: list[str],
        export_setup_instructions: list[str],
    ):
        self._reader_factory = reader_factory
        self._export_keyword = export_keyword
        self._file_extensions = file_extensions
        self._default_export_dir = default_export_dir
        self.example_queries = example_queries
        self._export_setup_instructions = export_setup_instructions

        self.max_items_default = -1
        self.embedding_model_default = "sentence-transformers/all-MiniLM-L6-v2"

        super().__init__(
            name=name,
            description=description,
            default_index_name=default_index_name,
        )

    def _add_specific_arguments(self, parser):
        group = parser.add_argument_group(f"{self.name} Parameters")
        group.add_argument(
            "--export-path",
            type=str,
            default=self._default_export_dir,
            help=f"Path to {self.name} export file or directory (default: {self._default_export_dir})",
        )
        group.add_argument(
            "--concatenate-conversations",
            action="store_true",
            default=True,
            help="Concatenate messages within conversations for better context (default: True)",
        )
        group.add_argument(
            "--separate-messages",
            action="store_true",
            help="Process each message as a separate document (overrides --concatenate-conversations)",
        )
        group.add_argument(
            "--chunk-size", type=int, default=512, help="Text chunk size (default: 512)"
        )
        group.add_argument(
            "--chunk-overlap", type=int, default=128, help="Text chunk overlap (default: 128)"
        )

    def _find_exports(self, export_path: Path) -> list[Path]:
        export_files: list[Path] = []
        if export_path.is_file():
            if export_path.suffix.lower() in self._file_extensions:
                export_files.append(export_path)
        elif export_path.is_dir():
            for ext in self._file_extensions:
                export_files.extend(export_path.glob(f"*{ext}"))
        return export_files

    async def load_data(self, args) -> list[dict[str, Any]]:
        export_path = Path(args.export_path)

        if not export_path.exists():
            print(f"{self.name} export path not found: {export_path}")
            print("Please ensure you have exported your data and placed it in the correct location.")
            for line in self._export_setup_instructions:
                print(line)
            return []

        export_files = self._find_exports(export_path)

        if not export_files:
            exts = ", ".join(self._file_extensions)
            print(f"No {self.name} export files ({exts}) found in: {export_path}")
            return []

        print(f"Found {len(export_files)} {self.name} export files")

        concatenate = args.concatenate_conversations and not args.separate_messages
        reader = self._reader_factory(concatenate)

        all_documents, _ = self._foreach_source(
            export_files,
            args,
            load=lambda src, mc: reader.load_data(
                **{
                    f"{self._export_keyword}_export_path": str(src),
                    "max_count": mc,
                    "include_metadata": True,
                }
            ),
            source_label="export file",
        )

        if not all_documents:
            print("No conversations found to process!")
            print("\nTroubleshooting:")
            print("- Ensure the export file is a valid export")
            return []

        print(f"\nTotal conversations processed: {len(all_documents)}")
        print("Now starting to split into text chunks... this may take some time")

        all_texts = create_text_chunks(
            all_documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        )

        print(f"Created {len(all_texts)} text chunks from {len(all_documents)} conversations")
        return all_texts
