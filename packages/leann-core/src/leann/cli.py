import argparse
import asyncio
from pathlib import Path
from typing import Union

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from tqdm import tqdm

from .api import LeannBuilder, LeannChat, LeannSearcher


def extract_pdf_text_with_pymupdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF for better quality."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except ImportError:
        # Fallback to default reader
        return None


def extract_pdf_text_with_pdfplumber(file_path: str) -> str:
    """Extract text from PDF using pdfplumber for better quality."""
    try:
        import pdfplumber

        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except ImportError:
        # Fallback to default reader
        return None


class LeannCLI:
    def __init__(self):
        # Always use project-local .leann directory (like .git)
        self.indexes_dir = Path.cwd() / ".leann" / "indexes"
        self.indexes_dir.mkdir(parents=True, exist_ok=True)

        # Default parser for documents
        self.node_parser = SentenceSplitter(
            chunk_size=256, chunk_overlap=128, separator=" ", paragraph_separator="\n\n"
        )

        # Code-optimized parser
        self.code_parser = SentenceSplitter(
            chunk_size=512,  # Larger chunks for code context
            chunk_overlap=50,  # Less overlap to preserve function boundaries
            separator="\n",  # Split by lines for code
            paragraph_separator="\n\n",  # Preserve logical code blocks
        )

    def get_index_path(self, index_name: str) -> str:
        index_dir = self.indexes_dir / index_name
        return str(index_dir / "documents.leann")

    def index_exists(self, index_name: str) -> bool:
        index_dir = self.indexes_dir / index_name
        meta_file = index_dir / "documents.leann.meta.json"
        return meta_file.exists()

    def create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="leann",
            description="The smallest vector index in the world. RAG Everything with LEANN!",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  leann build my-docs --docs ./documents                                  # Build index from directory
  leann build my-code --docs ./src ./tests ./config                      # Build index from multiple directories
  leann build my-files --docs ./file1.py ./file2.txt ./docs/             # Build index from files and directories
  leann build my-mixed --docs ./readme.md ./src/ ./config.json           # Build index from mixed files/dirs
  leann build my-ppts --docs ./ --file-types .pptx,.pdf                  # Index only PowerPoint and PDF files
  leann search my-docs "query"                                           # Search in my-docs index
  leann ask my-docs "question"                                           # Ask my-docs index
  leann list                                                             # List all stored indexes
  leann remove my-docs                                                   # Remove an index (local first, then global)
            """,
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Build command
        build_parser = subparsers.add_parser("build", help="Build document index")
        build_parser.add_argument(
            "index_name", nargs="?", help="Index name (default: current directory name)"
        )
        build_parser.add_argument(
            "--docs",
            type=str,
            nargs="+",
            default=["."],
            help="Documents directories and/or files (default: current directory)",
        )
        build_parser.add_argument(
            "--backend",
            type=str,
            default="hnsw",
            choices=["hnsw", "diskann"],
            help="Backend to use (default: hnsw)",
        )
        build_parser.add_argument(
            "--embedding-model",
            type=str,
            default="facebook/contriever",
            help="Embedding model (default: facebook/contriever)",
        )
        build_parser.add_argument(
            "--embedding-mode",
            type=str,
            default="sentence-transformers",
            choices=["sentence-transformers", "openai", "mlx", "ollama"],
            help="Embedding backend mode (default: sentence-transformers)",
        )
        build_parser.add_argument(
            "--force", "-f", action="store_true", help="Force rebuild existing index"
        )
        build_parser.add_argument(
            "--graph-degree", type=int, default=32, help="Graph degree (default: 32)"
        )
        build_parser.add_argument(
            "--complexity", type=int, default=64, help="Build complexity (default: 64)"
        )
        build_parser.add_argument("--num-threads", type=int, default=1)
        build_parser.add_argument(
            "--compact",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use compact storage (default: true). Must be `no-compact` for `no-recompute` build.",
        )
        build_parser.add_argument(
            "--recompute",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable recomputation (default: true)",
        )
        build_parser.add_argument(
            "--file-types",
            type=str,
            help="Comma-separated list of file extensions to include (e.g., '.txt,.pdf,.pptx'). If not specified, uses default supported types.",
        )
        build_parser.add_argument(
            "--include-hidden",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Include hidden files and directories (paths starting with '.') during indexing (default: false)",
        )
        build_parser.add_argument(
            "--doc-chunk-size",
            type=int,
            default=256,
            help="Document chunk size in tokens/characters (default: 256)",
        )
        build_parser.add_argument(
            "--doc-chunk-overlap",
            type=int,
            default=128,
            help="Document chunk overlap (default: 128)",
        )
        build_parser.add_argument(
            "--code-chunk-size",
            type=int,
            default=512,
            help="Code chunk size in tokens/lines (default: 512)",
        )
        build_parser.add_argument(
            "--code-chunk-overlap",
            type=int,
            default=50,
            help="Code chunk overlap (default: 50)",
        )

        # Search command
        search_parser = subparsers.add_parser("search", help="Search documents")
        search_parser.add_argument("index_name", help="Index name")
        search_parser.add_argument("query", help="Search query")
        search_parser.add_argument(
            "--top-k", type=int, default=5, help="Number of results (default: 5)"
        )
        search_parser.add_argument(
            "--complexity", type=int, default=64, help="Search complexity (default: 64)"
        )
        search_parser.add_argument("--beam-width", type=int, default=1)
        search_parser.add_argument("--prune-ratio", type=float, default=0.0)
        search_parser.add_argument(
            "--recompute",
            dest="recompute_embeddings",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable/disable embedding recomputation (default: enabled). Should not do a `no-recompute` search in a `recompute` build.",
        )
        search_parser.add_argument(
            "--pruning-strategy",
            choices=["global", "local", "proportional"],
            default="global",
            help="Pruning strategy (default: global)",
        )

        # Ask command
        ask_parser = subparsers.add_parser("ask", help="Ask questions")
        ask_parser.add_argument("index_name", help="Index name")
        ask_parser.add_argument(
            "--llm",
            type=str,
            default="ollama",
            choices=["simulated", "ollama", "hf", "openai"],
            help="LLM provider (default: ollama)",
        )
        ask_parser.add_argument(
            "--model", type=str, default="qwen3:8b", help="Model name (default: qwen3:8b)"
        )
        ask_parser.add_argument("--host", type=str, default="http://localhost:11434")
        ask_parser.add_argument(
            "--interactive", "-i", action="store_true", help="Interactive chat mode"
        )
        ask_parser.add_argument(
            "--top-k", type=int, default=20, help="Retrieval count (default: 20)"
        )
        ask_parser.add_argument("--complexity", type=int, default=32)
        ask_parser.add_argument("--beam-width", type=int, default=1)
        ask_parser.add_argument("--prune-ratio", type=float, default=0.0)
        ask_parser.add_argument(
            "--recompute",
            dest="recompute_embeddings",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable/disable embedding recomputation during ask (default: enabled)",
        )
        ask_parser.add_argument(
            "--pruning-strategy",
            choices=["global", "local", "proportional"],
            default="global",
        )
        ask_parser.add_argument(
            "--thinking-budget",
            type=str,
            choices=["low", "medium", "high"],
            default=None,
            help="Thinking budget for reasoning models (low/medium/high). Supported by GPT-Oss:20b and other reasoning models.",
        )

        # List command
        subparsers.add_parser("list", help="List all indexes")

        # Remove command
        remove_parser = subparsers.add_parser("remove", help="Remove an index")
        remove_parser.add_argument("index_name", help="Index name to remove")
        remove_parser.add_argument(
            "--force", "-f", action="store_true", help="Force removal without confirmation"
        )

        return parser

    def register_project_dir(self):
        """Register current project directory in global registry"""
        global_registry = Path.home() / ".leann" / "projects.json"
        global_registry.parent.mkdir(exist_ok=True)

        current_dir = str(Path.cwd())

        # Load existing registry
        projects = []
        if global_registry.exists():
            try:
                import json

                with open(global_registry) as f:
                    projects = json.load(f)
            except Exception:
                projects = []

        # Add current directory if not already present
        if current_dir not in projects:
            projects.append(current_dir)

        # Save registry
        import json

        with open(global_registry, "w") as f:
            json.dump(projects, f, indent=2)

    def _build_gitignore_parser(self, docs_dir: str):
        """Build gitignore parser using gitignore-parser library."""
        from gitignore_parser import parse_gitignore

        # Try to parse the root .gitignore
        gitignore_path = Path(docs_dir) / ".gitignore"

        if gitignore_path.exists():
            try:
                # gitignore-parser automatically handles all subdirectory .gitignore files!
                matches = parse_gitignore(str(gitignore_path))
                print(f"📋 Loaded .gitignore from {docs_dir} (includes all subdirectories)")
                return matches
            except Exception as e:
                print(f"Warning: Could not parse .gitignore: {e}")
        else:
            print("📋 No .gitignore found")

        # Fallback: basic pattern matching for essential files
        essential_patterns = {".git", ".DS_Store", "__pycache__", "node_modules", ".venv", "venv"}

        def basic_matches(file_path):
            path_parts = Path(file_path).parts
            return any(part in essential_patterns for part in path_parts)

        return basic_matches

    def _should_exclude_file(self, relative_path: Path, gitignore_matches) -> bool:
        """Check if a file should be excluded using gitignore parser."""
        return gitignore_matches(str(relative_path))

    def _is_git_submodule(self, path: Path) -> bool:
        """Check if a path is a git submodule."""
        try:
            # Find the git repo root
            current_dir = Path.cwd()
            while current_dir != current_dir.parent:
                if (current_dir / ".git").exists():
                    gitmodules_path = current_dir / ".gitmodules"
                    if gitmodules_path.exists():
                        # Read .gitmodules to check if this path is a submodule
                        gitmodules_content = gitmodules_path.read_text()
                        # Convert path to relative to git root
                        try:
                            relative_path = path.resolve().relative_to(current_dir)
                            # Check if this path appears in .gitmodules
                            return f"path = {relative_path}" in gitmodules_content
                        except ValueError:
                            # Path is not under git root
                            return False
                    break
                current_dir = current_dir.parent
            return False
        except Exception:
            # If anything goes wrong, assume it's not a submodule
            return False

    def list_indexes(self):
        # Get all project directories with .leann
        global_registry = Path.home() / ".leann" / "projects.json"
        all_projects = []

        if global_registry.exists():
            try:
                import json

                with open(global_registry) as f:
                    all_projects = json.load(f)
            except Exception:
                pass

        # Filter to only existing directories with .leann
        valid_projects = []
        for project_dir in all_projects:
            project_path = Path(project_dir)
            if project_path.exists() and (project_path / ".leann" / "indexes").exists():
                valid_projects.append(project_path)

        # Add current project if it has .leann but not in registry
        current_path = Path.cwd()
        if (current_path / ".leann" / "indexes").exists() and current_path not in valid_projects:
            valid_projects.append(current_path)

        # Separate current and other projects
        current_project = None
        other_projects = []

        for project_path in valid_projects:
            if project_path == current_path:
                current_project = project_path
            else:
                other_projects.append(project_path)

        print("📚 LEANN Indexes")
        print("=" * 50)

        total_indexes = 0
        current_indexes_count = 0

        # Show current project first (most important)
        if current_project:
            current_indexes_dir = current_project / ".leann" / "indexes"
            if current_indexes_dir.exists():
                current_index_dirs = [d for d in current_indexes_dir.iterdir() if d.is_dir()]

                print("\n🏠 Current Project")
                print(f"   {current_project}")
                print("   " + "─" * 45)

                if current_index_dirs:
                    for index_dir in current_index_dirs:
                        total_indexes += 1
                        current_indexes_count += 1
                        index_name = index_dir.name
                        meta_file = index_dir / "documents.leann.meta.json"
                        status = "✅" if meta_file.exists() else "❌"

                        print(f"   {current_indexes_count}. {index_name} {status}")
                        if meta_file.exists():
                            size_mb = sum(
                                f.stat().st_size for f in index_dir.iterdir() if f.is_file()
                            ) / (1024 * 1024)
                            print(f"      📦 Size: {size_mb:.1f} MB")
                else:
                    print("   📭 No indexes in current project")
        else:
            print("\n🏠 Current Project")
            print(f"   {current_path}")
            print("   " + "─" * 45)
            print("   📭 No indexes in current project")

        # Show other projects (reference information)
        if other_projects:
            print("\n\n🗂️  Other Projects")
            print("   " + "─" * 45)

            for project_path in other_projects:
                indexes_dir = project_path / ".leann" / "indexes"
                if not indexes_dir.exists():
                    continue

                index_dirs = [d for d in indexes_dir.iterdir() if d.is_dir()]
                if not index_dirs:
                    continue

                print(f"\n   📂 {project_path.name}")
                print(f"      {project_path}")

                for index_dir in index_dirs:
                    total_indexes += 1
                    index_name = index_dir.name
                    meta_file = index_dir / "documents.leann.meta.json"
                    status = "✅" if meta_file.exists() else "❌"

                    print(f"      • {index_name} {status}")
                    if meta_file.exists():
                        size_mb = sum(
                            f.stat().st_size for f in index_dir.iterdir() if f.is_file()
                        ) / (1024 * 1024)
                        print(f"        📦 {size_mb:.1f} MB")

        # Summary and usage info
        print("\n" + "=" * 50)
        if total_indexes == 0:
            print("💡 Get started:")
            print("   leann build my-docs --docs ./documents")
        else:
            projects_count = len(
                [
                    p
                    for p in valid_projects
                    if (p / ".leann" / "indexes").exists()
                    and list((p / ".leann" / "indexes").iterdir())
                ]
            )
            print(f"📊 Total: {total_indexes} indexes across {projects_count} projects")

            if current_indexes_count > 0:
                print("\n💫 Quick start (current project):")
                # Get first index from current project for example
                current_indexes_dir = current_path / ".leann" / "indexes"
                if current_indexes_dir.exists():
                    current_index_dirs = [d for d in current_indexes_dir.iterdir() if d.is_dir()]
                    if current_index_dirs:
                        example_name = current_index_dirs[0].name
                        print(f'   leann search {example_name} "your query"')
                        print(f"   leann ask {example_name} --interactive")
            else:
                print("\n💡 Create your first index:")
                print("   leann build my-docs --docs ./documents")

    def remove_index(self, index_name: str, force: bool = False):
        """Safely remove an index - always show all matches for transparency"""

        # Always do a comprehensive search for safety
        print(f"🔍 Searching for all indexes named '{index_name}'...")
        all_matches = self._find_all_matching_indexes(index_name)

        if not all_matches:
            print(f"❌ Index '{index_name}' not found in any project.")
            return False

        if len(all_matches) == 1:
            return self._remove_single_match(all_matches[0], index_name, force)
        else:
            return self._remove_from_multiple_matches(all_matches, index_name, force)

    def _find_all_matching_indexes(self, index_name: str):
        """Find all indexes with the given name across all projects"""
        matches = []

        # Get all registered projects
        global_registry = Path.home() / ".leann" / "projects.json"
        all_projects = []

        if global_registry.exists():
            try:
                import json

                with open(global_registry) as f:
                    all_projects = json.load(f)
            except Exception:
                pass

        # Always include current project
        current_path = Path.cwd()
        if str(current_path) not in all_projects:
            all_projects.append(str(current_path))

        # Search across all projects
        for project_dir in all_projects:
            project_path = Path(project_dir)
            if not project_path.exists():
                continue

            index_dir = project_path / ".leann" / "indexes" / index_name
            if index_dir.exists():
                is_current = project_path == current_path
                matches.append(
                    {"project_path": project_path, "index_dir": index_dir, "is_current": is_current}
                )

        # Sort: current project first, then by project name
        matches.sort(key=lambda x: (not x["is_current"], x["project_path"].name))
        return matches

    def _remove_single_match(self, match, index_name: str, force: bool):
        """Handle removal when only one match is found"""
        project_path = match["project_path"]
        index_dir = match["index_dir"]
        is_current = match["is_current"]

        if is_current:
            location_info = "current project"
            emoji = "🏠"
        else:
            location_info = f"other project '{project_path.name}'"
            emoji = "📂"

        print(f"✅ Found 1 index named '{index_name}':")
        print(f"   {emoji} Location: {location_info}")
        print(f"   📍 Path: {project_path}")

        if not force:
            if not is_current:
                print("\n⚠️  CROSS-PROJECT REMOVAL!")
                print("   This will delete the index from another project.")

            response = input(f"   ❓ Confirm removal from {location_info}? (y/N): ").strip().lower()
            if response not in ["y", "yes"]:
                print("   ❌ Removal cancelled.")
                return False

        return self._delete_index_directory(
            index_dir, index_name, project_path if not is_current else None
        )

    def _remove_from_multiple_matches(self, matches, index_name: str, force: bool):
        """Handle removal when multiple matches are found"""

        print(f"⚠️  Found {len(matches)} indexes named '{index_name}':")
        print("   " + "─" * 50)

        for i, match in enumerate(matches, 1):
            project_path = match["project_path"]
            is_current = match["is_current"]

            if is_current:
                print(f"   {i}. 🏠 Current project")
                print(f"      📍 {project_path}")
            else:
                print(f"   {i}. 📂 {project_path.name}")
                print(f"      📍 {project_path}")

            # Show size info
            try:
                size_mb = sum(
                    f.stat().st_size for f in match["index_dir"].iterdir() if f.is_file()
                ) / (1024 * 1024)
                print(f"      📦 Size: {size_mb:.1f} MB")
            except (OSError, PermissionError):
                pass

        print("   " + "─" * 50)

        if force:
            print("   ❌ Multiple matches found, but --force specified.")
            print("   Please run without --force to choose which one to remove.")
            return False

        try:
            choice = input(
                f"   ❓ Which one to remove? (1-{len(matches)}, or 'c' to cancel): "
            ).strip()
            if choice.lower() == "c":
                print("   ❌ Removal cancelled.")
                return False

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(matches):
                selected_match = matches[choice_idx]
                project_path = selected_match["project_path"]
                index_dir = selected_match["index_dir"]
                is_current = selected_match["is_current"]

                location = "current project" if is_current else f"'{project_path.name}' project"
                print(f"   🎯 Selected: Remove from {location}")

                # Final confirmation for safety
                confirm = input(
                    f"   ❓ FINAL CONFIRMATION - Type '{index_name}' to proceed: "
                ).strip()
                if confirm != index_name:
                    print("   ❌ Confirmation failed. Removal cancelled.")
                    return False

                return self._delete_index_directory(
                    index_dir, index_name, project_path if not is_current else None
                )
            else:
                print("   ❌ Invalid choice. Removal cancelled.")
                return False

        except (ValueError, KeyboardInterrupt):
            print("\n   ❌ Invalid input. Removal cancelled.")
            return False

    def _delete_index_directory(
        self, index_dir: Path, index_name: str, project_path: Path | None = None
    ):
        """Actually delete the index directory"""
        try:
            import shutil

            shutil.rmtree(index_dir)

            if project_path:
                print(f"✅ Index '{index_name}' removed from {project_path.name}")
            else:
                print(f"✅ Index '{index_name}' removed successfully")
            return True
        except Exception as e:
            print(f"❌ Error removing index '{index_name}': {e}")
            return False

    def load_documents(
        self,
        docs_paths: Union[str, list],
        custom_file_types: Union[str, None] = None,
        include_hidden: bool = False,
    ):
        # Handle both single path (string) and multiple paths (list) for backward compatibility
        if isinstance(docs_paths, str):
            docs_paths = [docs_paths]

        # Separate files and directories
        files = []
        directories = []
        for path in docs_paths:
            path_obj = Path(path)
            if path_obj.is_file():
                files.append(str(path_obj))
            elif path_obj.is_dir():
                # Check if this is a git submodule - if so, skip it
                if self._is_git_submodule(path_obj):
                    print(f"⚠️  Skipping git submodule: {path}")
                    continue
                directories.append(str(path_obj))
            else:
                print(f"⚠️  Warning: Path '{path}' does not exist, skipping...")
                continue

        # Print summary of what we're processing
        total_items = len(files) + len(directories)
        items_desc = []
        if files:
            items_desc.append(f"{len(files)} file{'s' if len(files) > 1 else ''}")
        if directories:
            items_desc.append(
                f"{len(directories)} director{'ies' if len(directories) > 1 else 'y'}"
            )

        print(f"Loading documents from {' and '.join(items_desc)} ({total_items} total):")
        if files:
            print(f"  📄 Files: {', '.join([Path(f).name for f in files])}")
        if directories:
            print(f"  📁 Directories: {', '.join(directories)}")

        if custom_file_types:
            print(f"Using custom file types: {custom_file_types}")

        all_documents = []

        # Helper to detect hidden path components
        def _path_has_hidden_segment(p: Path) -> bool:
            return any(part.startswith(".") and part not in [".", ".."] for part in p.parts)

        # First, process individual files if any
        if files:
            print(f"\n🔄 Processing {len(files)} individual file{'s' if len(files) > 1 else ''}...")

            # Load individual files using SimpleDirectoryReader with input_files
            # Note: We skip gitignore filtering for explicitly specified files
            try:
                # Group files by their parent directory for efficient loading
                from collections import defaultdict

                files_by_dir = defaultdict(list)
                for file_path in files:
                    file_path_obj = Path(file_path)
                    if not include_hidden and _path_has_hidden_segment(file_path_obj):
                        print(f"  ⚠️  Skipping hidden file: {file_path}")
                        continue
                    parent_dir = str(file_path_obj.parent)
                    files_by_dir[parent_dir].append(str(file_path_obj))

                # Load files from each parent directory
                for parent_dir, file_list in files_by_dir.items():
                    print(
                        f"  Loading {len(file_list)} file{'s' if len(file_list) > 1 else ''} from {parent_dir}"
                    )
                    try:
                        file_docs = SimpleDirectoryReader(
                            parent_dir,
                            input_files=file_list,
                            # exclude_hidden only affects directory scans; input_files are explicit
                            filename_as_id=True,
                        ).load_data()
                        all_documents.extend(file_docs)
                        print(
                            f"    ✅ Loaded {len(file_docs)} document{'s' if len(file_docs) > 1 else ''}"
                        )
                    except Exception as e:
                        print(f"    ❌ Warning: Could not load files from {parent_dir}: {e}")

            except Exception as e:
                print(f"❌ Error processing individual files: {e}")

        # Define file extensions to process
        if custom_file_types:
            # Parse custom file types from comma-separated string
            code_extensions = [ext.strip() for ext in custom_file_types.split(",") if ext.strip()]
            # Ensure extensions start with a dot
            code_extensions = [ext if ext.startswith(".") else f".{ext}" for ext in code_extensions]
        else:
            # Use default supported file types
            code_extensions = [
                # Original document types
                ".txt",
                ".md",
                ".docx",
                ".pptx",
                # Code files for Claude Code integration
                ".py",
                ".js",
                ".ts",
                ".jsx",
                ".tsx",
                ".java",
                ".cpp",
                ".c",
                ".h",
                ".hpp",
                ".cs",
                ".go",
                ".rs",
                ".rb",
                ".php",
                ".swift",
                ".kt",
                ".scala",
                ".r",
                ".sql",
                ".sh",
                ".bash",
                ".zsh",
                ".fish",
                ".ps1",
                ".bat",
                # Config and markup files
                ".json",
                ".yaml",
                ".yml",
                ".xml",
                ".toml",
                ".ini",
                ".cfg",
                ".conf",
                ".html",
                ".css",
                ".scss",
                ".less",
                ".vue",
                ".svelte",
                # Data science
                ".ipynb",
                ".R",
                ".py",
                ".jl",
            ]

        # Process each directory
        if directories:
            print(
                f"\n🔄 Processing {len(directories)} director{'ies' if len(directories) > 1 else 'y'}..."
            )

        for docs_dir in directories:
            print(f"Processing directory: {docs_dir}")
            # Build gitignore parser for each directory
            gitignore_matches = self._build_gitignore_parser(docs_dir)

            # Try to use better PDF parsers first, but only if PDFs are requested
            documents = []
            docs_path = Path(docs_dir)

            # Check if we should process PDFs
            should_process_pdfs = custom_file_types is None or ".pdf" in custom_file_types

            if should_process_pdfs:
                for file_path in docs_path.rglob("*.pdf"):
                    # Check if file matches any exclude pattern
                    try:
                        relative_path = file_path.relative_to(docs_path)
                        if not include_hidden and _path_has_hidden_segment(relative_path):
                            continue
                        if self._should_exclude_file(relative_path, gitignore_matches):
                            continue
                    except ValueError:
                        # Skip files that can't be made relative to docs_path
                        print(f"⚠️  Skipping file outside directory scope: {file_path}")
                        continue

                    print(f"Processing PDF: {file_path}")

                    # Try PyMuPDF first (best quality)
                    text = extract_pdf_text_with_pymupdf(str(file_path))
                    if text is None:
                        # Try pdfplumber
                        text = extract_pdf_text_with_pdfplumber(str(file_path))

                    if text:
                        # Create a simple document structure
                        from llama_index.core import Document

                        doc = Document(text=text, metadata={"source": str(file_path)})
                        documents.append(doc)
                    else:
                        # Fallback to default reader
                        print(f"Using default reader for {file_path}")
                        try:
                            default_docs = SimpleDirectoryReader(
                                str(file_path.parent),
                                exclude_hidden=not include_hidden,
                                filename_as_id=True,
                                required_exts=[file_path.suffix],
                            ).load_data()
                            documents.extend(default_docs)
                        except Exception as e:
                            print(f"Warning: Could not process {file_path}: {e}")

            # Load other file types with default reader
            try:
                # Create a custom file filter function using our PathSpec
                def file_filter(
                    file_path: str, docs_dir=docs_dir, gitignore_matches=gitignore_matches
                ) -> bool:
                    """Return True if file should be included (not excluded)"""
                    try:
                        docs_path_obj = Path(docs_dir)
                        file_path_obj = Path(file_path)
                        relative_path = file_path_obj.relative_to(docs_path_obj)
                        return not self._should_exclude_file(relative_path, gitignore_matches)
                    except (ValueError, OSError):
                        return True  # Include files that can't be processed

                other_docs = SimpleDirectoryReader(
                    docs_dir,
                    recursive=True,
                    encoding="utf-8",
                    required_exts=code_extensions,
                    file_extractor={},  # Use default extractors
                    exclude_hidden=not include_hidden,
                    filename_as_id=True,
                ).load_data(show_progress=True)

                # Filter documents after loading based on gitignore rules
                filtered_docs = []
                for doc in other_docs:
                    file_path = doc.metadata.get("file_path", "")
                    if file_filter(file_path):
                        filtered_docs.append(doc)

                documents.extend(filtered_docs)
            except ValueError as e:
                if "No files found" in str(e):
                    print(f"No additional files found for other supported types in {docs_dir}.")
                else:
                    raise e

            all_documents.extend(documents)
            print(f"Loaded {len(documents)} documents from {docs_dir}")

        documents = all_documents

        all_texts = []

        # Define code file extensions for intelligent chunking
        code_file_exts = {
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".hpp",
            ".cs",
            ".go",
            ".rs",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".scala",
            ".r",
            ".sql",
            ".sh",
            ".bash",
            ".zsh",
            ".fish",
            ".ps1",
            ".bat",
            ".json",
            ".yaml",
            ".yml",
            ".xml",
            ".toml",
            ".ini",
            ".cfg",
            ".conf",
            ".html",
            ".css",
            ".scss",
            ".less",
            ".vue",
            ".svelte",
            ".ipynb",
            ".R",
            ".jl",
        }

        print("start chunking documents")
        # Add progress bar for document chunking
        for doc in tqdm(documents, desc="Chunking documents", unit="doc"):
            # Check if this is a code file based on source path
            source_path = doc.metadata.get("source", "")
            is_code_file = any(source_path.endswith(ext) for ext in code_file_exts)

            # Use appropriate parser based on file type
            parser = self.code_parser if is_code_file else self.node_parser
            nodes = parser.get_nodes_from_documents([doc])

            for node in nodes:
                all_texts.append(node.get_content())

        print(f"Loaded {len(documents)} documents, {len(all_texts)} chunks")
        return all_texts

    async def build_index(self, args):
        docs_paths = args.docs
        # Use current directory name if index_name not provided
        if args.index_name:
            index_name = args.index_name
        else:
            index_name = Path.cwd().name
            print(f"Using current directory name as index: '{index_name}'")

        index_dir = self.indexes_dir / index_name
        index_path = self.get_index_path(index_name)

        # Display all paths being indexed with file/directory distinction
        files = [p for p in docs_paths if Path(p).is_file()]
        directories = [p for p in docs_paths if Path(p).is_dir()]

        print(f"📂 Indexing {len(docs_paths)} path{'s' if len(docs_paths) > 1 else ''}:")
        if files:
            print(f"  📄 Files ({len(files)}):")
            for i, file_path in enumerate(files, 1):
                print(f"    {i}. {Path(file_path).resolve()}")
        if directories:
            print(f"  📁 Directories ({len(directories)}):")
            for i, dir_path in enumerate(directories, 1):
                print(f"    {i}. {Path(dir_path).resolve()}")

        if index_dir.exists() and not args.force:
            print(f"Index '{index_name}' already exists. Use --force to rebuild.")
            return

        # Configure chunking based on CLI args before loading documents
        # Guard against invalid configurations
        doc_chunk_size = max(1, int(args.doc_chunk_size))
        doc_chunk_overlap = max(0, int(args.doc_chunk_overlap))
        if doc_chunk_overlap >= doc_chunk_size:
            print(
                f"⚠️  Adjusting doc chunk overlap from {doc_chunk_overlap} to {doc_chunk_size - 1} (must be < chunk size)"
            )
            doc_chunk_overlap = doc_chunk_size - 1

        code_chunk_size = max(1, int(args.code_chunk_size))
        code_chunk_overlap = max(0, int(args.code_chunk_overlap))
        if code_chunk_overlap >= code_chunk_size:
            print(
                f"⚠️  Adjusting code chunk overlap from {code_chunk_overlap} to {code_chunk_size - 1} (must be < chunk size)"
            )
            code_chunk_overlap = code_chunk_size - 1

        self.node_parser = SentenceSplitter(
            chunk_size=doc_chunk_size,
            chunk_overlap=doc_chunk_overlap,
            separator=" ",
            paragraph_separator="\n\n",
        )
        self.code_parser = SentenceSplitter(
            chunk_size=code_chunk_size,
            chunk_overlap=code_chunk_overlap,
            separator="\n",
            paragraph_separator="\n\n",
        )

        all_texts = self.load_documents(
            docs_paths, args.file_types, include_hidden=args.include_hidden
        )
        if not all_texts:
            print("No documents found")
            return

        index_dir.mkdir(parents=True, exist_ok=True)

        print(f"Building index '{index_name}' with {args.backend} backend...")

        builder = LeannBuilder(
            backend_name=args.backend,
            embedding_model=args.embedding_model,
            embedding_mode=args.embedding_mode,
            graph_degree=args.graph_degree,
            complexity=args.complexity,
            is_compact=args.compact,
            is_recompute=args.recompute,
            num_threads=args.num_threads,
        )

        for chunk_text in all_texts:
            builder.add_text(chunk_text)

        builder.build_index(index_path)
        print(f"Index built at {index_path}")

        # Register this project directory in global registry
        self.register_project_dir()

    async def search_documents(self, args):
        index_name = args.index_name
        query = args.query
        index_path = self.get_index_path(index_name)

        if not self.index_exists(index_name):
            print(
                f"Index '{index_name}' not found. Use 'leann build {index_name} --docs <dir> [<dir2> ...]' to create it."
            )
            return

        searcher = LeannSearcher(index_path=index_path)
        results = searcher.search(
            query,
            top_k=args.top_k,
            complexity=args.complexity,
            beam_width=args.beam_width,
            prune_ratio=args.prune_ratio,
            recompute_embeddings=args.recompute_embeddings,
            pruning_strategy=args.pruning_strategy,
        )

        print(f"Search results for '{query}' (top {len(results)}):")
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result.score:.3f}")
            print(f"   {result.text[:200]}...")
            print()

    async def ask_questions(self, args):
        index_name = args.index_name
        index_path = self.get_index_path(index_name)

        if not self.index_exists(index_name):
            print(
                f"Index '{index_name}' not found. Use 'leann build {index_name} --docs <dir> [<dir2> ...]' to create it."
            )
            return

        print(f"Starting chat with index '{index_name}'...")
        print(f"Using {args.model} ({args.llm})")

        llm_config = {"type": args.llm, "model": args.model}
        if args.llm == "ollama":
            llm_config["host"] = args.host

        chat = LeannChat(index_path=index_path, llm_config=llm_config)

        if args.interactive:
            print("LEANN Assistant ready! Type 'quit' to exit")
            print("=" * 40)

            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                if not user_input:
                    continue

                # Prepare LLM kwargs with thinking budget if specified
                llm_kwargs = {}
                if args.thinking_budget:
                    llm_kwargs["thinking_budget"] = args.thinking_budget

                response = chat.ask(
                    user_input,
                    top_k=args.top_k,
                    complexity=args.complexity,
                    beam_width=args.beam_width,
                    prune_ratio=args.prune_ratio,
                    recompute_embeddings=args.recompute_embeddings,
                    pruning_strategy=args.pruning_strategy,
                    llm_kwargs=llm_kwargs,
                )
                print(f"LEANN: {response}")
        else:
            query = input("Enter your question: ").strip()
            if query:
                # Prepare LLM kwargs with thinking budget if specified
                llm_kwargs = {}
                if args.thinking_budget:
                    llm_kwargs["thinking_budget"] = args.thinking_budget

                response = chat.ask(
                    query,
                    top_k=args.top_k,
                    complexity=args.complexity,
                    beam_width=args.beam_width,
                    prune_ratio=args.prune_ratio,
                    recompute_embeddings=args.recompute_embeddings,
                    pruning_strategy=args.pruning_strategy,
                    llm_kwargs=llm_kwargs,
                )
                print(f"LEANN: {response}")

    async def run(self, args=None):
        parser = self.create_parser()

        if args is None:
            args = parser.parse_args()

        if not args.command:
            parser.print_help()
            return

        if args.command == "list":
            self.list_indexes()
        elif args.command == "remove":
            self.remove_index(args.index_name, args.force)
        elif args.command == "build":
            await self.build_index(args)
        elif args.command == "search":
            await self.search_documents(args)
        elif args.command == "ask":
            await self.ask_questions(args)
        else:
            parser.print_help()


def main():
    import logging

    import dotenv

    dotenv.load_dotenv()

    # Set clean logging for CLI usage
    logging.getLogger().setLevel(logging.WARNING)  # Only show warnings and errors

    cli = LeannCLI()
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
