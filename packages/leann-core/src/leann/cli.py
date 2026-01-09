import argparse
import asyncio
import contextlib
import os
import time
from pathlib import Path
from typing import Any, Optional, Union

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from tqdm import tqdm

from .api import LeannBuilder, LeannChat, LeannSearcher
from .interactive_utils import create_cli_session
from .registry import register_project_directory
from .settings import (
    resolve_anthropic_base_url,
    resolve_ollama_host,
    resolve_openai_api_key,
    resolve_openai_base_url,
)


@contextlib.contextmanager
def suppress_cpp_output(suppress: bool = True):
    """Context manager to suppress C++ stdout/stderr output from FAISS/HNSW.

    This redirects file descriptors at the OS level to suppress native C++ output
    that cannot be controlled via Python's logging framework.
    """
    if not suppress:
        yield
        return

    # Save original file descriptors
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)

    try:
        # Open /dev/null for writing
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)  # Redirect stdout
        os.dup2(devnull, 2)  # Redirect stderr
        os.close(devnull)
        yield
    finally:
        # Restore original file descriptors
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)


def extract_pdf_text_with_pymupdf(file_path: str) -> str | None:
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


def extract_pdf_text_with_pdfplumber(file_path: str) -> str | None:
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
        # Respect LEANN_HOME if set, otherwise fallback to project-local .leann
        self.leann_home = Path(os.environ.get("LEANN_HOME", Path.cwd() / ".leann"))
        self.indexes_dir = self.leann_home / "indexes"
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

        # Global verbosity options
        verbosity_group = parser.add_mutually_exclusive_group()
        verbosity_group.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Show detailed output including C++ backend logs from FAISS/HNSW",
        )
        verbosity_group.add_argument(
            "-q",
            "--quiet",
            action="store_true",
            help="Suppress all non-essential output (default behavior)",
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
            "--backend-name",
            type=str,
            default="hnsw",
            choices=["hnsw", "diskann", "faiss"],
            help="Backend to use (default: hnsw)",
        )
        build_parser.add_argument(
            "--embedding-model",
            type=str,
            default="nomic-ai/nomic-embed-text-v1.5",
            help="Embedding model (default: nomic-ai/nomic-embed-text-v1.5)",
        )
        build_parser.add_argument(
            "--embedding-mode",
            type=str,
            default="sentence-transformers",
            choices=["sentence-transformers", "openai", "mlx", "ollama"],
            help="Embedding backend mode (default: sentence-transformers)",
        )
        build_parser.add_argument(
            "--embedding-host",
            type=str,
            default=None,
            help="Override Ollama-compatible embedding host",
        )
        build_parser.add_argument(
            "--embedding-api-base",
            type=str,
            default=None,
            help="Base URL for OpenAI-compatible embedding services",
        )
        build_parser.add_argument(
            "--embedding-api-key",
            type=str,
            default=None,
            help="API key for embedding service (defaults to OPENAI_API_KEY)",
        )
        build_parser.add_argument(
            "--embedding-prompt-template",
            type=str,
            default=None,
            help="Prompt template to prepend to all texts for embedding (e.g., 'query: ' for search)",
        )
        build_parser.add_argument(
            "--query-prompt-template",
            type=str,
            default=None,
            help="Prompt template for queries (different from build template for task-specific models)",
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
            help="Document chunk size in TOKENS (default: 256). Final chunks may be larger due to overlap. For 512 token models: recommended 350 tokens (350 + 128 overlap = 478 max)",
        )
        build_parser.add_argument(
            "--doc-chunk-overlap",
            type=int,
            default=128,
            help="Document chunk overlap in TOKENS (default: 128). Added to chunk size, not included in it",
        )
        build_parser.add_argument(
            "--code-chunk-size",
            type=int,
            default=512,
            help="Code chunk size in TOKENS (default: 512). Final chunks may be larger due to overlap. For 512 token models: recommended 400 tokens (400 + 50 overlap = 450 max)",
        )
        build_parser.add_argument(
            "--code-chunk-overlap",
            type=int,
            default=50,
            help="Code chunk overlap in TOKENS (default: 50). Added to chunk size, not included in it",
        )
        build_parser.add_argument(
            "--use-ast-chunking",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable AST-aware chunking for code files (requires astchunk) (default: true)",
        )
        build_parser.add_argument(
            "--ast-chunk-size",
            type=int,
            default=300,
            help="AST chunk size in CHARACTERS (non-whitespace) (default: 300). Final chunks may be larger due to overlap and expansion. For 512 token models: recommended 300 chars (300 + 64 overlap ~= 480 tokens)",
        )
        build_parser.add_argument(
            "--ast-chunk-overlap",
            type=int,
            default=64,
            help="AST chunk overlap in CHARACTERS (default: 64). Added to chunk size, not included in it. ~1.2 tokens per character for code",
        )
        build_parser.add_argument(
            "--ast-fallback-traditional",
            action="store_true",
            default=True,
            help="Fall back to traditional chunking if AST chunking fails (default: True)",
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
        search_parser.add_argument(
            "--non-interactive",
            action="store_true",
            help="Non-interactive mode: automatically select index without prompting",
        )
        search_parser.add_argument(
            "--show-metadata",
            action="store_true",
            help="Display file paths and metadata in search results",
        )
        search_parser.add_argument(
            "--embedding-prompt-template",
            type=str,
            default=None,
            help="Prompt template to prepend to query for embedding (e.g., 'query: ' for search)",
        )

        # Ask command
        ask_parser = subparsers.add_parser("ask", help="Ask questions")
        ask_parser.add_argument("index_name", help="Index name")
        ask_parser.add_argument(
            "query",
            nargs="?",
            help="Question to ask (omit for prompt or when using --interactive)",
        )
        ask_parser.add_argument(
            "--llm",
            type=str,
            default="ollama",
            choices=["simulated", "ollama", "hf", "openai", "anthropic"],
            help="LLM provider (default: ollama)",
        )
        ask_parser.add_argument(
            "--model", type=str, default="qwen3:8b", help="Model name (default: qwen3:8b)"
        )
        ask_parser.add_argument(
            "--host",
            type=str,
            default=None,
            help="Override Ollama-compatible host (defaults to LEANN_OLLAMA_HOST/OLLAMA_HOST)",
        )
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
        ask_parser.add_argument(
            "--api-base",
            type=str,
            default=None,
            help="Base URL for OpenAI-compatible APIs (e.g., http://localhost:10000/v1)",
        )
        ask_parser.add_argument(
            "--api-key",
            type=str,
            default=None,
            help="API key for cloud LLM providers (OpenAI, Anthropic)",
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
        register_project_directory()

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

    def _should_exclude_file(self, file_path: Path, gitignore_matches) -> bool:
        """Check if a file should be excluded using gitignore parser.

        Always match against absolute, posix-style paths for consistency with
        gitignore_parser expectations.
        """
        try:
            absolute_path = file_path.resolve()
        except Exception:
            absolute_path = Path(str(file_path))
        return gitignore_matches(absolute_path.as_posix())

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
        other_projects = []

        for project_path in valid_projects:
            if project_path != current_path:
                other_projects.append(project_path)

        print("📚 LEANN Indexes")
        print("=" * 50)

        total_indexes = 0
        current_indexes_count = 0

        # Show current project first (most important)
        print("\n🏠 Current Project")
        print(f"   {current_path}")
        print("   " + "─" * 45)

        current_indexes = self._discover_indexes_in_project(
            current_path, exclude_dirs=other_projects
        )
        if current_indexes:
            for idx in current_indexes:
                total_indexes += 1
                current_indexes_count += 1
                type_icon = "📁" if idx["type"] == "cli" else "📄"
                print(f"   {current_indexes_count}. {type_icon} {idx['name']} {idx['status']}")
                if idx["size_mb"] > 0:
                    print(f"      📦 Size: {idx['size_mb']:.1f} MB")
        else:
            print("   📭 No indexes in current project")

        # Show other projects (reference information)
        if other_projects:
            print("\n\n🗂️  Other Projects")
            print("   " + "─" * 45)

            for project_path in other_projects:
                project_indexes = self._discover_indexes_in_project(project_path)
                if not project_indexes:
                    continue

                print(f"\n   📂 {project_path.name}")
                print(f"      {project_path}")

                for idx in project_indexes:
                    total_indexes += 1
                    type_icon = "📁" if idx["type"] == "cli" else "📄"
                    print(f"      • {type_icon} {idx['name']} {idx['status']}")
                    if idx["size_mb"] > 0:
                        print(f"        📦 {idx['size_mb']:.1f} MB")

        # Summary and usage info
        print("\n" + "=" * 50)
        if total_indexes == 0:
            print("💡 Get started:")
            print("   leann build my-docs --docs ./documents")
        else:
            # Count only projects that have at least one discoverable index
            projects_count = 0
            for p in valid_projects:
                if p == current_path:
                    discovered = self._discover_indexes_in_project(p, exclude_dirs=other_projects)
                else:
                    discovered = self._discover_indexes_in_project(p)
                if len(discovered) > 0:
                    projects_count += 1
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

    def _discover_indexes_in_project(
        self, project_path: Path, exclude_dirs: Optional[list[Path]] = None
    ):
        """Discover all indexes in a project directory (both CLI and apps formats)

        exclude_dirs: when provided, skip any APP-format index files that are
        located under these directories. This prevents duplicates when the
        current project is a parent directory of other registered projects.
        """
        indexes = []
        exclude_dirs = exclude_dirs or []
        # normalize to resolved paths once for comparison
        try:
            exclude_dirs_resolved = [p.resolve() for p in exclude_dirs]
        except Exception:
            exclude_dirs_resolved = exclude_dirs

        # 1. CLI format: .leann/indexes/index_name/
        cli_indexes_dir = project_path / ".leann" / "indexes"
        if cli_indexes_dir.exists():
            for index_dir in cli_indexes_dir.iterdir():
                if index_dir.is_dir():
                    meta_file = index_dir / "documents.leann.meta.json"
                    status = "✅" if meta_file.exists() else "❌"

                    size_mb = 0
                    if meta_file.exists():
                        try:
                            size_mb = sum(
                                f.stat().st_size for f in index_dir.iterdir() if f.is_file()
                            ) / (1024 * 1024)
                        except (OSError, PermissionError):
                            pass

                    indexes.append(
                        {
                            "name": index_dir.name,
                            "type": "cli",
                            "status": status,
                            "size_mb": size_mb,
                            "path": index_dir,
                        }
                    )

        # 2. Apps format: *.leann.meta.json files anywhere in the project
        cli_indexes_dir = project_path / ".leann" / "indexes"
        for meta_file in project_path.rglob("*.leann.meta.json"):
            if meta_file.is_file():
                # Skip CLI-built indexes (which store meta under .leann/indexes/<name>/)
                try:
                    if cli_indexes_dir.exists() and cli_indexes_dir in meta_file.parents:
                        continue
                except Exception:
                    pass
                # Skip meta files that live under excluded directories
                try:
                    meta_parent_resolved = meta_file.parent.resolve()
                    if any(
                        meta_parent_resolved.is_relative_to(ex_dir)
                        for ex_dir in exclude_dirs_resolved
                    ):
                        continue
                except Exception:
                    # best effort; if resolve or comparison fails, do not exclude
                    pass
                # Use the parent directory name as the app index display name
                display_name = meta_file.parent.name
                # Extract file base used to store files
                file_base = meta_file.name.replace(".leann.meta.json", "")

                # Apps indexes are considered complete if the .leann.meta.json file exists
                status = "✅"

                # Calculate total size of all related files (use file base)
                size_mb = 0
                try:
                    index_dir = meta_file.parent
                    for related_file in index_dir.glob(f"{file_base}.leann*"):
                        size_mb += related_file.stat().st_size / (1024 * 1024)
                except (OSError, PermissionError):
                    pass

                indexes.append(
                    {
                        "name": display_name,
                        "type": "app",
                        "status": status,
                        "size_mb": size_mb,
                        "path": meta_file,
                    }
                )

        return indexes

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

            # 1) CLI-format index under .leann/indexes/<name>
            index_dir = project_path / ".leann" / "indexes" / index_name
            if index_dir.exists():
                is_current = project_path == current_path
                matches.append(
                    {
                        "project_path": project_path,
                        "index_dir": index_dir,
                        "is_current": is_current,
                        "kind": "cli",
                    }
                )

            # 2) App-format indexes
            # We support two ways of addressing apps:
            #   a) by the file base (e.g., `pdf_documents`)
            #   b) by the parent directory name (e.g., `new_txt`)
            seen_app_meta = set()

            # 2a) by file base
            for meta_file in project_path.rglob(f"{index_name}.leann.meta.json"):
                if meta_file.is_file():
                    # Skip CLI-built indexes' meta under .leann/indexes
                    try:
                        cli_indexes_dir = project_path / ".leann" / "indexes"
                        if cli_indexes_dir.exists() and cli_indexes_dir in meta_file.parents:
                            continue
                    except Exception:
                        pass
                    is_current = project_path == current_path
                    key = (str(project_path), str(meta_file))
                    if key in seen_app_meta:
                        continue
                    seen_app_meta.add(key)
                    matches.append(
                        {
                            "project_path": project_path,
                            "files_dir": meta_file.parent,
                            "meta_file": meta_file,
                            "is_current": is_current,
                            "kind": "app",
                            "display_name": meta_file.parent.name,
                            "file_base": meta_file.name.replace(".leann.meta.json", ""),
                        }
                    )

            # 2b) by parent directory name
            for meta_file in project_path.rglob("*.leann.meta.json"):
                if meta_file.is_file() and meta_file.parent.name == index_name:
                    # Skip CLI-built indexes' meta under .leann/indexes
                    try:
                        cli_indexes_dir = project_path / ".leann" / "indexes"
                        if cli_indexes_dir.exists() and cli_indexes_dir in meta_file.parents:
                            continue
                    except Exception:
                        pass
                    is_current = project_path == current_path
                    key = (str(project_path), str(meta_file))
                    if key in seen_app_meta:
                        continue
                    seen_app_meta.add(key)
                    matches.append(
                        {
                            "project_path": project_path,
                            "files_dir": meta_file.parent,
                            "meta_file": meta_file,
                            "is_current": is_current,
                            "kind": "app",
                            "display_name": meta_file.parent.name,
                            "file_base": meta_file.name.replace(".leann.meta.json", ""),
                        }
                    )

        # Sort: current project first, then by project name
        matches.sort(key=lambda x: (not x["is_current"], x["project_path"].name))
        return matches

    def _remove_single_match(self, match, index_name: str, force: bool):
        """Handle removal when only one match is found"""
        project_path = match["project_path"]
        is_current = match["is_current"]
        kind = match.get("kind", "cli")

        if is_current:
            location_info = "current project"
            emoji = "🏠"
        else:
            location_info = f"other project '{project_path.name}'"
            emoji = "📂"

        print(f"✅ Found 1 index named '{index_name}':")
        print(f"   {emoji} Location: {location_info}")
        if kind == "cli":
            print(f"   📍 Path: {project_path / '.leann' / 'indexes' / index_name}")
        else:
            print(f"   📍 Meta: {match['meta_file']}")

        if not force:
            if not is_current:
                print("\n⚠️  CROSS-PROJECT REMOVAL!")
                print("   This will delete the index from another project.")

            response = input(f"   ❓ Confirm removal from {location_info}? (y/N): ").strip().lower()
            if response not in ["y", "yes"]:
                print("   ❌ Removal cancelled.")
                return False

        if kind == "cli":
            return self._delete_index_directory(
                match["index_dir"],
                index_name,
                project_path if not is_current else None,
                is_app=False,
            )
        else:
            return self._delete_index_directory(
                match["files_dir"],
                match.get("display_name", index_name),
                project_path if not is_current else None,
                is_app=True,
                meta_file=match.get("meta_file"),
                app_file_base=match.get("file_base"),
            )

    def _remove_from_multiple_matches(self, matches, index_name: str, force: bool):
        """Handle removal when multiple matches are found"""

        print(f"⚠️  Found {len(matches)} indexes named '{index_name}':")
        print("   " + "─" * 50)

        for i, match in enumerate(matches, 1):
            project_path = match["project_path"]
            is_current = match["is_current"]
            kind = match.get("kind", "cli")

            if is_current:
                print(f"   {i}. 🏠 Current project ({'CLI' if kind == 'cli' else 'APP'})")
            else:
                print(f"   {i}. 📂 {project_path.name} ({'CLI' if kind == 'cli' else 'APP'})")

            # Show path details
            if kind == "cli":
                print(f"      📍 {project_path / '.leann' / 'indexes' / index_name}")
            else:
                print(f"      📍 {match['meta_file']}")

            # Show size info
            try:
                if kind == "cli":
                    size_mb = sum(
                        f.stat().st_size for f in match["index_dir"].iterdir() if f.is_file()
                    ) / (1024 * 1024)
                else:
                    file_base = match.get("file_base")
                    size_mb = 0.0
                    if file_base:
                        size_mb = sum(
                            f.stat().st_size
                            for f in match["files_dir"].glob(f"{file_base}.leann*")
                            if f.is_file()
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
                is_current = selected_match["is_current"]
                kind = selected_match.get("kind", "cli")

                location = "current project" if is_current else f"'{project_path.name}' project"
                print(f"   🎯 Selected: Remove from {location}")

                # Final confirmation for safety
                confirm = input(
                    f"   ❓ FINAL CONFIRMATION - Type '{index_name}' to proceed: "
                ).strip()
                if confirm != index_name:
                    print("   ❌ Confirmation failed. Removal cancelled.")
                    return False

                if kind == "cli":
                    return self._delete_index_directory(
                        selected_match["index_dir"],
                        index_name,
                        project_path if not is_current else None,
                        is_app=False,
                    )
                else:
                    return self._delete_index_directory(
                        selected_match["files_dir"],
                        selected_match.get("display_name", index_name),
                        project_path if not is_current else None,
                        is_app=True,
                        meta_file=selected_match.get("meta_file"),
                        app_file_base=selected_match.get("file_base"),
                    )
            else:
                print("   ❌ Invalid choice. Removal cancelled.")
                return False

        except (ValueError, KeyboardInterrupt):
            print("\n   ❌ Invalid input. Removal cancelled.")
            return False

    def _delete_index_directory(
        self,
        index_dir: Path,
        index_display_name: str,
        project_path: Optional[Path] = None,
        is_app: bool = False,
        meta_file: Optional[Path] = None,
        app_file_base: Optional[str] = None,
    ):
        """Delete a CLI index directory or APP index files safely."""
        try:
            if is_app:
                removed = 0
                errors = 0
                # Delete only files that belong to this app index (based on file base)
                pattern_base = app_file_base or ""
                for f in index_dir.glob(f"{pattern_base}.leann*"):
                    try:
                        f.unlink()
                        removed += 1
                    except Exception:
                        errors += 1
                # Best-effort: also remove the meta file if specified and still exists
                if meta_file and meta_file.exists():
                    try:
                        meta_file.unlink()
                        removed += 1
                    except Exception:
                        errors += 1

                if removed > 0 and errors == 0:
                    if project_path:
                        print(
                            f"✅ App index '{index_display_name}' removed from {project_path.name}"
                        )
                    else:
                        print(f"✅ App index '{index_display_name}' removed successfully")
                    return True
                elif removed > 0 and errors > 0:
                    print(
                        f"⚠️  App index '{index_display_name}' partially removed (some files couldn't be deleted)"
                    )
                    return True
                else:
                    print(
                        f"❌ No files found to remove for app index '{index_display_name}' in {index_dir}"
                    )
                    return False
            else:
                import shutil

                shutil.rmtree(index_dir)

                if project_path:
                    print(f"✅ Index '{index_display_name}' removed from {project_path.name}")
                else:
                    print(f"✅ Index '{index_display_name}' removed successfully")
                return True
        except Exception as e:
            print(f"❌ Error removing index '{index_display_name}': {e}")
            return False

    def load_documents(
        self,
        docs_paths: Union[str, list],
        custom_file_types: Union[str, None] = None,
        include_hidden: bool = False,
        args: Optional[dict[str, Any]] = None,
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
                        ).load_data(num_workers=os.cpu_count() or 1)
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

            # Use fd for fast file enumeration with native gitignore support
            # fd is a blazing-fast alternative to find, written in Rust
            fd_files = []
            use_fd = False

            try:
                import subprocess

                # Build fd command with extension filters
                # fd respects .gitignore by default and is extremely fast
                fd_cmd = ["fd", "--type", "f", "--absolute-path"]

                # Add extension filters if specified
                if code_extensions:
                    for ext in code_extensions:
                        # fd uses -e for extension (without the dot)
                        ext_clean = ext.lstrip(".")
                        fd_cmd.extend(["-e", ext_clean])

                # Execute fd
                result = subprocess.run(
                    fd_cmd, cwd=docs_dir, capture_output=True, text=True, check=True
                )

                fd_files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                use_fd = True
                print(f"⚡ fd: Found {len(fd_files)} files in {docs_dir}")

            except (subprocess.SubprocessError, FileNotFoundError) as e:
                # fd not available, fall back to standard traversal
                if os.environ.get("LEANN_LOG_LEVEL", "WARNING").upper() == "DEBUG":
                    print(f"⚠️  fd not available ({e}), using standard traversal")
                use_fd = False

            # Build gitignore parser ONLY as a fallback for standard traversal
            gitignore_matches = None
            if not use_fd:
                gitignore_matches = self._build_gitignore_parser(docs_dir)

            # Try to use better PDF parsers first, but only if PDFs are requested
            documents = []
            # Use resolved absolute paths to avoid mismatches (symlinks, relative vs absolute)
            docs_path = Path(docs_dir).resolve()

            # Check if we should process PDFs
            should_process_pdfs = custom_file_types is None or ".pdf" in custom_file_types

            if should_process_pdfs:
                for file_path in docs_path.rglob("*.pdf"):
                    # Check if file matches any exclude pattern
                    try:
                        # Ensure both paths are resolved before computing relativity
                        file_path_resolved = file_path.resolve()

                        # fd filter: strictly check if file is in fd_files if we used fd
                        if use_fd:
                            if str(file_path_resolved) not in fd_files:
                                continue
                        else:
                            # Fallback to manual gitignore parsing
                            # Determine directory scope using the non-resolved path to avoid
                            # misclassifying symlinked entries as outside the docs directory
                            relative_path = file_path.relative_to(docs_path)
                            if not include_hidden and _path_has_hidden_segment(relative_path):
                                continue
                            # Use absolute path for gitignore matching
                            if self._should_exclude_file(file_path_resolved, gitignore_matches):
                                continue

                        # ... rest of PDF processing ...
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

                    except ValueError:
                        # Skip files that can't be made relative to docs_path
                        print(f"⚠️  Skipping file outside directory scope: {file_path}")
                        continue

            # Load other file types with default reader
            # Exclude PDFs from code_extensions if they were already processed separately
            other_file_extensions = code_extensions
            if should_process_pdfs and ".pdf" in code_extensions:
                other_file_extensions = [ext for ext in code_extensions if ext != ".pdf"]

            try:
                # Only load other file types if there are extensions to process
                if other_file_extensions:
                    if use_fd and fd_files:
                        # High-performance path: fd already filtered by extension and gitignore
                        # Filter out PDFs if they were processed separately
                        if should_process_pdfs:
                            fd_files = [f for f in fd_files if not f.endswith(".pdf")]

                        if fd_files:
                            # Concatenate with previous message if possible, or just keep it simple
                            other_docs = SimpleDirectoryReader(
                                docs_dir,
                                input_files=fd_files,
                                recursive=False,  # Explicit file list provided
                                encoding="utf-8",
                                file_extractor={},
                                exclude_hidden=not include_hidden,
                                filename_as_id=True,
                            ).load_data(show_progress=True, num_workers=os.cpu_count() or 1)
                        else:
                            other_docs = []
                    else:
                        # Fallback: Standard recursive load with post-filtering
                        other_docs = SimpleDirectoryReader(
                            docs_dir,
                            recursive=True,
                            encoding="utf-8",
                            required_exts=other_file_extensions,
                            file_extractor={},  # Use default extractors
                            exclude_hidden=not include_hidden,
                            filename_as_id=True,
                        ).load_data(show_progress=True, num_workers=os.cpu_count() or 1)

                        # Filter documents (slow path - only when fd unavailable)
                        filtered_docs = []
                        for doc in tqdm(other_docs, desc="Filtering files", unit="file"):
                            file_path = doc.metadata.get("file_path", "")
                            file_path_obj = Path(file_path).resolve()
                            if not self._should_exclude_file(file_path_obj, gitignore_matches):
                                doc.metadata["source"] = file_path
                                filtered_docs.append(doc)
                        other_docs = filtered_docs
                else:
                    other_docs = []

                documents.extend(other_docs)
            except ValueError as e:
                if "No files found" in str(e):
                    print(f"No additional files found for other supported types in {docs_dir}.")
                else:
                    raise e

            all_documents.extend(documents)
            print(f"Loaded {len(documents)} documents from {docs_dir}")

        documents = all_documents

        # Path normalization: make paths relative to the documentation directory if possible
        # This ensures consistent metadata (e.g. src/server.py) instead of absolute paths.
        if directories:
            # Sort directories by length (descending) to match longest prefix first
            sorted_dirs = sorted([Path(d).resolve() for d in directories], key=lambda p: len(str(p)), reverse=True)
            for doc in documents:
                fpath = doc.metadata.get("file_path") or doc.metadata.get("source")
                if fpath:
                    fpath_obj = Path(fpath).resolve()
                    for d in sorted_dirs:
                        try:
                            rel_path = fpath_obj.relative_to(d)
                            doc.metadata["file_path"] = rel_path.as_posix()
                            break
                        except ValueError:
                            continue

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
            ".R",
            ".jl",
        }

        print("start chunking documents")

        # Check if AST chunking is requested
        use_ast = getattr(args, "use_ast_chunking", False)

        if use_ast:
            print("🧠 Using AST-aware chunking for code files")
        else:
            print("⚡ Using parallel chunking for documents")

        try:
            # Import enhanced chunking utilities from packaged module
            from .chunking_utils import create_text_chunks

            # Use enhanced chunking with parallel support (works for both AST and traditional)
            chunk_texts = create_text_chunks(
                documents,
                chunk_size=self.node_parser.chunk_size,
                chunk_overlap=self.node_parser.chunk_overlap,
                use_ast_chunking=use_ast,
                ast_chunk_size=getattr(args, "ast_chunk_size", 768),
                ast_chunk_overlap=getattr(args, "ast_chunk_overlap", 96),
                code_file_extensions=None,  # Use defaults
                ast_fallback_traditional=getattr(args, "ast_fallback_traditional", True),
            )

            # create_text_chunks now returns list[dict] with metadata preserved
            all_texts.extend(chunk_texts)

        except ImportError as e:
            print(
                f"⚠️  Chunking utilities not available in package ({e}), falling back to legacy serial chunking"
            )
            # Use traditional chunking logic (serial fallback)
            for doc in tqdm(documents, desc="Chunking documents", unit="doc"):
                # Check if this is a code file based on source path
                source_path = doc.metadata.get("source", "")
                file_path = doc.metadata.get("file_path", "")
                is_code_file = any(source_path.endswith(ext) for ext in code_file_exts)

                # Extract metadata to preserve with chunks
                chunk_metadata = {
                    "file_path": file_path or source_path,
                    "file_name": doc.metadata.get("file_name", ""),
                }

                # Add optional metadata if available
                if "creation_date" in doc.metadata:
                    chunk_metadata["creation_date"] = doc.metadata["creation_date"]
                if "last_modified_date" in doc.metadata:
                    chunk_metadata["last_modified_date"] = doc.metadata["last_modified_date"]

                # Use appropriate parser based on file type
                parser = self.code_parser if is_code_file else self.node_parser
                nodes = parser.get_nodes_from_documents([doc])

                for node in nodes:
                    all_texts.append({"text": node.get_content(), "metadata": chunk_metadata})

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
            docs_paths, args.file_types, include_hidden=args.include_hidden, args=args
        )
        if not all_texts:
            print("No documents found")
            return

        index_dir.mkdir(parents=True, exist_ok=True)

        print(f"Building index '{index_name}' with {args.backend_name} backend...")

        embedding_options: dict[str, Any] = {}
        if args.embedding_mode == "ollama":
            embedding_options["host"] = resolve_ollama_host(args.embedding_host)
        elif args.embedding_mode == "openai":
            embedding_options["base_url"] = resolve_openai_base_url(args.embedding_api_base)
            resolved_embedding_key = resolve_openai_api_key(args.embedding_api_key)
            if resolved_embedding_key:
                embedding_options["api_key"] = resolved_embedding_key
        if args.query_prompt_template:
            # New format: separate templates
            if args.embedding_prompt_template:
                embedding_options["build_prompt_template"] = args.embedding_prompt_template
            embedding_options["query_prompt_template"] = args.query_prompt_template
        elif args.embedding_prompt_template:
            # Old format: single template (backward compat)
            embedding_options["prompt_template"] = args.embedding_prompt_template

        builder = LeannBuilder(
            backend_name=args.backend_name,
            embedding_model=args.embedding_model,
            embedding_mode=args.embedding_mode,
            embedding_options=embedding_options or None,
            graph_degree=args.graph_degree,
            complexity=args.complexity,
            is_compact=args.compact,
            is_recompute=args.recompute,
            num_threads=args.num_threads,
        )

        for chunk in all_texts:
            builder.add_text(chunk["text"], metadata=chunk["metadata"])

        builder.build_index(index_path)
        print(f"Index built at {index_path}")

        # Register this project directory in global registry
        self.register_project_dir()

    async def search_documents(self, args):
        index_name = args.index_name
        query = args.query

        # First try to find the index in current project
        index_path = self.get_index_path(index_name)
        if self.index_exists(index_name):
            # Found in current project, use it
            pass
        else:
            # Search across all registered projects (like list_indexes does)
            all_matches = self._find_all_matching_indexes(index_name)
            if not all_matches:
                print(
                    f"Index '{index_name}' not found. Use 'leann build {index_name} --docs <dir> [<dir2> ...]' to create it."
                )
                return
            elif len(all_matches) == 1:
                # Found exactly one match, use it
                match = all_matches[0]
                if match["kind"] == "cli":
                    index_path = str(match["index_dir"] / "documents.leann")
                else:
                    # App format: use the meta file to construct the path
                    meta_file = match["meta_file"]
                    file_base = match["file_base"]
                    index_path = str(meta_file.parent / f"{file_base}.leann")

                project_info = (
                    "current project"
                    if match["is_current"]
                    else f"project '{match['project_path'].name}'"
                )
                print(f"Using index '{index_name}' from {project_info}")
            else:
                # Multiple matches found
                if args.non_interactive:
                    # Non-interactive mode: automatically select the best match
                    # Priority: current project first, then first available
                    current_matches = [m for m in all_matches if m["is_current"]]
                    if current_matches:
                        match = current_matches[0]
                        location_desc = "current project"
                    else:
                        match = all_matches[0]
                        location_desc = f"project '{match['project_path'].name}'"

                    if match["kind"] == "cli":
                        index_path = str(match["index_dir"] / "documents.leann")
                    else:
                        meta_file = match["meta_file"]
                        file_base = match["file_base"]
                        index_path = str(meta_file.parent / f"{file_base}.leann")

                    print(
                        f"Found {len(all_matches)} indexes named '{index_name}', using index from {location_desc}"
                    )
                else:
                    # Interactive mode: ask user to choose
                    print(f"Found {len(all_matches)} indexes named '{index_name}':")
                    for i, match in enumerate(all_matches, 1):
                        project_path = match["project_path"]
                        is_current = match["is_current"]
                        kind = match.get("kind", "cli")

                        if is_current:
                            print(
                                f"   {i}. 🏠 Current project ({'CLI' if kind == 'cli' else 'APP'})"
                            )
                        else:
                            print(
                                f"   {i}. 📂 {project_path.name} ({'CLI' if kind == 'cli' else 'APP'})"
                            )

                    try:
                        choice = input(f"Which index to search? (1-{len(all_matches)}): ").strip()
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(all_matches):
                            match = all_matches[choice_idx]
                            if match["kind"] == "cli":
                                index_path = str(match["index_dir"] / "documents.leann")
                            else:
                                meta_file = match["meta_file"]
                                file_base = match["file_base"]
                                index_path = str(meta_file.parent / f"{file_base}.leann")

                            project_info = (
                                "current project"
                                if match["is_current"]
                                else f"project '{match['project_path'].name}'"
                            )
                            print(f"Using index '{index_name}' from {project_info}")
                        else:
                            print("Invalid choice. Aborting search.")
                            return
                    except (ValueError, KeyboardInterrupt):
                        print("Invalid input. Aborting search.")
                        return

        # Build provider_options for runtime override
        provider_options = {}
        if args.embedding_prompt_template:
            provider_options["prompt_template"] = args.embedding_prompt_template

        searcher = LeannSearcher(index_path=index_path)
        results = searcher.search(
            query,
            top_k=args.top_k,
            complexity=args.complexity,
            beam_width=args.beam_width,
            prune_ratio=args.prune_ratio,
            recompute_embeddings=args.recompute_embeddings,
            pruning_strategy=args.pruning_strategy,
            provider_options=provider_options if provider_options else None,
        )

        print(f"Search results for '{query}' (top {len(results)}):")
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result.score:.3f}")

            # Display metadata if flag is set
            if args.show_metadata and result.metadata:
                file_path = result.metadata.get("file_path", "")
                if file_path:
                    print(f"   📄 File: {file_path}")

                file_name = result.metadata.get("file_name", "")
                if file_name and file_name != file_path:
                    print(f"   📝 Name: {file_name}")

                # Show timestamps if available
                if "creation_date" in result.metadata:
                    print(f"   🕐 Created: {result.metadata['creation_date']}")
                if "last_modified_date" in result.metadata:
                    print(f"   🕑 Modified: {result.metadata['last_modified_date']}")

            print(f"   {result.text[:200]}...")
            print(f"   Source: {result.metadata.get('source', '')}")
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
            llm_config["host"] = resolve_ollama_host(args.host)
        elif args.llm == "openai":
            llm_config["base_url"] = resolve_openai_base_url(args.api_base)
            resolved_api_key = resolve_openai_api_key(args.api_key)
            if resolved_api_key:
                llm_config["api_key"] = resolved_api_key
        elif args.llm == "anthropic":
            # For Anthropic, pass base_url and API key if provided
            if args.api_base:
                llm_config["base_url"] = resolve_anthropic_base_url(args.api_base)
            if args.api_key:
                llm_config["api_key"] = args.api_key

        chat = LeannChat(index_path=index_path, llm_config=llm_config)

        llm_kwargs: dict[str, Any] = {}
        if args.thinking_budget:
            llm_kwargs["thinking_budget"] = args.thinking_budget

        def _ask_once(prompt: str) -> None:
            query_start_time = time.time()
            response = chat.ask(
                prompt,
                top_k=args.top_k,
                complexity=args.complexity,
                beam_width=args.beam_width,
                prune_ratio=args.prune_ratio,
                recompute_embeddings=args.recompute_embeddings,
                pruning_strategy=args.pruning_strategy,
                llm_kwargs=llm_kwargs,
            )
            query_completion_time = time.time() - query_start_time
            print(f"LEANN: {response}")
            print(f"The query took {query_completion_time:.3f} seconds to finish")

        initial_query = (args.query or "").strip()

        if args.interactive:
            # Create interactive session
            session = create_cli_session(index_name)

            if initial_query:
                _ask_once(initial_query)

            session.run_interactive_loop(_ask_once)
        else:
            query = initial_query or input("Enter your question: ").strip()
            if not query:
                print("No question provided. Exiting.")
                return

            _ask_once(query)

    async def run(self, args=None):
        parser = self.create_parser()

        if args is None:
            args = parser.parse_args()

        if not args.command:
            parser.print_help()
            return

        # Determine whether to suppress C++ output
        # Default is to suppress (quiet mode), unless --verbose is specified
        suppress = not getattr(args, "verbose", False)

        if args.command == "list":
            self.list_indexes()
        elif args.command == "remove":
            self.remove_index(args.index_name, args.force)
        elif args.command == "build":
            with suppress_cpp_output(suppress):
                await self.build_index(args)
        elif args.command == "search":
            with suppress_cpp_output(suppress):
                await self.search_documents(args)
        elif args.command == "ask":
            with suppress_cpp_output(suppress):
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
