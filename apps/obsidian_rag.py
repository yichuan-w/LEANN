"""
Obsidian vault RAG example.

Indexes Markdown notes from an Obsidian vault while preserving note metadata
such as wiki links, embeds, tags, aliases, and frontmatter.
"""

import os
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Add parent directory to path for imports when executed as a script.
sys.path.insert(0, str(Path(__file__).parent))

from base_rag_example import BaseRAGExample
from chunking import create_text_chunks
from llama_index.core import Document

OBSIDIAN_EXCLUDED_DIRS = {".git", ".obsidian", ".trash"}
WIKI_EMBED_RE = re.compile(r"!\[\[([^\]]+)\]\]")
WIKI_LINK_RE = re.compile(r"(?<!!)\[\[([^\]]+)\]\]")
TAG_RE = re.compile(r"(?<![\w/])#([A-Za-z0-9_/-]+)")


class ObsidianRAG(BaseRAGExample):
    """RAG example for Obsidian vaults."""

    def __init__(self):
        super().__init__(
            name="Obsidian",
            description="Process and query an Obsidian Markdown vault with LEANN",
            default_index_name="obsidian_vault",
        )

    def _add_specific_arguments(self, parser):
        vault_group = parser.add_argument_group("Obsidian Vault Parameters")
        vault_group.add_argument(
            "--vault-dir",
            type=str,
            default=".",
            help="Obsidian vault directory to index (default: current directory)",
        )
        vault_group.add_argument(
            "--include-hidden",
            action="store_true",
            help="Include hidden Markdown files outside Obsidian internal folders",
        )
        vault_group.add_argument(
            "--chunk-size",
            type=int,
            default=384,
            help="Text chunk size for notes (default: 384)",
        )
        vault_group.add_argument(
            "--chunk-overlap",
            type=int,
            default=96,
            help="Text chunk overlap for notes (default: 96)",
        )

    async def load_data(self, args) -> list[dict[str, Any]]:
        vault_dir = Path(args.vault_dir)
        if not vault_dir.exists():
            raise ValueError(f"Obsidian vault directory not found: {args.vault_dir}")
        if not vault_dir.is_dir():
            raise ValueError(f"Obsidian vault path is not a directory: {args.vault_dir}")

        documents = load_obsidian_documents(
            vault_dir,
            include_hidden=bool(args.include_hidden),
        )
        if not documents:
            print(f"No Markdown notes found in {vault_dir}")
            return []

        chunks = create_text_chunks(
            documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            use_ast_chunking=False,
        )
        if args.max_items > 0 and len(chunks) > args.max_items:
            chunks = chunks[: args.max_items]
        print(f"Loaded {len(documents)} notes and generated {len(chunks)} chunks")
        return chunks


def load_obsidian_documents(
    vault_dir: str | Path,
    *,
    include_hidden: bool = False,
    excluded_dirs: Iterable[str] | None = None,
) -> list[Document]:
    vault_path = Path(vault_dir).expanduser().resolve()
    documents: list[Document] = []
    for note_path in iter_obsidian_notes(
        vault_path,
        include_hidden=include_hidden,
        excluded_dirs=excluded_dirs,
    ):
        text, metadata = parse_obsidian_note(note_path, vault_path)
        if text.strip():
            documents.append(Document(text=text, metadata=metadata))
    return documents


def iter_obsidian_notes(
    vault_dir: str | Path,
    *,
    include_hidden: bool = False,
    excluded_dirs: Iterable[str] | None = None,
) -> list[Path]:
    vault_path = Path(vault_dir).expanduser().resolve()
    excluded = set(excluded_dirs or OBSIDIAN_EXCLUDED_DIRS)
    notes: list[Path] = []
    for root, dirs, files in os.walk(vault_path):
        dirs[:] = sorted(
            dirname
            for dirname in dirs
            if dirname not in excluded and (include_hidden or not dirname.startswith("."))
        )
        root_path = Path(root)
        for filename in sorted(files):
            if not filename.endswith(".md"):
                continue
            if not include_hidden and filename.startswith("."):
                continue
            notes.append(root_path / filename)
    return sorted(notes)


def parse_obsidian_note(note_path: str | Path, vault_dir: str | Path) -> tuple[str, dict[str, Any]]:
    path = Path(note_path)
    vault_path = Path(vault_dir).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    frontmatter, body = split_frontmatter(text)
    relative_path = path.resolve().relative_to(vault_path).as_posix()
    frontmatter_tags = _frontmatter_values(frontmatter.get("tags"))
    inline_tags = extract_obsidian_tags(body)
    aliases = _frontmatter_values(frontmatter.get("aliases"))
    links = extract_wiki_targets(body, embeds=False)
    embeds = extract_wiki_targets(body, embeds=True)
    metadata = {
        "source": str(path),
        "file_path": str(path),
        "file_name": path.name,
        "obsidian_note": True,
        "obsidian_vault_path": str(vault_path),
        "obsidian_relative_path": relative_path,
        "obsidian_title": str(frontmatter.get("title") or path.stem),
        "obsidian_aliases": aliases,
        "obsidian_tags": sorted(set(frontmatter_tags + inline_tags)),
        "obsidian_links": links,
        "obsidian_embeds": embeds,
        "obsidian_frontmatter": frontmatter,
    }
    return body, metadata


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            frontmatter = parse_frontmatter_lines(lines[1:index])
            body = "\n".join(lines[index + 1 :])
            if text.endswith("\n"):
                body += "\n"
            return frontmatter, body
    return {}, text


def parse_frontmatter_lines(lines: list[str]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_key: str | None = None
    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith((" ", "\t")) and current_key:
            item = line.strip()
            if item.startswith("- "):
                existing = data.setdefault(current_key, [])
                if isinstance(existing, list):
                    existing.append(_parse_frontmatter_scalar(item[2:].strip()))
            continue
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        current_key = key
        data[key] = _parse_frontmatter_value(raw_value)
    return data


def extract_wiki_targets(text: str, *, embeds: bool) -> list[str]:
    pattern = WIKI_EMBED_RE if embeds else WIKI_LINK_RE
    targets = [_normalize_wiki_target(match) for match in pattern.findall(text)]
    return sorted({target for target in targets if target})


def extract_obsidian_tags(text: str) -> list[str]:
    return sorted(set(TAG_RE.findall(text)))


def _normalize_wiki_target(raw_target: str) -> str:
    return raw_target.split("|", 1)[0].strip()


def _frontmatter_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip().lstrip("#") for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [item.strip().lstrip("#") for item in value.split(",") if item.strip()]
    return []


def _parse_frontmatter_value(raw_value: str) -> Any:
    if raw_value == "":
        return []
    if raw_value.startswith("[") and raw_value.endswith("]"):
        inner = raw_value[1:-1].strip()
        if not inner:
            return []
        return [_parse_frontmatter_scalar(item.strip()) for item in inner.split(",")]
    return _parse_frontmatter_scalar(raw_value)


def _parse_frontmatter_scalar(value: str) -> str:
    return value.strip().strip("\"'")


if __name__ == "__main__":
    import asyncio

    print("\nObsidian RAG Example")
    print("=" * 50)
    print("\nUsage examples:")
    print(
        "  python -m apps.obsidian_rag --vault-dir ~/Notes --query 'What did I learn about LEANN?'"
    )
    print("  python -m apps.obsidian_rag --vault-dir ~/Notes --include-hidden")
    print("\nOr run without --query for interactive mode\n")

    rag = ObsidianRAG()
    asyncio.run(rag.run())
