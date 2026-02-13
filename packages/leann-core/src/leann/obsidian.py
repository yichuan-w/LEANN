"""Obsidian vault reader for LEANN (#96).

Parses Obsidian markdown files and extracts wikilinks (``[[page]]``,
``[[page|alias]]``) and YAML frontmatter.  Builds a reverse backlink map
so each chunk carries its connectivity context as metadata — enabling
graph-aware search over personal knowledge bases.

Usage with the CLI::

    leann build my-vault --docs /path/to/obsidian-vault --obsidian

Or programmatically::

    from leann.obsidian import ObsidianVaultReader
    reader = ObsidianVaultReader("/path/to/vault")
    for chunk in reader.iter_chunks():
        builder.add_text(chunk["text"], metadata=chunk["metadata"])
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

# Regex patterns for Obsidian-specific syntax
# Matches [[page]], [[page|alias]], [[page#heading]], [[page#heading|alias]]
_WIKILINK_RE = re.compile(
    r"\[\["
    r"(?P<target>[^\]|#]+)"        # target page name
    r"(?:#(?P<heading>[^\]|]+))?"  # optional #heading
    r"(?:\|(?P<alias>[^\]]+))?"    # optional |alias
    r"\]\]"
)

# YAML frontmatter delimiter
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Obsidian tag syntax: #tag or #nested/tag (not inside code blocks)
_TAG_RE = re.compile(r"(?<!\S)#([a-zA-Z][a-zA-Z0-9_/-]*)")


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from a markdown file.

    Returns ``(frontmatter_dict, body_text)`` where body_text is the
    content after the frontmatter block.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    try:
        import yaml

        fm = yaml.safe_load(match.group(1))
        if not isinstance(fm, dict):
            fm = {}
    except ImportError:
        # Fallback: basic key: value parsing
        fm = {}
        for line in match.group(1).splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                if value:
                    fm[key] = value
    except Exception:
        fm = {}

    body = text[match.end():]
    return fm, body


def extract_wikilinks(text: str) -> list[dict[str, str]]:
    """Extract all ``[[wikilinks]]`` from markdown text.

    Returns a list of dicts with keys: ``target``, ``heading`` (optional),
    ``alias`` (optional), ``raw`` (the full matched text).
    """
    links = []
    for m in _WIKILINK_RE.finditer(text):
        link: dict[str, str] = {"target": m.group("target").strip(), "raw": m.group(0)}
        if m.group("heading"):
            link["heading"] = m.group("heading").strip()
        if m.group("alias"):
            link["alias"] = m.group("alias").strip()
        links.append(link)
    return links


def extract_tags(text: str) -> list[str]:
    """Extract Obsidian ``#tags`` from text (excluding code blocks)."""
    # Remove code blocks first
    cleaned = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"`[^`]+`", "", cleaned)
    return list(dict.fromkeys(m.group(1) for m in _TAG_RE.finditer(cleaned)))


def build_backlink_map(
    vault_dir: Path, notes: dict[str, list[dict[str, str]]]
) -> dict[str, list[str]]:
    """Build a reverse backlink map from wikilink data.

    Args:
        vault_dir: Root directory of the vault (for resolving relative names).
        notes: Mapping of ``{note_name: [wikilinks...]}`` where each note_name
               is the stem (no extension) of the markdown file.

    Returns:
        Mapping of ``{target_note: [list of notes that link to it]}``.
    """
    backlinks: dict[str, list[str]] = {}
    for source_note, links in notes.items():
        for link in links:
            target = link["target"]
            # Normalize target (Obsidian is case-insensitive for links)
            target_lower = target.lower()
            backlinks.setdefault(target_lower, [])
            if source_note not in backlinks[target_lower]:
                backlinks[target_lower].append(source_note)
    return backlinks


class ObsidianVaultReader:
    """Read and chunk an Obsidian vault with wikilink/backlink awareness.

    This reader scans a directory for ``.md`` files, parses frontmatter and
    wikilinks, builds a vault-wide backlink map, and produces chunks that
    carry connectivity metadata.

    Args:
        vault_path: Path to the Obsidian vault root.
        chunk_size: Target chunk size in characters (default 1024).
        chunk_overlap: Overlap between consecutive chunks (default 128).
        include_hidden: Whether to include hidden files (default False).
    """

    def __init__(
        self,
        vault_path: str | Path,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        include_hidden: bool = False,
    ):
        self.vault_path = Path(vault_path)
        if not self.vault_path.is_dir():
            raise ValueError(f"Vault path is not a directory: {self.vault_path}")
        self.chunk_size = max(1, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size - 1))
        self.include_hidden = include_hidden

        # Pre-scan vault to build the backlink map
        self._note_wikilinks: dict[str, list[dict[str, str]]] = {}
        self._note_metadata: dict[str, dict[str, Any]] = {}
        self._backlinks: dict[str, list[str]] = {}
        self._scan_vault()

    def _iter_md_files(self) -> Iterator[Path]:
        """Iterate over all markdown files in the vault."""
        for md_file in sorted(self.vault_path.rglob("*.md")):
            # Skip hidden files/directories
            if not self.include_hidden:
                parts = md_file.relative_to(self.vault_path).parts
                if any(p.startswith(".") for p in parts):
                    continue
            yield md_file

    def _scan_vault(self) -> None:
        """Pre-scan the entire vault to build the backlink map."""
        # Track stems we've already seen to warn about duplicates
        seen_stems: dict[str, str] = {}  # stem -> first relative path

        for md_file in self._iter_md_files():
            note_name = md_file.stem
            rel_path = str(md_file.relative_to(self.vault_path))

            # Use relative path as the key to avoid collisions between
            # e.g. notes/A.md and subfolder/A.md
            note_key = rel_path

            if note_name in seen_stems and seen_stems[note_name] != rel_path:
                logger.warning(
                    "Duplicate note name '%s': %s and %s — using full path as key",
                    note_name,
                    seen_stems[note_name],
                    rel_path,
                )
            seen_stems.setdefault(note_name, rel_path)

            try:
                text = md_file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                logger.warning("Could not read %s, skipping", md_file)
                continue

            frontmatter, body = parse_frontmatter(text)
            wikilinks = extract_wikilinks(body)
            tags = extract_tags(body)

            # Merge tags from frontmatter
            fm_tags = frontmatter.get("tags", [])
            if isinstance(fm_tags, str):
                fm_tags = [t.strip() for t in fm_tags.split(",")]
            elif not isinstance(fm_tags, list):
                fm_tags = []
            all_tags = list(dict.fromkeys(tags + fm_tags))

            self._note_wikilinks[note_key] = wikilinks
            self._note_metadata[note_key] = {
                "frontmatter": frontmatter,
                "tags": all_tags,
                "wikilinks": [link["target"] for link in wikilinks],
                "file_path": rel_path,
                "note_name": note_name,
            }

        self._backlinks = build_backlink_map(self.vault_path, self._note_wikilinks)
        logger.info(
            "Scanned %d notes, found %d backlink targets",
            len(self._note_wikilinks),
            len(self._backlinks),
        )

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size

            # Try to break at paragraph, then sentence, then word boundary
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind("\n\n", start, end)
                if para_break > start + self.chunk_size // 2:
                    end = para_break + 2
                else:
                    # Look for sentence break
                    for sep in (". ", ".\n", "! ", "? "):
                        sent_break = text.rfind(sep, start, end)
                        if sent_break > start + self.chunk_size // 2:
                            end = sent_break + len(sep)
                            break
                    else:
                        # Fall back to word boundary
                        word_break = text.rfind(" ", start, end)
                        if word_break > start:
                            end = word_break + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap
            if start >= len(text):
                break

        return chunks

    def iter_chunks(self) -> Iterator[dict[str, Any]]:
        """Iterate over all chunks from the vault with metadata.

        Yields dicts with keys:
            - ``text``: The chunk text.
            - ``metadata``: Dict with ``note_name``, ``file_path``, ``tags``,
              ``wikilinks``, ``backlinks``, ``frontmatter``, ``chunk_index``.
        """
        for md_file in self._iter_md_files():
            note_name = md_file.stem
            rel_path = str(md_file.relative_to(self.vault_path))
            note_key = rel_path
            note_meta = self._note_metadata.get(note_key, {})

            try:
                text = md_file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            _, body = parse_frontmatter(text)

            # Get backlinks for this note
            backlinks = self._backlinks.get(note_name.lower(), [])

            chunks = self._chunk_text(body)
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{note_name}:{i}"
                yield {
                    "text": chunk_text,
                    "metadata": {
                        "id": chunk_id,
                        "note_name": note_name,
                        "file_path": note_meta.get("file_path", ""),
                        "tags": note_meta.get("tags", []),
                        "wikilinks": note_meta.get("wikilinks", []),
                        "backlinks": backlinks,
                        "frontmatter": note_meta.get("frontmatter", {}),
                        "chunk_index": i,
                        "source": "obsidian",
                    },
                }

    @property
    def note_count(self) -> int:
        """Number of notes in the vault."""
        return len(self._note_wikilinks)

    @property
    def backlink_map(self) -> dict[str, list[str]]:
        """The computed backlink map."""
        return dict(self._backlinks)
