"""Tests for Obsidian vault reader (#96).

Tests wikilink parsing, backlink map construction, frontmatter extraction,
chunking, and the CLI --obsidian flag.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub out C++ backend so we can import leann.obsidian in CI
# ---------------------------------------------------------------------------
_mod = sys.modules.get("leann_backend_hnsw.convert_to_csr")
if _mod is not None and not hasattr(_mod, "prune_hnsw_embeddings_inplace"):
    _mod.prune_hnsw_embeddings_inplace = lambda *a, **kw: True
if "leann_backend_hnsw" not in sys.modules:
    stub = MagicMock()
    sys.modules["leann_backend_hnsw"] = stub
    sys.modules["leann_backend_hnsw.convert_to_csr"] = stub.convert_to_csr
    stub.convert_to_csr.prune_hnsw_embeddings_inplace = lambda *a, **kw: True

from leann.obsidian import (
    ObsidianVaultReader,
    build_backlink_map,
    extract_tags,
    extract_wikilinks,
    parse_frontmatter,
)


class TestParseWikilinks:
    def test_basic_wikilink(self):
        links = extract_wikilinks("Check out [[My Page]] for details.")
        assert len(links) == 1
        assert links[0]["target"] == "My Page"
        assert links[0]["raw"] == "[[My Page]]"

    def test_aliased_wikilink(self):
        links = extract_wikilinks("See [[My Page|this page]] for more.")
        assert len(links) == 1
        assert links[0]["target"] == "My Page"
        assert links[0]["alias"] == "this page"

    def test_wikilink_with_heading(self):
        links = extract_wikilinks("Refer to [[Page#Section]] above.")
        assert len(links) == 1
        assert links[0]["target"] == "Page"
        assert links[0]["heading"] == "Section"

    def test_wikilink_with_heading_and_alias(self):
        links = extract_wikilinks("See [[Page#Section|alias text]].")
        assert len(links) == 1
        assert links[0]["target"] == "Page"
        assert links[0]["heading"] == "Section"
        assert links[0]["alias"] == "alias text"

    def test_multiple_wikilinks(self):
        text = "Links to [[Page A]], [[Page B|alias]], and [[Page C#heading]]."
        links = extract_wikilinks(text)
        assert len(links) == 3
        targets = [l["target"] for l in links]
        assert targets == ["Page A", "Page B", "Page C"]

    def test_no_wikilinks(self):
        links = extract_wikilinks("No links here, just regular [markdown](http://example.com).")
        assert len(links) == 0


class TestParseFrontmatter:
    def test_basic_frontmatter(self):
        text = """---
title: My Note
tags: [tag1, tag2]
---
# Content here
Some body text."""
        fm, body = parse_frontmatter(text)
        assert fm.get("title") == "My Note"
        assert "Content here" in body

    def test_no_frontmatter(self):
        text = "# Just a heading\nSome text."
        fm, body = parse_frontmatter(text)
        assert fm == {}
        assert body == text

    def test_frontmatter_with_tags_string(self):
        text = """---
tags: philosophy, logic
---
Body"""
        fm, body = parse_frontmatter(text)
        assert "tags" in fm
        assert "Body" in body

    def test_empty_frontmatter(self):
        text = """---
---
Body"""
        fm, body = parse_frontmatter(text)
        assert fm == {}
        assert "Body" in body


class TestExtractTags:
    def test_basic_tags(self):
        tags = extract_tags("Some text #philosophy and #logic here.")
        assert "philosophy" in tags
        assert "logic" in tags

    def test_nested_tags(self):
        tags = extract_tags("A #topic/subtopic tag.")
        assert "topic/subtopic" in tags

    def test_no_tags_in_code(self):
        tags = extract_tags("Normal text. `code with #notag` and more text.")
        assert "notag" not in tags

    def test_no_tags_in_code_block(self):
        tags = extract_tags("Before\n```\n#notag\n```\nAfter #realtag")
        assert "notag" not in tags
        assert "realtag" in tags

    def test_hash_in_heading_not_tag(self):
        # Headings like "# Title" should not produce tags because # is followed by space
        tags = extract_tags("# Heading\nSome text #actual-tag")
        # The regex requires #[a-zA-Z], so "# Heading" won't match
        assert "actual-tag" in tags


class TestBacklinkMap:
    def test_basic_backlinks(self):
        notes = {
            "NoteA": [{"target": "NoteB"}, {"target": "NoteC"}],
            "NoteB": [{"target": "NoteC"}],
            "NoteC": [],
        }
        backlinks = build_backlink_map(Path("/vault"), notes)
        assert "noteb" in backlinks
        assert "NoteA" in backlinks["noteb"]
        assert "notec" in backlinks
        assert set(backlinks["notec"]) == {"NoteA", "NoteB"}

    def test_no_duplicate_backlinks(self):
        notes = {
            "NoteA": [{"target": "NoteB"}, {"target": "NoteB"}],
        }
        backlinks = build_backlink_map(Path("/vault"), notes)
        assert backlinks["noteb"] == ["NoteA"]

    def test_empty_vault(self):
        backlinks = build_backlink_map(Path("/vault"), {})
        assert backlinks == {}


class TestObsidianVaultReader:
    @pytest.fixture
    def vault(self, tmp_path):
        """Create a minimal Obsidian vault for testing."""
        # Note A links to Note B
        (tmp_path / "Note A.md").write_text(
            """---
title: Note A
tags: [physics]
---
# Note A

This is Note A. It links to [[Note B]] and [[Note C|C alias]].

Some more content about #quantum mechanics.
""",
            encoding="utf-8",
        )
        # Note B links to Note C
        (tmp_path / "Note B.md").write_text(
            """# Note B

This is Note B. See also [[Note C#section]].
""",
            encoding="utf-8",
        )
        # Note C has no outgoing links
        (tmp_path / "Note C.md").write_text(
            """---
tags: math, logic
---
# Note C

This is Note C. No outgoing wikilinks.
""",
            encoding="utf-8",
        )
        # Hidden file should be skipped
        hidden_dir = tmp_path / ".obsidian"
        hidden_dir.mkdir()
        (hidden_dir / "config.md").write_text("Hidden config", encoding="utf-8")

        return tmp_path

    def test_note_count(self, vault):
        reader = ObsidianVaultReader(vault)
        assert reader.note_count == 3

    def test_backlink_map(self, vault):
        reader = ObsidianVaultReader(vault)
        backlinks = reader.backlink_map
        # Note B is linked from Note A (source keys are now relative paths)
        assert "note b" in backlinks
        assert any("Note A" in src for src in backlinks["note b"])
        # Note C is linked from both Note A and Note B
        assert "note c" in backlinks
        assert len(backlinks["note c"]) == 2

    def test_chunks_have_metadata(self, vault):
        reader = ObsidianVaultReader(vault)
        chunks = list(reader.iter_chunks())
        assert len(chunks) > 0

        # Check first chunk from Note A
        note_a_chunks = [c for c in chunks if c["metadata"]["note_name"] == "Note A"]
        assert len(note_a_chunks) > 0
        meta = note_a_chunks[0]["metadata"]
        assert meta["source"] == "obsidian"
        assert "Note B" in meta["wikilinks"]
        assert "Note C" in meta["wikilinks"]
        assert "physics" in meta["tags"] or "quantum" in meta["tags"]

    def test_chunks_have_backlinks(self, vault):
        reader = ObsidianVaultReader(vault)
        chunks = list(reader.iter_chunks())

        # Note C chunks should have backlinks from A and B
        note_c_chunks = [c for c in chunks if c["metadata"]["note_name"] == "Note C"]
        assert len(note_c_chunks) > 0
        backlinks = note_c_chunks[0]["metadata"]["backlinks"]
        assert len(backlinks) == 2

    def test_hidden_files_excluded(self, vault):
        reader = ObsidianVaultReader(vault)
        chunks = list(reader.iter_chunks())
        note_names = {c["metadata"]["note_name"] for c in chunks}
        assert "config" not in note_names

    def test_hidden_files_included(self, vault):
        reader = ObsidianVaultReader(vault, include_hidden=True)
        chunks = list(reader.iter_chunks())
        note_names = {c["metadata"]["note_name"] for c in chunks}
        assert "config" in note_names

    def test_invalid_vault_path(self):
        with pytest.raises(ValueError, match="not a directory"):
            ObsidianVaultReader("/nonexistent/path")


class TestCliObsidianFlag:
    def test_obsidian_flag_parsed(self):
        from leann.cli import LeannCLI

        cli = LeannCLI()
        parser = cli.create_parser()
        args = parser.parse_args(["build", "my-vault", "--docs", "/path/to/vault", "--obsidian"])
        assert args.obsidian is True

    def test_obsidian_flag_default_false(self):
        from leann.cli import LeannCLI

        cli = LeannCLI()
        parser = cli.create_parser()
        args = parser.parse_args(["build", "my-docs", "--docs", "/path/to/docs"])
        assert args.obsidian is False
