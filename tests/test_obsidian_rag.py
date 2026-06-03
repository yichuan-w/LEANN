import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps import obsidian_rag
from apps.obsidian_rag import (
    ObsidianRAG,
    extract_obsidian_tags,
    extract_wiki_targets,
    iter_obsidian_notes,
    load_obsidian_documents,
    parse_obsidian_note,
    split_frontmatter,
)


def test_split_frontmatter_parses_common_obsidian_shapes():
    text = """---
title: Alpha
tags:
  - research
  - "#local-first"
aliases: [Alpha Note, "Project Alpha"]
cssclasses: knowledge, evergreen
---
# Alpha

Body text.
"""

    frontmatter, body = split_frontmatter(text)

    assert body.startswith("# Alpha")
    assert frontmatter == {
        "title": "Alpha",
        "tags": ["research", "#local-first"],
        "aliases": ["Alpha Note", "Project Alpha"],
        "cssclasses": "knowledge, evergreen",
    }


def test_extract_obsidian_links_embeds_and_tags():
    body = """
Link to [[Beta Note|beta]] and [[Folder/Gamma#Section]].
Embed ![[diagram.png]] and keep #project/local-first.
Ignore a URL fragment like https://example.com/#section.
"""

    assert extract_wiki_targets(body, embeds=False) == ["Beta Note", "Folder/Gamma#Section"]
    assert extract_wiki_targets(body, embeds=True) == ["diagram.png"]
    assert extract_obsidian_tags(body) == ["project/local-first"]


def test_iter_obsidian_notes_skips_internal_and_hidden_dirs(tmp_path):
    (tmp_path / "visible.md").write_text("visible", encoding="utf-8")
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "workspace.md").write_text("internal", encoding="utf-8")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "secret.md").write_text("hidden", encoding="utf-8")

    assert iter_obsidian_notes(tmp_path) == [tmp_path / "visible.md"]
    assert iter_obsidian_notes(tmp_path, include_hidden=True) == [
        tmp_path / ".hidden" / "secret.md",
        tmp_path / "visible.md",
    ]


def test_load_obsidian_documents_preserves_vault_metadata(tmp_path):
    note = tmp_path / "Projects" / "Alpha.md"
    note.parent.mkdir()
    note.write_text(
        """---
tags: [research, "#ai"]
aliases:
  - Alpha Note
---
# Alpha

Link to [[Beta Note|beta]] and ![[diagram.png]] #project/x
""",
        encoding="utf-8",
    )

    documents = load_obsidian_documents(tmp_path)

    assert len(documents) == 1
    document = documents[0]
    metadata = document.metadata
    assert document.text.startswith("# Alpha")
    assert metadata["obsidian_note"] is True
    assert metadata["obsidian_vault_path"] == str(tmp_path.resolve())
    assert metadata["obsidian_relative_path"] == "Projects/Alpha.md"
    assert metadata["obsidian_title"] == "Alpha"
    assert metadata["obsidian_aliases"] == ["Alpha Note"]
    assert metadata["obsidian_tags"] == ["ai", "project/x", "research"]
    assert metadata["obsidian_links"] == ["Beta Note"]
    assert metadata["obsidian_embeds"] == ["diagram.png"]


@pytest.mark.parametrize(
    ("frontmatter", "expected_title"),
    [
        ("title: Custom Title", "Custom Title"),
        ("", "Untitled"),
    ],
)
def test_parse_obsidian_note_title_fallback(tmp_path, frontmatter, expected_title):
    note = tmp_path / "Untitled.md"
    note.write_text(f"---\n{frontmatter}\n---\nBody", encoding="utf-8")

    _, metadata = parse_obsidian_note(note, tmp_path)

    assert metadata["obsidian_title"] == expected_title


def test_obsidian_rag_load_data_uses_chunking(monkeypatch, tmp_path):
    (tmp_path / "Alpha.md").write_text("# Alpha\n\n[[Beta]] #tag", encoding="utf-8")
    captured = {}

    def fake_create_text_chunks(documents, *, chunk_size, chunk_overlap, use_ast_chunking):
        captured["documents"] = documents
        captured["chunk_size"] = chunk_size
        captured["chunk_overlap"] = chunk_overlap
        captured["use_ast_chunking"] = use_ast_chunking
        return [
            {
                "text": documents[0].text,
                "metadata": documents[0].metadata,
            }
        ]

    monkeypatch.setattr(obsidian_rag, "create_text_chunks", fake_create_text_chunks)

    args = SimpleNamespace(
        vault_dir=str(tmp_path),
        include_hidden=False,
        chunk_size=256,
        chunk_overlap=64,
        max_items=-1,
    )
    chunks = asyncio.run(ObsidianRAG().load_data(args))

    assert len(chunks) == 1
    assert captured["chunk_size"] == 256
    assert captured["chunk_overlap"] == 64
    assert captured["use_ast_chunking"] is False
    assert chunks[0]["metadata"]["obsidian_links"] == ["Beta"]
    assert chunks[0]["metadata"]["obsidian_tags"] == ["tag"]


def test_obsidian_rag_parser_accepts_vault_options(tmp_path):
    args = ObsidianRAG().parser.parse_args(
        [
            "--vault-dir",
            str(tmp_path),
            "--include-hidden",
            "--chunk-size",
            "128",
            "--chunk-overlap",
            "16",
        ]
    )

    assert args.vault_dir == str(tmp_path)
    assert args.include_hidden is True
    assert args.chunk_size == 128
    assert args.chunk_overlap == 16
