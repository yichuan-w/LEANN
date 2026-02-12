"""
Tests for the `leann reindex` CLI command and ``LeannBuilder.from_meta``.
"""

import json
import os
import tempfile

import pytest


def test_from_meta_creates_builder():
    """Unit test: LeannBuilder.from_meta reads a meta.json and returns a
    correctly configured builder instance."""
    from leann.api import LeannBuilder

    meta = {
        "version": "1.0",
        "backend_name": "hnsw",
        "embedding_model": "facebook/contriever",
        "dimensions": 768,
        "backend_kwargs": {
            "graph_degree": 32,
            "complexity": 64,
            "is_compact": True,
            "is_recompute": True,
            "num_threads": 1,
        },
        "embedding_mode": "sentence-transformers",
        "embedding_options": {"prompt_template": "query: "},
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".meta.json", delete=False
    ) as f:
        json.dump(meta, f)
        meta_path = f.name

    try:
        builder = LeannBuilder.from_meta(meta_path)

        assert builder.backend_name == "hnsw"
        assert builder.embedding_model == "facebook/contriever"
        assert builder.embedding_mode == "sentence-transformers"
        assert builder.embedding_options == {"prompt_template": "query: "}
        assert builder.backend_kwargs.get("graph_degree") == 32
        assert builder.backend_kwargs.get("complexity") == 64
        assert builder.backend_kwargs.get("is_compact") is True
        assert builder.backend_kwargs.get("is_recompute") is True
        assert builder.backend_kwargs.get("num_threads") == 1
        # Builder should start with no chunks
        assert builder.chunks == []
    finally:
        os.unlink(meta_path)


def test_from_meta_defaults_embedding_mode():
    """from_meta should default embedding_mode to 'sentence-transformers'
    when the field is missing from the meta file."""
    from leann.api import LeannBuilder

    meta = {
        "version": "1.0",
        "backend_name": "hnsw",
        "embedding_model": "facebook/contriever",
        "dimensions": 768,
        "backend_kwargs": {},
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".meta.json", delete=False
    ) as f:
        json.dump(meta, f)
        meta_path = f.name

    try:
        builder = LeannBuilder.from_meta(meta_path)
        assert builder.embedding_mode == "sentence-transformers"
        # No embedding_options in meta => builder should have empty dict
        assert builder.embedding_options == {}
    finally:
        os.unlink(meta_path)


def test_reindex_cli_exists():
    """The 'reindex' subcommand should be registered in the CLI parser."""
    from leann.cli import LeannCLI

    cli = LeannCLI()
    parser = cli.create_parser()

    # Parse a minimal reindex invocation to verify the subcommand exists
    args = parser.parse_args(["reindex", "my-index", "--docs", "./src"])
    assert args.command == "reindex"
    assert args.index_name == "my-index"
    assert args.docs == ["./src"]


def test_reindex_cli_multiple_docs():
    """The 'reindex' subcommand should accept multiple --docs directories."""
    from leann.cli import LeannCLI

    cli = LeannCLI()
    parser = cli.create_parser()

    args = parser.parse_args(
        ["reindex", "my-index", "--docs", "./src", "./tests", "./config"]
    )
    assert args.command == "reindex"
    assert args.index_name == "my-index"
    assert args.docs == ["./src", "./tests", "./config"]
