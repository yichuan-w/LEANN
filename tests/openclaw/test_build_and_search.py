"""
End-to-end test: build a LEANN index on OpenClaw-style memory files,
then search with --json and verify results.

Requires the embedding model (~90 MB download on first run).
Marked 'slow' — skip with: pytest -m "not slow"
"""

import asyncio
import json
import os

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(os.environ.get("CI") == "true", reason="Skip in CI — needs model download"),
]


@pytest.fixture
def cli_instance(leann_index_dir):
    """Create a LeannCLI instance pointing at a temporary index directory."""
    from leann.cli import LeannCLI

    cli = LeannCLI()
    cli.indexes_dir = leann_index_dir
    cli.indexes_dir.mkdir(parents=True, exist_ok=True)
    return cli


BUILD_ARGV_TEMPLATE = [
    "build",
    "openclaw-memory",
    "--docs",
    "{docs_dir}",
    "--backend-name",
    "hnsw",
    "--no-compact",
    "--embedding-model",
    "all-MiniLM-L6-v2",
    "--embedding-mode",
    "sentence-transformers",
]


def _parse_args(cli, argv: list[str]):
    parser = cli.create_parser()
    return parser.parse_args(argv)


def _build_argv(docs_dir: str) -> list[str]:
    return [a.format(docs_dir=docs_dir) for a in BUILD_ARGV_TEMPLATE]


def test_build_memory_index(cli_instance, memory_fixtures):
    """Build a non-compact HNSW index on the memory fixtures."""
    args = _parse_args(cli_instance, _build_argv(str(memory_fixtures)))
    asyncio.get_event_loop().run_until_complete(cli_instance.build_index(args))

    index_dir = cli_instance.indexes_dir / "openclaw-memory"
    assert index_dir.exists(), "Index directory was not created"

    meta_files = list(index_dir.glob("*.meta.json"))
    assert len(meta_files) >= 1, "Missing meta.json"

    passages_files = list(index_dir.glob("*.passages.jsonl"))
    assert len(passages_files) >= 1, "Missing passages.jsonl"


def _build_and_search(cli_instance, memory_fixtures, capsys, query, top_k=3):
    """Helper: build index, clear capsys, search, return parsed JSON results."""
    build_args = _parse_args(cli_instance, _build_argv(str(memory_fixtures)))
    asyncio.get_event_loop().run_until_complete(cli_instance.build_index(build_args))
    capsys.readouterr()  # discard build output

    search_args = _parse_args(
        cli_instance,
        ["search", "openclaw-memory", query, "--top-k", str(top_k), "--json", "--non-interactive"],
    )
    asyncio.get_event_loop().run_until_complete(cli_instance.search_documents(search_args))

    captured = capsys.readouterr()
    assert captured.out.strip(), f"Search produced no stdout (stderr: {captured.err[:200]})"
    return json.loads(captured.out)


def test_search_returns_json(cli_instance, memory_fixtures, capsys):
    """Build then search with --json; output must be valid JSON with results."""
    results = _build_and_search(cli_instance, memory_fixtures, capsys, "gRPC migration")
    assert isinstance(results, list)
    assert len(results) > 0
    assert all({"id", "score", "text", "metadata"} <= set(r.keys()) for r in results)


def test_search_relevance(cli_instance, memory_fixtures, capsys):
    """Top result for 'payment gateway' should come from the Feb 20 memory."""
    results = _build_and_search(
        cli_instance, memory_fixtures, capsys, "payment gateway timeout fix"
    )
    top_text = results[0]["text"].lower()
    assert "payment" in top_text or "gateway" in top_text or "hotfix" in top_text


def test_idempotent_rebuild(cli_instance, memory_fixtures, capsys):
    """Running build twice should detect no changes on the second run."""
    args1 = _parse_args(cli_instance, _build_argv(str(memory_fixtures)))
    asyncio.get_event_loop().run_until_complete(cli_instance.build_index(args1))
    capsys.readouterr()  # discard first build output

    args2 = _parse_args(cli_instance, _build_argv(str(memory_fixtures)))
    asyncio.get_event_loop().run_until_complete(cli_instance.build_index(args2))

    captured = capsys.readouterr()
    assert "up to date" in captured.out.lower() or "no changes" in captured.out.lower()


def test_incremental_add(cli_instance, memory_fixtures, capsys):
    """Adding a new file should trigger incremental update, not full rebuild."""
    args1 = _parse_args(cli_instance, _build_argv(str(memory_fixtures)))
    asyncio.get_event_loop().run_until_complete(cli_instance.build_index(args1))
    capsys.readouterr()  # discard first build output

    new_file = memory_fixtures / "memory" / "2026-02-26.md"
    new_file.write_text(
        "# 2026-02-26 Thursday\n\n- Tested LEANN integration with OpenClaw.\n"
        "- The semantic search returned highly relevant memory results.\n",
        encoding="utf-8",
    )

    args2 = _parse_args(cli_instance, _build_argv(str(memory_fixtures)))
    asyncio.get_event_loop().run_until_complete(cli_instance.build_index(args2))
    capsys.readouterr()  # discard rebuild output

    search_args = _parse_args(
        cli_instance,
        [
            "search",
            "openclaw-memory",
            "LEANN OpenClaw integration test",
            "--top-k",
            "3",
            "--json",
            "--non-interactive",
        ],
    )
    asyncio.get_event_loop().run_until_complete(cli_instance.search_documents(search_args))

    captured = capsys.readouterr()
    assert captured.out.strip(), f"Search produced no stdout (stderr: {captured.err[:200]})"
    results = json.loads(captured.out)
    assert any("leann" in r["text"].lower() or "openclaw" in r["text"].lower() for r in results)
